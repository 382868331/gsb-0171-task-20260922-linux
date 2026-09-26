"""演示: 导频短暂消失时 PLL 的 tracking / holdover / reacquire 状态机。

直接运行: python demo_iteration.py
纯标准库、确定性合成信号、真实逐样本计算(无 sleep、无预录), 约 0.1 秒完成。

场景: 已知相位的合成导频 -> 缺口(低幅) -> 一次假恢复 -> 再次缺口 ->
带缓慢频率漂移的真恢复。展示:
1. holdover 期间冻结中位频率外推相位, 并如实报告外推相位与真实相位的误差
   (参考丢失后 PLL 无法知道真实相位, 只能报告保持误差);
2. reacquire 期间每样本相位校正有界, 不瞬间重置;
3. 连续满足相位+频率双条件才重新锁定;
4. 任意分块与一次性输入逐点一致;
5. 边界情况: 幅度恰好等于门限 amin 判为有效样本。
"""

import math

from pll_capture import (
    PLL,
    PLLParams,
    STATE_HOLDOVER,
    STATE_REACQUIRE,
    STATE_TRACKING,
    wrap_phase,
)

PARAMS = PLLParams(kp=0.15, ki=0.01, wmax=1.0, amin=0.3, eps=0.04,
                   lock_count=20, holdover_samples=8, median_window=11,
                   max_phase_step=0.05, freq_eps=0.01, reacquire_count=25)


def segment(amplitude, phase0, offset, n, drift=0.0, k0=0):
    """确定性导频段; 返回 (样本, 每样本真实相位)。"""
    samples, truth = [], []
    for i in range(n):
        k = k0 + i
        phi = phase0 + offset * k + 0.5 * drift * k * k
        samples.append(complex(amplitude * math.cos(phi),
                               amplitude * math.sin(phi)))
        truth.append(wrap_phase(phi))
    return samples, truth


def build_scenario():
    """拼接完整场景, 返回 (samples, true_phases, 分段标签边界)。"""
    parts = [
        ("跟踪段(锁定)",      segment(1.0, 0.5, 0.18, 400)),
        ("缺口1(低幅噪声)",   segment(0.05, 0.5, 0.18, 30, k0=400)),
        ("假恢复(相位错误)",  segment(0.9, 2.1, 0.18, 6, k0=430)),
        ("缺口2",            segment(0.04, 0.5, 0.18, 25, k0=436)),
        ("真恢复(频率缓漂)",  segment(1.0, 0.5, 0.18, 400, drift=0.0002, k0=461)),
    ]
    samples, truth, marks = [], [], []
    for label, (s, t) in parts:
        marks.append((len(samples), label))
        samples.extend(s)
        truth.extend(t)
    return samples, truth, marks


def main():
    samples, truth, marks = build_scenario()
    pll = PLL(PARAMS)
    result = pll.run(samples)
    diags = result.diagnostics

    print("=== 1. 状态机轨迹(分段) ===")
    for start, label in marks:
        d = diags[start]
        print(f"  idx={start:>4} 进入[{label}] 状态={d.state}")
    counts = {s: sum(1 for d in diags if d.state == s)
              for s in (STATE_TRACKING, STATE_HOLDOVER, STATE_REACQUIRE)}
    print(f"  各状态样本数: tracking={counts[STATE_TRACKING]}, "
          f"holdover={counts[STATE_HOLDOVER]}, "
          f"reacquire={counts[STATE_REACQUIRE]}")
    assert counts[STATE_HOLDOVER] > 0 and counts[STATE_REACQUIRE] > 0

    print("\n=== 2. holdover 保持误差(诚实口径) ===")
    print("  参考丢失后 PLL 无法知道真实相位, 只能用冻结的中位频率外推;")
    print("  以下报告的是保持期间外推相位与真实相位的偏差, 而非'仍跟踪真值'。")
    hold_idx = [d.index for d in diags if d.state == STATE_HOLDOVER]
    for i in (hold_idx[0], hold_idx[len(hold_idx) // 2], hold_idx[-1]):
        err = wrap_phase(diags[i].theta - truth[i])
        print(f"  idx={i:>4} 外推相位误差 = {err:+.5f} rad "
              f"(omega_hold={diags[i].omega:.6f})")

    print("\n=== 3. reacquire 有界校正与重锁定 ===")
    max_step = 0.0
    for prev, cur in zip(diags, diags[1:]):
        if cur.state == STATE_REACQUIRE and cur.valid:
            step = abs(wrap_phase(cur.theta - prev.theta) - cur.omega)
            max_step = max(max_step, step)
    print(f"  reacquire 每样本相位校正上限 max_phase_step = "
          f"{PARAMS.max_phase_step}")
    print(f"  实际最大每样本校正 = {max_step:.6f} rad "
          f"(<= 上限: {max_step <= PARAMS.max_phase_step + 1e-12})")
    assert max_step <= PARAMS.max_phase_step + 1e-12
    # 假恢复未重锁定(不足 reacquire_count 且双条件不满足)
    false_start = marks[2][0]
    false_states = [diags[false_start + i].state for i in range(6)]
    print(f"  假恢复 6 个样本的状态: {set(false_states)}, "
          f"期间 locked={any(diags[false_start + i].locked for i in range(6))}")
    assert not any(diags[false_start + i].locked for i in range(6))
    print(f"  真恢复后重新锁定: locked={result.locked}, "
          f"state={result.state}, "
          f"omega={result.omega:.6f} rad/sample")
    assert result.locked and result.state == STATE_TRACKING

    print("\n=== 4. 分块与一次性输入逐点一致 ===")
    chunked = PLL(PARAMS)
    cdiags = []
    for lo, hi in ((0, 1), (1, 30), (30, 318), (318, 337), (337, 350),
                   (350, len(samples))):
        cdiags.extend(chunked.run(samples[lo:hi]).diagnostics)
    identical = (tuple(cdiags) == diags and chunked.omega == result.omega
                 and chunked.theta == result.theta
                 and chunked.state == result.state)
    print(f"  7 个任意 chunk 与整批处理逐点一致: {identical}")
    assert identical

    print("\n=== 5. 边界情况: 幅度恰好等于门限 ===")
    edge = PLL(PARAMS)
    edge.run(samples[:400])
    edge.run([0.02 + 0j] * 20)          # 进入 holdover
    d = edge.step(complex(PARAMS.amin, 0.0))   # amp == amin
    print(f"  amp == amin(={PARAMS.amin}) 判为有效样本: valid={d.valid}, "
          f"状态 {STATE_HOLDOVER} -> {d.state}(不冻结, 走有效这一边)")
    assert d.valid and d.state == STATE_REACQUIRE

    print("\ndemo_iteration 完成, 全部断言通过。")


if __name__ == "__main__":
    main()
