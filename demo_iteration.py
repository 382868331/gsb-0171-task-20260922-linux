"""演示: 导频短暂消失时 PLL 的 holdover/reacquire 状态机。

场景: 已知真值的合成导频, 包含一次幅度缺口、缓慢频率漂移和一次假恢复。
直接运行: python demo_iteration.py (纯标准库, 确定性计算, 约 1 秒内完成)。
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

PARAMS = PLLParams(kp=0.12, ki=0.008, wmax=1.2, amin=0.2, eps=0.03,
                   lock_count=40, holdover_enter=8, holdover_window=32,
                   reacquire_step=0.08, freq_eps=0.06, reacquire_count=30)


def synth(n, amp, omega, phase0=0.6):
    """确定性合成导频; 返回 (samples, 每样本真实相位, 每样本真实频率)。"""
    samples, phases, omegas = [], [], []
    phi = phase0
    for k in range(n):
        a = amp(k) if callable(amp) else amp
        w = omega(k) if callable(omega) else omega
        samples.append(complex(a * math.cos(phi), a * math.sin(phi)))
        phases.append(phi)
        omegas.append(w)
        phi += w
    return samples, phases, omegas


def build_scenario():
    """锁定 300 + 缺口 25 + 假恢复 3 + 再缺口 12 + 恢复 400, 全程缓慢漂移。"""
    segs = [(300, 1.0), (25, 0.0), (3, 1.0), (12, 0.0), (400, 1.0)]
    amps = [a for n, a in segs for _ in range(n)]
    drift = 1e-4  # 频率缓慢漂移: 0.30 -> 约 0.374 rad/sample
    return synth(len(amps), lambda k: amps[k], lambda k: 0.30 + drift * k)


def demo_holdover_reacquire():
    print("=== 1. 缺口 + 漂移 + 假恢复: tracking/holdover/reacquire ===")
    samples, phases, omegas = build_scenario()
    pll = PLL(PARAMS)
    diags = pll.run(samples).diagnostics

    # 状态迁移概览
    print("状态迁移(样本区间):")
    prev, start = None, 0
    transitions = []
    for d in diags:
        if d.state != prev:
            if prev is not None:
                transitions.append((start, d.index - 1, prev))
            prev, start = d.state, d.index
    transitions.append((start, diags[-1].index, prev))
    for lo, hi, st in transitions:
        print(f"  [{lo:>3}, {hi:>3}] {st}")

    # 保持期间误差: 诚实报告——PLL 在参考丢失后无法获知真实相位,
    # 以下是与合成真值事后对比的外推误差。
    hold = [d for d in diags if d.state == STATE_HOLDOVER]
    errs = [abs(wrap_phase(d.theta - phases[d.index])) for d in hold]
    print(f"holdover 样本数: {len(hold)}, 冻结频率 omega_hold = "
          f"{hold[0].omega:.6f} rad/sample")
    print(f"保持期间相位外推误差(事后对照真值): 进入时 {errs[0]:.4f} rad, "
          f"峰值 {max(errs):.4f} rad, 离开时 {errs[-1]:.4f} rad")
    print("(参考丢失期间 PLL 只能按冻结频率外推, 该误差真实存在, 不掩盖)")

    # 重捕获: 有界每样本校正
    reacq = [d for d in diags if d.state == STATE_REACQUIRE]
    worst = max(abs(wrap_phase(d.theta - diags[d.index - 1].theta - d.omega))
                for d in reacq)
    print(f"reacquire 样本数: {len(reacq)}, 最大每样本校正量 "
          f"{worst:.4f} rad <= 上限 {PARAMS.reacquire_step} rad")
    assert worst <= PARAMS.reacquire_step + 1e-12
    assert not any(d.locked for d in reacq), "reacquire 期间不应标记锁定"

    print(f"最终: state={pll.state}, locked={pll.locked}, "
          f"omega={pll.omega:.6f} (真值 {omegas[-1]:.6f}), "
          f"失锁样本={pll.lost_samples}")
    assert pll.locked and pll.state == STATE_TRACKING
    assert abs(pll.omega - omegas[-1]) < 5e-3

    # 分块与一次性输入逐点一致
    chunked = PLL(PARAMS)
    cds = []
    for lo, hi in ((0, 7), (7, 311), (311, 329), (329, len(samples))):
        cds.extend(chunked.run(samples[lo:hi]).diagnostics)
    same = tuple(cds) == tuple(diags)
    print(f"任意分块(跨缺口/假恢复)与一次性输入逐点一致: {same}")
    assert same
    print()


def demo_threshold_tie():
    print("=== 2. 边界情况: 幅度恰好等于门限 ===")
    # 契约: 无效判定是 amp < amin, 故 amp == amin 的样本按有效处理。
    pll = PLL(PLLParams(kp=0.2, ki=0.02, wmax=1.0, amin=0.5, eps=0.05,
                        lock_count=5, holdover_enter=2))
    d = pll.step(complex(0.5, 0.0))  # |z| == amin
    print(f"|z| = 0.5 == amin: valid={d.valid}, state={d.state}, "
          f"lost_samples={pll.lost_samples}")
    assert d.valid and pll.lost_samples == 0
    d = pll.step(complex(0.5 - 1e-15, 0.0))  # |z| 略低于 amin
    print(f"|z| = 0.5 - 1e-15 < amin: valid={d.valid}, "
          f"lost_samples={pll.lost_samples}")
    assert not d.valid and pll.lost_samples == 1
    print("门限相等走有效一侧(< 才算低幅), 契约与测试均已固定。")


def main():
    demo_holdover_reacquire()
    demo_threshold_tie()
    print("demo_iteration 完成。")


if __name__ == "__main__":
    main()
