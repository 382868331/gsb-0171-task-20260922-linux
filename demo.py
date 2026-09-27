"""演示: 二阶离散 PLL 的捕获过程与非法输入拒绝。

直接运行: python demo.py (纯标准库, 确定性计算, 数秒内完成)。
"""

import math

from pll_capture import (
    DualReferencePLL,
    PLL,
    PLLInputError,
    PLLParamError,
    PLLParams,
)


def make_pilot(amplitude, phase, offset, n):
    """确定性合成复数导频: z_k = A * exp(j*(phase + offset*k))。"""
    return [
        complex(amplitude * math.cos(phase + offset * k),
                amplitude * math.sin(phase + offset * k))
        for k in range(n)
    ]


def demo_acquisition():
    print("=== 1. 正常捕获过程 ===")
    sample_rate = 48_000.0
    true_offset = 0.35          # rad/sample
    true_phase = -2.4           # rad
    params = PLLParams(kp=0.12, ki=0.008, wmax=1.2, amin=0.2,
                       eps=0.03, lock_count=40)
    samples = make_pilot(1.0, true_phase, true_offset, 600)
    pll = PLL(params)
    result = pll.run(samples)

    print(f"真值: 频偏 {true_offset} rad/sample "
          f"({true_offset * sample_rate / (2 * math.pi):.1f} Hz @ {sample_rate:.0f} Hz), "
          f"初相 {true_phase} rad")
    print(f"{'idx':>4} {'error(rad)':>11} {'omega':>9} {'theta':>9} {'locked':>6}")
    for d in result.diagnostics[::60]:
        print(f"{d.index:>4} {d.error:>11.5f} {d.omega:>9.5f} "
              f"{d.theta:>9.5f} {str(d.locked):>6}")
    last = result.diagnostics[-1]
    print(f"{last.index:>4} {last.error:>11.5f} {last.omega:>9.5f} "
          f"{last.theta:>9.5f} {str(last.locked):>6}")

    est_hz = result.frequency_offset(sample_rate)
    print(f"估计: 频偏 {result.omega:.6f} rad/sample ({est_hz:.1f} Hz), "
          f"locked={result.locked}, 失锁样本={result.lost_samples}")
    assert result.locked, "600 个样本内应完成捕获"
    assert abs(result.omega - true_offset) < 1e-6, "频偏估计应收敛到真值"

    # 任意分块处理结果一致
    chunked = PLL(params)
    for lo, hi in ((0, 7), (7, 250), (250, 600)):
        chunked.run(samples[lo:hi])
    consistent = (chunked.omega == result.omega
                  and chunked.theta == result.theta
                  and chunked.locked == result.locked)
    print(f"分块(7/243/350)与整批结果逐位一致: {consistent}")
    assert consistent
    print()


def demo_rejection():
    print("=== 2. 非法输入实际触发拒绝 ===")
    try:
        PLLParams(kp=1.5, ki=0.008, wmax=1.2, amin=0.2,
                  eps=0.03, lock_count=40)
    except PLLParamError as e:
        print(f"超范围参数被拒绝 (PLLParamError): {e}")

    pll = PLL(PLLParams(kp=0.12, ki=0.008, wmax=1.2, amin=0.2,
                        eps=0.03, lock_count=40))
    try:
        pll.run([1 + 0j, complex(float("nan"), 0.0)])
    except PLLInputError as e:
        print(f"NaN 样本被拒绝 (PLLInputError): {e}")
    print(f"拒绝后状态未变: processed={pll.processed}, omega={pll.omega}")


def demo_dual_reference():
    print("=== 3. 双参考无扰动切换 ===")
    params = PLLParams(kp=0.2, ki=0.02, wmax=1.0, amin=0.5,
                       eps=0.05, lock_count=10)
    d = DualReferencePLL(params)
    # 参考 0:恒相位 0.3;参考 1:恒相位 -1.6(同采样速率,成对喂入)。
    warm = [(make_pilot(1.0, 0.3, 0.0, 120)[k], 0.1 + 0j)
            for k in range(120)]
    d.run(warm)
    print(f"预热后: selected={d.selected}, mode={d.mode}, "
          f"输出相位={d.theta:.6f} rad")

    # 请求切换:目标参考 1 先独立跟踪,稳定前继续输出原参考。
    d.request_switch(1)
    z0 = make_pilot(1.0, 0.3, 0.0, 60)
    z1 = make_pilot(1.0, -1.6, 0.0, 60)
    result = d.run(list(zip(z0, z1)))
    hi = next(i for i, x in enumerate(result.diagnostics) if x.handoff)
    print(f"切换请求后目标独立跟踪 {hi} 个样本才确认稳定;")
    print(f"交接前输出相位={result.diagnostics[hi - 1].theta:.12f}")
    print(f"交接样本相位  ={result.diagnostics[hi].theta:.12f} "
          f"(active={result.diagnostics[hi].active}, 跳变=0)")
    print(f"目标裸相位={d.trackers[1].theta:.6f}, "
          f"对齐偏置={d.offset(1):.6f} rad")
    assert result.diagnostics[hi].theta == result.diagnostics[hi - 1].theta

    # 切换窗口内原参考失效,且目标尚未稳定:进入 holdover,明确不可信。
    d2 = DualReferencePLL(params)
    d2.run(warm)
    d2.request_switch(1)
    gap = 6  # 仅 6 个样本,目标尚不足 lock_count
    d2.run([(0j, z) for z in make_pilot(1.0, -1.6, 0.0, gap)])
    print(f"原参考失效且目标未稳: mode={d2.mode}, trusted={d2.trusted}, "
          f"holdover_samples={d2.holdover_samples}")
    assert d2.mode == "holdover" and not d2.trusted

    # 目标继续独立跟踪并稳定:从保持态无扰动恢复,交接样本不跳变。
    frozen = d2.theta
    resume = d2.run([(0j, z) for z in make_pilot(1.0, -1.6, 0.0, 60)])
    handoff_diag = next(x for x in resume.diagnostics if x.handoff)
    handoff_theta = handoff_diag.theta
    d2.run([(0j, z) for z in make_pilot(1.0, -1.6, 0.0, 300)])
    print(f"目标稳定后恢复: selected={d2.selected}, mode={d2.mode}, "
          f"trusted={d2.trusted}")
    print(f"保持前相位={frozen:.12f}, 交接样本相位={handoff_theta:.12f}, "
          f"继续收敛后={d2.theta:.12f}")
    assert d2.trusted and d2.selected == 1
    # 交接样本严格不跳变(偏置即时对齐);之后随目标指数收敛到同相物理输出。
    assert handoff_theta == frozen
    # 偏置只反映两参考在交接瞬间的真实相位差(锁定残差 < eps),不含人为累积。
    assert abs(d2.offset(1) - 1.9) < params.eps

    # 来回切换:每次按绝对相位重新计算一次性偏置,不累积人为偏移。
    before = d.theta
    d.request_switch(0)
    d.run(list(zip(z0, z1)))
    d.request_switch(1)
    last = d.run(list(zip(z0, z1)))
    handoff_diag = next(x for x in reversed(last.diagnostics) if x.handoff)
    print(f"0->1->0->1 来回切换后输出相位={d.theta:.9f} "
          f"(首次切到 1 后={before:.9f}), 无累积偏移")
    # 差异仅为各次交接瞬间锁定残差(< eps),偏置本身不叠加。
    assert abs(d.theta - before) < params.eps
    # 每次偏置都仅由当次两路绝对相位差重算(单次 wrap, [-pi, pi))。
    assert abs(handoff_diag.offset) <= math.pi
    assert abs(d.offset(1) - handoff_diag.offset) < 1e-12
    print()


def main():
    demo_acquisition()
    demo_dual_reference()
    demo_rejection()
    print("demo 完成。")


if __name__ == "__main__":
    main()
