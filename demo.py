"""演示: 二阶离散 PLL 的捕获过程与非法输入拒绝。

直接运行: python demo.py (纯标准库, 确定性计算, 数秒内完成)。
"""

import math

from pll_capture import (
    PLL,
    PLLInputError,
    PLLParamError,
    PLLParams,
    ReferenceSwitcher,
    wrap_phase,
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


def demo_reference_switch():
    print("=== 3. 双参考切换(相位连续交接) ===")
    offset = 0.3                       # 两路参考采样速率相同, 频偏一致
    phase_a, phase_b = 0.5, -1.1
    ref_a = make_pilot(1.0, phase_a, offset, 2200)
    ref_b = make_pilot(1.0, phase_b, offset, 2200)
    params = PLLParams(kp=0.12, ki=0.008, wmax=1.2, amin=0.2,
                       eps=0.03, lock_count=40)
    sw = ReferenceSwitcher(params)
    sw.run(ref_a[:400])
    print(f"原参考 A 已锁定: theta={sw.theta:+.6f}, omega={sw.omega:.6f}, "
          f"status={sw.status}")

    # 目标未稳定: 候选只喂 10 个样本, 交接被拒绝, 继续跟踪原参考
    sw.begin_switch()
    sw.run_candidate(ref_b[400:410])
    accepted = sw.complete_switch()
    print(f"候选仅 10 样本(未稳定): complete_switch -> {accepted}, "
          f"仍跟踪原参考, theta={sw.theta:+.6f}")
    assert not accepted and sw.switching

    # 候选继续独立跟踪新参考直至稳定, 然后交接
    sw.run_candidate(ref_b[410:700])
    theta_before = sw.theta
    accepted = sw.complete_switch()
    print(f"候选稳定后: complete_switch -> {accepted}, "
          f"交接前后 theta 差={abs(sw.theta - theta_before):.2e} (无跳变)")
    assert accepted
    assert sw.theta == theta_before

    # 交接后跟踪新参考 B, 相位逐渐收敛到 B 的真实相位
    result = sw.run(ref_b[700:1100])
    true_b = wrap_phase(phase_b + offset * 1100)
    print(f"切换后跟踪 B: locked={result.locked}, "
          f"theta={sw.theta:+.6f} (B 真值 {true_b:+.6f}), "
          f"omega={sw.omega:.6f}")
    assert result.locked
    assert abs(wrap_phase(sw.theta - true_b)) < 1e-3

    # 来回切换不累积人为相位偏移
    cursor = 1100
    for ref, phase in ((ref_a, phase_a), (ref_b, phase_b)):
        sw.begin_switch()
        sw.run_candidate(ref[cursor:cursor + 250])
        cursor += 250
        assert sw.complete_switch()
        sw.run(ref[cursor:cursor + 250])
        cursor += 250
    drift = abs(wrap_phase(sw.theta - wrap_phase(phase_b + offset * cursor)))
    print(f"往返切换 {sw.switch_count} 次后相位残差: {drift:.2e} rad "
          f"(不累积人为偏移)")
    assert drift < 1e-3

    # 切换期间原参考失效: 如实报告未锁定
    sw.begin_switch()
    sw.run([0j] * 3)
    print(f"切换期间原参考失效: status={sw.status}, locked={sw.locked} "
          f"(不假装仍锁定)")
    assert not sw.locked
    print()


def main():
    demo_acquisition()
    demo_rejection()
    demo_reference_switch()
    print("demo 完成。")


if __name__ == "__main__":
    main()
