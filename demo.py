"""演示: 二阶离散 PLL 的捕获过程与非法输入拒绝。

直接运行: python demo.py (纯标准库, 确定性计算, 数秒内完成)。
"""

import math

from pll_capture import PLL, PLLInputError, PLLParamError, PLLParams


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


def main():
    demo_acquisition()
    demo_rejection()
    print("demo 完成。")


if __name__ == "__main__":
    main()
