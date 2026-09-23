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


def demo_rate_switch():
    print("=== 2. 运行中切换采样周期(相位连续) ===")
    fs1 = 48_000.0
    ts1 = 1.0 / fs1
    freq_hz = 1500.0
    omega_phys = 2.0 * math.pi * freq_hz   # 物理角频率 rad/s
    phi0 = 0.7
    params = PLLParams(kp=0.12, ki=0.008, wmax=1.2, amin=0.2,
                       eps=0.03, lock_count=40)
    pll = PLL(params, sample_period=ts1)

    # 第一段: 48 kHz 采样, 无噪声固定频率导频 z(t) = exp(j*(phi0 + Omega*t))
    n1 = 800
    s1 = [complex(math.cos(phi0 + omega_phys * k * ts1),
                  math.sin(phi0 + omega_phys * k * ts1)) for k in range(n1)]
    r1 = pll.run(s1)
    print(f"48 kHz 段: omega={r1.omega:.6f} rad/sample "
          f"({r1.frequency_offset():.2f} Hz), locked={r1.locked}")
    assert r1.locked

    # 切换到 24 kHz: r = Ts_new/Ts_old = 2
    theta_before = pll.theta
    change = pll.set_sample_period(2.0 * ts1)
    print(f"切换 48 kHz -> 24 kHz (r={change.ratio:.1f}): "
          f"omega {change.omega_before:.6f} -> {change.omega_after:.6f} rad/sample, "
          f"kp {change.old_params.kp:.3f} -> {change.new_params.kp:.3f}, "
          f"ki {change.old_params.ki:.4f} -> {change.new_params.ki:.4f}")
    print(f"物理角频率不变: {change.angular_frequency:.2f} rad/s, "
          f"theta 逐位不变: {pll.theta == theta_before}")
    assert pll.theta == theta_before

    # 第二段: 24 kHz 采样, 信号相位在时间上连续
    n2 = 400
    ts2 = 2.0 * ts1
    s2 = [complex(math.cos(phi0 + omega_phys * (n1 * ts1 + m * ts2)),
                  math.sin(phi0 + omega_phys * (n1 * ts1 + m * ts2)))
          for m in range(n2)]
    r2 = pll.run(s2)
    max_err = max(abs(d.error) for d in r2.diagnostics)
    print(f"24 kHz 段: 切换后最大相位误差 {max_err:.2e} rad (eps=0.03), "
          f"全程 locked={all(d.locked for d in r2.diagnostics)}")
    print(f"收敛: omega={r2.omega:.6f} rad/sample "
          f"({r2.frequency_offset():.2f} Hz, 真值 {freq_hz:.1f} Hz)")
    assert max_err < 1e-4, "相位连续: 切换后不应出现重新捕获瞬态"
    assert abs(r2.omega - omega_phys * ts2) < 1e-9, "omega 应收敛到新周期真值"
    assert r2.locked

    # 非法切换: 周期非正 / 缩放后增益失稳, 拒绝且状态不变
    for bad, why in ((0.0, "周期非正"), (0.02, "r=480, 缩放后 kp>=1 失稳")):
        snapshot = (pll.params, pll.omega, pll.theta)
        try:
            pll.set_sample_period(bad)
        except PLLParamError as e:
            unchanged = (pll.params, pll.omega, pll.theta) == snapshot
            print(f"非法切换被拒绝({why}): {e} [状态未变: {unchanged}]")
            assert unchanged
    print()


def demo_rejection():
    print("=== 3. 非法输入实际触发拒绝 ===")
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
    demo_rate_switch()
    demo_rejection()
    print("demo 完成。")


if __name__ == "__main__":
    main()
