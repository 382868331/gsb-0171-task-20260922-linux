"""演示: 二阶离散 PLL 的捕获、运行中采样周期切换与非法输入拒绝。

直接运行: python demo.py (纯标准库, 确定性计算, 数秒内完成)。
"""

import math

from pll_capture import (
    PLL,
    PLLInputError,
    PLLParamError,
    PLLParams,
    PLLSwitchError,
    SamplePeriod,
)


def make_pilot(amplitude, phase, offset, n):
    """确定性合成复数导频: z_k = A * exp(j*(phase + offset*k))。"""
    return [
        complex(amplitude * math.cos(phase + offset * k),
                amplitude * math.sin(phase + offset * k))
        for k in range(n)
    ]


def demo_acquisition():
    print("=== 1. 正常捕获过程(旧固定速率接口) ===")
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
    print("=== 2. 运行中切换采样周期(48 kHz -> 96 kHz) ===")
    fs1, fs2 = 48_000.0, 96_000.0
    ts1, ts2 = 1.0 / fs1, 1.0 / fs2
    r = ts2 / ts1  # 0.5
    f_hz = 300.0
    n1, n2 = 600, 400
    # 信号真值(独立公式, 非被测实现输出):
    w1_true = 2.0 * math.pi * f_hz * ts1  # 48k 速率下 rad/sample
    w2_true = 2.0 * math.pi * f_hz * ts2  # 96k 速率下 rad/sample

    params = PLLParams(kp=0.12, ki=0.008, wmax=1.2, amin=0.2,
                       eps=0.03, lock_count=40,
                       ts=SamplePeriod.seconds(ts1))
    pll = PLL(params)
    pll.run(make_pilot(1.0, 0.0, w1_true, n1))
    print(f"切换前 @{fs1:.0f} Hz: omega={pll.omega:.8f} rad/sample, "
          f"f={pll.frequency_hz:.4f} Hz, theta={pll.theta:.8f} rad, "
          f"locked={pll.locked}")
    assert pll.locked
    assert abs(pll.omega - w1_true) < 1e-6

    theta_before, omega_before, streak_before = pll.theta, pll.omega, pll.streak
    rep = pll.set_sample_period(SamplePeriod.seconds(ts2))
    print(f"切换: r=Ts2/Ts1={rep.ratio:.3f}; "
          f"kp {rep.params_old.kp}->{rep.params_new.kp}, "
          f"ki {rep.params_old.ki}->{rep.params_new.ki:.6f}, "
          f"wmax {rep.params_old.wmax}->{rep.params_new.wmax}")
    print(f"切换后 @{fs2:.0f} Hz: omega={pll.omega:.8f} rad/sample, "
          f"f={pll.frequency_hz:.4f} Hz, theta={pll.theta:.8f} rad, "
          f"locked={pll.locked}")
    # 解析关系核对
    assert rep.ratio == 0.5
    assert pll.theta == theta_before, "theta 必须逐位连续"
    assert abs(pll.omega - omega_before * r) < 1e-15
    assert abs(pll.frequency_hz - f_hz) < 1e-6, "物理频偏 Hz 不变"
    assert pll.streak == streak_before and pll.locked
    # 物理动态不变量: omega_n=sqrt(ki)/Ts, zeta=kp/(2 sqrt(ki))
    wn_before = math.sqrt(rep.params_old.ki) / rep.period_old
    wn_after = math.sqrt(rep.params_new.ki) / rep.period_new
    assert abs(wn_after - wn_before) < 1e-7 * wn_before

    # 第二段: 时间戳从"上一个样本之后隔一个新周期"继续(无噪声固定频率)。
    seg2 = [
        complex(math.cos(2.0 * math.pi * f_hz * ((n1 - 1) * ts1 + m * ts2)),
                math.sin(2.0 * math.pi * f_hz * ((n1 - 1) * ts1 + m * ts2)))
        for m in range(1, n2 + 1)
    ]
    res2 = pll.run(seg2)
    first, last = res2.diagnostics[0], res2.diagnostics[-1]
    # 首个新速率样本误差的解析预测: w2 - w1 = (r-1)*w1
    first_err_true = (r - 1.0) * w1_true
    print(f"新速率首样本误差: 实测 {first.error:+.8f} rad, "
          f"解析预测 (r-1)*w1 = {first_err_true:+.8f} rad; "
          f"locked 未中断={first.locked}")
    assert abs(first.error - first_err_true) < 1e-6
    assert first.locked, "|(r-1)*w1| < eps 时锁定应保持"
    print(f"再处理 {n2} 个样本后: omega={pll.omega:.8f} (真值 {w2_true:.8f}), "
          f"末端误差 {last.error:+.2e}, locked={res2.locked}, "
          f"失锁样本={res2.lost_samples}")
    assert abs(pll.omega - w2_true) < 1e-7
    assert abs(last.error) < 1e-6
    # Hz 容差由 omega 容差换算: 1e-7/(2*pi*Ts2) ≈ 1.5e-3 Hz
    assert abs(pll.frequency_hz - f_hz) < 5e-3

    # 边界: r=1 幂等, 状态逐位不变
    omega0, theta0, params0 = pll.omega, pll.theta, pll.params
    rep1 = pll.set_sample_period(SamplePeriod.seconds(ts2))
    assert rep1.ratio == 1.0 and pll.omega == omega0
    assert pll.theta == theta0 and pll.params == params0
    print(f"边界 r=1: 幂等, 状态逐位不变; 便捷写法 set_sample_period({ts2!r}) 同样接受")
    print()


def demo_rejection():
    print("=== 3. 非法输入实际触发拒绝(状态不变) ===")
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
    assert pll.processed == 0 and pll.omega == 0.0
    print(f"拒绝后状态未变: processed={pll.processed}, omega={pll.omega}")

    # 非法采样周期
    sp = PLL(PLLParams(kp=0.12, ki=0.008, wmax=1.2, amin=0.2,
                       eps=0.03, lock_count=40,
                       ts=SamplePeriod.seconds(1.0 / 48000.0)),
             omega0=0.05)
    for bad in (0.0, float("nan"), (1e-3, "weeks")):
        try:
            sp.set_sample_period(bad)
        except PLLSwitchError as e:
            print(f"非法周期 {bad!r} 被拒绝 (PLLSwitchError): {e}")
    assert sp.sample_rate == 48000.0 and sp.omega == 0.05
    print(f"拒绝后状态未变: fs={sp.sample_rate:.0f} Hz, omega={sp.omega}")

    # 大比率降采样使缩放后 kp'=0.6*2=1.2 不满足 0<kp<1
    crit = PLL(PLLParams(kp=0.6, ki=0.05, wmax=1.0, amin=0.2,
                         eps=0.03, lock_count=10))
    try:
        crit.set_sample_period(SamplePeriod.seconds(2.0))
    except PLLSwitchError as e:
        print(f"不稳定切换被拒绝 (PLLSwitchError): {e}")
    assert crit.params.kp == 0.6 and crit.sample_period_seconds == 1.0
    print(f"拒绝后参数未变: kp={crit.params.kp}, Ts={crit.sample_period_seconds}")


def main():
    demo_acquisition()
    demo_rate_switch()
    demo_rejection()
    print("demo 完成。")


if __name__ == "__main__":
    main()
