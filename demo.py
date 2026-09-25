"""演示: 二阶离散 PLL 的捕获过程与非法输入拒绝。

直接运行: python demo.py (纯标准库, 确定性计算, 数秒内完成)。
"""

import math

from pll_capture import (
    CycleSlipConfig,
    PLL,
    PLLInputError,
    PLLParamError,
    PLLParams,
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


def demo_cycle_slip():
    print("=== 3. 周跳候选检测: 导频方向突然反转 ===")
    params = PLLParams(kp=0.12, ki=0.008, wmax=1.2, amin=0.2,
                       eps=0.03, lock_count=40)
    # 导频在样本 600 处相位反转(在连续的频率斜坡上加 pi rad 阶跃)
    samples = [
        complex(math.cos(-2.4 + 0.35 * k + (math.pi if k >= 600 else 0.0)),
                math.sin(-2.4 + 0.35 * k + (math.pi if k >= 600 else 0.0)))
        for k in range(1200)
    ]
    cfg = dict(threshold=0.5, confirm_count=3, deg_threshold=0.3)
    # 以真值初始化, 使捕获暂态不产生额外事件, 只观察反转引起的周跳
    init = dict(theta0=-2.4, omega0=0.35)

    print("--- 仅报告模式 (compensate=False) ---")
    reporter = PLL(params, slip_config=CycleSlipConfig(compensate=False, **cfg),
                   **init)
    rep = reporter.run(samples)
    ev0 = rep.slip_events[0]
    print(f"首个事件: 起始样本 {ev0.start_index}, 确认号 {ev0.confirm_index}, "
          f"累计有符号误差 {ev0.error_sum:+.5f} rad, "
          f"圆均值 {ev0.circular_mean:+.5f} rad, "
          f"合向量长度 {ev0.resultant:.4f}, ambiguous={ev0.ambiguous}")
    print(f"反转后环路缓慢重新跟踪, 期间共确认 {len(rep.slip_events)} 次事件")
    assert ev0.start_index == 600 and ev0.confirm_index == 602
    # 候选误差是环路跟踪中衰减的误差(首个 ≈ -pi, 随后幅度回落),
    # 其圆均值应指向反转方向且幅度接近 pi 量级
    assert 2.0 < abs(ev0.circular_mean) <= math.pi

    print("--- 补偿模式 (compensate=True) ---")
    comp = PLL(params, slip_config=CycleSlipConfig(compensate=True, **cfg),
               **init)
    comp_result = comp.run(samples)
    ev_c = comp_result.slip_events[0]
    print(f"首个事件: 确认号 {ev_c.confirm_index}, "
          f"补偿量 {ev_c.correction:+.5f} rad (= 候选误差圆均值), "
          f"compensated={ev_c.compensated}, "
          f"补偿后总事件数 {len(comp_result.slip_events)}")
    assert ev_c.compensated and not ev_c.ambiguous
    # 确认后锁定计数清零、频率估计保留, 随后重新锁定并继续跟踪
    assert comp_result.diagnostics[ev_c.confirm_index].streak == 0
    assert comp_result.locked
    assert abs(comp_result.omega - 0.35) < 1e-3
    print(f"确认后锁定计数清零并重新捕获: locked={comp_result.locked}, "
          f"omega={comp_result.omega:.5f} (真值 0.35, 频率估计保留)")

    print("--- 合向量退化: 只报告 ambiguous ---")
    # 误差 ±1.8 对称抵消, 合向量长度 |cos(1.8)| ≈ 0.227 < 0.3, 不挑方向
    amb = PLL(PLLParams(kp=0.2, ki=0.02, wmax=1.0, amin=0.5,
                        eps=0.05, lock_count=5),
              slip_config=CycleSlipConfig(threshold=1.0, confirm_count=2,
                                          deg_threshold=0.3, compensate=True))
    amb.run(make_pilot(1.0, 0.3, 0.0, 60))
    for e in (1.8, -1.8):
        amb.step(complex(math.cos(amb.theta + e), math.sin(amb.theta + e)))
    ev_a = amb.slip_events[0]
    print(f"事件: 合向量长度 {ev_a.resultant:.4f} < 0.3 -> "
          f"ambiguous={ev_a.ambiguous}, compensated={ev_a.compensated}, "
          f"circular_mean={ev_a.circular_mean}")
    assert ev_a.ambiguous and not ev_a.compensated
    print("注: 以上是对合成导频的离线检测演示, 不构成真实硬件诊断保证。")
    print()


def main():
    demo_acquisition()
    demo_rejection()
    demo_cycle_slip()
    print("demo 完成。")


if __name__ == "__main__":
    main()
