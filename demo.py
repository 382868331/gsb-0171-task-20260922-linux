"""演示: 二阶离散 PLL 的捕获过程、周跳候选检测与非法输入拒绝。

直接运行: ./.venv/bin/python demo.py
(纯标准库, 确定性计算, 约 0.1 秒完成;不连接任何外部服务。)
"""

import math

from pll_capture import (
    PLL,
    PLLInputError,
    PLLParamError,
    PLLParams,
    circular_rmean,
    wrap_phase,
)


def make_pilot(amplitude, phase, offset, n):
    """确定性合成复数导频: z_k = A * exp(j*(phase + offset*k))。"""
    return [
        complex(amplitude * math.cos(phase + offset * k),
                amplitude * math.sin(phase + offset * k))
        for k in range(n)
    ]


def feed_errors(pll, errors):
    """按期望误差合成样本: z = exp(j*(本步更新前 theta + e))。"""
    out = []
    for e in errors:
        z = complex(math.cos(pll.theta + e), math.sin(pll.theta + e))
        out.append(pll.step(z))
    return out


JP = dict(kp=0.2, ki=0.02, wmax=1.0, amin=0.5, eps=0.05, lock_count=30,
          jump_k=3, jump_eps=0.3, r_min=0.5)


def demo_acquisition():
    print("=== 1. 正常捕获过程(无周跳检测, 行为与旧版一致) ===")
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


def _print_event(tag, ev):
    mean = "None(退化)" if ev.circular_mean is None else f"{ev.circular_mean:+.4f}"
    print(f"  [{tag}] 事件 起始={ev.start_index} 确认={ev.confirm_index} "
          f"误差累计={ev.error_sum:+.4f} 圆均值={mean} r={ev.r:.4f} "
          f"ambiguous={ev.ambiguous} 已补偿={ev.compensated}")


def demo_jump_detection():
    print("=== 2. 可选周跳候选检测(jump_k=3, jump_eps=0.3, r_min=0.5) ===")

    # (a) 孤立毛刺: 1 次越限后恢复 -> 候选被打断, 不确认
    a = PLL(PLLParams(**JP))
    a.run(make_pilot(1.0, 0.3, 0.0, 60))
    d = feed_errors(a, [1.0, -0.1, -0.1, -0.1])
    print("(a) 孤立毛刺后立即恢复:")
    print(f"    毛刺样本 candidate={d[0].candidate}, jump={d[0].jump}; "
          f"恢复样本 candidate={d[1].candidate}")
    assert d[0].candidate and d[0].jump is None
    assert not d[1].candidate and a.jump_count == 0

    # (b) 接近 ±pi 的跨界误差: 算术和方向相反, 圆均值取最短弧
    b = PLL(PLLParams(**JP))
    cross = (3.05, 3.12, -3.08)
    ev = feed_errors(b, cross)[-1].jump
    print(f"(b) 跨界误差 {cross}:")
    _print_event("跨界", ev)
    assert ev is not None and not ev.ambiguous and ev.circular_mean > 3.1
    assert not ev.compensated  # 默认仅报告

    # (c) 真实阶跃: 确认后锁定计数清零、频率估计保留, 随后重新锁定
    c = PLL(PLLParams(kp=0.3, ki=0.01, wmax=1.0, amin=0.5, eps=0.05,
                      lock_count=30, jump_k=3, jump_eps=0.3, r_min=0.5),
            theta0=0.3, omega0=0.1)
    stream = make_pilot(1.0, 0.3, 0.1, 900)
    for k in range(400, 900):          # 样本 400 起永久 +0.7 阶跃
        phi = 0.3 + 0.1 * k + 0.7
        stream[k] = complex(math.cos(phi), math.sin(phi))
    c.run(stream[:400])
    rc = c.run(stream[400:])
    ev = rc.jumps[0]
    print("(c) 样本 400 真实 +0.7 rad 阶跃(频偏 0.1 保留):")
    _print_event("阶跃/仅报告", ev)
    print(f"    确认样本 streak={rc.diagnostics[2].streak}, "
          f"omega={rc.diagnostics[2].omega:.4f}(非零保留); "
          f"末尾 locked={c.locked}, omega={c.omega:.4f}")
    assert ev.start_index == 400 and ev.confirm_index == 402
    assert rc.diagnostics[2].streak == 0 and c.locked
    assert abs(c.omega - 0.1) < 1e-4

    # 同样的阶跃, 补偿模式: 确认样本的原更新之后叠加圆均值
    cc = PLL(PLLParams(kp=0.3, ki=0.01, wmax=1.0, amin=0.5, eps=0.05,
                       lock_count=30, jump_k=3, jump_eps=0.3, r_min=0.5),
             theta0=0.3, omega0=0.1, jump_compensate=True)
    cc.run(stream[:400])
    # 对照: 仅报告路径同一样本的原更新结果
    ref = PLL(PLLParams(kp=0.3, ki=0.01, wmax=1.0, amin=0.5, eps=0.05,
                        lock_count=30), theta0=0.3, omega0=0.1)
    ref.run(stream[:400])
    rcc = cc.run(stream[400:])
    rr = ref.run(stream[400:])
    evc = rcc.jumps[0]
    print("(c') 同阶跃, jump_compensate=True:")
    _print_event("阶跃/补偿", evc)
    expect_theta = wrap_phase(rr.diagnostics[2].theta + evc.circular_mean)
    print(f"    确认样本 theta={rcc.diagnostics[2].theta:+.4f} "
          f"== wrap(原更新 {rr.diagnostics[2].theta:+.4f} "
          f"+ 圆均值 {evc.circular_mean:+.4f}) = {expect_theta:+.4f}")
    assert evc.compensated
    assert abs(rcc.diagnostics[2].theta - expect_theta) < 1e-12
    assert rcc.diagnostics[2].omega == rr.diagnostics[2].omega  # 频率不动

    # (d) 低幅度中断: 打断候选, 不确认、不伪造有效相位
    dd = PLL(PLLParams(**JP))
    feed_errors(dd, [0.9, 0.9])
    theta_frozen = dd.theta
    dz = dd.step(0j)
    after = feed_errors(dd, [0.9, 0.9, 0.9])
    print("(d) 候选中插入零幅度样本:")
    print(f"    无效样本 valid={dz.valid}, error={dz.error}, "
          f"candidate={dz.candidate}, theta 冻结={dz.theta == theta_frozen}; "
          f"重新计候选后确认起始={after[2].jump.start_index}")
    assert not dz.valid and dz.error is None and not dz.candidate
    assert after[2].jump.start_index == 3

    # (e) 合向量退化: r < r_min -> ambiguous, 补偿模式也不改 theta
    opp = (math.pi / 2, -math.pi / 2, math.pi / 2, -math.pi / 2)
    mean, r = circular_rmean(opp, 0.5)
    e_rep = feed_errors(PLL(PLLParams(kp=0.2, ki=0.02, wmax=1.0, amin=0.5,
                                      eps=0.05, lock_count=30,
                                      jump_k=4, jump_eps=0.3)), opp)[-1].jump
    print(f"(e) 方向对消的候选 {tuple(round(x, 2) for x in opp)}:")
    print(f"    circular_rmean -> (mean={mean}, r={r:.3f}); "
          f"事件 ambiguous={e_rep.ambiguous}, 不挑方向、不补偿")
    assert mean is None and e_rep.ambiguous and not e_rep.compensated

    # 分块一致: 候选状态跨 chunk 保存
    whole = PLL(PLLParams(**JP))
    whole.run(make_pilot(1.0, 0.3, 0.0, 60))
    split = PLL(PLLParams(**JP))
    split.run(make_pilot(1.0, 0.3, 0.0, 60))
    seq = [3.05, 3.12, -3.08]
    wd = feed_errors(whole, seq)
    sd = feed_errors(split, seq[:2])
    assert split.candidate_active
    sd += feed_errors(split, seq[2:])
    print(f"(f) 候选跨切块边界: 切块确认={sd[-1].jump.confirm_index}, "
          f"整批确认={wd[-1].jump.confirm_index}, 事件逐位相等="
          f"{sd[-1].jump == wd[-1].jump}")
    assert sd[-1].jump == wd[-1].jump
    print()


def demo_rejection():
    print("=== 3. 非法输入实际触发拒绝 ===")
    try:
        PLLParams(kp=1.5, ki=0.008, wmax=1.2, amin=0.2,
                  eps=0.03, lock_count=40)
    except PLLParamError as e:
        print(f"超范围参数被拒绝 (PLLParamError): {e}")

    try:  # 只给 jump_k 不给 jump_eps
        PLLParams(kp=0.2, ki=0.02, wmax=1.0, amin=0.5, eps=0.05,
                  lock_count=10, jump_k=3)
    except PLLParamError as e:
        print(f"检测参数不成对被拒绝 (PLLParamError): {e}")

    try:  # 退化阈值越界
        PLLParams(kp=0.2, ki=0.02, wmax=1.0, amin=0.5, eps=0.05,
                  lock_count=10, r_min=1.5)
    except PLLParamError as e:
        print(f"r_min 越界被拒绝 (PLLParamError): {e}")

    pll = PLL(PLLParams(kp=0.12, ki=0.008, wmax=1.2, amin=0.2,
                        eps=0.03, lock_count=40))
    try:
        pll.run([1 + 0j, complex(float("nan"), 0.0)])
    except PLLInputError as e:
        print(f"NaN 样本被拒绝 (PLLInputError): {e}")
    print(f"拒绝后状态未变: processed={pll.processed}, omega={pll.omega}")


def main():
    demo_acquisition()
    demo_jump_detection()
    demo_rejection()
    print("demo 完成。")
    print("提醒: 周跳检测只是对合成相位误差序列的启发式标注,")
    print("      不构成真实硬件的周跳/失锁诊断结论或保证。")


if __name__ == "__main__":
    main()
