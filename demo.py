#!/usr/bin/env python3
"""离线演示：整数采样二阶数字锁相环（DPLL）的捕获诊断。

数据全部由本文件确定性生成（合成复数导频 exp(j(phi0+dphi*n))），
无网络、无第三方依赖、无 sleep、无预录结果；所有数字均为当场计算。

运行：
    python demo.py
"""

from __future__ import annotations

import math
import time

from pll_diag import PLL, PLLConfigError, PLLParams, track

# ---- 固定演示配置（同时是 README 中的可复现样例） ----
KP, KI, WMAX, EPS, L_PL, AMIN = 0.3, 0.05, 0.5, 2e-2, 8, 1e-9
N = 240
DPHI_TRUE, PHI0_TRUE = 0.05, 0.3      # 真值：频偏 0.05 rad/样本，初相 0.3
ZERO_AT = 40                          # 在第 40 个样本制造一次失锁
CHUNK = 60                            # 分块大小：4 个任意分块


def pilot(n: int, dphi: float, phi0: float, zero_at: int | None) -> list[complex]:
    """确定性合成复数导频，可在 zero_at 处置零。"""
    out: list[complex] = []
    for k in range(n):
        if k == zero_at:
            out.append(0j)
        else:
            ph = phi0 + dphi * k
            out.append(complex(math.cos(ph), math.sin(ph)))
    return out


def wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def fmt(s) -> str:
    if not s.valid:
        return (f"n={s.index:3d}  失锁冻结 | 幅值={s.amplitude:.3g} | "
                f"theta={s.theta:+.6f} omega={s.omega:+.6f} | 计数清零")
    flag = " [限幅]" if s.omega_saturated else ""
    return (f"n={s.index:3d}  error={s.error:+.6f} | "
            f"theta={s.theta:+.6f} omega={s.omega:+.6f} | "
            f"连续合格={s.streak_after:2d} "
            f"{'LOCKED' if s.locked_after else '      '}{flag}")


def main() -> None:
    t0 = time.perf_counter()
    params = PLLParams(kp=KP, ki=KI, wmax=WMAX, eps=EPS, L=L_PL, amin=AMIN)
    samples = pilot(N, DPHI_TRUE, PHI0_TRUE, ZERO_AT)

    print("=" * 72)
    print("整数采样二阶 DPLL 捕获诊断演示（离线、确定性合成导频）")
    print("=" * 72)
    print(f"参数: kp={KP}  ki={KI}  wmax={WMAX}  eps={EPS}  L={L_PL}  "
          f"amin={AMIN:g}")
    print(f"导频: N={N}  真值频偏={DPHI_TRUE} rad/样本  真值初相={PHI0_TRUE} rad"
          f"  第 {ZERO_AT} 样本置零")
    print("-" * 72)

    # 1) 一次性处理，展示捕获过程。
    diag = track(samples, params)
    print("【捕获过程（节选逐样本诊断）】")
    milestones = [0, 1, 2, 4, 8, 12]
    first_lock = diag.locked_at
    for i in milestones:
        print("  " + fmt(diag.samples[i]))
    print("  ...")
    print(f"  首次锁定：n={first_lock}（连续 {L_PL} 个 |error| <= {EPS:g}）")
    print("  ...")
    for i in range(ZERO_AT - 1, ZERO_AT + 4):
        print("  " + fmt(diag.samples[i]))

    # 重新锁定位置：零样本后首个 locked_after 的下标。
    relock = next(
        (s.index for s in diag.samples[ZERO_AT + 1:] if s.locked_after), None
    )
    print(f"  失锁后重新锁定：n={relock}")

    # 2) 终态估计 vs 解析真值（紧口径，不用宽容差）。
    om = diag.frequency_offset
    # 更新式 theta'=theta+omega 是对“下一”样本（序号 N）相位的预测。
    phase_truth = wrap(PHI0_TRUE + DPHI_TRUE * N)
    om_err = abs(om - DPHI_TRUE)
    ph_err = abs(wrap(diag.phase - phase_truth))
    print("-" * 72)
    print("【估计结果 vs 解析真值】")
    print(f"  频偏估计 = {om:+.12f} rad/样本，真值 {DPHI_TRUE:+.6f}，"
          f"绝对误差 {om_err:.3e}")
    print(f"  相位估计 = {diag.phase:+.12f} rad（对下一样本的预测），"
          f"序号 {N} 处真值 {phase_truth:+.12f}，绝对误差 {ph_err:.3e}")
    print(f"  终态状态 = {diag.status}；累计失锁 {diag.final_state.lost_count} 个；"
          f"限幅样本 {len(diag.saturated_indices)} 个")
    assert diag.status == "locked"
    assert om_err < 1e-9 and ph_err < 1e-9, "紧口径校验失败"

    # 3) 任意分块结果一致性：4 个 60 样本分块 vs 一次性。
    pll = PLL(params)
    chunk_diags = []
    for start in range(0, N, CHUNK):
        chunk_diags.append(pll.update(samples[start:start + CHUNK]))
    joined = tuple(s for d in chunk_diags for s in d.samples)
    same_state = pll.state == diag.final_state
    same_steps = joined == diag.samples
    print("-" * 72)
    print(f"【分块一致性】4 个 {CHUNK} 样本分块 vs 一次性 {N} 样本：")
    print(f"  终态完全一致: {same_state}；逐样本诊断完全一致: {same_steps}")
    assert same_state and same_steps

    # 4) 实际触发积分器限幅：大初相 + 小 wmax。
    print("-" * 72)
    print("【积分器饱和】初相 2.6 rad、wmax=0.05：前若干步 omega 必须被钳住")
    sat_params = PLLParams(kp=0.3, ki=0.05, wmax=0.05, eps=2e-2, L=8)
    sat_sig = pilot(10, 0.0, 2.6, None)
    sat_diag = track(sat_sig, sat_params)
    for s in sat_diag.samples[:4]:
        print("  " + fmt(s))
    n_sat = len(sat_diag.saturated_indices)
    print(f"  前 10 步中限幅 {n_sat} 步；omega 始终满足 |omega| <= wmax")
    assert n_sat > 0
    assert all(abs(s.omega) <= 0.05 + 1e-15 for s in sat_diag.samples)

    # 5) 实际触发的拒绝边界：超稳定范围参数。
    print("-" * 72)
    print("【超范围参数拒绝边界】")
    for bad_kwargs, why in (
        (dict(kp=1.2, ki=0.05, wmax=0.5, eps=EPS, L=L_PL),
         "kp=1.2 违反 0 < kp < 1"),
        (dict(kp=0.3, ki=0.3, wmax=0.5, eps=EPS, L=L_PL),
         "ki=0.3 违反 0 < ki < kp"),
    ):
        try:
            PLL(PLLParams(**bad_kwargs))
        except PLLConfigError as exc:
            print(f"  拒绝（{why}）：{exc}")
        else:
            raise AssertionError("非法参数必须被拒绝")

    dt = time.perf_counter() - t0
    print("=" * 72)
    print(f"演示完成，实际计算耗时 {dt*1000:.1f} ms（无 sleep、无预录结果）。")


if __name__ == "__main__":
    main()
