"""跟踪正确性：独立参考逐样本比对 + 解析参考 + 边界场景。

期望值来源：
* tests/reference.py 中独立编写的参考实现（不调用被测核心）；
* 对恒相位 / 固定频偏导频可解析得到的真值（稳态 omega→dphi、
  theta→相位），断言用紧数值口径，不靠宽容差掩盖边界错误。
"""

from __future__ import annotations

import cmath
import math
import random
import unittest

from pll_diag import PLL, PLLParams, track

from reference import ref_pilot, ref_run

P = PLLParams(kp=0.3, ki=0.05, wmax=0.5, eps=2e-2, L=8, amin=1e-9)


class TestAgainstIndependentReference(unittest.TestCase):
    """逐样本与独立 oracle 对齐（运算次序不同，允许极小舍入差）。"""

    def assertMatchesReference(self, samples, params=P, theta0=0.0, omega0=0.0):
        ref = ref_run(
            list(samples),
            kp=params.kp, ki=params.ki, wmax=params.wmax,
            eps=params.eps, L=params.L, amin=params.amin,
            theta0=theta0, omega0=omega0,
        )
        init = None
        if theta0 != 0.0 or omega0 != 0.0:
            from pll_diag import PLLState
            init = PLLState(theta=theta0, omega=omega0, streak=0,
                            locked=False, processed=0, lost_count=0)
        diag = track(samples, params, init)

        self.assertEqual(len(diag.samples), len(ref["steps"]))
        for got, exp in zip(diag.samples, ref["steps"]):
            self.assertEqual(got.index, exp["index"])
            self.assertEqual(got.valid, exp["valid"])
            self.assertTrue(math.isfinite(got.amplitude))
            self.assertAlmostEqual(got.amplitude, exp["amplitude"], places=14)
            if exp["valid"]:
                # 两边三角运算次序不同，比对用 8 ulp 量级的紧口径。
                self.assertAlmostEqual(got.error, exp["error"], places=12)
                self.assertAlmostEqual(got.theta, exp["theta"], places=12)
                self.assertAlmostEqual(got.omega, exp["omega"], places=12)
                self.assertEqual(got.omega_saturated, exp["saturated"])
            else:
                self.assertIsNone(got.error)
                self.assertFalse(got.omega_saturated)
            self.assertEqual(got.streak_after, exp["streak"])
            self.assertEqual(got.locked_after, exp["locked"])

        self.assertAlmostEqual(diag.phase, ref["theta"], places=12)
        self.assertAlmostEqual(diag.frequency_offset, ref["omega"], places=12)
        self.assertEqual(diag.final_state.streak, ref["streak"])
        self.assertEqual(diag.locked, ref["locked"])
        self.assertEqual(diag.final_state.processed, ref["processed"])
        self.assertEqual(diag.final_state.lost_count, ref["lost_count"])
        self.assertEqual(diag.locked_at, ref["locked_at"])
        self.assertEqual(diag.lost_indices, ref["lost_indices"])
        self.assertEqual(diag.saturated_indices, ref["saturated_indices"])
        if ref["max_abs_error"] is None:
            self.assertIsNone(diag.max_abs_error)
        else:
            self.assertAlmostEqual(diag.max_abs_error, ref["max_abs_error"], places=12)

    def test_constant_phase(self):
        self.assertMatchesReference(ref_pilot(200, 0.0, 0.7))

    def test_fixed_frequency_offset(self):
        self.assertMatchesReference(ref_pilot(400, 0.05, 0.3))

    def test_negative_frequency_offset(self):
        self.assertMatchesReference(ref_pilot(300, -0.04, -0.9))

    def test_with_zero_sample(self):
        self.assertMatchesReference(ref_pilot(120, 0.05, 0.3, zero_at=5))
        self.assertMatchesReference(ref_pilot(120, 0.02, 0.1, zero_at=0))
        self.assertMatchesReference(ref_pilot(120, 0.02, 0.1, zero_at=119))

    def test_dense_zero_block_resets_streak(self):
        sig = ref_pilot(60, 0.05, 0.3)
        sig[20] = 0j
        sig[21] = 0j
        self.assertMatchesReference(sig)

    def test_saturating_loop(self):
        # 大初相 + 小 wmax：积分器必然先撞到限幅。
        self.assertMatchesReference(ref_pilot(80, 0.0, 2.6),
                                    PLLParams(0.3, 0.05, 0.05, 2e-2, 8))

    def test_crossing_pi_phase_walk(self):
        # 正频偏让相位连续累加、反复跨过 ±pi。
        sig = ref_pilot(300, 0.12, 3.0)
        self.assertMatchesReference(sig)

    def test_randomized_against_reference(self):
        rng = random.Random(20260922)
        for _ in range(12):
            n = rng.randint(1, 90)
            dphi = rng.uniform(-0.15, 0.15)
            phi0 = rng.uniform(-math.pi, math.pi)
            sig = [cmath.exp(1j * (phi0 + dphi * k)) * rng.uniform(0.5, 2.0)
                   for k in range(n)]
            for _ in range(rng.randint(0, 3)):
                sig[rng.randrange(n)] = 0j
            params = PLLParams(
                kp=rng.uniform(0.05, 0.8),
                ki=rng.uniform(1e-4, 0.08),
                wmax=rng.uniform(0.05, 1.0),
                eps=rng.uniform(1e-3, 0.1),
                L=rng.randint(1, 10),
                amin=1e-9,
            )
            # 仅保留 ki < kp 的合法随机参数（其余由校验测试覆盖）。
            if not (0.0 < params.ki < params.kp):
                continue
            self.assertMatchesReference(sig, params)


class TestAnalyticReferences(unittest.TestCase):
    """解析小数据参考：恒相位与固定频偏的稳态真值。"""

    def test_constant_phase_converges_to_truth(self):
        phi0 = 0.7
        sig = ref_pilot(200, 0.0, phi0)
        diag = track(sig, P)
        self.assertTrue(diag.locked)
        self.assertIsNotNone(diag.locked_at)
        # 频偏真值 0；相位真值 phi0。口径 1e-9（实测 ~1e-12）。
        self.assertLessEqual(abs(diag.frequency_offset), 1e-9)
        self.assertLessEqual(abs(wrapdiff(diag.phase, phi0)), 1e-9)
        tail = [s for s in diag.samples[-40:] if s.valid]
        self.assertLessEqual(max(abs(s.error) for s in tail), 1e-9)

    def test_fixed_offset_converges_to_truth(self):
        dphi, phi0 = 0.05, 0.3
        sig = ref_pilot(400, dphi, phi0)
        diag = track(sig, P)
        self.assertTrue(diag.locked)
        # omega 真值 = dphi；更新后 theta'=theta+omega 是对“下一”样本
        # 相位的预测，故处理完样本 k 的 theta 对应相位真值 phi0+dphi*(k+1)。
        self.assertLessEqual(abs(diag.frequency_offset - dphi), 1e-9)
        phase_truth = ref_wrap_local(phi0 + dphi * len(sig))
        self.assertLessEqual(abs(wrapdiff(diag.phase, phase_truth)), 1e-9)
        tail = [s for s in diag.samples[-80:] if s.valid]
        self.assertLessEqual(max(abs(s.error) for s in tail), 1e-9)
        self.assertLessEqual(max(abs(s.omega - dphi) for s in tail), 1e-9)

    def test_estimated_phase_matches_pilot_sample_by_sample(self):
        # 锁定后：处理完样本 k 的 theta 预测第 k+1 个导频相位。
        dphi = 0.03
        sig = ref_pilot(250, dphi, 1.1)
        diag = track(sig, PLLParams(0.3, 0.05, 0.5, 5e-3, 8))
        self.assertTrue(diag.locked)
        for k in range(120, 250):
            truth = ref_wrap_local(1.1 + dphi * (k + 1))
            self.assertLessEqual(
                abs(wrapdiff(diag.samples[k].theta, truth)), 1e-6,
                msg=f"样本 {k} 相位估计偏离真值",
            )

    def test_first_step_matches_closed_form(self):
        # 首步解析值：error0 = phi0；omega1 = clip(ki*phi0)；
        # theta1 = wrap(omega1 + kp*phi0)。
        phi0 = 0.4
        z0 = cmath.exp(1j * phi0)
        diag = track([z0], P)
        s0 = diag.samples[0]
        self.assertAlmostEqual(s0.error, phi0, places=14)
        self.assertAlmostEqual(s0.omega, P.ki * phi0, places=14)
        self.assertAlmostEqual(
            s0.theta,
            ref_wrap_local(P.ki * phi0 + P.kp * phi0),
            places=14,
        )
        self.assertEqual(s0.streak_after, 0)  # phi0 > eps
        self.assertFalse(s0.locked_after)


def ref_wrap_local(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def wrapdiff(a: float, b: float) -> float:
    """两个角度的有向最短差，落在 [-pi, pi)。"""
    d = (a - b + math.pi) % (2.0 * math.pi) - math.pi
    return d


class TestAngleRangeAndLockSemantics(unittest.TestCase):
    def test_theta_and_error_always_in_half_open_range(self):
        rng = random.Random(7)
        sig = [cmath.exp(1j * rng.uniform(-20.0, 20.0)) for _ in range(500)]
        diag = track(sig, PLLParams(0.2, 0.02, 1.0, 0.5, 3))
        for s in diag.samples:
            self.assertGreaterEqual(s.theta, -math.pi)
            self.assertLess(s.theta, math.pi)
            if s.valid:
                self.assertGreaterEqual(s.error, -math.pi)
                self.assertLess(s.error, math.pi)

    def test_error_at_exactly_minus_pi_boundary(self):
        # 样本相位恰为 pi（atan2(0,-1)=pi），契约要求误差归到 -pi 一侧。
        diag = track([complex(-1.0, 0.0)], P)
        self.assertEqual(diag.samples[0].error, -math.pi)
        self.assertGreaterEqual(diag.samples[0].theta, -math.pi)
        self.assertLess(diag.samples[0].theta, math.pi)

    def test_lock_requires_L_consecutive_and_clears_on_exceed(self):
        params = PLLParams(0.3, 0.05, 0.5, eps=1e-2, L=4, amin=1e-9)
        # 先 L 个完全同相样本：进入锁定。
        pll = PLL(params)
        pll.update([1 + 0j] * params.L)
        self.assertTrue(pll.locked)
        # 注入 pi/2 相位跳变（绝对误差 > eps）：立即解锁、计数清零。
        jump = pll.update([cmath.exp(1j * math.pi / 2)])
        self.assertFalse(jump.samples[0].locked_after)
        self.assertEqual(jump.samples[0].streak_after, 0)
        # 独立闭式递推跳变后的状态（不调用被测代码）：
        #   e=pi/2, omega1=ki*e, theta1=wrap(omega1+kp*e)；
        # 之后误差恒为 0 的样本满足 theta_{k+1}=theta_k+omega1。
        e = math.pi / 2
        omega1 = params.ki * e
        theta1 = ref_wrap_local(omega1 + params.kp * e)
        good = []
        th = theta1
        for _ in range(params.L):
            good.append(cmath.exp(1j * th))
            th = ref_wrap_local(th + omega1)
        # 前 L-1 个：仍未锁定，计数逐拍 +1。
        diag2 = pll.update(good[:-1])
        for s in diag2.samples:
            self.assertAlmostEqual(s.error, 0.0, places=9)
        self.assertEqual(diag2.samples[-1].streak_after, params.L - 1)
        self.assertFalse(diag2.samples[-1].locked_after)
        # 第 L 个：锁定。
        diag3 = pll.update(good[-1:])
        self.assertEqual(diag3.samples[0].streak_after, params.L)
        self.assertTrue(diag3.samples[0].locked_after)

    def test_eps_boundary_is_inclusive(self):
        # 恰好误差为 0 时连续 L 个必锁定（<= eps 含等号）。
        diag = track([1 + 0j] * 5, PLLParams(0.3, 0.05, 0.5, 1e-6, 5))
        self.assertTrue(diag.locked)
        self.assertEqual(diag.locked_at, 4)


class TestChunkingEquivalence(unittest.TestCase):
    """任意分块必须与一次性处理逐样本一致。"""

    def _assert_chunks_match(self, samples, params=P, cuts=(1, 1, 3, 10, 64, 200, 401)):
        whole = track(samples, params)
        for cut in cuts:
            pll = PLL(params)
            pieces = []
            for start in range(0, len(samples), cut):
                pieces.append(pll.update(samples[start:start + cut]))
            # 终态一致。
            self.assertEqual(pll.state, whole.final_state,
                             msg=f"cut={cut} 终态不一致")
            # 逐样本诊断拼接后一致。
            joined = [s for d in pieces for s in d.samples]
            self.assertEqual(len(joined), len(whole.samples))
            for a, b in zip(joined, whole.samples):
                self.assertEqual(a, b, msg=f"cut={cut}")
            # 聚合字段（下标为全局下标，跨块直接拼接）。
            self.assertEqual(pll.locked, whole.locked)
            total_lost = tuple(
                i for d in pieces for i in d.lost_indices
            )
            self.assertEqual(total_lost, whole.lost_indices)

    def test_constant_phase_chunks(self):
        self._assert_chunks_match(ref_pilot(200, 0.0, 0.7))

    def test_offset_chunks(self):
        self._assert_chunks_match(ref_pilot(400, 0.05, 0.3))

    def test_with_zeros_chunks(self):
        sig = ref_pilot(150, 0.05, 0.3, zero_at=0)
        sig[70] = 0j
        sig[149] = 0j
        self._assert_chunks_match(sig, cuts=(1, 2, 7, 37, 100, 151))

    def test_saturation_chunks(self):
        params = PLLParams(0.3, 0.05, 0.05, 2e-2, 8)
        self._assert_chunks_match(
            ref_pilot(80, 0.0, 2.6), params, cuts=(1, 3, 17, 81))

    def test_single_sample_chunks(self):
        self._assert_chunks_match(
            ref_pilot(120, -0.04, 2.0), cuts=(1,))

    def test_empty_chunks_are_neutral(self):
        pll = PLL(P)
        before = pll.state
        d = pll.update([])
        self.assertEqual(pll.state, before)
        self.assertEqual(d.samples, ())
        self.assertIsNone(d.max_abs_error)
        self.assertFalse(d.locked)
        # 空块后再处理，与直接处理完全一致。
        sig = ref_pilot(50, 0.05, 0.3)
        pll.update([])
        d2 = pll.update(sig)
        whole = track(sig, P)
        self.assertEqual(d2.samples, whole.samples)
        self.assertEqual(pll.state, whole.final_state)


class TestZeroAmplitudeFreeze(unittest.TestCase):
    def test_zero_freezes_state_exactly(self):
        pll = PLL(P)
        pll.update(ref_pilot(20, 0.05, 0.3))
        snap = pll.state
        d = pll.update([0j, 0j, 0j])
        for s in d.samples:
            self.assertFalse(s.valid)
            self.assertIsNone(s.error)
            self.assertEqual(s.theta, snap.theta)
            self.assertEqual(s.omega, snap.omega)
            self.assertEqual(s.streak_after, 0)
            self.assertFalse(s.locked_after)
        self.assertEqual(pll.state.theta, snap.theta)
        self.assertEqual(pll.state.omega, snap.omega)
        self.assertEqual(pll.state.lost_count, snap.lost_count + 3)
        self.assertEqual(pll.state.processed, snap.processed + 3)
        self.assertEqual(d.lost_indices, (20, 21, 22))

    def test_amin_boundary_is_strict(self):
        # 幅度恰为 amin 仍然有效（严格小于才失锁）。
        params = PLLParams(0.3, 0.05, 0.5, 1.0, 1, amin=0.5)
        diag = track([0.5 + 0j], params)
        self.assertTrue(diag.samples[0].valid)
        diag2 = track([0.499999999999 + 0j], params)
        self.assertFalse(diag2.samples[0].valid)


class TestSaturation(unittest.TestCase):
    def test_omega_clipped_to_wmax(self):
        params = PLLParams(0.3, 0.05, 0.05, 2e-2, 8, amin=1e-9)
        diag = track(ref_pilot(20, 0.0, 2.6), params)
        sat = [i for i, s in enumerate(diag.samples) if s.omega_saturated]
        self.assertTrue(sat, "大初相下应出现限幅")
        for i in sat:
            self.assertEqual(abs(diag.samples[i].omega), params.wmax)
        for s in diag.samples:
            self.assertLessEqual(abs(s.omega), params.wmax)

    def test_negative_saturation(self):
        params = PLLParams(0.3, 0.05, 0.05, 2e-2, 8)
        diag = track(ref_pilot(20, 0.0, -2.6), params)
        self.assertTrue(any(s.omega_saturated for s in diag.samples))
        for s in diag.samples:
            self.assertGreaterEqual(s.omega, -params.wmax)


if __name__ == "__main__":
    unittest.main()
