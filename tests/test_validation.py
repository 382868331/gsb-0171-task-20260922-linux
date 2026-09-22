"""参数与输入校验、错误语义、原子性与接口边界测试。"""

from __future__ import annotations

import math
import unittest

from pll_diag import (
    PLL,
    PLLConfigError,
    PLLInputError,
    PLLParams,
    PLLState,
    track,
)
from pll_diag.core import (
    MAX_COMPONENT_MAGNITUDE,
    MAX_L,
    MAX_SAMPLES_PER_CALL,
    wrap_angle,
)


class TestParamValidation(unittest.TestCase):
    def _reject(self, kwargs):
        with self.assertRaises(PLLConfigError):
            PLL(PLLParams(**kwargs))

    def test_stable_range_kp(self):
        for kp in (0.0, -0.1, 1.0, 1.5):
            self._reject(dict(kp=kp, ki=0.05, wmax=0.5, eps=1e-2, L=8))

    def test_stable_range_ki(self):
        for ki in (0.0, -0.01, 0.3, 0.31):  # ki>=kp 或非正
            self._reject(dict(kp=0.3, ki=ki, wmax=0.5, eps=1e-2, L=8))

    def test_positive_auxiliary_params(self):
        self._reject(dict(kp=0.3, ki=0.05, wmax=0.0, eps=1e-2, L=8))
        self._reject(dict(kp=0.3, ki=0.05, wmax=-1.0, eps=1e-2, L=8))
        self._reject(dict(kp=0.3, ki=0.05, wmax=0.5, eps=0.0, L=8))
        self._reject(dict(kp=0.3, ki=0.05, wmax=0.5, eps=-1e-3, L=8))
        self._reject(dict(kp=0.3, ki=0.05, wmax=0.5, eps=1e-2, L=8, amin=0.0))

    def test_L_must_be_positive_real_int(self):
        for L in (0, -1, 2.0, "8"):
            self._reject(dict(kp=0.3, ki=0.05, wmax=0.5, eps=1e-2, L=L))

    def test_L_bool_rejected_even_though_subclass_of_int(self):
        self._reject(dict(kp=0.3, ki=0.05, wmax=0.5, eps=1e-2, L=True))
        self._reject(dict(kp=0.3, ki=0.05, wmax=0.5, eps=1e-2, L=False))

    def test_gain_bool_rejected(self):
        self._reject(dict(kp=True, ki=0.05, wmax=0.5, eps=1e-2, L=8))
        self._reject(dict(kp=0.3, ki=False, wmax=0.5, eps=1e-2, L=8))

    def test_nan_inf_rejected_in_params(self):
        bads = (math.nan, math.inf, -math.inf)
        for field_name in ("kp", "ki", "wmax", "eps", "amin"):
            for val in bads:
                kw = dict(kp=0.3, ki=0.05, wmax=0.5, eps=1e-2, L=8, amin=1e-9)
                kw[field_name] = val
                self._reject(kw)

    def test_L_upper_bound(self):
        self._reject(dict(kp=0.3, ki=0.05, wmax=0.5, eps=1e-2, L=MAX_L + 1))

    def test_error_message_is_locatable(self):
        try:
            PLL(PLLParams(kp=1.0, ki=0.05, wmax=0.5, eps=1e-2, L=8))
        except PLLConfigError as exc:
            self.assertIn("kp", str(exc))
        else:
            self.fail("应抛出 PLLConfigError")
        try:
            PLL(PLLParams(kp=0.3, ki=0.4, wmax=0.5, eps=1e-2, L=8))
        except PLLConfigError as exc:
            self.assertIn("ki", str(exc))
        else:
            self.fail("应抛出 PLLConfigError")

    def test_params_wrong_type(self):
        with self.assertRaises(PLLConfigError):
            PLL(("kp", 0.3))  # type: ignore[arg-type]

    def test_valid_params_accepted(self):
        pll = PLL(PLLParams(kp=0.3, ki=0.05, wmax=0.5, eps=1e-2, L=8))
        self.assertFalse(pll.locked)


class TestSampleValidation(unittest.TestCase):
    def setUp(self):
        self.pll = PLL(PLLParams(0.3, 0.05, 0.5, 1e-2, 8))

    def test_nan_rejected_with_index(self):
        with self.assertRaises(PLLInputError) as cm:
            self.pll.update([1 + 0j, complex(math.nan, 0.0)])
        self.assertIn("samples[1]", str(cm.exception))

    def test_inf_rejected_with_index(self):
        with self.assertRaises(PLLInputError) as cm:
            self.pll.update([complex(0.0, math.inf)])
        self.assertIn("samples[0]", str(cm.exception))

    def test_non_numeric_sample_rejected(self):
        for bad in ("1+0j", None, [1.0, 0.0], object()):
            with self.assertRaises(PLLInputError):
                self.pll.update([bad])

    def test_bool_sample_rejected(self):
        with self.assertRaises(PLLInputError):
            self.pll.update([True])

    def test_component_magnitude_limit(self):
        with self.assertRaises(PLLInputError):
            self.pll.update([complex(MAX_COMPONENT_MAGNITUDE * 10, 0.0)])
        # 边界内可处理。
        d = self.pll.update([complex(MAX_COMPONENT_MAGNITUDE, 0.0)])
        self.assertTrue(d.samples[0].valid)

    def test_sample_count_limit_sized(self):
        # 临时把接口上限调小，避免真的分配千万级列表。
        import pll_diag.core as core
        old = core.MAX_SAMPLES_PER_CALL
        core.MAX_SAMPLES_PER_CALL = 4
        try:
            with self.assertRaises(PLLInputError) as cm:
                self.pll.update([0j] * 5)
            self.assertIn("上限", str(cm.exception))
            self.assertEqual(self.pll.state.processed, 0)  # 原子性
        finally:
            core.MAX_SAMPLES_PER_CALL = old

    def test_sample_count_limit_generator(self):
        import pll_diag.core as core
        old = core.MAX_SAMPLES_PER_CALL
        core.MAX_SAMPLES_PER_CALL = 4

        def gen():
            yield from range(5)  # 无 __len__，走物化后长度检查路径

        try:
            with self.assertRaises(PLLInputError):
                self.pll.update(gen())
            self.assertEqual(self.pll.state.processed, 0)
        finally:
            core.MAX_SAMPLES_PER_CALL = old

    def test_advertised_limit_constant(self):
        # 固定公开的接口边界值（README 同步声明）。
        self.assertEqual(MAX_SAMPLES_PER_CALL, 10_000_000)
        self.assertEqual(MAX_COMPONENT_MAGNITUDE, 1e100)

    def test_non_iterable_rejected(self):
        with self.assertRaises(PLLInputError):
            self.pll.update(1234)

    def test_string_rejected(self):
        with self.assertRaises(PLLInputError):
            self.pll.update("1+0j")


class TestAtomicity(unittest.TestCase):
    """失败不能留下未说明的部分变更。"""

    def setUp(self):
        self.params = PLLParams(0.3, 0.05, 0.5, 1e-2, 8)
        self.pll = PLL(self.params)
        self.pll.update([complex(math.cos(0.3 * k), math.sin(0.3 * k))
                         for k in range(20)])
        self.snapshot = self.pll.state

    def test_bad_block_leaves_state_untouched(self):
        with self.assertRaises(PLLInputError):
            self.pll.update([1 + 0j, 2 + 0j, complex(math.nan, 0.0)])
        self.assertEqual(self.pll.state, self.snapshot)

    def test_oversized_block_leaves_state_untouched(self):
        import pll_diag.core as core
        old = core.MAX_SAMPLES_PER_CALL
        core.MAX_SAMPLES_PER_CALL = 4
        try:
            with self.assertRaises(PLLInputError):
                self.pll.update([0j] * 5)
            self.assertEqual(self.pll.state, self.snapshot)
        finally:
            core.MAX_SAMPLES_PER_CALL = old

    def test_empty_block_is_neutral(self):
        d = self.pll.update(())
        self.assertEqual(self.pll.state, self.snapshot)
        self.assertEqual(d.final_state, self.snapshot)

    def test_reset_bad_state_keeps_current(self):
        with self.assertRaises(PLLInputError):
            self.pll.reset(PLLState(theta=math.nan, omega=0.0, streak=0,
                                   locked=False, processed=0, lost_count=0))
        self.assertEqual(self.pll.state, self.snapshot)


class TestInitialStateValidation(unittest.TestCase):
    P = PLLParams(0.3, 0.05, 0.5, 1e-2, 8)

    def test_theta_out_of_range(self):
        with self.assertRaises(PLLInputError):
            PLL(self.P, PLLState(math.pi, 0.0, 0, False, 0, 0))
        with self.assertRaises(PLLInputError):
            PLL(self.P, PLLState(-math.pi - 1e-3, 0.0, 0, False, 0, 0))

    def test_omega_out_of_wmax(self):
        with self.assertRaises(PLLInputError):
            PLL(self.P, PLLState(0.0, 0.6, 0, False, 0, 0))

    def test_inconsistent_locked_streak(self):
        with self.assertRaises(PLLInputError):
            PLL(self.P, PLLState(0.0, 0.0, 3, True, 0, 0))  # 3 < L=8
        with self.assertRaises(PLLInputError):
            PLL(self.P, PLLState(0.0, 0.0, 9, False, 0, 0))  # 9 >= L 但未锁

    def test_negative_counts_and_bool(self):
        with self.assertRaises(PLLInputError):
            PLL(self.P, PLLState(0.0, 0.0, -1, False, 0, 0))
        with self.assertRaises(PLLInputError):
            PLL(self.P, PLLState(0.0, 0.0, True, False, 0, 0))
        with self.assertRaises(PLLInputError):
            PLL(self.P, PLLState(0.0, 0.0, 0, True, 0, 0))  # locked 类型语义矛盾

    def test_valid_explicit_state_resumes(self):
        st = PLLState(theta=0.2, omega=0.01, streak=9, locked=True,
                      processed=100, lost_count=2)
        pll = PLL(self.P, st)
        self.assertEqual(pll.state, st)
        d = pll.update([0j])  # 失锁应清 streak
        self.assertFalse(pll.locked)
        self.assertEqual(pll.state.processed, 101)
        self.assertEqual(pll.state.lost_count, 3)


class TestWrapAngle(unittest.TestCase):
    def test_half_open_range(self):
        for a in (-3*math.pi, -2*math.pi, -math.pi, 0.0, math.pi,
                  2*math.pi, 12.345, -12.345):
            w = wrap_angle(a)
            self.assertGreaterEqual(w, -math.pi)
            self.assertLess(w, math.pi)

    def test_pi_maps_to_minus_pi(self):
        self.assertEqual(wrap_angle(math.pi), -math.pi)

    def test_rejects_nonfinite_and_bool(self):
        for bad in (math.nan, math.inf, -math.inf):
            with self.assertRaises(PLLInputError):
                wrap_angle(bad)
        with self.assertRaises(PLLInputError):
            wrap_angle(True)


class TestDiagnosisStructure(unittest.TestCase):
    def test_immutability_and_views(self):
        d = track([1 + 0j] * 8,
                  PLLParams(0.3, 0.05, 0.5, 1e-2, 8))
        self.assertEqual(d.status, "locked")
        self.assertEqual(d.frequency_offset, d.final_state.omega)
        self.assertEqual(d.phase, d.final_state.theta)
        with self.assertRaises(Exception):
            d.locked = False  # type: ignore[misc]
        with self.assertRaises(Exception):
            d.samples[0] = None  # type: ignore[index]

    def test_empty_input_result_fields(self):
        d = track([], PLLParams(0.3, 0.05, 0.5, 1e-2, 8))
        self.assertEqual(d.samples, ())
        self.assertIsNone(d.locked_at)
        self.assertIsNone(d.max_abs_error)
        self.assertEqual(d.lost_indices, ())
        self.assertEqual(d.saturated_indices, ())
        self.assertFalse(d.locked)
        self.assertEqual(d.final_state.processed, 0)

    def test_all_lost_input(self):
        d = track([0j, 0j], PLLParams(0.3, 0.05, 0.5, 1e-2, 8))
        self.assertIsNone(d.max_abs_error)
        self.assertEqual(d.lost_indices, (0, 1))
        self.assertEqual(d.status, "unlocked")
        self.assertEqual(d.final_state.lost_count, 2)


if __name__ == "__main__":
    unittest.main()
