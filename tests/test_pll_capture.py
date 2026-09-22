"""pll_capture 模块单元测试。

参考期望值来源说明:
- tests/data/*.json 中的期望值是导频信号的理论真值(生成时使用的相位/频偏),
  不是调用被测核心得到的;
- test_first_step_analytic 的期望值由题面更新方程在测试内独立逐步推导。
"""

import json
import math
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pll_capture
from pll_capture import (
    MAX_SAMPLES,
    PLL,
    PLLInputError,
    PLLParamError,
    PLLParams,
    wrap_phase,
)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def make_pilot(amplitude, phase, offset, n):
    """确定性生成复数导频: z_k = A * exp(j*(phase + offset*k))。"""
    return [
        complex(amplitude * math.cos(phase + offset * k),
                amplitude * math.sin(phase + offset * k))
        for k in range(n)
    ]


def load_reference(name):
    with open(os.path.join(DATA_DIR, name), "r", encoding="utf-8") as f:
        return json.load(f)


def params_from(d):
    p = d["params"]
    return PLLParams(kp=p["kp"], ki=p["ki"], wmax=p["wmax"],
                     amin=p["amin"], eps=p["eps"], lock_count=p["lock_count"])


GOOD = dict(kp=0.2, ki=0.02, wmax=1.0, amin=0.5, eps=0.05, lock_count=5)


class TestReferences(unittest.TestCase):
    """解析恒相位与固定频偏参考数据, 对照理论真值。"""

    def test_constant_phase_reference(self):
        ref = load_reference("constant_phase.json")
        exp = ref["expected"]
        samples = make_pilot(ref["amplitude"], ref["phase"],
                             ref["freq_offset"], ref["n_samples"])
        result = PLL(params_from(ref)).run(samples)
        self.assertTrue(result.locked is exp["locked"])
        self.assertLessEqual(abs(result.omega - exp["omega"]), exp["omega_tol"])
        self.assertLessEqual(
            abs(wrap_phase(result.theta - exp["theta"])), exp["theta_tol"])
        self.assertLessEqual(
            abs(result.diagnostics[-1].error), exp["final_error_max"])
        self.assertEqual(result.lost_samples, 0)

    def test_fixed_frequency_reference(self):
        ref = load_reference("fixed_frequency.json")
        exp = ref["expected"]
        samples = make_pilot(ref["amplitude"], ref["phase"],
                             ref["freq_offset"], ref["n_samples"])
        result = PLL(params_from(ref)).run(samples)
        self.assertTrue(result.locked is exp["locked"])
        self.assertLessEqual(abs(result.omega - exp["omega"]), exp["omega_tol"])
        self.assertLessEqual(
            abs(result.diagnostics[-1].error), exp["final_error_max"])
        self.assertEqual(result.lost_samples, 0)


class TestUpdateEquations(unittest.TestCase):
    """更新方程的独立逐步推导(不依赖被测核心计算期望值)。"""

    def test_first_step_analytic(self):
        # z = exp(j*0.6), theta0 = omega0 = 0
        # error = 0.6; omega' = ki*0.6; theta' = wrap(ki*0.6 + kp*0.6)
        kp, ki = 0.2, 0.02
        pll = PLL(PLLParams(kp=kp, ki=ki, wmax=1.0, amin=0.5,
                            eps=0.05, lock_count=3))
        d = pll.step(complex(math.cos(0.6), math.sin(0.6)))
        self.assertAlmostEqual(d.error, 0.6, delta=1e-15)
        self.assertAlmostEqual(d.omega, ki * 0.6, delta=1e-15)
        self.assertAlmostEqual(d.theta, ki * 0.6 + kp * 0.6, delta=1e-15)
        self.assertFalse(d.saturated)
        self.assertEqual(d.streak, 0)  # 0.6 > eps

    def test_error_exactly_pi_maps_to_minus_pi(self):
        # atan2(0, -1) = pi, 契约要求误差落在 [-pi, pi), 即映射为 -pi。
        pll = PLL(PLLParams(**GOOD))
        d = pll.step(complex(-1.0, 0.0))
        self.assertEqual(d.error, -math.pi)

    def test_cross_pi_lock(self):
        # 初相 3.1 与信号相位 -3.1 跨 ±pi 边界, 初始误差应取最短路径。
        pll = PLL(PLLParams(kp=0.2, ki=0.02, wmax=1.0, amin=0.5,
                            eps=0.05, lock_count=10), theta0=3.1)
        samples = make_pilot(1.0, -3.1, 0.0, 200)
        result = pll.run(samples)
        expected_first_error = wrap_phase(-3.1 - 3.1)  # 独立计算
        self.assertAlmostEqual(result.diagnostics[0].error,
                               expected_first_error, delta=1e-15)
        self.assertTrue(result.locked)
        self.assertLessEqual(abs(result.omega), 1e-9)

    def test_streak_reset_on_error_spike(self):
        # 任一有效样本 |error| > eps 立即清零计数并退出 locked。
        params = PLLParams(kp=0.2, ki=0.02, wmax=1.0, amin=0.5,
                           eps=0.1, lock_count=5)
        pll = PLL(params)
        pll.run(make_pilot(1.0, 0.3, 0.0, 30))
        self.assertTrue(pll.locked)
        # 注入相位跳变 0.5 rad (> eps) 的样本
        d = pll.step(complex(math.cos(0.8), math.sin(0.8)))
        self.assertEqual(d.streak, 0)
        self.assertFalse(d.locked)
        self.assertFalse(pll.locked)
        # 恢复正常相位后重新计数并重新锁定
        pll.run(make_pilot(1.0, 0.3, 0.0, 30))
        self.assertTrue(pll.locked)


class TestInvalidSamples(unittest.TestCase):
    def test_zero_amplitude_freeze_and_loss(self):
        params = PLLParams(kp=0.2, ki=0.02, wmax=1.0, amin=0.5,
                           eps=0.05, lock_count=10)
        pll = PLL(params)
        pll.run(make_pilot(1.0, 0.4, 0.0, 60))
        omega_before, theta_before = pll.omega, pll.theta
        self.assertTrue(pll.locked)
        # 5 个零幅度样本: 冻结 + 计失锁 + 清零计数
        result = pll.run([0j] * 5)
        self.assertEqual(result.lost_samples, 5)
        self.assertEqual(pll.lost_samples, 5)
        for d in result.diagnostics:
            self.assertFalse(d.valid)
            self.assertIsNone(d.error)
            self.assertEqual(d.omega, omega_before)
            self.assertEqual(d.theta, theta_before)
            self.assertEqual(d.streak, 0)
            self.assertFalse(d.locked)
        self.assertFalse(pll.locked)
        self.assertEqual(pll.omega, omega_before)
        self.assertEqual(pll.theta, theta_before)
        # 恢复有效样本后重新捕获
        pll.run(make_pilot(1.0, 0.4, 0.0, 60))
        self.assertTrue(pll.locked)

    def test_amplitude_equal_amin_is_valid(self):
        # 无效判定是 amp < amin, 等于 amin 的样本有效。
        pll = PLL(PLLParams(**GOOD))
        d = pll.step(complex(0.5, 0.0))
        self.assertTrue(d.valid)
        self.assertEqual(pll.lost_samples, 0)


class TestSaturation(unittest.TestCase):
    def test_omega_clips_at_wmax(self):
        # 频偏 0.8 rad/sample 超出 wmax=0.5: omega 限幅在 0.5, 相位持续滑脱, 无法锁定。
        params = PLLParams(kp=0.1, ki=0.05, wmax=0.5, amin=0.5,
                           eps=0.05, lock_count=10)
        result = PLL(params).run(make_pilot(1.0, 0.0, 0.8, 400))
        self.assertTrue(any(d.saturated for d in result.diagnostics))
        self.assertTrue(all(abs(d.omega) <= 0.5 for d in result.diagnostics))
        self.assertTrue(any(d.saturated and d.omega == 0.5
                            for d in result.diagnostics))
        self.assertFalse(result.locked)


class TestChunking(unittest.TestCase):
    def test_arbitrary_chunks_identical(self):
        params = PLLParams(kp=0.15, ki=0.01, wmax=1.0, amin=0.5,
                           eps=0.03, lock_count=20)
        # 带确定性扰动的导频, 避免平凡输入
        samples = [
            complex((1.0 + 0.05 * math.sin(0.3 * k)) * math.cos(1.0 + 0.4 * k)
                    + 0.01 * math.cos(1.7 * k),
                    (1.0 + 0.05 * math.sin(0.3 * k)) * math.sin(1.0 + 0.4 * k)
                    + 0.01 * math.sin(1.7 * k))
            for k in range(577)
        ]
        whole = PLL(params).run(samples)
        chunked = PLL(params)
        diags = []
        for lo, hi in ((0, 1), (1, 64), (64, 314), (314, 577)):
            diags.extend(chunked.run(samples[lo:hi]).diagnostics)
        # 同一操作序列, 浮点结果应逐位一致
        self.assertEqual(tuple(diags), whole.diagnostics)
        self.assertEqual(chunked.omega, whole.omega)
        self.assertEqual(chunked.theta, whole.theta)
        self.assertEqual(chunked.locked, whole.locked)
        # step() 与 run() 混合亦一致
        mixed = PLL(params)
        mixed_d = [mixed.step(z) for z in samples[:100]]
        mixed_d.extend(mixed.run(samples[100:]).diagnostics)
        self.assertEqual(tuple(mixed_d), whole.diagnostics)


class TestParamValidation(unittest.TestCase):
    def test_bad_params_rejected(self):
        bad = [
            dict(kp=0.0), dict(kp=1.0), dict(kp=-0.1), dict(kp=1.5),
            dict(kp=True), dict(kp=float("nan")), dict(kp=float("inf")),
            dict(ki=0.0), dict(ki=-0.01), dict(ki=0.2),   # ki == kp
            dict(ki=0.5), dict(ki=False), dict(ki=float("nan")),
            dict(wmax=0.0), dict(wmax=-1.0), dict(wmax=float("inf")),
            dict(amin=-0.1), dict(amin=float("nan")),
            dict(eps=0.0), dict(eps=-0.1), dict(eps=float("inf")),
            dict(lock_count=0), dict(lock_count=-3), dict(lock_count=True),
            dict(lock_count=2.5), dict(lock_count="5"),
        ]
        for override in bad:
            with self.subTest(override=override):
                kwargs = {**GOOD, **override}
                with self.assertRaises(PLLParamError):
                    PLLParams(**kwargs)

    def test_param_error_is_value_error_and_locatable(self):
        with self.assertRaises(ValueError) as ctx:
            PLLParams(**{**GOOD, "kp": 1.5})
        self.assertIn("kp", str(ctx.exception))

    def test_int_params_normalized_to_float(self):
        p = PLLParams(kp=0.2, ki=0.02, wmax=1, amin=0, eps=0.05, lock_count=5)
        self.assertIsInstance(p.wmax, float)
        self.assertIsInstance(p.amin, float)

    def test_bad_initial_state_rejected(self):
        with self.assertRaises(PLLParamError):
            PLL(PLLParams(**GOOD), theta0=float("nan"))
        with self.assertRaises(PLLParamError):
            PLL(PLLParams(**GOOD), omega0=float("inf"))
        with self.assertRaises(PLLParamError):
            PLL(PLLParams(**GOOD), omega0=1.5)  # |omega0| > wmax, 不静默限幅
        with self.assertRaises(PLLParamError):
            PLL("not-params")


class TestSampleValidation(unittest.TestCase):
    def test_bad_samples_rejected(self):
        pll = PLL(PLLParams(**GOOD))
        for bad in (True, None, "1+2j", float("nan"), float("inf"),
                    complex(float("nan"), 0.0), complex(0.0, float("inf")),
                    object()):
            with self.subTest(bad=repr(bad)):
                with self.assertRaises(PLLInputError):
                    pll.step(bad)
        self.assertEqual(pll.processed, 0)  # 全部拒绝, 无部分变更

    def test_run_rejects_bad_batch_atomically(self):
        pll = PLL(PLLParams(**GOOD))
        batch = make_pilot(1.0, 0.3, 0.0, 10) + [complex(float("nan"), 0.0)]
        with self.assertRaises(PLLInputError):
            pll.run(batch)
        # 整批校验先于更新: 状态保持调用前原样
        self.assertEqual(pll.processed, 0)
        self.assertEqual(pll.omega, 0.0)
        self.assertEqual(pll.theta, 0.0)

    def test_run_rejects_non_iterable_and_string(self):
        pll = PLL(PLLParams(**GOOD))
        with self.assertRaises(PLLInputError):
            pll.run(42)
        with self.assertRaises(PLLInputError):
            pll.run("1+1j")

    def test_run_respects_max_samples(self):
        pll = PLL(PLLParams(**GOOD))
        with mock.patch.object(pll_capture, "MAX_SAMPLES", 3):
            with self.assertRaises(PLLInputError):
                pll.run([1 + 0j] * 4)
        self.assertEqual(pll.processed, 0)
        self.assertEqual(MAX_SAMPLES, 1_000_000)

    def test_real_sample_treated_as_complex(self):
        pll = PLL(PLLParams(**GOOD))
        d = pll.step(1.0)  # 实数 -> 虚部 0
        self.assertTrue(d.valid)
        self.assertAlmostEqual(d.amplitude, 1.0, delta=1e-15)


if __name__ == "__main__":
    unittest.main()
