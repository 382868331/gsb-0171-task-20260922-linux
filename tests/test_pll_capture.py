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


class TestSamplePeriodSwitch(unittest.TestCase):
    """运行中切换采样周期: 缩放律、相位连续、收敛与失败原子性。

    期望值来源: 缩放关系(omega*r, kp*r, ki*r^2, wmax*r)是题面解析关系,
    相位连续性由信号模型 z = exp(j*(phi0 + Omega*t)) 直接推出, 不经过被测实现。
    """

    PARAMS = dict(kp=0.12, ki=0.008, wmax=1.2, amin=0.2, eps=0.03, lock_count=40)

    def test_scaling_law_analytic(self):
        # 手算: z = exp(j*0.6), theta0 = omega0 = 0
        # error = 0.6; omega = ki*0.6 = theta = ki*0.6 + kp*0.6
        kp, ki = 0.2, 0.02
        pll = PLL(PLLParams(kp=kp, ki=ki, wmax=1.0, amin=0.5,
                            eps=0.05, lock_count=3), sample_period=1e-3)
        pll.step(complex(math.cos(0.6), math.sin(0.6)))
        omega_pre = ki * 0.6
        theta_pre = ki * 0.6 + kp * 0.6
        self.assertAlmostEqual(pll.omega, omega_pre, delta=1e-15)
        change = pll.set_sample_period(2e-3)   # r = 2
        # 解析缩放律: omega*2, kp*2, ki*4, wmax*2; theta 不变
        self.assertAlmostEqual(pll.omega, omega_pre * 2.0, delta=1e-15)
        self.assertAlmostEqual(pll.theta, theta_pre, delta=1e-15)
        self.assertAlmostEqual(pll.params.kp, kp * 2.0, delta=1e-15)
        self.assertAlmostEqual(pll.params.ki, ki * 4.0, delta=1e-15)
        self.assertAlmostEqual(pll.params.wmax, 2.0, delta=1e-15)
        # 未缩放的字段保持不变
        self.assertEqual(pll.params.amin, 0.5)
        self.assertEqual(pll.params.eps, 0.05)
        self.assertEqual(pll.params.lock_count, 3)
        self.assertEqual(pll.sample_period, 2e-3)
        # 物理角频率 Omega = omega/Ts 不变: 12 rad/s
        self.assertAlmostEqual(pll.omega / pll.sample_period,
                               omega_pre / 1e-3, delta=1e-9)
        self.assertAlmostEqual(change.angular_frequency, 12.0, delta=1e-9)

    def test_phase_continuity_and_convergence_after_switch(self):
        # 无噪声固定频率导频: z(t) = exp(j*(phi0 + Omega*t)), f = 1500 Hz
        fs1 = 48000.0
        ts1 = 1.0 / fs1
        freq_hz = 1500.0
        omega_phys = 2.0 * math.pi * freq_hz
        phi0 = 0.7
        n1, n2 = 800, 400
        pll = PLL(PLLParams(**self.PARAMS), sample_period=ts1)
        s1 = [complex(math.cos(phi0 + omega_phys * k * ts1),
                      math.sin(phi0 + omega_phys * k * ts1)) for k in range(n1)]
        r1 = pll.run(s1)
        self.assertTrue(r1.locked)
        self.assertLessEqual(abs(r1.omega - omega_phys * ts1), 1e-9)

        theta_before = pll.theta
        ts2 = 2.0 * ts1   # 48 kHz -> 24 kHz, r = 2
        pll.set_sample_period(ts2)
        # 切换本身不改变相位状态(逐位不变)
        self.assertEqual(pll.theta, theta_before)

        # 信号在新采样时刻 t = n1*ts1 + m*ts2 上连续采样
        s2 = [complex(math.cos(phi0 + omega_phys * (n1 * ts1 + m * ts2)),
                      math.sin(phi0 + omega_phys * (n1 * ts1 + m * ts2)))
              for m in range(n2)]
        r2 = pll.run(s2)
        errors = [abs(d.error) for d in r2.diagnostics]
        # 相位连续: 切换后无重新捕获瞬态, 误差始终在稳态残差量级(<< eps)
        self.assertLess(max(errors), 1e-4)
        self.assertTrue(all(d.locked for d in r2.diagnostics))
        # 收敛: omega 收敛到新周期下的真值 Omega*ts2
        self.assertLessEqual(abs(r2.omega - omega_phys * ts2), 1e-9)
        self.assertTrue(r2.locked)
        # 换算 Hz 与真值一致
        self.assertAlmostEqual(r2.frequency_offset(), freq_hz, delta=1e-6)

    def test_switch_same_period_is_noop(self):
        pll = PLL(PLLParams(**self.PARAMS), sample_period=1e-3)
        pll.run(make_pilot(1.0, 0.4, 0.2, 100))
        omega, theta = pll.omega, pll.theta
        change = pll.set_sample_period(1e-3)
        self.assertEqual(change.ratio, 1.0)
        self.assertEqual(pll.omega, omega)
        self.assertEqual(pll.theta, theta)
        self.assertEqual(pll.params, pll.params.__class__(**self.PARAMS))

    def test_downscale_boundary(self):
        # 边界: r 很小(周期缩小 1000 倍), 增益与频率字同步缩小但仍合法。
        pll = PLL(PLLParams(**self.PARAMS), sample_period=1e-3)
        pll.run(make_pilot(1.0, 0.4, 0.2, 400))
        self.assertTrue(pll.locked)
        omega = pll.omega
        pll.set_sample_period(1e-6)
        self.assertAlmostEqual(pll.omega, omega * 1e-3, delta=1e-18)
        self.assertAlmostEqual(pll.params.kp, 0.12 * 1e-3, delta=1e-18)
        self.assertAlmostEqual(pll.params.ki, 0.008 * 1e-6, delta=1e-24)
        self.assertTrue(pll.locked)  # 捕获计数不受切换影响

    def test_invalid_new_period_rejected_state_unchanged(self):
        pll = PLL(PLLParams(**self.PARAMS), sample_period=1e-3)
        pll.run(make_pilot(1.0, 0.4, 0.2, 100))
        params, omega, theta = pll.params, pll.omega, pll.theta
        for bad in (0.0, -1e-3, float("nan"), float("inf"), True, "2e-3", None):
            with self.subTest(bad=repr(bad)):
                with self.assertRaises(PLLParamError):
                    pll.set_sample_period(bad)
        self.assertIs(pll.params, params)
        self.assertEqual(pll.omega, omega)
        self.assertEqual(pll.theta, theta)
        self.assertEqual(pll.sample_period, 1e-3)

    def test_unstable_scaling_rejected_state_unchanged(self):
        # kp*r >= 1: r = 10 -> kp = 1.2, 拒绝且不留半更新状态
        pll = PLL(PLLParams(kp=0.2, ki=0.02, wmax=1.0, amin=0.5,
                            eps=0.05, lock_count=5), sample_period=1e-3)
        pll.run(make_pilot(1.0, 0.4, 0.2, 50))
        params, omega, theta = pll.params, pll.omega, pll.theta
        with self.assertRaises(PLLParamError):
            pll.set_sample_period(1e-2)
        self.assertIs(pll.params, params)
        self.assertEqual(pll.omega, omega)
        self.assertEqual(pll.theta, theta)
        self.assertEqual(pll.sample_period, 1e-3)
        # ki*r^2 >= kp*r 分支: kp=0.5, ki=0.3, r=1.8 -> kp*r=0.9 < 1 但
        # ki*r^2 = 0.972 >= 0.9, 同样拒绝
        pll2 = PLL(PLLParams(kp=0.5, ki=0.3, wmax=1.0, amin=0.5,
                             eps=0.05, lock_count=5), sample_period=1e-3)
        with self.assertRaises(PLLParamError):
            pll2.set_sample_period(1.8e-3)
        self.assertEqual(pll2.params.kp, 0.5)
        self.assertEqual(pll2.sample_period, 1e-3)

    def test_bad_constructor_sample_period(self):
        for bad in (0.0, -1.0, float("nan"), float("inf"), True, "1e-3"):
            with self.subTest(bad=repr(bad)):
                with self.assertRaises(PLLParamError):
                    PLL(PLLParams(**self.PARAMS), sample_period=bad)

    def test_result_tracks_period_and_frequency_offset(self):
        # 切换后 run() 结果快照携带新周期; frequency_offset() 省略参数时用
        # 跟踪的周期换算, 显式给 sample_rate 的旧用法不变。
        ts1, ts2 = 1.0 / 48000.0, 1.0 / 24000.0
        pll = PLL(PLLParams(**self.PARAMS), sample_period=ts1)
        pll.run(make_pilot(1.0, 0.4, 0.2, 10))
        pll.set_sample_period(ts2)
        result = pll.run(make_pilot(1.0, 0.4, 0.4, 10))
        self.assertEqual(result.sample_period, ts2)
        self.assertAlmostEqual(result.frequency_offset(),
                               result.omega * 24000.0 / (2.0 * math.pi),
                               delta=1e-15)
        self.assertAlmostEqual(result.frequency_offset(48000.0),
                               result.omega * 48000.0 / (2.0 * math.pi),
                               delta=1e-15)
        # 旧默认行为: 未指定周期时默认 1.0, frequency_offset() == omega/(2*pi)
        legacy = PLL(PLLParams(**self.PARAMS)).run(make_pilot(1.0, 0.4, 0.2, 5))
        self.assertEqual(legacy.sample_period, 1.0)
        self.assertAlmostEqual(legacy.frequency_offset(),
                               legacy.omega / (2.0 * math.pi), delta=1e-15)

    def test_report_fields(self):
        pll = PLL(PLLParams(**self.PARAMS), sample_period=1e-3)
        pll.run(make_pilot(1.0, 0.4, 0.2, 100))
        omega, theta = pll.omega, pll.theta
        change = pll.set_sample_period(4e-3)   # r = 4
        self.assertEqual(change.old_period, 1e-3)
        self.assertEqual(change.new_period, 4e-3)
        self.assertEqual(change.ratio, 4.0)
        self.assertEqual(change.omega_before, omega)
        self.assertEqual(change.omega_after, omega * 4.0)
        self.assertEqual(change.theta, theta)
        self.assertEqual(change.old_params.kp, 0.12)
        self.assertEqual(change.new_params.kp, 0.48)
        # 切换前后物理角频率一致
        self.assertAlmostEqual(change.omega_before / change.old_period,
                               change.omega_after / change.new_period,
                               delta=1e-9)


if __name__ == "__main__":
    unittest.main()
