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
    PLLSwitchError,
    RateSwitchReport,
    SamplePeriod,
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


def physical_pilot(amplitude, phase0, freq_hz, ts, n):
    """按物理时间确定性生成导频: z_k = A*exp(j*(phase0 + 2*pi*f*k*Ts))。

    期望值(相位/频偏)由物理模型直接给出,不经过被测核心。
    """
    w = 2.0 * math.pi * freq_hz * ts  # rad/sample
    return make_pilot(amplitude, phase0, w, n), w


class TestSamplePeriod(unittest.TestCase):
    def test_seconds_and_default(self):
        sp = SamplePeriod.seconds(1.0 / 48000.0)
        self.assertEqual(sp.unit, "seconds")
        self.assertAlmostEqual(sp.to_seconds(), 1.0 / 48000.0, delta=1e-18)
        self.assertEqual(SamplePeriod().to_seconds(), 1.0)
        # 无量纲单位仅允许 1.0
        sp2 = SamplePeriod(1.0, "samples")
        self.assertEqual(sp2.to_seconds(), 1.0)

    def test_bad_period_rejected(self):
        for bad in (0.0, -1e-9, float("nan"), float("inf"), True, "1e-3", None):
            with self.subTest(bad=repr(bad)):
                with self.assertRaises(PLLParamError):
                    SamplePeriod.seconds(bad)
        with self.assertRaises(PLLParamError):
            SamplePeriod(2.0, "samples")
        with self.assertRaises(PLLParamError):
            SamplePeriod(1.0, "weeks")

    def test_params_ts_validation(self):
        p = PLLParams(**GOOD, ts=SamplePeriod.seconds(2e-5))
        self.assertAlmostEqual(p.ts.to_seconds(), 2e-5, delta=1e-20)
        with self.assertRaises(PLLParamError):
            PLLParams(**GOOD, ts=2e-5)
        # 不传 ts 与传单位周期都不改变增益数值
        self.assertEqual(PLLParams(**GOOD).kp, p.kp)
        self.assertEqual(PLLParams(**GOOD).ki, p.ki)


class TestRateSwitchAnalytic(unittest.TestCase):
    """切换缩放的解析式核对(期望值全部手算,不用被测核心生成)。"""

    def test_gain_frequency_phase_scaling_exact(self):
        # r = 0.5(周期减半/采样率加倍): kp'=0.5kp, ki'=0.25ki,
        # wmax'=0.5wmax, omega'=0.5omega, theta 数值不变。
        params = PLLParams(kp=0.2, ki=0.02, wmax=1.0, amin=0.5,
                           eps=0.05, lock_count=5,
                           ts=SamplePeriod.seconds(1.0))
        pll = PLL(params, theta0=0.4, omega0=0.3)
        theta_initial = pll.theta  # wrap_phase(0.4) 的双精度结果
        pll._streak = 3  # 直接造一个非平凡计数(经属性核对用)
        self.assertEqual(pll.streak, 3)
        rep = pll.set_sample_period(SamplePeriod.seconds(0.5))
        self.assertIsInstance(rep, RateSwitchReport)
        # 手算期望值(0.3*0.5、0.02/4 等在二进制下为精确缩放)
        self.assertEqual(rep.ratio, 0.5)
        self.assertEqual(pll.params.kp, 0.1)
        self.assertEqual(pll.params.ki, 0.005)
        self.assertEqual(pll.params.wmax, 0.5)
        self.assertEqual(pll.omega, 0.15)
        self.assertAlmostEqual(pll.theta, 0.4, delta=1e-15)
        self.assertEqual(pll.theta, theta_initial)
        self.assertEqual(rep.theta_new, rep.theta_old)
        self.assertEqual(rep.omega_new, 0.15)
        self.assertEqual(rep.omega_old, 0.3)
        # eps/amin/lock_count 与计数状态不随采样周期改变
        self.assertEqual(pll.params.eps, 0.05)
        self.assertEqual(pll.params.amin, 0.5)
        self.assertEqual(pll.params.lock_count, 5)
        self.assertEqual(pll.streak, 3)
        self.assertFalse(pll.locked)
        self.assertEqual(pll.processed, 0)
        self.assertEqual(pll.sample_period_seconds, 0.5)
        self.assertEqual(pll.sample_rate, 2.0)

    def test_physical_dynamic_invariants(self):
        # 连续化不变量: omega_n = sqrt(ki)/Ts (rad/s), zeta = kp/(2 sqrt(ki))。
        ts1, ts2 = 1.0 / 48000.0, 1.0 / 24000.0  # r = 2
        kp, ki = 0.12, 0.008
        params = PLLParams(kp=kp, ki=ki, wmax=1.2, amin=0.2,
                           eps=0.03, lock_count=20,
                           ts=SamplePeriod.seconds(ts1))
        pll = PLL(params, omega0=-0.07)
        r = ts2 / ts1
        pll.set_sample_period(SamplePeriod.seconds(ts2))
        p1, p2 = params, pll.params
        wn1 = math.sqrt(ki) / ts1
        wn2 = math.sqrt(p2.ki) / ts2
        zeta1 = kp / (2.0 * math.sqrt(ki))
        zeta2 = p2.kp / (2.0 * math.sqrt(p2.ki))
        self.assertAlmostEqual(wn2, wn1, delta=1e-7 * wn1)
        self.assertAlmostEqual(zeta2, zeta1, delta=1e-12)
        # 手算: kp'=0.24, ki'=0.032, omega'=-0.14
        self.assertAlmostEqual(p2.kp, kp * r, delta=1e-15)
        self.assertAlmostEqual(p2.ki, ki * r * r, delta=1e-15)
        self.assertAlmostEqual(pll.omega, -0.14, delta=1e-15)

    def test_physical_frequency_hz_invariant(self):
        # omega(rad/sample)随周期缩放,物理频偏 Hz 不变。
        ts1, ts2 = 1.0 / 48000.0, 1.0 / 96000.0  # r = 0.5
        f_hz = 1500.0
        w1 = 2.0 * math.pi * f_hz * ts1
        params = PLLParams(kp=0.2, ki=0.02, wmax=1.0, amin=0.0,
                           eps=0.2, lock_count=5,
                           ts=SamplePeriod.seconds(ts1))
        pll = PLL(params, omega0=w1)
        self.assertAlmostEqual(pll.frequency_hz, f_hz, delta=1e-9)
        rep = pll.set_sample_period(SamplePeriod.seconds(ts2))
        self.assertAlmostEqual(pll.frequency_hz, f_hz, delta=1e-9)
        self.assertAlmostEqual(rep.physical_frequency_hz, f_hz, delta=1e-9)
        # 新频率字应等于信号在新速率下的真值 2*pi*f*Ts2(独立公式)
        self.assertAlmostEqual(pll.omega, 2.0 * math.pi * f_hz * ts2,
                               delta=1e-15)

    def test_convenience_input_forms(self):
        params = PLLParams(kp=0.2, ki=0.02, wmax=1.0, amin=0.5,
                           eps=0.05, lock_count=5)
        p1 = PLL(params).set_sample_period(0.5).params_new
        p2 = PLL(params).set_sample_period((0.5, "seconds")).params_new
        p3 = PLL(params).set_sample_period(SamplePeriod.seconds(0.5)).params_new
        self.assertEqual((p1.kp, p1.ki, p1.wmax),
                         (p2.kp, p2.ki, p2.wmax))
        self.assertEqual((p1.kp, p1.ki, p1.wmax),
                         (p3.kp, p3.ki, p3.wmax))

    def test_unitless_to_seconds_switch(self):
        # 无量纲单位(视为 1 秒)切到真实秒周期也合法。
        params = PLLParams(kp=0.2, ki=0.02, wmax=1.0, amin=0.5,
                           eps=0.05, lock_count=5,
                           ts=SamplePeriod(1.0, "samples"))
        pll = PLL(params, omega0=0.4)
        rep = pll.set_sample_period(SamplePeriod.seconds(2e-5))
        self.assertAlmostEqual(rep.ratio, 2e-5, delta=1e-20)
        self.assertAlmostEqual(pll.omega, 0.4 * 2e-5, delta=1e-18)


class TestRateSwitchContinuity(unittest.TestCase):
    """无噪声固定频率 + 一次切换:相位连续性、锁定保持与收敛误差。"""

    def _locked_pll(self, ts, f_hz, n, kp=0.2, ki=0.02, eps=0.2):
        params = PLLParams(kp=kp, ki=ki, wmax=1.0, amin=0.0,
                           eps=eps, lock_count=20,
                           ts=SamplePeriod.seconds(ts))
        pll = PLL(params)
        seg, w = physical_pilot(1.0, 0.0, f_hz, ts, n)
        res = pll.run(seg)
        return pll, params, res, w

    def test_phase_continuous_and_relock(self):
        fs1, fs2 = 48000.0, 96000.0
        ts1, ts2 = 1.0 / fs1, 1.0 / fs2
        r = ts2 / ts1  # 0.5
        f_hz = 1000.0
        n1, n2 = 600, 400
        pll, _, _, w1 = self._locked_pll(ts1, f_hz, n1)
        self.assertTrue(pll.locked)
        self.assertAlmostEqual(pll.omega, w1, delta=1e-6)

        theta_before = pll.theta
        omega_before = pll.omega
        streak_before = pll.streak
        rep = pll.set_sample_period(SamplePeriod.seconds(ts2))
        # 物理相位连续:theta 逐位不变
        self.assertEqual(pll.theta, theta_before)
        self.assertEqual(rep.theta_new, rep.theta_old)
        self.assertAlmostEqual(rep.omega_new, omega_before * r, delta=1e-15)
        self.assertEqual(pll.streak, streak_before)
        self.assertTrue(pll.locked)  # 切换本身不消耗样本、不清计数

        # 第二段样本: 时间戳从"上一个样本之后隔一个新周期"继续。
        # 最后处理的样本在 t=(n1-1)*Ts1, 下一个在 t=(n1-1)*Ts1+Ts2。
        seg2 = [
            complex(math.cos(2.0 * math.pi * f_hz * ((n1 - 1) * ts1 + m * ts2)),
                    math.sin(2.0 * math.pi * f_hz * ((n1 - 1) * ts1 + m * ts2)))
            for m in range(1, n2 + 1)
        ]
        res2 = pll.run(seg2)
        first = res2.diagnostics[0]
        # 独立推导的首个新速率样本误差: 锁定时 theta 领先一个旧增量,
        # 预测相位为 phi0+n1*w1; 实际相位为 phi0+(n1-1)w1+w2;
        # error = w2 - w1 = (r-1)*w1。
        w2 = 2.0 * math.pi * f_hz * ts2
        expected_first_error = wrap_phase(w2 - w1)
        self.assertAlmostEqual(first.error, expected_first_error, delta=1e-6)
        # |(r-1)*w1| <= eps 时锁定不中断
        self.assertLessEqual(abs(expected_first_error), 0.2)
        self.assertTrue(first.locked)
        self.assertTrue(res2.locked)
        # 新速率下重新收敛到信号真值,物理频偏仍是 f_hz
        self.assertAlmostEqual(res2.omega, w2, delta=1e-7)
        self.assertLessEqual(abs(res2.diagnostics[-1].error), 1e-6)
        # Hz 容差由 omega 容差换算: 1e-7/(2*pi*Ts2)
        self.assertAlmostEqual(pll.frequency_hz, f_hz,
                               delta=1e-7 / (2.0 * math.pi * ts2) * 5)
        self.assertEqual(res2.lost_samples, 0)

    def test_switch_then_chunk_equivalence(self):
        # 同一切换操作序列,整批/逐步必须逐位一致。
        ts1, ts2 = 1.0 / 48000.0, 1.0 / 24000.0
        f_hz = 800.0
        n1, n2 = 300, 200

        def build():
            seg1, _ = physical_pilot(1.0, 0.3, f_hz, ts1, n1)
            seg2 = [
                complex(math.cos(0.3 + 2.0 * math.pi * f_hz *
                                 ((n1 - 1) * ts1 + m * ts2)),
                        math.sin(0.3 + 2.0 * math.pi * f_hz *
                                 ((n1 - 1) * ts1 + m * ts2)))
                for m in range(1, n2 + 1)
            ]
            return seg1, seg2

        def make_pll():
            return PLL(PLLParams(kp=0.15, ki=0.01, wmax=1.0, amin=0.0,
                                 eps=0.05, lock_count=20,
                                 ts=SamplePeriod.seconds(ts1)))

        seg1, seg2 = build()
        a = make_pll()
        ra = a.run(seg1)
        a.set_sample_period(SamplePeriod.seconds(ts2))
        rb = a.run(seg2)

        b = make_pll()
        d1 = [b.step(z) for z in seg1]
        b.set_sample_period(SamplePeriod.seconds(ts2))
        d2 = [b.step(z) for z in seg2]
        self.assertEqual(tuple(d1), ra.diagnostics)
        self.assertEqual(tuple(d2), rb.diagnostics)
        self.assertEqual(b.omega, a.omega)
        self.assertEqual(b.theta, a.theta)
        self.assertEqual(b.params, a.params)


class TestRateSwitchBoundary(unittest.TestCase):
    def test_ratio_one_is_idempotent(self):
        params = PLLParams(kp=0.2, ki=0.02, wmax=1.0, amin=0.5,
                           eps=0.05, lock_count=5,
                           ts=SamplePeriod.seconds(0.25))
        pll = PLL(params, theta0=-1.1, omega0=0.3)
        pll.run(make_pilot(1.0, 0.0, 0.3, 30))
        omega0, theta0, p0 = pll.omega, pll.theta, pll.params
        rep = pll.set_sample_period(SamplePeriod.seconds(0.25))
        self.assertEqual(rep.ratio, 1.0)
        self.assertEqual(pll.omega, omega0)
        self.assertEqual(pll.theta, theta0)
        self.assertEqual(pll.params, p0)
        self.assertEqual(pll.params.kp, 0.2)

    def test_switch_at_zero_samples(self):
        pll = PLL(PLLParams(kp=0.2, ki=0.02, wmax=1.0, amin=0.5,
                            eps=0.05, lock_count=5))
        rep = pll.set_sample_period(0.5)
        self.assertEqual(rep.index, 0)
        self.assertEqual(pll.processed, 0)
        self.assertEqual(pll.theta, 0.0)

    def test_saturated_omega_scales_with_rail(self):
        # 频偏 0.8、wmax=0.5: omega 贴在 0.5 限幅上。
        params = PLLParams(kp=0.1, ki=0.05, wmax=0.5, amin=0.5,
                           eps=0.05, lock_count=10)
        pll = PLL(params)
        pll.run(make_pilot(1.0, 0.0, 0.8, 400))
        self.assertEqual(pll.omega, 0.5)
        # r=0.5: 新限幅 0.25,缩放后 omega 恰好落在新限幅上。
        rep = pll.set_sample_period(SamplePeriod.seconds(0.5))
        self.assertEqual(pll.omega, 0.25)
        self.assertEqual(pll.omega, pll.params.wmax)
        self.assertEqual(rep.params_new.wmax, 0.25)
        # r=2 反向(从单位周期): 限幅与 omega 同步翻倍仍贴幅。
        # 取 ki=0.02 以保证 4ki < kp, 缩放后仍在稳定范围内。
        params_up = PLLParams(kp=0.1, ki=0.02, wmax=0.5, amin=0.5,
                              eps=0.05, lock_count=10)
        pll2 = PLL(params_up)
        pll2.run(make_pilot(1.0, 0.0, 0.8, 400))
        rep2 = pll2.set_sample_period(SamplePeriod.seconds(2.0))
        self.assertEqual(rep2.params_new.kp, 0.2)
        self.assertEqual(rep2.params_new.ki, 0.08)
        self.assertEqual(rep2.params_new.wmax, 1.0)
        self.assertEqual(pll2.omega, 1.0)


class TestRateSwitchRejection(unittest.TestCase):
    def test_bad_period_rejected_atomically(self):
        params = PLLParams(kp=0.2, ki=0.02, wmax=1.0, amin=0.5,
                           eps=0.05, lock_count=5,
                           ts=SamplePeriod.seconds(1.0))
        pll = PLL(params, theta0=0.9, omega0=0.3)
        pll.run(make_pilot(1.0, 0.0, 0.3, 30))
        snap = (pll.omega, pll.theta, pll.streak, pll.processed,
                pll.params.kp, pll.params.ki, pll.params.wmax,
                pll.sample_period_seconds)
        bad_inputs = [
            0.0, -0.5, float("nan"), float("inf"), True, "0.5", None,
            (0.5,), (0.5, "weeks"), (0.0, "seconds"),
            (float("inf"), "seconds"), object(),
        ]
        for bad in bad_inputs:
            with self.subTest(bad=repr(bad)):
                with self.assertRaises(PLLSwitchError):
                    pll.set_sample_period(bad)
        self.assertEqual(
            (pll.omega, pll.theta, pll.streak, pll.processed,
             pll.params.kp, pll.params.ki, pll.params.wmax,
             pll.sample_period_seconds),
            snap,
        )

    def test_instability_kp_rejected_atomically(self):
        # kp*r >= 1: 拒绝, 状态不变。
        params = PLLParams(kp=0.6, ki=0.05, wmax=1.0, amin=0.5,
                           eps=0.05, lock_count=5)
        pll = PLL(params, omega0=0.2)
        with self.assertRaises(PLLSwitchError) as ctx:
            pll.set_sample_period(2.0)  # r=2 -> kp'=1.2
        self.assertIn("kp", str(ctx.exception))
        self.assertEqual(pll.params.kp, 0.6)
        self.assertEqual(pll.omega, 0.2)
        self.assertEqual(pll.sample_period_seconds, 1.0)
        # 边界: kp*r 恰好 1.0 也拒绝(要求严格 < 1)
        pll2 = PLL(PLLParams(kp=0.5, ki=0.02, wmax=1.0, amin=0.5,
                             eps=0.05, lock_count=5))
        with self.assertRaises(PLLSwitchError):
            pll2.set_sample_period(2.0)

    def test_instability_ki_rejected_atomically(self):
        # kp'=0.22 < 1 但 ki'=0.19*1.21=0.2299 >= kp': 命中第二条稳定性检查。
        params = PLLParams(kp=0.2, ki=0.19, wmax=1.0, amin=0.5,
                           eps=0.05, lock_count=5)
        pll = PLL(params)
        with self.assertRaises(PLLSwitchError) as ctx:
            pll.set_sample_period(1.1)
        self.assertIn("ki", str(ctx.exception))
        self.assertEqual(pll.params.kp, 0.2)
        self.assertEqual(pll.params.ki, 0.19)
        self.assertEqual(pll.sample_period_seconds, 1.0)

    def test_switch_error_is_value_error(self):
        pll = PLL(PLLParams(**GOOD))
        with self.assertRaises(ValueError):
            pll.set_sample_period(0.0)

    def test_failed_switch_then_behavior_matches_control(self):
        # 拒绝切换后继续处理,轨迹应与从未尝试切换的对照实例逐位一致。
        params = PLLParams(kp=0.6, ki=0.05, wmax=1.0, amin=0.5,
                           eps=0.05, lock_count=5)
        a, b = PLL(params), PLL(params)
        samples = make_pilot(1.0, 0.3, 0.2, 40)
        with self.assertRaises(PLLSwitchError):
            a.set_sample_period(2.0)
        self.assertEqual(a.run(samples).diagnostics,
                         b.run(samples).diagnostics)


class TestPhysicalUnits(unittest.TestCase):
    def test_frequency_properties_and_legacy_conversion(self):
        ts = 1.0 / 48000.0
        w = 0.3
        params = PLLParams(kp=0.2, ki=0.02, wmax=1.0, amin=0.5,
                           eps=0.05, lock_count=5,
                           ts=SamplePeriod.seconds(ts))
        pll = PLL(params, omega0=w)
        self.assertAlmostEqual(pll.sample_rate, 48000.0, delta=1e-6)
        self.assertAlmostEqual(pll.frequency_hz,
                               w * 48000.0 / (2.0 * math.pi), delta=1e-9)
        # 旧换算入口保持可用且口径一致
        res = pll.run([1 + 0j])
        self.assertAlmostEqual(res.frequency_offset(48000.0),
                               pll.frequency_hz, delta=1e-9)
        with self.assertRaises(PLLInputError):
            res.frequency_offset(0.0)


if __name__ == "__main__":
    unittest.main()
