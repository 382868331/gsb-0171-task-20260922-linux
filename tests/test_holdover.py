"""tracking / holdover / reacquire 三态状态机的单元测试。

所有期望值独立于被测核心推导(真值由生成信号的参数直接给出)。
"""

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pll_capture import (
    PLL,
    PLLInputError,
    PLLParamError,
    PLLParams,
    STATE_HOLDOVER,
    STATE_REACQUIRE,
    STATE_TRACKING,
    wrap_phase,
)


def pilot(amplitude, phase, offset, n, drift=0.0):
    """z_k = A * exp(j*(phase + offset*k + 0.5*drift*k^2))(已知相位)。"""
    return [
        complex(amplitude * math.cos(phase + offset * k + 0.5 * drift * k * k),
                amplitude * math.sin(phase + offset * k + 0.5 * drift * k * k))
        for k in range(n)
    ]


HPARAMS = dict(kp=0.15, ki=0.01, wmax=1.0, amin=0.3, eps=0.04,
               lock_count=20, holdover_samples=8, median_window=11,
               max_phase_step=0.05, freq_eps=0.01, reacquire_count=25)


def build_p():
    return PLLParams(**HPARAMS)


class TestHoldoverEntry(unittest.TestCase):
    def test_disabled_by_default_matches_legacy(self):
        # 默认参数: 没有 holdover_samples, 状态恒为 tracking。
        pll = PLL(PLLParams(kp=0.2, ki=0.02, wmax=1.0, amin=0.5,
                            eps=0.05, lock_count=10))
        result = pll.run(pilot(1.0, 0.4, 0.0, 40) + [0j] * 5)
        self.assertEqual(result.state, STATE_TRACKING)
        self.assertTrue(all(d.state == STATE_TRACKING for d in result.diagnostics))

    def test_state_machine_and_phase_extrapolation(self):
        pll = PLL(build_p())
        pll.run(pilot(1.0, 0.3, 0.25, 500))
        self.assertTrue(pll.locked)
        result = pll.run([0.02 + 0j] * 20)
        states = [d.state for d in result.diagnostics]
        # 前 holdover_samples-1 个仍在 tracking(冻结), 第 8 个起进入 holdover。
        self.assertEqual(states[:7], [STATE_TRACKING] * 7)
        self.assertEqual(states[7:], [STATE_HOLDOVER] * 13)
        # holdover 期间 omega 冻结为常量, 相位以该常量推进
        frozen = result.diagnostics[7].omega
        for d in result.diagnostics[8:]:
            self.assertEqual(d.omega, frozen)
            self.assertFalse(d.valid)
        # 第一个 holdover 样本的相位步进恰为冻结频率(独立校验更新方程)
        d_prev, d_first = result.diagnostics[6], result.diagnostics[7]
        self.assertAlmostEqual(
            wrap_phase(d_first.theta - d_prev.theta), frozen, delta=1e-15)

    def test_frozen_omega_is_window_median(self):
        pll = PLL(build_p())
        pll.run(pilot(1.0, 0.3, 0.25, 500))
        window = sorted(pll._omega_window)
        n = len(window)
        expected_median = window[n // 2] if n % 2 else \
            (window[n // 2 - 1] + window[n // 2]) / 2.0
        pll.run([0.02 + 0j] * 8)
        self.assertEqual(pll.state, STATE_HOLDOVER)
        self.assertEqual(pll.omega, expected_median)
        self.assertEqual(pll.omega_hold, expected_median)

    def test_low_count_persists_across_chunks(self):
        pll = PLL(build_p())
        pll.run(pilot(1.0, 0.3, 0.2, 400))
        # 把 8 个低幅样本拆到三个 chunk: 计数必须跨 chunk 延续。
        pll.run([0.02 + 0j] * 3)
        self.assertEqual(pll.state, STATE_TRACKING)
        pll.run([0.02 + 0j] * 2)
        self.assertEqual(pll.state, STATE_TRACKING)
        r = pll.run([0.02 + 0j] * 3)
        self.assertEqual(pll.state, STATE_HOLDOVER)
        self.assertEqual(r.diagnostics[-1].state, STATE_HOLDOVER)

    def test_no_integrator_update_during_holdover(self):
        pll = PLL(build_p())
        pll.run(pilot(1.0, 0.3, 0.2, 400))
        noise = [complex(0.01 * math.cos(3.7 * k), 0.01 * math.sin(3.7 * k))
                 for k in range(30)]
        result = pll.run(noise)
        omegas = {d.omega for d in result.diagnostics if d.state == STATE_HOLDOVER}
        self.assertEqual(len(omegas), 1)  # 噪声样本不更新积分项


class TestReacquire(unittest.TestCase):
    def test_phase_correction_bounded_per_sample(self):
        pll = PLL(build_p())
        pll.run(pilot(1.0, 0.3, 0.2, 400))
        pll.run([0.02 + 0j] * 20)
        self.assertEqual(pll.state, STATE_HOLDOVER)
        # 恢复时给出明显相位偏差: 只能逐步有界追上
        result = pll.run(pilot(1.0, 0.9, 0.2, 200, drift=0.0001))
        reacq = [d for d in result.diagnostics if d.state == STATE_REACQUIRE]
        self.assertTrue(len(reacq) > 1)
        for prev, cur in zip(result.diagnostics, result.diagnostics[1:]):
            if cur.state == STATE_REACQUIRE and cur.valid:
                step = wrap_phase(cur.theta - prev.theta)
                correction = step - cur.omega
                self.assertLessEqual(abs(correction),
                                     HPARAMS["max_phase_step"] + 1e-12)
        self.assertTrue(pll.locked)
        self.assertEqual(pll.state, STATE_TRACKING)

    def test_false_recovery_does_not_relock(self):
        pll = PLL(build_p())
        pll.run(pilot(1.0, 0.3, 0.2, 400))
        pll.run([0.02 + 0j] * 20)
        # 假恢复: 6 个高幅但相位错误的样本(不足 reacquire_count), 随后缺口再现
        fake = pilot(0.9, 2.0, 0.2, 6)
        r1 = pll.run(fake)
        self.assertTrue(any(d.state == STATE_REACQUIRE
                            for d in r1.diagnostics))
        self.assertFalse(pll.locked)
        r2 = pll.run([0.02 + 0j] * 12)
        self.assertEqual(pll.state, STATE_HOLDOVER)
        self.assertTrue(all(not d.locked for d in r2.diagnostics))

    def test_relock_requires_both_conditions(self):
        pll = PLL(build_p())
        pll.run(pilot(1.0, 0.3, 0.2, 400))
        pll.run([0.02 + 0j] * 20)
        # 恒定相位但频率错误: 相位误差可能小, 但频率误差条件不满足
        bad_freq = pilot(1.0, 0.0, 0.55, 100)
        pll.run(bad_freq)
        self.assertEqual(pll.state, STATE_REACQUIRE)
        self.assertFalse(pll.locked)

    def test_amplitude_equal_amin_is_valid_and_exits_holdover(self):
        # 门限相等归属: amp == amin 为有效(== 走有效这一边)。
        pll = PLL(build_p())
        pll.run(pilot(1.0, 0.3, 0.2, 400))
        pll.run([0.02 + 0j] * 20)
        d = pll.step(complex(HPARAMS["amin"], 0.0))
        self.assertTrue(d.valid)
        self.assertEqual(d.state, STATE_REACQUIRE)


class TestChunkEquivalence(unittest.TestCase):
    def test_chunked_bitwise_identical_with_gaps_and_false_recovery(self):
        p = build_p()
        samples = (
            pilot(1.0, 0.5, 0.18, 300)
            + pilot(0.05, 0.0, 0.18, 30)          # 缺口(低幅噪声)
            + pilot(0.9, 2.1, 0.18, 6)            # 假恢复
            + pilot(0.04, 0.0, 0.18, 25)          # 再次缺口
            + pilot(1.0, 0.5, 0.18, 400, drift=0.0002)  # 真恢复, 频率缓漂
        )
        whole = PLL(p).run(samples)

        chunked = PLL(p)
        diags = []
        bounds = (0, 1), (1, 30), (30, 318), (318, 337), (337, 350), (350, 761)
        for lo, hi in bounds:
            diags.extend(chunked.run(samples[lo:hi]).diagnostics)
        self.assertEqual(tuple(diags), whole.diagnostics)
        self.assertEqual(chunked.omega, whole.omega)
        self.assertEqual(chunked.theta, whole.theta)
        self.assertEqual(chunked.state, whole.state)
        self.assertEqual(chunked.locked, whole.locked)

        # step()/run() 混合同样逐点一致
        mixed = PLL(p)
        md = [mixed.step(z) for z in samples[:123]]
        md.extend(mixed.run(samples[123:]).diagnostics)
        self.assertEqual(tuple(md), whole.diagnostics)


class TestHoldoverParams(unittest.TestCase):
    def test_bad_new_params_rejected(self):
        for bad in (
            dict(holdover_samples=-1), dict(holdover_samples=True),
            dict(holdover_samples=2.0), dict(median_window=0),
            dict(median_window=True), dict(reacquire_count=0),
            dict(reacquire_count=False), dict(max_phase_step=0.0),
            dict(max_phase_step=-1.0), dict(max_phase_step=float("nan")),
            dict(freq_eps=0.0), dict(freq_eps=float("inf")),
        ):
            with self.subTest(bad=bad):
                with self.assertRaises(PLLParamError):
                    PLLParams(**{**HPARAMS, **bad})


if __name__ == "__main__":
    unittest.main()
