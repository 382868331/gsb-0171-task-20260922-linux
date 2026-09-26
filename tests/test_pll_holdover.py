"""holdover/reacquire 状态机的单元测试。

期望值来源说明:
- 合成信号的真值(相位/频率)由测试内确定性生成器独立给出, 不调用被测核心;
- 中位数期望由测试内独立排序计算;
- 分块一致性以 == 断言(同一操作序列, 浮点逐位一致)。
"""

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pll_capture import (
    PLL,
    PLLParamError,
    PLLParams,
    STATE_HOLDOVER,
    STATE_REACQUIRE,
    STATE_TRACKING,
    wrap_phase,
)

# 启用 holdover 的参数; 锁定/重捕获计数较小以便测试。
HO = dict(kp=0.12, ki=0.008, wmax=1.2, amin=0.2, eps=0.03, lock_count=20,
          holdover_enter=8, holdover_window=16, reacquire_step=0.08,
          freq_eps=0.02, reacquire_count=15)


def synth(n, amp, omega, phase0=0.6):
    """确定性合成导频。amp/omega 为常数或 k 的函数。

    返回 (samples, phases): phases[k] 是生成第 k 个样本所用的真实相位。
    """
    samples, phases = [], []
    phi = phase0
    for k in range(n):
        a = amp(k) if callable(amp) else amp
        w = omega(k) if callable(omega) else omega
        samples.append(complex(a * math.cos(phi), a * math.sin(phi)))
        phases.append(phi)
        phi += w
    return samples, phases


def median(values):
    s = sorted(values)
    n = len(s)
    m = n // 2
    return s[m] if n % 2 else (s[m - 1] + s[m]) / 2.0


def gap_signal(lock_n=300, gap_n=40, tail_n=120, offset=0.30,
               false_recover=None):
    """锁定段 + 低幅缺口 + 恢复段; 可选在中间插入一次假恢复。

    false_recover: (valid_n, low_n) 在缺口中段先恢复 valid_n 个样本,
    再跌落 low_n 个样本, 然后才真正恢复。
    """
    segs = [(lock_n, 1.0)]
    if false_recover:
        v, lo = false_recover
        segs += [(gap_n, 0.0), (v, 1.0), (lo, 0.0)]
    else:
        segs += [(gap_n, 0.0)]
    segs += [(tail_n, 1.0)]
    amp_of, points = [], []
    for n, a in segs:
        amp_of.extend([a] * n)
    return synth(len(amp_of), lambda k: amp_of[k], offset)


class TestHoldoverEntry(unittest.TestCase):
    def test_freeze_below_threshold_then_holdover(self):
        # 前 holdover_enter-1 个低幅样本完全冻结(旧行为),
        # 第 holdover_enter 个起进入 holdover 并以冻结频率推进相位。
        params = PLLParams(**HO)
        pll = PLL(params)
        lock_diags = pll.run(synth(200, 1.0, 0.30)[0]).diagnostics
        self.assertTrue(pll.locked)
        theta_frozen = pll.theta
        omega_frozen = median([d.omega for d in lock_diags[-16:]])
        result = pll.run([0j] * 7)  # 7 < holdover_enter=8
        for d in result.diagnostics:
            self.assertEqual(d.state, STATE_TRACKING)
            self.assertEqual(d.theta, theta_frozen)
            self.assertEqual(d.omega, pll.omega)
        d8 = pll.step(0j)  # 第 8 个: 进入 holdover
        self.assertEqual(d8.state, STATE_HOLDOVER)
        self.assertFalse(d8.valid)
        self.assertIsNone(d8.error)
        self.assertEqual(d8.omega, omega_frozen)  # 频率冻结
        self.assertNotEqual(d8.theta, theta_frozen)  # 相位继续推进
        d9 = pll.step(0j)
        step = wrap_phase(d9.theta - d8.theta)
        # 每样本恰好推进 omega_hold(wrap 往返允许 1 ulp 量级残差)
        self.assertAlmostEqual(step, d8.omega, delta=1e-12)

    def test_holdover_freezes_median_of_recent_valid_omega(self):
        # 冻结频率 = 最近 holdover_window 个有效 omega 的中位数(独立计算)。
        params = PLLParams(**{**HO, "holdover_window": 10})
        pll = PLL(params)
        # 频率缓变的导频, 使最近若干 omega 互不相同
        samples, _ = synth(300, 1.0, lambda k: 0.30 + 1e-4 * k)
        result = pll.run(samples)
        recent = [d.omega for d in result.diagnostics[-10:]]
        expected = median(recent)
        pll.run([0j] * (HO["holdover_enter"] - 1))
        d = pll.step(0j)
        self.assertEqual(d.state, STATE_HOLDOVER)
        self.assertEqual(d.omega, expected)
        # 保持期间积分项不再被噪声更新: omega 恒定
        d2 = pll.step(0j)
        self.assertEqual(d2.omega, expected)

    def test_low_amp_counter_persists_across_chunks(self):
        # 低幅计数跨 run() 调用延续: 4 + 4 个低幅样本同样触发 holdover。
        params = PLLParams(**HO)
        pll = PLL(params)
        pll.run(synth(200, 1.0, 0.30)[0])
        r1 = pll.run([0j] * 4)
        self.assertEqual(r1.state, STATE_TRACKING)
        r2 = pll.run([0j] * 4)
        self.assertEqual(r2.diagnostics[-1].state, STATE_HOLDOVER)
        # 中途出现有效样本则连续计数清零
        pll2 = PLL(PLLParams(**HO))
        pll2.run(synth(200, 1.0, 0.30)[0])
        pll2.run([0j] * 7)
        pll2.run(synth(1, 1.0, 0.30)[0])
        d = pll2.step(0j)
        self.assertEqual(d.state, STATE_TRACKING)  # 重新计数, 未进 holdover


class TestReacquire(unittest.TestCase):
    def test_bounded_correction_and_gradual_catchup(self):
        # 恢复后不瞬间重置相位: 每样本校正量 <= reacquire_step。
        params = PLLParams(**HO)
        pll = PLL(params)
        samples, phases = gap_signal()
        diags = pll.run(samples).diagnostics
        reacq = [d for d in diags if d.state == STATE_REACQUIRE]
        self.assertTrue(len(reacq) > 1)  # 逐渐追上, 非单样本跳变
        for i, d in enumerate(reacq):
            prev_theta = diags[d.index - 1].theta
            correction = wrap_phase(d.theta - prev_theta - d.omega)
            self.assertLessEqual(abs(correction),
                                 HO["reacquire_step"] + 1e-12)
        # 进入 reacquire 的第一个样本不重置相位: 误差不会立刻消失
        self.assertGreater(abs(reacq[0].error), HO["eps"])

    def test_relock_requires_consecutive_dual_conditions(self):
        # 只有连续 reacquire_count 个样本同时满足相位/频率条件才重新锁定。
        params = PLLParams(**HO)
        pll = PLL(params)
        samples, _ = gap_signal()
        diags = pll.run(samples).diagnostics
        # reacquire 期间一律未锁定
        for d in diags:
            if d.state == STATE_REACQUIRE:
                self.assertFalse(d.locked)
        # 最终重新锁定且状态回到 tracking
        self.assertTrue(pll.locked)
        self.assertEqual(pll.state, STATE_TRACKING)
        # 锁定前的最后 reacquire_count 个样本都满足双条件
        last_lock = max(d.index for d in diags if not d.locked)
        for d in diags[last_lock - HO["reacquire_count"] + 1:last_lock + 1]:
            self.assertLessEqual(abs(d.error), HO["eps"])

    def test_false_recovery_returns_to_holdover(self):
        # 假恢复: 短暂有效后再次跌落, 立即回到 holdover。
        params = PLLParams(**HO)
        pll = PLL(params)
        samples, _ = gap_signal(gap_n=20, false_recover=(3, 10))
        diags = pll.run(samples).diagnostics
        states = [d.state for d in diags]
        self.assertIn(STATE_HOLDOVER, states)
        self.assertIn(STATE_REACQUIRE, states)
        # 假恢复后的第一个低幅样本立即是 holdover(无需重新计满门限)
        first_reacq = states.index(STATE_REACQUIRE)
        after = states[first_reacq:]
        low_after = next(i for i, d in enumerate(diags[first_reacq:])
                         if not d.valid)
        self.assertEqual(after[low_after], STATE_HOLDOVER)
        # 假恢复期间不应标记锁定
        self.assertFalse(any(d.locked for d in diags[first_reacq:
                                                    first_reacq + low_after]))


class TestDriftAndHonestError(unittest.TestCase):
    def test_holdover_error_grows_with_drift(self):
        # 频率缓慢漂移 + 缺口: 保持期间 PLL 按冻结频率外推,
        # 与真实相位的误差会增长——测试报告该误差而非假装能跟踪。
        drift = 1e-4
        n_lock, n_gap, n_tail = 300, 60, 400
        total = n_lock + n_gap + n_tail
        omega_of = lambda k: 0.30 + drift * k
        amp_of = lambda k: 1.0 if k < n_lock or k >= n_lock + n_gap else 0.0
        samples, phases = synth(total, amp_of, omega_of)
        # freq_eps 需容下稳态 omega 与冻结中位数之间的漂移偏差
        pll = PLL(PLLParams(**{**HO, "freq_eps": 0.06}))
        diags = pll.run(samples).diagnostics
        hold = [d for d in diags if d.state == STATE_HOLDOVER]
        self.assertTrue(hold)
        errs = [abs(wrap_phase(d.theta - phases[d.index])) for d in hold]
        # 保持期间误差单调趋势: 末尾误差显著大于进入时
        self.assertGreater(errs[-1], errs[0])
        self.assertGreater(errs[-1], 1e-3)  # 误差真实存在, 不掩盖
        # 恢复后重新锁定, 频率估计跟上漂移后的真值
        self.assertTrue(pll.locked)
        true_final = omega_of(total - 1)
        self.assertLess(abs(pll.omega - true_final), 5e-3)


class TestChunkConsistency(unittest.TestCase):
    def test_chunked_identical_to_oneshot(self):
        # 跨缺口/假恢复/重捕获区域任意分块, 与一次性输入逐点一致。
        params = PLLParams(**HO)
        samples, _ = gap_signal(gap_n=20, false_recover=(3, 10))
        whole = PLL(params).run(samples)
        chunked = PLL(params)
        diags = []
        for lo, hi in ((0, 5), (5, 310), (310, 311), (311, 333),
                       (333, len(samples))):
            diags.extend(chunked.run(samples[lo:hi]).diagnostics)
        self.assertEqual(tuple(diags), whole.diagnostics)
        self.assertEqual(chunked.omega, whole.omega)
        self.assertEqual(chunked.theta, whole.theta)
        self.assertEqual(chunked.locked, whole.locked)
        self.assertEqual(chunked.state, whole.state)


class TestLegacyDefault(unittest.TestCase):
    def test_default_params_keep_freeze_behavior(self):
        # 默认 holdover_enter=None: 低幅样本完全冻结, 状态保持 tracking。
        pll = PLL(PLLParams(kp=0.2, ki=0.02, wmax=1.0, amin=0.5,
                            eps=0.05, lock_count=10))
        pll.run(synth(60, 1.0, 0.0, phase0=0.4)[0])
        theta, omega = pll.theta, pll.omega
        result = pll.run([0j] * 5)
        for d in result.diagnostics:
            self.assertEqual(d.state, STATE_TRACKING)
            self.assertEqual(d.theta, theta)
            self.assertEqual(d.omega, omega)


class TestHoldoverParamValidation(unittest.TestCase):
    def test_bad_holdover_params_rejected(self):
        bad = [
            dict(holdover_enter=0), dict(holdover_enter=-1),
            dict(holdover_enter=True), dict(holdover_enter=2.5),
            dict(holdover_window=0), dict(holdover_window=False),
            dict(holdover_window=1.5),
            dict(reacquire_step=0.0), dict(reacquire_step=-0.1),
            dict(reacquire_step=float("nan")),
            dict(freq_eps=0.0), dict(freq_eps=-1.0),
            dict(freq_eps=float("inf")),
            dict(reacquire_count=0), dict(reacquire_count=True),
            dict(reacquire_count=3.5),
        ]
        for override in bad:
            with self.subTest(override=override):
                with self.assertRaises(PLLParamError):
                    PLLParams(**{**HO, **override})

    def test_holdover_enter_none_allowed(self):
        p = PLLParams(**{k: v for k, v in HO.items()
                         if k != "holdover_enter"})
        self.assertIsNone(p.holdover_enter)


if __name__ == "__main__":
    unittest.main()
