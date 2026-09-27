"""DualReferencePLL 双参考无扰动交接单元测试。

两路参考同采样速率、成对喂入。期望值由独立合成导频与独立的单路 PLL
对照得到,不直接调用被测交接器生成期望。
"""

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pll_capture import (
    MODE_ACQUIRING,
    MODE_HOLDOVER,
    MODE_TRACKING,
    PLL,
    PLLInputError,
    PLLParamError,
    PLLParams,
    DualReferencePLL,
    wrap_phase,
)


def pilot(amplitude, phase, offset, n):
    """确定性合成复数导频: z_k = A * exp(j*(phase + offset*k))。"""
    return [
        complex(amplitude * math.cos(phase + offset * k),
                amplitude * math.sin(phase + offset * k))
        for k in range(n)
    ]


PARAMS = PLLParams(kp=0.2, ki=0.02, wmax=1.0, amin=0.5,
                   eps=0.05, lock_count=10)
ZERO = 0j


class TestIndependentAcquisition(unittest.TestCase):
    """目标参考先独立跟踪,确认稳定前继续使用原参考。"""

    def test_target_unstable_keeps_original(self):
        d = DualReferencePLL(PARAMS)
        # 参考 0 先完成捕获;参考 1 在请求切换后才开始接收有效信号,
        # 从首个有效样本起独立计数,确认稳定前继续输出原参考。
        d.run([(z, 0.1 + 0j) for z in pilot(1.0, 0.3, 0.0, 120)])
        self.assertEqual(d.selected, 0)
        self.assertTrue(d.trusted)

        # 请求切到参考 1,但参考 1 刚开始捕获(尚未连续 L 个在门限样本)。
        d.request_switch(1)
        window = 60  # 目标从裸初相 0 捕获到 1.8 rad 约需 40 个样本
        z0b = pilot(1.0, 0.3, 0.0, window)
        z1b = pilot(1.0, 1.8, 0.0, window)
        result = d.run(list(zip(z0b, z1b)))
        handoff_idx = next(i for i, x in enumerate(result.diagnostics)
                           if x.handoff)
        # 交接恰好发生在目标首次 locked 的样本,之前仍输出原参考。
        self.assertGreaterEqual(handoff_idx, PARAMS.lock_count)
        for diag in result.diagnostics[:handoff_idx]:
            self.assertEqual(diag.active, 0)
            self.assertTrue(diag.trusted)
            self.assertEqual(diag.mode, MODE_TRACKING)
            self.assertEqual(diag.requested, 1)
        self.assertEqual(d.selected, 1)

    def test_both_trackers_updated_independently(self):
        d = DualReferencePLL(PARAMS)
        z0 = pilot(1.0, 0.3, 0.0, 120)
        z1 = pilot(1.0, -1.2, 0.0, 120)
        d.run(list(zip(z0, z1)))
        t0, t1 = d.trackers
        # 非活动参考也在独立跟踪,其裸相位对应各自导频。
        self.assertTrue(t0.locked and t1.locked)
        self.assertLessEqual(abs(wrap_phase(t0.theta - 0.3)), 1e-6)
        self.assertLessEqual(abs(wrap_phase(t1.theta - (-1.2))), 1e-5)

    def test_initial_acquiring_untrusted(self):
        d = DualReferencePLL(PARAMS)
        result = d.step((pilot(1.0, 0.3, 0.0, 1)[0],
                         pilot(1.0, 1.8, 0.0, 1)[0]))
        self.assertIsNone(d.selected)
        self.assertFalse(result.trusted)
        self.assertEqual(result.mode, MODE_ACQUIRING)


class TestSeamlessHandoff(unittest.TestCase):
    def test_output_phase_does_not_jump(self):
        d = DualReferencePLL(PARAMS)
        z0 = pilot(1.0, 0.3, 0.0, 200)
        z1 = pilot(1.0, 1.8, 0.0, 200)
        d.run(list(zip(z0, z1)))
        d.request_switch(1)
        theta_before = d.theta
        result = d.run(list(zip(z0, z1)))
        hi = next(i for i, x in enumerate(result.diagnostics) if x.handoff)
        # 交接样本输出相位与上一样本完全相等(不跳到参考 1 的裸相位)。
        expected_prev = theta_before if hi == 0 else result.diagnostics[hi - 1].theta
        self.assertEqual(result.diagnostics[hi].theta, expected_prev)
        self.assertTrue(abs(result.diagnostics[hi].offset) <= math.pi)
        # 交接后输出平滑跟随参考 1:对齐相位仍约等于原物理输出。
        self.assertLessEqual(
            abs(wrap_phase(result.diagnostics[-1].theta - theta_before)), 1e-9)

    def test_back_and_forth_does_not_accumulate_offset(self):
        # 两路为完全相同的导频:反复切换不应引入任何人为偏置,
        # 输出应与一个从头跑到尾的独立单路 PLL 逐样本一致。
        n = 400
        z = pilot(1.0, 0.3, 0.0, n)
        ref_pll = PLL(PARAMS)
        ref_pll.run(z)

        d = DualReferencePLL(PARAMS)
        d.run([(z[k], z[k]) for k in range(300)])
        for i in range(1, 8):
            lo, hi = 300 + i * 10, 310 + i * 10
            d.request_switch(i % 2)
            d.run([(z[k], z[k]) for k in range(lo, hi)])
        # 所有偏置每次按绝对裸相位重算;相同导频下应恒为 0。
        self.assertLessEqual(abs(d.offset(0)), 1e-9)
        self.assertLessEqual(abs(d.offset(1)), 1e-9)
        self.assertLessEqual(abs(wrap_phase(d.theta - ref_pll.theta)), 1e-9)

        # 不同恒相位导频间多次切换:偏置始终在单次 wrap 范围内,不累积。
        d2 = DualReferencePLL(PARAMS)
        a = pilot(1.0, 0.3, 0.0, n)
        b = pilot(1.0, 2.2, 0.0, n)
        d2.run(list(zip(a, b))[:120])
        seq = [1, 0, 1, 0]  # 偶数次切换,最终回到参考 0
        for i, tgt in enumerate(seq):
            d2.request_switch(tgt)
            lo, hi = 120 + i * 40, 160 + i * 40
            result = d2.run(list(zip(a[lo:hi], b[lo:hi])))
            for diag in result.diagnostics:
                if diag.handoff:
                    self.assertTrue(abs(diag.offset) <= math.pi)
        # 最终回到参考 0,输出对齐的物理相位仍是 0.3(无累积偏差)。
        self.assertEqual(d2.selected, 0)
        self.assertLessEqual(abs(wrap_phase(d2.theta - 0.3)), 1e-5)


class TestHoldover(unittest.TestCase):
    """切换期间原参考失效:明确进入保持态而非假装锁定。"""

    def _setup_pending_switch(self):
        d = DualReferencePLL(PARAMS)
        d.run([(z, 0.1 + 0j) for z in pilot(1.0, 0.3, 0.0, 120)])
        self.assertEqual(d.selected, 0)
        d.request_switch(1)  # 参考 1 尚不稳定(首个有效样本都还未出现)
        return d

    def test_original_failure_enters_holdover(self):
        d = self._setup_pending_switch()
        n = PARAMS.lock_count - 1  # 参考 1 仍差 1 个有效样本才锁定
        z0 = [ZERO] * n           # 原参考在切换窗口内失效
        z1 = pilot(1.0, 1.8, 0.0, n)
        frozen_theta, frozen_omega = d.theta, d.omega
        result = d.run(list(zip(z0, z1)))
        self.assertEqual(d.mode, MODE_HOLDOVER)
        self.assertFalse(d.trusted)
        self.assertIsNone(d.selected)
        self.assertEqual(d.holdover_samples, n)
        self.assertEqual(d.requested, 1)  # 切换请求保留,目标继续独立跟踪
        for diag in result.diagnostics:
            self.assertEqual(diag.mode, MODE_HOLDOVER)
            self.assertFalse(diag.trusted)
            self.assertIsNone(diag.active)
            self.assertEqual(diag.theta, frozen_theta)  # 输出冻结
            self.assertEqual(diag.omega, frozen_omega)

    def test_holdover_resumes_without_jump(self):
        d = self._setup_pending_switch()
        frozen = d.theta
        # 原参考持续失效,直到参考 1 完成锁定 -> 当样本无扰动交接。
        # 目标从裸初相 0 捕获到 1.8 rad 约需 40 个样本;原参考全程失效。
        n = 60
        z0 = [ZERO] * n
        z1 = pilot(1.0, 1.8, 0.0, n)
        result = d.run(list(zip(z0, z1)))
        hi = next(i for i, x in enumerate(result.diagnostics) if x.handoff)
        self.assertEqual(result.diagnostics[hi].theta, frozen)
        self.assertEqual(d.selected, 1)
        self.assertTrue(d.trusted)
        self.assertEqual(d.mode, MODE_TRACKING)
        self.assertEqual(d.handoff_count, 2)  # 初次接管 + 本次交接

    def test_active_failure_without_pending_holds_until_reacquire(self):
        d = DualReferencePLL(PARAMS)
        z0 = pilot(1.0, 0.3, 0.0, 120)
        z1 = [0.1 + 0j] * 120
        d.run(list(zip(z0, z1)))
        frozen = d.theta
        out = d.run([(ZERO, ZERO)] * 5)
        self.assertEqual(d.mode, MODE_HOLDOVER)
        self.assertFalse(d.trusted)
        self.assertEqual(d.holdover_samples, 5)
        for diag in out.diagnostics:
            self.assertEqual(diag.theta, frozen)
        # 原参考恢复并重新锁定后再次接管。
        z0b = pilot(1.0, 0.3, 0.0, PARAMS.lock_count + 3)
        z1b = [0.1 + 0j] * (PARAMS.lock_count + 3)
        d.run(list(zip(z0b, z1b)))
        self.assertTrue(d.trusted)
        self.assertEqual(d.selected, 0)


class TestValidationAndAtomicity(unittest.TestCase):
    def test_bad_pair_rejected_atomically(self):
        d = DualReferencePLL(PARAMS)
        good = pilot(1.0, 0.3, 0.0, 4)
        bad_batch = list(zip(good, good)) + [(1 + 0j, complex(float("nan"), 0))]
        with self.assertRaises(PLLInputError):
            d.run(bad_batch)
        self.assertEqual(d.processed, 0)
        self.assertEqual(d.trackers[0].processed, 0)
        self.assertEqual(d.trackers[1].processed, 0)
        self.assertEqual(d.mode, MODE_ACQUIRING)

    def test_pair_shape_and_ref_validation(self):
        d = DualReferencePLL(PARAMS)
        for bad in ((1 + 0j,), (1 + 0j, 2 + 0j, 3 + 0j), "ab", 42, [1, 2]):
            with self.subTest(bad=bad):
                with self.assertRaises(PLLInputError):
                    d.step(bad)
        for bad in (2, -1, True, "0"):
            with self.subTest(bad=bad):
                with self.assertRaises(PLLParamError):
                    d.request_switch(bad)
        with self.assertRaises(PLLParamError):
            d.offset(5)


if __name__ == "__main__":
    unittest.main()
