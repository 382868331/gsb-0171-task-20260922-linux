"""ReferenceSwitcher 双参考切换的单元测试。

期望值来源:两路参考均为本地确定性合成的复数导频,真值(相位/频偏)
在生成时指定,不调用被测核心得到。
"""

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pll_capture import (
    PLLError,
    PLLParams,
    ReferenceSwitcher,
    wrap_phase,
)


def make_pilot(amplitude, phase, offset, n):
    """确定性生成复数导频: z_k = A * exp(j*(phase + offset*k))。"""
    return [
        complex(amplitude * math.cos(phase + offset * k),
                amplitude * math.sin(phase + offset * k))
        for k in range(n)
    ]


PARAMS = PLLParams(kp=0.12, ki=0.008, wmax=1.2, amin=0.2,
                   eps=0.03, lock_count=40)
OFFSET = 0.3          # 两路参考相同的频偏(rad/sample, 采样速率相同)
PHASE_A = 0.5
PHASE_B = -1.1

# 两路参考各自是一条时间连续的导频流; 测试按时间游标切片,
# 模拟真实切换中"原参考继续到达 / 新参考同步独立跟踪"的采样序列。
REF_A = make_pilot(1.0, PHASE_A, OFFSET, 8000)
REF_B = make_pilot(1.0, PHASE_B, OFFSET, 8000)


def true_theta(phase, k):
    """第 k 个样本(0 起)处理后环路应收敛到的相位: theta 在误差测量
    之后再前进一个 omega, 故稳态为 phase + OFFSET*(k+1)。"""
    return wrap_phase(phase + OFFSET * (k + 1))


def make_switcher(n=400):
    sw = ReferenceSwitcher(PARAMS)
    sw.run(REF_A[:n])
    assert sw.locked
    return sw


class TestSwitchFlow(unittest.TestCase):
    def test_phase_continuity_on_handover(self):
        sw = make_switcher()
        theta_before = sw.theta
        sw.begin_switch()
        self.assertTrue(sw.switching)
        self.assertEqual(sw.status, "switching")
        sw.run_candidate(REF_B[400:700])
        self.assertTrue(sw.candidate_locked)
        self.assertTrue(sw.complete_switch())
        self.assertFalse(sw.switching)
        # 交接瞬间输出相位不跳变
        self.assertAlmostEqual(sw.theta, theta_before, delta=1e-12)
        self.assertAlmostEqual(sw.omega, OFFSET, delta=1e-3)
        self.assertEqual(sw.switch_count, 1)
        # 交接后环路把真实相位差逐渐拉入, 最终收敛到新参考相位
        result = sw.run(REF_B[700:1100])
        self.assertTrue(result.locked)
        self.assertLessEqual(
            abs(wrap_phase(sw.theta - true_theta(PHASE_B, 1099))), 1e-3)
        self.assertLessEqual(abs(sw.omega - OFFSET), 1e-6)

    def test_unstable_target_keeps_old_reference(self):
        sw = make_switcher()
        theta_before, omega_before = sw.theta, sw.omega
        sw.begin_switch()
        # 只喂少量样本, 候选环路尚未 locked
        sw.run_candidate(REF_B[400:410])
        self.assertFalse(sw.candidate_locked)
        self.assertFalse(sw.complete_switch())
        # 继续跟踪原参考, 状态不变
        self.assertTrue(sw.switching)
        self.assertEqual(sw.theta, theta_before)
        self.assertEqual(sw.omega, omega_before)
        result = sw.run(REF_A[400:450])
        self.assertTrue(result.locked)
        sw.abort_switch()
        self.assertFalse(sw.switching)
        self.assertEqual(sw.status, "locked")

    def test_source_loss_during_switch_reports_honestly(self):
        sw = make_switcher()
        sw.begin_switch()
        sw.run_candidate(REF_B[400:420])
        # 切换期间原参考失效(零幅度样本)
        result = sw.run([0j] * 5)
        self.assertFalse(result.locked)
        self.assertFalse(sw.locked)
        self.assertEqual(sw.status, "switching")
        self.assertEqual(result.lost_samples, 5)

    def test_repeated_switching_no_offset_accumulation(self):
        sw = make_switcher()
        # 两路参考内容一致, 来回切换 6 次
        cursor = 400
        for _ in range(6):
            sw.begin_switch()
            sw.run_candidate(REF_A[cursor:cursor + 200])
            cursor += 200
            self.assertTrue(sw.complete_switch())
            result = sw.run(REF_A[cursor:cursor + 200])
            cursor += 200
            self.assertTrue(result.locked)
        self.assertEqual(sw.switch_count, 6)
        # 不累积人为相位偏移: 仍贴合参考真值相位
        self.assertLessEqual(
            abs(wrap_phase(sw.theta - true_theta(PHASE_A, cursor - 1))), 1e-3)
        self.assertLessEqual(abs(sw.omega - OFFSET), 1e-6)

    def test_alternating_phase_references_no_accumulation(self):
        sw = make_switcher()
        # A/B 相位不同, 各跟踪一段后切换, 往返 3 轮
        cursor = 400
        for ref, phase in ((REF_B, PHASE_B), (REF_A, PHASE_A)) * 3:
            sw.begin_switch()
            sw.run_candidate(ref[cursor:cursor + 300])
            cursor += 300
            self.assertTrue(sw.complete_switch())
            result = sw.run(ref[cursor:cursor + 400])
            cursor += 400
            self.assertTrue(result.locked)
            self.assertLessEqual(
                abs(wrap_phase(sw.theta - true_theta(phase, cursor - 1))), 1e-3)
            self.assertLessEqual(abs(sw.omega - OFFSET), 1e-6)

    def test_switch_requires_begin(self):
        sw = make_switcher()
        with self.assertRaises(PLLError):
            sw.complete_switch()
        with self.assertRaises(PLLError):
            sw.step_candidate(1 + 0j)
        with self.assertRaises(PLLError):
            sw.run_candidate([1 + 0j])

    def test_candidate_validates_input(self):
        from pll_capture import PLLInputError
        sw = make_switcher()
        sw.begin_switch()
        with self.assertRaises(PLLInputError):
            sw.run_candidate([1 + 0j, complex(float("nan"), 0.0)])


if __name__ == "__main__":
    unittest.main()
