"""pll_capture 周跳候选检测单元测试。

期望值来源说明:
- 注入误差序列的测试用 pll.theta 实时合成样本,使每个样本的旋转误差
  精确等于预设值,触发位置(起始号/确认号)由预设序列手工数出;
- 圆均值/合向量长度的期望值在测试内用 math 按定义独立计算;
- 真实阶跃测试的触发位置对照诊断中记录的逐样本误差手工核对,
  不调用被测核心的检测逻辑得到期望值。
"""

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pll_capture import (
    CycleSlipConfig,
    PLL,
    PLLParamError,
    PLLParams,
    wrap_phase,
)

PARAMS = PLLParams(kp=0.2, ki=0.02, wmax=1.0, amin=0.5, eps=0.05, lock_count=5)


def make_pilot(amplitude, phase, offset, n):
    """确定性生成复数导频: z_k = A * exp(j*(phase + offset*k))。"""
    return [
        complex(amplitude * math.cos(phase + offset * k),
                amplitude * math.sin(phase + offset * k))
        for k in range(n)
    ]


def slip_cfg(**kw):
    base = dict(threshold=0.5, confirm_count=3, deg_threshold=0.3,
                compensate=True)
    base.update(kw)
    return CycleSlipConfig(**base)


def locked_pll(cfg=None):
    """构造已在 0.3 rad 导频上锁定的 PLL, 返回 (pll, 已处理样本数)。"""
    pll = PLL(PARAMS, slip_config=cfg)
    pll.run(make_pilot(1.0, 0.3, 0.0, 60))
    assert pll.locked
    return pll, pll.processed


def inject(pll, error):
    """合成一个样本, 使其旋转后的误差精确等于 error(相对当前 theta)。"""
    return pll.step(complex(math.cos(pll.theta + error),
                            math.sin(pll.theta + error)))


def circular_mean_of(errors):
    """按定义独立计算圆均值与平均合向量长度。"""
    ms = sum(math.sin(e) for e in errors) / len(errors)
    mc = sum(math.cos(e) for e in errors) / len(errors)
    return math.atan2(ms, mc), math.hypot(ms, mc)


class TestCandidateTriggering(unittest.TestCase):
    """候选的开启、打断与确认位置。"""

    def test_isolated_glitch_does_not_confirm(self):
        # 孤立毛刺: 单次超限达不到 K=3, 不得确认事件。
        pll, _ = locked_pll(slip_cfg())
        inject(pll, 2.0)                    # 孤立超限
        for _ in range(10):
            inject(pll, 0.01)               # 回到门限内
        self.assertEqual(pll.slip_events, ())
        self.assertTrue(pll.locked)         # 毛刺后重新锁定

    def test_candidate_broken_by_small_valid_error(self):
        # 有效样本误差回到门限内会打断候选, 重新计数。
        pll, base = locked_pll(slip_cfg())
        for e in (0.8, 0.8, 0.01, 0.8, 0.8, 0.8):
            inject(pll, e)
        self.assertEqual(len(pll.slip_events), 1)
        ev = pll.slip_events[0]
        # 手工数出: 前两次超限被 0.01 打断, 候选从第 4 个注入样本重新开始。
        self.assertEqual(ev.start_index, base + 3)
        self.assertEqual(ev.confirm_index, base + 5)
        self.assertEqual(ev.count, 3)
        self.assertAlmostEqual(ev.error_sum, 2.4, delta=1e-12)
        mean, resultant = circular_mean_of([0.8, 0.8, 0.8])
        self.assertAlmostEqual(ev.circular_mean, mean, delta=1e-12)
        self.assertAlmostEqual(ev.resultant, resultant, delta=1e-12)
        self.assertFalse(ev.ambiguous)
        self.assertTrue(ev.compensated)
        self.assertAlmostEqual(ev.correction, mean, delta=1e-12)

    def test_low_amplitude_interrupts_candidate(self):
        # 低幅度样本打断候选: 不累计、不确认, 也不伪造有效相位。
        pll, base = locked_pll(slip_cfg())
        inject(pll, 0.8)
        inject(pll, 0.8)
        d_invalid = pll.step(0j)            # 零幅度: 无效样本
        for _ in range(3):
            inject(pll, 0.8)
        # 无效样本不伪造相位/误差
        self.assertFalse(d_invalid.valid)
        self.assertIsNone(d_invalid.error)
        self.assertIsNone(d_invalid.slip_event)
        self.assertEqual(pll.lost_samples, 1)
        # 若无效样本未打断候选, 确认会落在 base+4; 实际应从零重计。
        self.assertEqual(len(pll.slip_events), 1)
        ev = pll.slip_events[0]
        self.assertEqual(ev.start_index, base + 3)
        self.assertEqual(ev.confirm_index, base + 5)

    def test_confirm_sample_cannot_restart_candidate(self):
        # 确认候选的样本不会再次开启新候选(防止连发):
        # 连续 6 次超限、K=2, 事件只能确认在第 2/4/6 个样本。
        pll, base = locked_pll(slip_cfg(confirm_count=2, compensate=False))
        for _ in range(6):
            inject(pll, 0.8)
        events = pll.slip_events
        self.assertEqual(len(events), 3)
        self.assertEqual([e.start_index for e in events],
                         [base + 0, base + 2, base + 4])
        self.assertEqual([e.confirm_index for e in events],
                         [base + 1, base + 3, base + 5])


class TestCircularMeanAndAmbiguity(unittest.TestCase):
    """跨 ±pi 的圆均值与合向量退化。"""

    def test_cross_pi_circular_mean(self):
        # 误差分布在 ±pi 两侧: 有符号算术和(3.09)与算术均值(1.03)都
        # 严重偏离真实方向(≈±pi), 圆均值才能给出正确补偿方向。
        pll, _ = locked_pll(slip_cfg(threshold=3.0))
        errors = [3.10, -3.12, 3.11]
        for e in errors:
            inject(pll, e)
        self.assertEqual(len(pll.slip_events), 1)
        ev = pll.slip_events[0]
        mean, resultant = circular_mean_of(errors)
        self.assertGreater(ev.circular_mean, 3.0)   # 靠 +pi 一侧
        self.assertAlmostEqual(ev.circular_mean, mean, delta=1e-12)
        self.assertAlmostEqual(ev.resultant, resultant, delta=1e-12)
        self.assertGreater(ev.resultant, 0.99)      # 方向高度一致
        # 累计有符号误差按原样记录(跨 ±pi 不代表真实偏移, 仅供记录)
        self.assertAlmostEqual(ev.error_sum, 3.09, delta=1e-12)
        self.assertNotAlmostEqual(ev.error_sum / 3, ev.circular_mean, delta=1.0)
        # 补偿量等于圆均值, 施加在原更新之后的相位上
        self.assertTrue(ev.compensated)
        self.assertAlmostEqual(ev.correction, mean, delta=1e-12)

    def test_ambiguous_resultant_reports_only(self):
        # 误差 ±1.8 对称抵消: 合向量长度 |cos(1.8)| ≈ 0.227 < deg_threshold,
        # 判定 ambiguous, 即使 compensate=True 也不补偿、不挑方向。
        pll, _ = locked_pll(slip_cfg(threshold=1.0, confirm_count=2))
        inject(pll, 1.8)
        inject(pll, -1.8)
        self.assertEqual(len(pll.slip_events), 1)
        ev = pll.slip_events[0]
        self.assertTrue(ev.ambiguous)
        self.assertIsNone(ev.circular_mean)
        self.assertFalse(ev.compensated)
        self.assertEqual(ev.correction, 0.0)
        self.assertAlmostEqual(ev.resultant, abs(math.cos(1.8)), delta=1e-12)
        # ambiguous 时 theta 与仅报告/未启用检测的轨迹完全一致
        twin, _ = locked_pll(slip_cfg(threshold=1.0, confirm_count=2,
                                      compensate=False))
        inject(twin, 1.8)
        inject(twin, -1.8)
        self.assertEqual(pll.theta, twin.theta)
        self.assertEqual(pll.omega, twin.omega)

    def test_degenerate_threshold_is_explicit(self):
        # 同一组误差, 退化阈值取不同显式值时结论相反。
        errors = (1.8, -1.8)    # resultant ≈ 0.227
        low, _ = locked_pll(slip_cfg(threshold=1.0, confirm_count=2,
                                     deg_threshold=0.2))
        high, _ = locked_pll(slip_cfg(threshold=1.0, confirm_count=2,
                                      deg_threshold=0.3))
        for e in errors:
            inject(low, e)
            inject(high, e)
        self.assertFalse(low.slip_events[0].ambiguous)   # 0.227 >= 0.2
        self.assertTrue(high.slip_events[0].ambiguous)   # 0.227 < 0.3


class TestConfirmEffects(unittest.TestCase):
    """确认后的状态处理: 锁定计数清零、频率保留、补偿时机。"""

    def test_lock_reset_and_frequency_preserved(self):
        pll, base = locked_pll(slip_cfg())
        omega_before = pll.omega
        diags = [inject(pll, 0.8) for _ in range(3)]
        ev = pll.slip_events[0]
        self.assertEqual(ev.confirm_index, base + 2)
        # 确认样本的诊断: 锁定计数清零、退出 locked
        confirm_diag = diags[-1]
        self.assertEqual(confirm_diag.streak, 0)
        self.assertFalse(confirm_diag.locked)
        self.assertIsNotNone(confirm_diag.slip_event)
        # 频率估计保留: 确认只清零计数/补偿相位, omega 延续正常更新,
        # 既没有被清零也没有被额外改动(与仅报告模式逐位一致)。
        self.assertNotEqual(confirm_diag.omega, 0.0)
        self.assertNotEqual(confirm_diag.omega, omega_before)  # 正常积分在继续
        reporter, _ = locked_pll(slip_cfg(compensate=False))
        rep_diags = [inject(reporter, 0.8) for _ in range(3)]
        self.assertEqual(confirm_diag.omega, rep_diags[-1].omega)

    def test_compensation_applied_after_normal_update(self):
        # 补偿发生在该样本原更新之后: 确认样本的 theta == 仅报告轨迹 + 圆均值。
        comp, _ = locked_pll(slip_cfg())
        repo, _ = locked_pll(slip_cfg(compensate=False))
        for _ in range(3):
            inject(comp, 0.8)
            inject(repo, 0.8)
        ev = comp.slip_events[0]
        self.assertEqual(repo.slip_events[0].circular_mean, ev.circular_mean)
        self.assertEqual(comp.theta, wrap_phase(repo.theta + ev.correction))
        # 确认之后的样本从补偿后的相位继续(轨迹分叉, 证明补偿真实生效)
        inject(comp, 0.01)
        inject(repo, 0.01)
        self.assertNotEqual(comp.theta, repo.theta)

    def test_report_only_matches_disabled(self):
        # 仅报告模式不得改变环路本身: 与完全未启用检测的轨迹逐位一致。
        stream = (make_pilot(1.0, 0.3, 0.0, 60)
                  + make_pilot(1.0, 0.3 + 1.2, 0.0, 40)
                  + [0j] * 3
                  + make_pilot(1.0, 0.3 + 1.2, 0.0, 40))
        reporter = PLL(PARAMS, slip_config=slip_cfg(compensate=False))
        disabled = PLL(PARAMS)
        r = reporter.run(stream)
        d = disabled.run(stream)
        self.assertEqual(len(r.slip_events) > 0, True)
        self.assertEqual(d.slip_events, ())
        for dr, dd in zip(r.diagnostics, d.diagnostics):
            self.assertEqual((dr.valid, dr.error, dr.omega, dr.theta,
                              dr.streak, dr.locked, dr.saturated),
                             (dd.valid, dd.error, dd.omega, dd.theta,
                              dd.streak, dd.locked, dd.saturated))


class TestRealPhaseStep(unittest.TestCase):
    """真实导频相位阶跃: 手工核对触发位置与后续重新锁定。"""

    def _run_step(self, compensate):
        cfg = slip_cfg(compensate=compensate)
        pll = PLL(PARAMS, slip_config=cfg)
        samples = (make_pilot(1.0, 0.3, 0.0, 60)
                   + make_pilot(1.0, 0.3 + 1.2, 0.0, 250))
        return pll, pll.run(samples)

    def test_step_trigger_position_and_relock(self):
        pll, result = self._run_step(compensate=False)
        self.assertEqual(len(result.slip_events), 1)
        ev = result.slip_events[0]
        diags = result.diagnostics
        # 手工核对触发位置: 阶跃前末样本误差在 eps 内, 阶跃后第 1 个样本
        # 误差 ≈ 1.2 超过门限 0.5, 连续 3 次超限确认。
        self.assertEqual(ev.start_index, 60)
        self.assertEqual(ev.confirm_index, 62)
        self.assertLessEqual(abs(diags[59].error), PARAMS.eps)
        for i in range(ev.start_index, ev.confirm_index + 1):
            self.assertGreater(abs(diags[i].error), 0.5)
        self.assertEqual(ev.confirm_index - ev.start_index + 1, 3)
        # 确认样本: 锁定计数清零; 圆均值与记录误差独立重算一致
        self.assertEqual(diags[ev.confirm_index].streak, 0)
        self.assertFalse(diags[ev.confirm_index].locked)
        errors = [diags[i].error for i in range(60, 63)]
        mean, _ = circular_mean_of(errors)
        self.assertAlmostEqual(ev.circular_mean, mean, delta=1e-12)
        self.assertAlmostEqual(ev.error_sum, sum(errors), delta=1e-12)
        # 频率估计保留: 确认前后 omega 延续正常积分, 未被清零
        self.assertNotEqual(diags[ev.confirm_index].omega, 0.0)
        # 后续重新锁定并跟踪到新相位
        self.assertTrue(result.locked)
        self.assertLessEqual(abs(wrap_phase(result.theta - 1.5)), 1e-6)
        self.assertLessEqual(abs(result.omega), 1e-6)

    def test_step_compensation_then_relock(self):
        comp, comp_result = self._run_step(compensate=True)
        repo, repo_result = self._run_step(compensate=False)
        ev_c = comp_result.slip_events[0]
        ev_r = repo_result.slip_events[0]
        # 同一信号, 触发位置一致; 补偿量等于圆均值
        self.assertEqual((ev_c.start_index, ev_c.confirm_index),
                         (ev_r.start_index, ev_r.confirm_index))
        self.assertTrue(ev_c.compensated)
        self.assertFalse(ev_r.compensated)
        self.assertEqual(ev_c.correction, ev_c.circular_mean)
        # 确认样本的 theta = 仅报告轨迹 + 圆均值; omega 轨迹在确认前一致
        d_c = comp_result.diagnostics[ev_c.confirm_index]
        d_r = repo_result.diagnostics[ev_r.confirm_index]
        self.assertEqual(d_c.theta, wrap_phase(d_r.theta + ev_c.correction))
        self.assertEqual(d_c.omega, d_r.omega)
        # 补偿后环路重新锁定并跟踪到新相位
        self.assertTrue(comp_result.locked)
        self.assertLessEqual(abs(wrap_phase(comp_result.theta - 1.5)), 1e-6)


class TestChunkedConsistency(unittest.TestCase):
    """候选状态跨 chunk 保存, 任意切块结果一致。"""

    def test_arbitrary_chunks_identical_with_slip(self):
        cfg = slip_cfg()
        samples = (make_pilot(1.0, 0.3, 0.0, 60)
                   + make_pilot(1.0, 0.3 + 1.2, 0.0, 30)
                   + [0j] * 3
                   + make_pilot(1.0, 0.3 + 1.2, 0.0, 40)
                   + make_pilot(1.0, 0.3 + 1.2 + 2.5, 0.0, 50))
        whole = PLL(PARAMS, slip_config=cfg).run(samples)
        chunked = PLL(PARAMS, slip_config=cfg)
        diags = []
        for lo, hi in ((0, 1), (1, 64), (64, 130), (130, 183)):
            diags.extend(chunked.run(samples[lo:hi]).diagnostics)
        # 同一操作序列, 诊断(含 slip_event)与事件列表逐位一致
        self.assertEqual(tuple(diags), whole.diagnostics)
        self.assertEqual(chunked.slip_events, whole.slip_events)
        self.assertEqual(chunked.omega, whole.omega)
        self.assertEqual(chunked.theta, whole.theta)
        self.assertEqual(chunked.locked, whole.locked)
        # step() 与 run() 混合亦一致
        mixed = PLL(PARAMS, slip_config=cfg)
        mixed_d = [mixed.step(z) for z in samples[:100]]
        mixed_d.extend(mixed.run(samples[100:]).diagnostics)
        self.assertEqual(tuple(mixed_d), whole.diagnostics)
        self.assertEqual(mixed.slip_events, whole.slip_events)


class TestCycleSlipConfigValidation(unittest.TestCase):
    def test_bad_config_rejected(self):
        good = dict(threshold=0.5, confirm_count=3, deg_threshold=0.3,
                    compensate=True)
        bad = [
            dict(threshold=0.0), dict(threshold=-0.1),
            dict(threshold=math.pi + 0.1), dict(threshold=float("nan")),
            dict(threshold=float("inf")), dict(threshold=True),
            dict(confirm_count=0), dict(confirm_count=-2),
            dict(confirm_count=2.5), dict(confirm_count=True),
            dict(confirm_count="3"),
            dict(deg_threshold=-0.1), dict(deg_threshold=1.01),
            dict(deg_threshold=float("nan")), dict(deg_threshold=True),
            dict(compensate=0), dict(compensate=1),
            dict(compensate="yes"), dict(compensate=None),
        ]
        for override in bad:
            with self.subTest(override=override):
                with self.assertRaises(PLLParamError):
                    CycleSlipConfig(**{**good, **override})

    def test_boundary_values_accepted(self):
        CycleSlipConfig(threshold=math.pi, confirm_count=1,
                        deg_threshold=0.0, compensate=False)
        CycleSlipConfig(threshold=0.5, confirm_count=3,
                        deg_threshold=1.0, compensate=True)

    def test_error_is_value_error_and_locatable(self):
        with self.assertRaises(ValueError) as ctx:
            CycleSlipConfig(threshold=0.0, confirm_count=3,
                            deg_threshold=0.3, compensate=True)
        self.assertIn("threshold", str(ctx.exception))

    def test_pll_rejects_bad_slip_config_type(self):
        with self.assertRaises(PLLParamError):
            PLL(PARAMS, slip_config="not-a-config")


if __name__ == "__main__":
    unittest.main()
