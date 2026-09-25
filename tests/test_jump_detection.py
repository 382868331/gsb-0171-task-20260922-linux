"""可选周跳候选检测的单元测试。

手工核对的四类场景(触发位置为精确断言, 非统计容差):
- 孤立毛刺:单次越限后被正常样本打断, 不确认;
- 接近 ±pi 的跨界误差:圆均值取最短弧方向, 算术累计仅供参考;
- 真实阶跃:K 个连续越限确认, 锁定计数清零、频率估计保留, 随后重新锁定;
- 低幅度中断:无效样本打断候选, 不确认、不伪造有效相位。

期望值均在测试内独立推导/构造, 不调用被测核心产生期望值:
- feed_errors 按 z = exp(j*(更新前 theta + e)) 直接合成指定误差样本;
- 无检测并行 PLL 仅用于对照"检测路径不应扰动环路状态", 其构造参数不含
  jump_k/jump_eps, 是两条独立配置的执行轨迹比较, 而非用被测结果当期望。
"""

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pll_capture import (
    Diagnostic,
    JumpCandidate,
    PLL,
    PLLInputError,
    PLLParamError,
    PLLParams,
    circular_rmean,
    wrap_phase,
)


def make_pilot(amplitude, phase, offset, n, phase_step=0.0, step_at=None):
    """确定性合成导频; phase_step 在 step_at 起永久叠加(真实阶跃)。"""
    out = []
    for k in range(n):
        phi = phase + offset * k
        if step_at is not None and k >= step_at:
            phi += phase_step
        out.append(complex(amplitude * math.cos(phi),
                           amplitude * math.sin(phi)))
    return out


def feed_errors(pll, errors, amplitude=1.0):
    """按期望误差序列合成样本: z = A*exp(j*(本步更新前 theta + e))。

    旋转后 atan2 误差即 e(仅浮点舍入), 与 theta/omega 的内部演化无关。
    """
    diags = []
    for e in errors:
        z = complex(amplitude * math.cos(pll.theta + e),
                    amplitude * math.sin(pll.theta + e))
        diags.append(pll.step(z))
    return diags


JP = dict(kp=0.2, ki=0.02, wmax=1.0, amin=0.5, eps=0.05,
          lock_count=30, jump_k=3, jump_eps=0.3)


def core_tuple(diags):
    """抽取与检测无关的核心字段, 用于报告模式的环路无扰动对照。"""
    return tuple((d.valid, d.error, d.omega, d.theta, d.streak,
                  d.locked, d.saturated) for d in diags)


class TestDetectionDisabledByDefault(unittest.TestCase):
    def test_default_params_disable_detection(self):
        p = PLLParams(kp=0.2, ki=0.02, wmax=1.0, amin=0.5,
                      eps=0.05, lock_count=5)
        self.assertIsNone(p.jump_k)
        self.assertIsNone(p.jump_eps)

    def test_report_mode_loop_bit_identical_to_disabled(self):
        # 同样输入下, 仅报告模式的核心更新与完全禁用检测逐位一致。
        sig = make_pilot(1.0, 0.3, 0.0, 60, phase_step=0.8, step_at=40)
        plain = PLL(PLLParams(kp=0.2, ki=0.02, wmax=1.0, amin=0.5,
                              eps=0.05, lock_count=30))
        rpt = PLL(PLLParams(**JP), jump_compensate=False)
        d_plain = plain.run(sig).diagnostics
        d_rpt = rpt.run(sig).diagnostics
        self.assertEqual(core_tuple(d_rpt), core_tuple(d_plain))
        # 禁用实例上检测字段保持默认值
        self.assertTrue(all(d.jump is None and d.candidate is False
                            for d in d_plain))

    def test_compensate_flag_without_detection_has_no_effect(self):
        p = PLLParams(kp=0.2, ki=0.02, wmax=1.0, amin=0.5,
                      eps=0.05, lock_count=5)
        pll = PLL(p, jump_compensate=True)  # 未启用检测:允许, 但永不补偿
        r = pll.run(make_pilot(1.0, 0.3, 0.0, 20))
        self.assertEqual(r.jumps, ())
        self.assertEqual(r.jump_count, 0)


class TestIsolatedGlitch(unittest.TestCase):
    def setUp(self):
        self.params = PLLParams(**JP)
        self.pll = PLL(self.params)
        self.pll.run(make_pilot(1.0, 0.3, 0.0, 60))
        assert self.pll.locked

    def test_single_glitch_interrupted_no_event(self):
        # 样本 60: 孤立 +1.0 rad 毛刺; 其后恢复恒相位导频。
        d_glitch = self.pll.step(
            complex(math.cos(1.3), math.sin(1.3)))
        self.assertGreater(abs(d_glitch.error), 0.9)
        self.assertTrue(d_glitch.candidate)
        self.assertIsNone(d_glitch.jump)
        # 下一正常样本误差约 -0.2(PLL 被毛刺带动 kp*1.0), 打断候选。
        back = self.pll.run(make_pilot(1.0, 0.3, 0.0, 9))
        self.assertLess(abs(back.diagnostics[0].error), 0.3)
        self.assertFalse(back.diagnostics[0].candidate)
        self.assertEqual(back.jumps, ())
        self.assertEqual(self.pll.jump_count, 0)
        self.assertFalse(self.pll.candidate_active)
        # 候选被打断后, 后续真实阶跃从新位置重新计候选。
        step = self.pll.run(make_pilot(1.0, 1.0, 0.0, 60))
        event = next(d.jump for d in step.diagnostics if d.jump)
        self.assertEqual(event.start_index, 70)
        self.assertEqual(event.confirm_index, 72)
        self.assertFalse(event.ambiguous)
        self.assertIsNotNone(event.circular_mean)
        self.assertFalse(event.compensated)  # 默认仅报告
        # 确认样本本身不重开候选(防止连发);其后的事件只可能来自更晚样本
        self.assertTrue(all(d.index != 72 or not d.candidate
                            for d in step.diagnostics))


class TestCrossPi(unittest.TestCase):
    ERRORS = (3.05, 3.12, -3.08)  # 算术和 3.09, 圆方向却在 +pi 附近

    def test_circular_mean_takes_short_arc(self):
        # 独立计算合向量(测试内直接算, 不调用被测函数)
        sx = sum(math.cos(e) for e in self.ERRORS)
        sy = sum(math.sin(e) for e in self.ERRORS)
        exp_r = math.hypot(sx, sy) / 3
        exp_mean = wrap_phase(math.atan2(sy, sx))
        self.assertGreater(exp_r, 0.99)
        self.assertGreater(exp_mean, 3.1)

        pll = PLL(PLLParams(**JP))
        diags = feed_errors(pll, self.ERRORS)
        event = diags[-1].jump
        self.assertIsNotNone(event)
        self.assertEqual(event.start_index, 0)
        self.assertEqual(event.confirm_index, 2)
        self.assertEqual(event.errors, self.ERRORS)
        self.assertAlmostEqual(event.error_sum, sum(self.ERRORS), delta=1e-14)
        self.assertAlmostEqual(event.r, exp_r, delta=1e-15)
        self.assertAlmostEqual(event.circular_mean, exp_mean, delta=1e-14)
        self.assertFalse(event.ambiguous)
        self.assertFalse(event.compensated)
        # 原始跨界误差仍保留在诊断里, 不被圆均值替换
        self.assertAlmostEqual(diags[-1].error, -3.08, delta=1e-14)
        # 确认样本锁定计数清零
        self.assertEqual(diags[-1].streak, 0)

    def test_compensation_uses_circular_mean_after_normal_update(self):
        plain = PLL(PLLParams(kp=0.2, ki=0.02, wmax=1.0, amin=0.5,
                              eps=0.05, lock_count=30))
        comp = PLL(PLLParams(**JP), jump_compensate=True)
        d_plain = feed_errors(plain, self.ERRORS)
        d_comp = feed_errors(comp, self.ERRORS)
        event = d_comp[-1].jump
        self.assertTrue(event.compensated)
        mean = event.circular_mean
        # 补偿 = 原更新结果之上叠加圆均值, 且 omega 不动
        self.assertAlmostEqual(
            d_comp[-1].theta,
            wrap_phase(d_plain[-1].theta + mean), delta=1e-14)
        self.assertEqual(d_comp[-1].omega, d_plain[-1].omega)
        # 原更新当步照常完成:补偿前的中间轨迹与仅报告路径一致
        self.assertAlmostEqual(d_comp[0].theta, d_plain[0].theta, delta=1e-14)
        # 同一样本不再重开候选
        self.assertFalse(d_comp[-1].candidate)

    def test_opposing_errors_ambiguous_no_direction_chosen(self):
        # 严格反相: ±pi/2 交替, 合向量在数值上为零(退化)。
        errors = (math.pi / 2, -math.pi / 2, math.pi / 2, -math.pi / 2)
        # 仅报告
        rpt = PLL(PLLParams(kp=0.2, ki=0.02, wmax=1.0, amin=0.5, eps=0.05,
                            lock_count=30, jump_k=4, jump_eps=0.3))
        ev_rpt = feed_errors(rpt, errors)[-1].jump
        self.assertTrue(ev_rpt.ambiguous)
        self.assertIsNone(ev_rpt.circular_mean)
        self.assertAlmostEqual(ev_rpt.r, 0.0, delta=1e-15)
        # 补偿模式也不挑方向: theta 与无检测轨迹一致
        plain = PLL(PLLParams(kp=0.2, ki=0.02, wmax=1.0, amin=0.5,
                              eps=0.05, lock_count=30))
        comp = PLL(PLLParams(kp=0.2, ki=0.02, wmax=1.0, amin=0.5, eps=0.05,
                             lock_count=30, jump_k=4, jump_eps=0.3),
                   jump_compensate=True)
        d_plain = feed_errors(plain, errors)
        d_comp = feed_errors(comp, errors)
        ev = d_comp[-1].jump
        self.assertTrue(ev.ambiguous)
        self.assertFalse(ev.compensated)
        self.assertEqual(d_comp[-1].theta, d_plain[-1].theta)


class TestRealStep(unittest.TestCase):
    def test_confirm_clears_lock_keeps_frequency_then_relocks(self):
        # 单条连续导频: 0.1 rad/sample 频偏, 样本 400 起相位永久阶跃 +0.7。
        # kp=0.3 使阶跃后误差单调收敛, 除确认候选外不再产生第二次事件。
        pj = PLLParams(kp=0.3, ki=0.01, wmax=1.0, amin=0.5, eps=0.05,
                       lock_count=30, jump_k=3, jump_eps=0.3)
        pp = PLLParams(kp=0.3, ki=0.01, wmax=1.0, amin=0.5, eps=0.05,
                       lock_count=30)
        stream = make_pilot(1.0, 0.3, 0.1, 900, phase_step=0.7, step_at=400)
        # 从已知锁定点启动(theta0/omega0 与信号一致), 避免捕获瞬态触发事件,
        # 精确隔离"样本 400 的真实阶跃"这一场景;omega0=0.1 仍在 wmax 内。
        rpt, plain = PLL(pj, 0.3, 0.1), PLL(pp, 0.3, 0.1)
        r0 = rpt.run(stream[:400])
        plain.run(stream[:400])
        self.assertEqual(r0.jumps, ())
        self.assertTrue(rpt.locked)
        self.assertAlmostEqual(rpt.omega, 0.1, delta=1e-9)

        d = rpt.run(stream[400:]).diagnostics
        d_plain = plain.run(stream[400:]).diagnostics
        event = d[2].jump
        self.assertEqual(event.start_index, 400)
        self.assertEqual(event.confirm_index, 402)
        self.assertEqual(len(event.errors), 3)
        self.assertTrue(all(e > 0.3 for e in event.errors))
        self.assertTrue(all(e > 0 for e in event.errors))
        self.assertAlmostEqual(event.error_sum, sum(event.errors),
                               delta=1e-14)
        # 确认样本: 锁定计数清零, omega 与无检测轨迹逐位相等(频率估计保留)
        self.assertEqual(d[2].streak, 0)
        self.assertFalse(d[2].locked)
        self.assertEqual(d[2].omega, d_plain[2].omega)
        self.assertNotAlmostEqual(d[2].omega, 0.0)  # 未被清零
        # 仅报告: theta/omega 全程与无检测路径逐位相等;全程仅一次事件
        self.assertEqual(core_tuple(d), core_tuple(d_plain))
        self.assertEqual(rpt.jump_count, 1)
        self.assertFalse(event.compensated)
        # 阶跃是真实的: 环路最终重新锁定且频率估计回到 0.1 附近
        self.assertTrue(rpt.locked)
        self.assertAlmostEqual(rpt.omega, 0.1, delta=1e-4)
        self.assertLess(abs(d[-1].error), 0.05)


class TestLowAmplitudeInterrupt(unittest.TestCase):
    def test_low_amplitude_breaks_candidate_no_forged_phase(self):
        pll = PLL(PLLParams(kp=0.2, ki=0.02, wmax=1.0, amin=0.5, eps=0.05,
                            lock_count=30, jump_k=3, jump_eps=0.3))
        d01 = feed_errors(pll, [0.9, 0.9])
        self.assertTrue([d.candidate for d in d01] == [True, True])
        theta_before, omega_before = pll.theta, pll.omega
        # 低幅度样本打断候选: 无效、冻结、无误差、不确认
        dz = pll.step(0j)
        self.assertFalse(dz.valid)
        self.assertIsNone(dz.error)
        self.assertFalse(dz.candidate)
        self.assertIsNone(dz.jump)
        self.assertEqual(dz.theta, theta_before)
        self.assertEqual(dz.omega, omega_before)
        self.assertEqual(pll.lost_samples, 1)
        self.assertFalse(pll.candidate_active)
        # 候选重新从样本 3 计起, 3、4、5 连续越限 -> 5 确认(起点不是 0)
        d = feed_errors(pll, [0.9, 0.9, 0.9])
        self.assertTrue([x.candidate for x in d[:2]] == [True, True])
        event = d[2].jump
        self.assertIsNotNone(event)
        self.assertEqual(event.start_index, 3)
        self.assertEqual(event.confirm_index, 5)
        self.assertEqual(pll.jump_count, 1)


class TestChunkInvariance(unittest.TestCase):
    def _stream(self):
        s = make_pilot(1.0, 0.3, 0.05, 200)  # 先捕获
        s += [0j, 0j]                          # 低幅度中断
        s += make_pilot(1.0, 0.9, 0.05, 80)    # 恢复后相位已偏 -> 候选/事件
        s[260] = complex(math.cos(2.5), math.sin(2.5))  # 一处孤立毛刺
        return s

    def _assert_same_trace(self, chunked, whole_pll, whole_diags, diags):
        self.assertEqual(tuple(diags), tuple(whole_diags))  # 含 jump 对象
        self.assertEqual(chunked.omega, whole_pll.omega)
        self.assertEqual(chunked.theta, whole_pll.theta)
        self.assertEqual(chunked.streak, whole_pll.streak)
        self.assertEqual(chunked.locked, whole_pll.locked)
        self.assertEqual(chunked.jump_count, whole_pll.jump_count)
        self.assertEqual(chunked.candidate_active,
                         whole_pll.candidate_active)

    def test_chunks_identical_report_mode(self):
        params = PLLParams(kp=0.2, ki=0.02, wmax=1.0, amin=0.5, eps=0.05,
                           lock_count=20, jump_k=3, jump_eps=0.25)
        sig = self._stream()
        whole_pll = PLL(params)
        whole_diags = whole_pll.run(sig).diagnostics
        for bounds in ((0, 1, 2, 203, len(sig)),
                       (0, 100, 259, 261, len(sig)),
                       (0, 205, len(sig))):
            chunked = PLL(params)
            diags = []
            for lo, hi in zip(bounds, bounds[1:]):
                diags.extend(chunked.run(sig[lo:hi]).diagnostics)
            self._assert_same_trace(chunked, whole_pll, whole_diags, diags)

    def test_chunks_identical_compensate_mode(self):
        params = PLLParams(kp=0.2, ki=0.02, wmax=1.0, amin=0.5, eps=0.05,
                           lock_count=20, jump_k=3, jump_eps=0.25)
        sig = self._stream()
        whole_pll = PLL(params, jump_compensate=True)
        whole_res = whole_pll.run(sig)
        chunked = PLL(params, jump_compensate=True)
        diags = []
        batch_jumps = []
        for lo, hi in ((0, 202), (202, 260), (260, len(sig))):
            r = chunked.run(sig[lo:hi])
            diags.extend(r.diagnostics)
            batch_jumps.append(r.jumps)
        self._assert_same_trace(chunked, whole_pll, whole_res.diagnostics,
                                diags)
        # 事件只在确认所在批次出现一次, 不跨批重复携带
        flat = tuple(e for batch in batch_jumps for e in batch)
        self.assertEqual(flat, whole_res.jumps)
        self.assertEqual(
            [len(b) for b in batch_jumps],
            [sum(1 for e in whole_res.jumps if lo <= e.confirm_index < hi)
             for lo, hi in ((0, 202), (202, 260), (260, len(sig)))])

    def test_step_run_mixed_identical(self):
        params = PLLParams(kp=0.2, ki=0.02, wmax=1.0, amin=0.5, eps=0.05,
                           lock_count=20, jump_k=3, jump_eps=0.25)
        sig = self._stream()
        whole = PLL(params).run(sig).diagnostics
        mixed = PLL(params)
        diags = [mixed.step(z) for z in sig[:150]]
        diags.extend(mixed.run(sig[150:]).diagnostics)
        self.assertEqual(tuple(diags), tuple(whole))

    def test_candidate_state_carries_across_chunk_boundary(self):
        # 候选恰好跨在切块边界: 前批 2 个越限, 后批第 1 个确认。
        params = PLLParams(kp=0.2, ki=0.02, wmax=1.0, amin=0.5, eps=0.05,
                           lock_count=10, jump_k=3, jump_eps=0.3)
        ref = feed_errors(PLL(params), [0.9, 0.9, 0.9])
        split = PLL(params)
        d1 = feed_errors(split, [0.9, 0.9])
        self.assertTrue(split.candidate_active)
        self.assertEqual(split.jump_count, 0)
        d2 = feed_errors(split, [0.9])
        self.assertEqual(tuple(d1 + d2), tuple(ref))
        event = d2[0].jump
        self.assertEqual((event.start_index, event.confirm_index), (0, 2))


class TestThresholdSemantics(unittest.TestCase):
    def test_strict_greater_than_threshold(self):
        # |error| 恰好等于 jump_eps 不计数: 打断候选。
        params = PLLParams(kp=0.2, ki=0.02, wmax=1.0, amin=0.5, eps=0.05,
                           lock_count=10, jump_k=2, jump_eps=0.3)
        pll = PLL(params)
        ds = feed_errors(pll, [0.31, 0.30, 0.31, 0.31])
        # 0.31 起候选; 0.30(恰等门限)打断; 后两个 0.31 重新计满确认
        self.assertTrue(ds[0].candidate and ds[0].jump is None)
        self.assertFalse(ds[1].candidate)
        self.assertIsNone(ds[1].jump)
        self.assertTrue(ds[2].candidate)
        ev = ds[3].jump
        self.assertIsNotNone(ev)
        self.assertEqual((ev.start_index, ev.confirm_index), (2, 3))

    def test_k_equals_1_confirms_immediately_no_same_sample_reopen(self):
        params = PLLParams(kp=0.2, ki=0.02, wmax=1.0, amin=0.5, eps=0.05,
                           lock_count=10, jump_k=1, jump_eps=0.3)
        pll = PLL(params)
        ds = feed_errors(pll, [0.5, 0.5, 0.0])
        self.assertEqual([d.jump is not None for d in ds], [True, True, False])
        # 每个确认样本本身不挂候选(不连发), 事件各自独立计数
        self.assertTrue(all(not d.candidate for d in ds))
        self.assertEqual([d.jump.start_index for d in ds[:2]], [0, 1])
        self.assertEqual(pll.jump_count, 2)

    def test_confirm_clears_streak_even_when_within_eps(self):
        # jump_eps < eps: 确认候选的样本可能同时满足 |error| <= eps,
        # 但确认后锁定计数必须清零(题面硬性要求)。
        params = PLLParams(kp=0.2, ki=0.02, wmax=1.0, amin=0.5, eps=0.2,
                           lock_count=3, jump_k=1, jump_eps=0.05)
        pll = PLL(params)
        feed_errors(pll, [0.01, 0.01, 0.01])
        self.assertTrue(pll.locked)
        d = feed_errors(pll, [0.1])[0]  # > jump_eps 但 <= eps
        self.assertIsNotNone(d.jump)
        self.assertEqual(d.streak, 0)
        self.assertFalse(d.locked)

    def test_low_amplitude_does_not_count_toward_k(self):
        # K 个越限中间夹一个无效样本, 必须重新数满 K 个有效样本。
        params = PLLParams(kp=0.2, ki=0.02, wmax=1.0, amin=0.5, eps=0.05,
                           lock_count=10, jump_k=3, jump_eps=0.3)
        pll = PLL(params)
        ds = feed_errors(pll, [0.5, 0.5])
        dz = pll.step(0.1 + 0j)  # 幅度 0.1 < amin
        ds2 = feed_errors(pll, [0.5, 0.5])
        # 两个有效越限仍未确认
        self.assertIsNone(ds2[1].jump)
        self.assertTrue(ds2[1].candidate)
        self.assertEqual(pll.jump_count, 0)
        self.assertFalse(dz.valid)
        ds3 = feed_errors(pll, [0.5])
        self.assertEqual((ds3[0].jump.start_index, ds3[0].jump.confirm_index),
                         (3, 5))


    def test_rejected_batch_leaves_candidate_intact(self):
        # 批次含非法样本时整批拒绝: 已挂起的候选与计数不得部分变更。
        params = PLLParams(kp=0.2, ki=0.02, wmax=1.0, amin=0.5, eps=0.05,
                           lock_count=10, jump_k=3, jump_eps=0.3)
        pll = PLL(params)
        feed_errors(pll, [0.6, 0.6])
        self.assertTrue(pll.candidate_active)
        bad_batch = [complex(math.cos(pll.theta + 0.6),
                             math.sin(pll.theta + 0.6)),
                     complex(float("nan"), 0.0)]
        with self.assertRaises(PLLInputError):
            pll.run(bad_batch)
        self.assertTrue(pll.candidate_active)
        self.assertEqual(pll.jump_count, 0)
        self.assertEqual(pll.processed, 2)
        # 候选仍可从断点继续: 再一个越限样本即确认, 起点为 0
        ev = feed_errors(pll, [0.6])[0].jump
        self.assertIsNotNone(ev)
        self.assertEqual((ev.start_index, ev.confirm_index), (0, 2))


class TestResultPlumbing(unittest.TestCase):
    def test_jumps_batch_local_count_cumulative(self):
        params = PLLParams(kp=0.2, ki=0.02, wmax=1.0, amin=0.5, eps=0.05,
                           lock_count=10, jump_k=2, jump_eps=0.3)
        pll = PLL(params)
        r1 = pll.run(make_pilot(1.0, 0.3, 0.0, 30))
        self.assertEqual(r1.jumps, ())
        self.assertEqual(r1.jump_count, 0)
        # step() 连续 2 个越限误差 -> 确认一次(诊断携带事件)
        d2 = feed_errors(pll, [0.5, 0.5])
        self.assertEqual([d.jump is not None for d in d2], [False, True])
        self.assertIsInstance(d2[1].jump, JumpCandidate)
        self.assertEqual(pll.jump_count, 1)
        # 回到当前跟踪相位的恒相位导频: run() 本批无事件, 累计计数保留
        r3 = pll.run(make_pilot(1.0, pll.theta, 0.0, 40))
        self.assertEqual(r3.jumps, ())
        self.assertEqual(r3.jump_count, 1)
        self.assertEqual(pll.jump_count, 1)
        self.assertTrue(pll.locked)


class TestCircularRmean(unittest.TestCase):
    def test_basic_and_threshold_boundary(self):
        mean, r = circular_rmean([0.0, 0.0], 0.5)
        self.assertEqual(mean, 0.0)
        self.assertAlmostEqual(r, 1.0, delta=1e-15)
        # 两夹角 90°: r = cos(45°) ≈ 0.7071
        r45 = math.cos(math.pi / 4)
        mean, r = circular_rmean([0.0, math.pi / 2], r45 + 1e-12)
        self.assertIsNone(mean)  # 阈值在 r 之上 -> 退化
        self.assertAlmostEqual(r, r45, delta=1e-15)
        mean2, r2 = circular_rmean([0.0, math.pi / 2], r45)
        self.assertAlmostEqual(mean2, math.pi / 4, delta=1e-15)  # r >= 阈值
        self.assertEqual(r2, r)

    def test_validation(self):
        with self.assertRaises(PLLParamError):
            circular_rmean([0.0], -0.01)
        with self.assertRaises(PLLParamError):
            circular_rmean([0.0], 1.01)
        with self.assertRaises(PLLParamError):
            circular_rmean([0.0], float("nan"))
        with self.assertRaises(PLLInputError):
            circular_rmean([], 0.5)
        with self.assertRaises(PLLInputError):
            circular_rmean([0.0, float("inf")], 0.5)
        with self.assertRaises(PLLInputError):
            circular_rmean("0.1", 0.5)
        with self.assertRaises(PLLInputError):
            circular_rmean(123, 0.5)

    def test_diagnostic_default_fields(self):
        # 旧的 9 字段位置构造仍可用(向后兼容)
        d = Diagnostic(0, False, 0.0, None, 0.0, 0.0, 0, False, False)
        self.assertFalse(d.candidate)
        self.assertIsNone(d.jump)


class TestJumpParamValidation(unittest.TestCase):
    BASE = dict(kp=0.2, ki=0.02, wmax=1.0, amin=0.5, eps=0.05, lock_count=5)

    def test_partial_pair_rejected(self):
        with self.assertRaises(PLLParamError):
            PLLParams(**self.BASE, jump_k=3)
        with self.assertRaises(PLLParamError):
            PLLParams(**self.BASE, jump_eps=0.3)

    def test_bad_jump_values(self):
        bad_kwargs = [
            dict(jump_k=0, jump_eps=0.3),
            dict(jump_k=-1, jump_eps=0.3),
            dict(jump_k=True, jump_eps=0.3),
            dict(jump_k=2.5, jump_eps=0.3),
            dict(jump_k="3", jump_eps=0.3),
            dict(jump_k=3, jump_eps=0.0),
            dict(jump_k=3, jump_eps=-0.1),
            dict(jump_k=3, jump_eps=float("nan")),
            dict(jump_k=3, jump_eps=float("inf")),
        ]
        for kw in bad_kwargs:
            with self.subTest(kw=kw):
                with self.assertRaises(PLLParamError):
                    PLLParams(**self.BASE, **kw)

    def test_bad_rmin_even_when_disabled(self):
        with self.assertRaises(PLLParamError):
            PLLParams(**self.BASE, r_min=-0.01)
        with self.assertRaises(PLLParamError):
            PLLParams(**self.BASE, r_min=1.01)
        with self.assertRaises(PLLParamError):
            PLLParams(**self.BASE, r_min=float("inf"))
        p = PLLParams(**self.BASE, r_min=0)      # 边界允许
        self.assertEqual(p.r_min, 0.0)
        p = PLLParams(**self.BASE, r_min=1)      # 边界允许(int 规范化)
        self.assertEqual(p.r_min, 1.0)

    def test_bad_compensate_flag(self):
        p = PLLParams(**self.BASE, jump_k=3, jump_eps=0.3)
        with self.assertRaises(PLLParamError):
            PLL(p, jump_compensate=1)
        with self.assertRaises(PLLParamError):
            PLL(p, jump_compensate="true")


if __name__ == "__main__":
    unittest.main()
