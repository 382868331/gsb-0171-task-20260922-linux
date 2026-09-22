"""独立参考实现（oracle）。

按题面契约从头编写，**不 import pll_diag 的任何内容**，用于给测试
提供期望值。与被测核心的刻意差异：

* 用 ``cmath.exp`` 做复数旋转（核心用 cos/sin 的实虚部分式）；
* 用 ``math.fmod`` 做角度回绕（核心用 ``% (2π)``）；
* atan2 后用条件平移到半开区间；
* 限幅判定显式比较（核心用钳位结果是否变化）。

输入为“原始样本 + 参数”，输出按步 dict 列表与终态 dict，
方便测试逐字段比较。
"""

from __future__ import annotations

import cmath
import math
from typing import Sequence


def ref_wrap(a: float) -> float:
    w = math.fmod(a + math.pi, 2.0 * math.pi)
    if w < 0.0:
        w += 2.0 * math.pi
    return w - math.pi


def ref_run(
    samples: Sequence[complex],
    kp: float,
    ki: float,
    wmax: float,
    eps: float,
    L: int,
    amin: float = 1e-12,
    theta0: float = 0.0,
    omega0: float = 0.0,
) -> dict:
    theta, omega = theta0, omega0
    streak = 0
    processed = 0
    lost_count = 0
    steps = []
    locked_at = None
    lost_indices = []
    saturated_indices = []
    max_abs_error = None

    for i, z in enumerate(samples):
        amp = abs(z)
        if amp < amin:
            lost_count += 1
            streak = 0
            steps.append({
                "index": i,
                "amplitude": amp,
                "valid": False,
                "error": None,
                "theta": theta,
                "omega": omega,
                "saturated": False,
                "streak": 0,
                "locked": False,
            })
            lost_indices.append(i)
            processed += 1
            continue

        r = z * cmath.exp(-1j * theta)
        error = math.atan2(r.imag, r.real)
        if error >= math.pi:
            error -= 2.0 * math.pi
        elif error < -math.pi:
            error += 2.0 * math.pi

        unc = omega + ki * error
        if unc > wmax:
            omega_new, saturated = wmax, True
        elif unc < -wmax:
            omega_new, saturated = -wmax, True
        else:
            omega_new, saturated = unc, False

        theta_new = ref_wrap(theta + omega_new + kp * error)

        if abs(error) <= eps:
            streak += 1
        else:
            streak = 0
        locked = streak >= L
        if locked and locked_at is None:
            locked_at = i
        if max_abs_error is None or abs(error) > max_abs_error:
            max_abs_error = abs(error)
        if saturated:
            saturated_indices.append(i)

        theta, omega = theta_new, omega_new
        processed += 1
        steps.append({
            "index": i,
            "amplitude": amp,
            "valid": True,
            "error": error,
            "theta": theta,
            "omega": omega,
            "saturated": saturated,
            "streak": streak,
            "locked": locked,
        })

    return {
        "steps": steps,
        "theta": theta,
        "omega": omega,
        "streak": streak,
        "locked": streak >= L,
        "processed": processed,
        "lost_count": lost_count,
        "locked_at": locked_at,
        "lost_indices": tuple(lost_indices),
        "saturated_indices": tuple(saturated_indices),
        "max_abs_error": max_abs_error,
    }


def ref_pilot(
    n: int,
    dphi: float,
    phi0: float,
    zero_at: int | None = None,
) -> list[complex]:
    """确定性合成导频 z[k] = exp(j(phi0 + dphi*k))；可在 zero_at 置零。"""
    out = []
    for k in range(n):
        if k == zero_at:
            out.append(0j)
        else:
            out.append(cmath.exp(1j * (phi0 + dphi * k)))
    return out
