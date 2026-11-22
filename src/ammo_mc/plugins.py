"""Extensibility interfaces and reference plugins for 6-DOF rigid-body flight & Guided Munitions (App_Spec R28)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable
import numpy as np

from ammo_mc.environment import AtmosphereEnvironment
from ammo_mc.exterior import BatchTrajectoryConfig, integrate_trajectories_batch
from ammo_mc.launch import da_03_cep_circular


class GuidancePlugin(ABC):
    """Abstract interface for guided-munition / course-correction fuze models (AR-05, MS-01)."""

    name: str = "base_guidance"

    @abstractmethod
    def apply_guidance(
        self,
        ballistic_range_m: np.ndarray,
        ballistic_deflection_m: np.ndarray,
        aim_range_m: float,
        aim_deflection_m: float,
        params: dict[str, Any],
        rng: np.random.Generator,
    ) -> dict[str, np.ndarray]:
        """Transform unguided ballistic impacts into guided impacts with failure-fallback branch."""


class SixDOFPlugin(ABC):
    """Reserved interface for McCoy 6-DOF rigid-body trajectory integrators (App_Spec R28)."""

    name: str = "base_6dof"

    @abstractmethod
    def integrate_batch(
        self,
        initial_states: np.ndarray,
        masses_kg: np.ndarray,
        form_factors: np.ndarray,
        env: AtmosphereEnvironment,
        cfg: BatchTrajectoryConfig,
        extra_params: dict[str, Any],
        fuze_times_s: np.ndarray | None = None,
        terrain_fn: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
    ) -> dict[str, Any]:
        """Integrate rigid-body trajectories and return standard impact dictionary."""


class CourseCorrectionGuidancePlugin(GuidancePlugin):
    """Course-Correction Fuze (CCF / PGK) & Precision-Guided Shell plugin (AR-05).

    Models aerodynamic control authority (`max_authority_m`), GNSS/SAL guidance accuracy
    (`cep_guided_m`), occasion bias (`gnss_bias_sigma_m`), and guidance failure fallback (`p_guidance_fail`)
    onto the unguided ballistic trajectory.
    """

    name = "course_correction"

    def apply_guidance(
        self,
        ballistic_range_m: np.ndarray,
        ballistic_deflection_m: np.ndarray,
        aim_range_m: float,
        aim_deflection_m: float,
        params: dict[str, Any],
        rng: np.random.Generator,
    ) -> dict[str, np.ndarray]:
        n = ballistic_range_m.size
        cep_guided = float(params.get("cep_guided_m", 8.0))
        sigma_g = cep_guided / np.sqrt(2.0 * np.log(2.0))
        max_auth = float(params.get("max_authority_m", 250.0))
        p_fail = float(params.get("p_guidance_fail", 0.03))
        bias_sigma = float(params.get("gnss_bias_sigma_m", 2.0))

        b_r = float(rng.normal(0.0, bias_sigma)) if bias_sigma > 0 else 0.0
        b_d = float(rng.normal(0.0, bias_sigma)) if bias_sigma > 0 else 0.0

        desired_r = aim_range_m + b_r + rng.normal(0.0, sigma_g, size=n)
        desired_d = aim_deflection_m + b_d + rng.normal(0.0, sigma_g, size=n)

        corr_r = desired_r - ballistic_range_m
        corr_d = desired_d - ballistic_deflection_m
        corr_mag = np.hypot(corr_r, corr_d)

        scale = np.where(corr_mag > max_auth, max_auth / np.maximum(corr_mag, 1e-9), 1.0)
        guided_r = ballistic_range_m + corr_r * scale
        guided_d = ballistic_deflection_m + corr_d * scale

        failed = rng.random(size=n) < p_fail
        final_r = np.where(failed, ballistic_range_m, guided_r)
        final_d = np.where(failed, ballistic_deflection_m, guided_d)
        _ = da_03_cep_circular(sigma_g)
        return {
            "range_m": final_r,
            "deflection_m": final_d,
            "guidance_failed": failed,
        }


class SeekerMissDistancePlugin(GuidancePlugin):
    """Effects-level Guided Missile miss-distance plugin (MS-01: ATGM / SAM / Cruise)."""

    name = "seeker_miss"

    def apply_guidance(
        self,
        ballistic_range_m: np.ndarray,
        ballistic_deflection_m: np.ndarray,
        aim_range_m: float,
        aim_deflection_m: float,
        params: dict[str, Any],
        rng: np.random.Generator,
    ) -> dict[str, np.ndarray]:
        n = ballistic_range_m.size
        cep_m = float(params.get("cep_guided_m", 1.2))
        sigma_m = cep_m / np.sqrt(2.0 * np.log(2.0))
        p_fail = float(params.get("p_guidance_fail", 0.04))
        fail_sigma_m = float(params.get("fail_miss_sigma_m", 85.0))

        failed = rng.random(size=n) < p_fail
        r_ok = aim_range_m + rng.normal(0.0, sigma_m, size=n)
        d_ok = aim_deflection_m + rng.normal(0.0, sigma_m, size=n)
        r_bad = aim_range_m + rng.normal(0.0, fail_sigma_m, size=n)
        d_bad = aim_deflection_m + rng.normal(0.0, fail_sigma_m, size=n)
        _ = da_03_cep_circular(sigma_m)
        return {
            "range_m": np.where(failed, r_bad, r_ok),
            "deflection_m": np.where(failed, d_bad, d_ok),
            "guidance_failed": failed,
        }


class McCoyEpicyclic6DOFAdapter(SixDOFPlugin):
    """Reference 6-DOF rigid-body adapter coupling McCoy epicyclic initial yaw drag with MPM."""

    name = "mccoy_6dof_adapter"

    def integrate_batch(
        self,
        initial_states: np.ndarray,
        masses_kg: np.ndarray,
        form_factors: np.ndarray,
        env: AtmosphereEnvironment,
        cfg: BatchTrajectoryConfig,
        extra_params: dict[str, Any],
        fuze_times_s: np.ndarray | None = None,
        terrain_fn: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
    ) -> dict[str, Any]:
        yaw0_rad = float(extra_params.get("initial_yaw_rms_rad", 0.015))
        epicyclic_factor = 1.0 + 0.5 * cfg.c_da2 * (yaw0_rad**2)
        mpm_cfg = BatchTrajectoryConfig(**{**cfg.__dict__, "fidelity": "L2"})
        return integrate_trajectories_batch(
            initial_states=initial_states,
            masses_kg=masses_kg,
            form_factors=form_factors * epicyclic_factor,
            env=env,
            cfg=mpm_cfg,
            fuze_times_s=fuze_times_s,
            terrain_fn=terrain_fn,
        )


GUIDANCE_PLUGINS: dict[str, GuidancePlugin] = {}
SIX_DOF_PLUGINS: dict[str, SixDOFPlugin] = {}


def register_guidance_plugin(plugin: GuidancePlugin) -> None:
    GUIDANCE_PLUGINS[plugin.name] = plugin


def register_6dof_plugin(plugin: SixDOFPlugin) -> None:
    SIX_DOF_PLUGINS[plugin.name] = plugin


def get_guidance_plugin(name: str) -> GuidancePlugin:
    if name not in GUIDANCE_PLUGINS:
        raise KeyError(f"Unknown guidance plugin '{name}'. Available: {list(GUIDANCE_PLUGINS.keys())}")
    return GUIDANCE_PLUGINS[name]


def get_6dof_plugin(name: str) -> SixDOFPlugin:
    if name not in SIX_DOF_PLUGINS:
        raise KeyError(f"Unknown 6-DOF plugin '{name}'. Available: {list(SIX_DOF_PLUGINS.keys())}")
    return SIX_DOF_PLUGINS[name]


def list_plugins() -> dict[str, list[str]]:
    return {
        "guidance_plugins": sorted(GUIDANCE_PLUGINS.keys()),
        "six_dof_plugins": sorted(SIX_DOF_PLUGINS.keys()),
    }


register_guidance_plugin(CourseCorrectionGuidancePlugin())
register_guidance_plugin(SeekerMissDistancePlugin())
register_6dof_plugin(McCoyEpicyclic6DOFAdapter())
