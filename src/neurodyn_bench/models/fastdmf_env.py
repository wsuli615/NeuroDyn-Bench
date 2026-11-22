"""FastDMF Whole-Brain Dynamic Mean Field simulator wrapper (canonical C++ core).

Wraps upstream concog/fastdmf (Mediano, Luppi & Rosas / Herzog et al. 2024,
Network Neuroscience) via our native pybind11 extension `_fastdmf_ext` with
bundled Schaefer-100 HCP structural connectome, analytic Feedback Inhibition
Control (FIC), and parallel Balloon-Windkessel BOLD hemodynamics.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
from scipy.stats import ks_2samp

from neurodyn_bench.models.base import BaseSimulator, ParameterSpec, SimulationResult


def _default_connectome_path() -> Path:
    here = Path(__file__).resolve()
    repo_root = here.parents[3]
    cand = repo_root / "data" / "connectomes" / "DTI_fiber_consensus_HCP.csv"
    if cand.exists():
        return cand
    fallback = repo_root / "third_party" / "fastdmf_gitlab" / "python" / "fastdmf" / "DTI_fiber_consensus_HCP.csv"
    if fallback.exists():
        return fallback
    raise FileNotFoundError(
        "Could not locate DTI_fiber_consensus_HCP.csv. Run scripts/build_fastdmf.py first."
    )


def compute_fc_and_fcd(
    bold: np.ndarray,
    burn_in_trs: int = 20,
    fcd_window: int = 15,
    fcd_step: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute static Functional Connectivity (FC) and sliding-window FCD upper triangle.

    Args:
        bold: Array of shape (N, T_bold)
        burn_in_trs: Number of initial TRs discarded as hemodynamic transient
        fcd_window: Sliding window length in TRs for FCD
        fcd_step: Step size in TRs for FCD windows

    Returns:
        fc: (N, N) Pearson correlation matrix
        fcd_triu: 1D array of upper-triangular window-to-window FC correlations
    """
    if bold.shape[1] > burn_in_trs + 5:
        b = bold[:, burn_in_trs:]
    else:
        b = bold
    n_rois, n_trs = b.shape
    fc = np.corrcoef(b)
    iu = np.triu_indices(n_rois, k=1)

    if n_trs < fcd_window + fcd_step:
        return fc, np.array([], dtype=np.float64)

    win_vecs = []
    for start in range(0, n_trs - fcd_window + 1, fcd_step):
        w_fc = np.corrcoef(b[:, start : start + fcd_window])
        vec = w_fc[iu]
        if np.all(np.isfinite(vec)):
            win_vecs.append(vec)

    if len(win_vecs) < 3:
        return fc, np.array([], dtype=np.float64)

    mat = np.vstack(win_vecs)
    fcd = np.corrcoef(mat)
    w_iu = np.triu_indices(fcd.shape[0], k=1)
    fcd_triu = fcd[w_iu]
    fcd_triu = fcd_triu[np.isfinite(fcd_triu)]
    return fc, fcd_triu


def _format_dmf_dict(d: dict) -> dict[str, np.ndarray]:
    q: dict[str, np.ndarray] = {}
    for k, v in d.items():
        if isinstance(v, np.ndarray):
            arr = v.reshape(1).astype(np.float64) if v.ndim == 0 else v.astype(np.float64)
            q[k] = np.asfortranarray(arr)
        elif isinstance(v, (list, tuple)):
            q[k] = np.asfortranarray(np.asarray(v, dtype=np.float64))
        elif np.isscalar(v):
            q[k] = np.asfortranarray(np.array([float(v)], dtype=np.float64))
        else:
            raise ValueError(f"Unsupported parameter value for '{k}': {type(v)}")
    return q


class FastDMFSimulator(BaseSimulator):
    """Whole-brain Dynamic Mean Field simulator backed by canonical C++ FastDMF."""

    def __init__(
        self,
        connectome_path: str | Path | None = None,
        c_scale: float = 0.2,
        nb_steps: int = 160_000,  # 160 s @ 1 ms rate step -> 80 TRs at TR=2s
        batch_size: int = 5000,
        tr_s: float = 2.0,
        dt_ms: float = 0.1,
        burn_in_trs: int = 15,
        target_fc: np.ndarray | None = None,
        target_fcd_triu: np.ndarray | None = None,
        n_hetero_groups: int = 0,
    ) -> None:
        from neurodyn_bench.models import _fastdmf_ext

        self._ext = _fastdmf_ext
        self.name = "fastdmf_wholebrain" if n_hetero_groups == 0 else f"fastdmf_hetero{n_hetero_groups}"
        c_path = Path(connectome_path) if connectome_path else _default_connectome_path()
        c_mat = np.loadtxt(c_path, delimiter=",", dtype=np.float64)
        c_mat = float(c_scale) * (c_mat / np.max(c_mat))
        self.c_scale = float(c_scale)
        self.C = np.asfortranarray(c_mat)
        self.n_rois = int(self.C.shape[0])
        self.node_strength = self.C.sum(axis=0).squeeze()
        self.nb_steps = int(nb_steps)
        self.batch_size = int(batch_size)
        self.tr_s = float(tr_s)
        self.dt_ms = float(dt_ms)
        self.burn_in_trs = int(burn_in_trs)
        self.target_fc = target_fc
        self.target_fcd_triu = target_fcd_triu
        self.n_hetero_groups = int(n_hetero_groups)

        # Build deterministic contiguous ROI groups for optional heterogeneous FIC/gain tasks
        if self.n_hetero_groups > 0:
            self._roi_groups = np.array_split(np.arange(self.n_rois), self.n_hetero_groups)
        else:
            self._roi_groups = []

    def parameter_specs(self) -> list[ParameterSpec]:
        specs = [
            ParameterSpec("G", "AU", 0.5, 4.0, 0.0, 10.0, "linear", 2.2, "Global structural coupling"),
            ParameterSpec("w", "AU", 0.7, 2.2, 0.1, 4.0, "linear", 1.4, "Local excitatory recurrence"),
            ParameterSpec("JN", "nA", 0.08, 0.25, 0.01, 0.8, "linear", 0.15, "Excitatory NMDA coupling"),
            ParameterSpec("I0", "nA", 0.28, 0.48, 0.05, 1.0, "linear", 0.382, "Effective external input"),
            ParameterSpec("taon", "ms", 50.0, 180.0, 20.0, 300.0, "log", 100.0, "NMDA synaptic decay constant"),
            ParameterSpec("taog", "ms", 5.0, 25.0, 2.0, 60.0, "log", 10.0, "GABA synaptic decay constant"),
            ParameterSpec("sigma", "nA", 0.002, 0.012, 0.0005, 0.05, "log", 0.007, "Noise standard deviation"),
            ParameterSpec("fic_alpha", "AU", 0.4, 1.2, 0.1, 3.0, "linear", 0.75, "FIC linear slope (J = alpha*G*S + 1)"),
        ]
        for g_idx in range(self.n_hetero_groups):
            specs.append(
                ParameterSpec(
                    f"J_scale_g{g_idx}",
                    "AU",
                    0.7,
                    1.3,
                    0.3,
                    2.5,
                    "linear",
                    1.0,
                    f"Regional FIC multiplier for ROI group {g_idx}",
                )
            )
        return specs

    def default_params(self) -> dict[str, float]:
        return {spec.name: float(spec.nominal) for spec in self.parameter_specs() if spec.nominal is not None}

    def _build_cpp_params(self, params: dict[str, float], seed: int) -> dict[str, np.ndarray]:
        p = self.default_params()
        p.update({k: float(v) for k, v in params.items()})

        j_vec = p["fic_alpha"] * p["G"] * self.node_strength + 1.0
        if self.n_hetero_groups > 0:
            j_vec = j_vec.copy()
            for g_idx, rois in enumerate(self._roi_groups):
                j_vec[rois] *= p[f"J_scale_g{g_idx}"]

        raw = {
            "C": self.C,
            "receptors": 0.0,
            "dt": self.dt_ms,
            "taon": p["taon"],
            "taog": p["taog"],
            "gamma": 0.641,
            "sigma": p["sigma"],
            "JN": p["JN"],
            "I0": p["I0"],
            "Jexte": 1.0,
            "Jexti": 0.7,
            "w": p["w"],
            "de": 0.16,
            "Ie": 125.0 / 310.0,
            "g_e": 310.0,
            "di": 0.087,
            "Ii": 177.0 / 615.0,
            "g_i": 615.0,
            "wgaine": 0.0,
            "wgaini": 0.0,
            "G": p["G"],
            "J": j_vec,
            "TR": self.tr_s,
            "dtt": 0.001,
            "batch_size": float(self.batch_size),
            "seed": float(int(seed)),
        }
        return _format_dmf_dict(raw)

    def simulate_raw(
        self,
        params: dict[str, float],
        seed: int = 0,
        duration_scale: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Run C++ FastDMF and return (rate_buffer, bold, wall_time_s)."""
        t0 = time.perf_counter()
        # Ensure steps is a multiple of batch_size and >= 2 * batch_size for circular buffer safety
        raw_steps = round(self.nb_steps * float(duration_scale))
        steps = max(2 * self.batch_size, (raw_steps // self.batch_size) * self.batch_size)
        cpp_params = self._build_cpp_params(params, seed=seed)
        rate_buf, bold = self._ext.run_dmf(cpp_params, steps, self.n_rois, False, True)
        return rate_buf, bold, time.perf_counter() - t0

    def simulate(
        self,
        params: dict[str, float],
        seed: int = 0,
        duration_scale: float = 1.0,
    ) -> SimulationResult:
        p = self.default_params()
        p.update({k: float(v) for k, v in params.items()})
        if min(p["taon"], p["taog"], p["sigma"], p["JN"], p["w"], p["fic_alpha"]) <= 0.0 or p["G"] < 0.0:
            return SimulationResult(
                observables={},
                feasible=False,
                failure_reason="non_positive_param",
                wall_time_s=0.0,
                sim_units=float(duration_scale),
                seed=seed,
            )

        rate_buf, bold, wall_s = self.simulate_raw(p, seed=seed, duration_scale=duration_scale)

        if not (np.all(np.isfinite(rate_buf)) and np.all(np.isfinite(bold))):
            return SimulationResult(
                observables={},
                feasible=False,
                failure_reason="nan_or_inf",
                wall_time_s=wall_s,
                sim_units=float(duration_scale),
                seed=seed,
            )

        mean_rate = float(np.mean(rate_buf))
        max_rate = float(np.max(np.mean(rate_buf, axis=1)))
        std_rate = float(np.std(np.mean(rate_buf, axis=1)))

        burn = min(self.burn_in_trs, max(0, bold.shape[1] // 4))
        bold_post = bold[:, burn:]
        bold_std_mean = float(np.mean(np.std(bold_post, axis=1)))

        if max_rate > 60.0 or mean_rate > 35.0:
            return SimulationResult(
                observables={
                    "mean_rate_hz": mean_rate,
                    "max_rate_hz": max_rate,
                    "bold_std_mean": bold_std_mean,
                },
                feasible=False,
                failure_reason="rate_runaway",
                wall_time_s=wall_s,
                sim_units=float(duration_scale),
                seed=seed,
            )

        if mean_rate < 0.05 or bold_std_mean < 1e-6:
            return SimulationResult(
                observables={
                    "mean_rate_hz": mean_rate,
                    "max_rate_hz": max_rate,
                    "bold_std_mean": bold_std_mean,
                },
                feasible=False,
                failure_reason="silent_or_flat_bold",
                wall_time_s=wall_s,
                sim_units=float(duration_scale),
                seed=seed,
            )

        fc, fcd_triu = compute_fc_and_fcd(bold, burn_in_trs=burn)
        iu = np.triu_indices(self.n_rois, k=1)
        fc_triu = fc[iu]
        if not np.all(np.isfinite(fc_triu)):
            return SimulationResult(
                observables={"mean_rate_hz": mean_rate, "max_rate_hz": max_rate},
                feasible=False,
                failure_reason="degenerate_fc",
                wall_time_s=wall_s,
                sim_units=float(duration_scale),
                seed=seed,
            )

        obs: dict[str, float] = {
            "mean_rate_hz": mean_rate,
            "max_rate_hz": max_rate,
            "std_rate_hz": std_rate,
            "bold_std_mean": bold_std_mean,
            "fc_mean": float(np.mean(fc_triu)),
            "fc_std": float(np.std(fc_triu)),
            "fcd_mean": float(np.mean(fcd_triu)) if fcd_triu.size > 0 else float("nan"),
            "fcd_std": float(np.std(fcd_triu)) if fcd_triu.size > 0 else float("nan"),
        }

        if self.target_fc is not None:
            t_triu = self.target_fc[iu]
            obs["fc_corr"] = float(np.corrcoef(fc_triu, t_triu)[0, 1])
            obs["fc_rmse"] = float(np.sqrt(np.mean((fc_triu - t_triu) ** 2)))
        if self.target_fcd_triu is not None and fcd_triu.size > 0 and self.target_fcd_triu.size > 0:
            obs["fcd_ks"] = float(ks_2samp(fcd_triu, self.target_fcd_triu).statistic)

        feasible = all(
            math.isfinite(obs[k])
            for k in ("mean_rate_hz", "max_rate_hz", "fc_mean", "fc_std")
        )
        return SimulationResult(
            observables=obs,
            feasible=feasible,
            failure_reason=None if feasible else "non_finite_stats",
            wall_time_s=wall_s,
            sim_units=float(duration_scale),
            seed=seed,
            meta={"n_rois": self.n_rois, "n_bold_trs": int(bold.shape[1])},
        )
