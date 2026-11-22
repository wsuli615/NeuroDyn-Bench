"""Small E-I Spiking Network simulator (8-10D, Brunel / SNOPS CBN & SBN paradigm).

Implements a Numba-accelerated current-based LIF network with Dale's law,
exponential excitatory/inhibitory synapses, conduction delays, refractory
period, and exact SNOPS-style 6 population activity statistics:
  1. fr_e   - mean excitatory firing rate (Hz)
  2. ff     - mean single-neuron Fano factor
  3. rsc    - mean pairwise spike-count correlation
  4. pct_sh - percent shared variance (%sh) via Factor Analysis
  5. dsh    - shared dimensionality (95% shared variance)
  6. es     - shared eigenspectrum (top-1 share + full normalized spectrum)
"""

from __future__ import annotations

import math
import time
from typing import Literal

import numba as nb
import numpy as np
from sklearn.decomposition import FactorAnalysis
from sklearn.model_selection import KFold

from neurodyn_bench.models.base import BaseSimulator, ParameterSpec, SimulationResult


@nb.njit(cache=True)
def _build_cbn_csr(
    n_e: int,
    n_i: int,
    p_conn: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build Erdős-Rényi (CBN) outgoing CSR adjacency list deterministically."""
    np.random.seed(seed)
    n_total = n_e + n_i
    # Estimate capacity with safety margin
    est = int(n_total * n_total * p_conn * 1.25) + 1024
    indices = np.empty(est, dtype=np.int32)
    indptr = np.empty(n_total + 1, dtype=np.int32)
    cursor = 0
    for pre in range(n_total):
        indptr[pre] = cursor
        for post in range(n_total):
            if pre == post:
                continue
            if np.random.random() < p_conn:
                if cursor >= indices.shape[0]:
                    new_idx = np.empty(indices.shape[0] * 2, dtype=np.int32)
                    new_idx[:cursor] = indices[:cursor]
                    indices = new_idx
                indices[cursor] = post
                cursor += 1
    indptr[n_total] = cursor
    return indptr, indices[:cursor].copy()


@nb.njit(cache=True)
def _build_sbn_csr(
    n_e: int,
    n_i: int,
    p_conn: float,
    sigma_e: float,
    sigma_i: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build 2D periodic Spatial Balanced Network (SBN) outgoing CSR adjacency list."""
    np.random.seed(seed)
    n_total = n_e + n_i
    pos_x = np.random.random(n_total)
    pos_y = np.random.random(n_total)

    est = int(n_total * n_total * p_conn * 1.4) + 1024
    indices = np.empty(est, dtype=np.int32)
    indptr = np.empty(n_total + 1, dtype=np.int32)
    cursor = 0

    for pre in range(n_total):
        indptr[pre] = cursor
        sig = sigma_e if pre < n_e else sigma_i
        inv_2sig2 = 0.5 / (sig * sig)
        # Compute Gaussian kernel weights to match target mean degree p_conn * (N - 1)
        weights = np.empty(n_total, dtype=np.float64)
        w_sum = 0.0
        for post in range(n_total):
            if pre == post:
                weights[post] = 0.0
                continue
            dx = abs(pos_x[pre] - pos_x[post])
            if dx > 0.5:
                dx = 1.0 - dx
            dy = abs(pos_y[pre] - pos_y[post])
            if dy > 0.5:
                dy = 1.0 - dy
            w = math.exp(-(dx * dx + dy * dy) * inv_2sig2)
            weights[post] = w
            w_sum += w

        scale = (p_conn * (n_total - 1)) / max(w_sum, 1e-12)
        for post in range(n_total):
            if pre == post:
                continue
            p = min(weights[post] * scale, 1.0)
            if np.random.random() < p:
                if cursor >= indices.shape[0]:
                    new_idx = np.empty(indices.shape[0] * 2, dtype=np.int32)
                    new_idx[:cursor] = indices[:cursor]
                    indices = new_idx
                indices[cursor] = post
                cursor += 1
    indptr[n_total] = cursor
    return indptr, indices[:cursor].copy()


@nb.njit(cache=True)
def _make_poisson_cdf(lam: float) -> np.ndarray:
    """Exact Poisson(lam) CDF lookup table up to 1 - 1e-12."""
    if lam <= 0.0:
        return np.ones(1, dtype=np.float64)
    k_max = max(16, int(lam + 8.0 * math.sqrt(lam) + 12.0))
    cdf = np.empty(k_max + 1, dtype=np.float64)
    p = math.exp(-lam)
    s = p
    cdf[0] = s
    for k in range(1, k_max):
        p *= lam / k
        s += p
        cdf[k] = s
        if s >= 1.0 - 1e-12:
            cdf[k] = 1.0
            return cdf[: k + 1].copy()
    cdf[k_max] = 1.0
    return cdf


@nb.njit(cache=True)
def _simulate_lif_ei(
    indptr: np.ndarray,
    indices: np.ndarray,
    n_e: int,
    n_i: int,
    j_ee: float,
    j_ie: float,
    j_ei: float,
    j_ii: float,
    tau_e: float,
    tau_i: float,
    tau_m_e: float,
    tau_m_i: float,
    nu_ext_e: float,
    nu_ext_i: float,
    j_ext: float,
    dt: float,
    n_steps: int,
    burn_in_steps: int,
    delay_steps: int,
    ref_steps: int,
    sample_idx: np.ndarray,
    bin_steps: int,
    pop_bin_steps: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int, bool]:
    """Run current-based LIF E-I spiking simulation with delayed exponential synapses.

    Returns:
        counts_sample: (n_sample, n_bins) spike count matrix for SNOPS statistics
        pop_e_rate_bins: (n_pop_bins,) population E spike counts per fine bin (for PSD)
        isi_sum, isi_sq_sum, isi_n: per-sampled-neuron ISI moments (for ISI CV)
        total_e_spikes, total_i_spikes: total post-burn-in spike counts
        aborted_runaway: True if firing exploded above safety ceiling
    """
    np.random.seed(seed)
    n_total = n_e + n_i
    e_l = -70.0
    v_th = -50.0
    v_reset = -65.0

    v = e_l + (v_th - e_l) * np.random.random(n_total)
    i_exc = np.zeros(n_total, dtype=np.float64)
    i_inh = np.zeros(n_total, dtype=np.float64)
    ref_cnt = np.zeros(n_total, dtype=np.int32)

    ring_len = delay_steps + 1
    ring_exc = np.zeros((ring_len, n_total), dtype=np.float64)
    ring_inh = np.zeros((ring_len, n_total), dtype=np.float64)

    decay_e = math.exp(-dt / tau_e)
    decay_i = math.exp(-dt / tau_i)
    dt_over_tm_e = dt / tau_m_e
    dt_over_tm_i = dt / tau_m_i
    lam_ext_e = nu_ext_e * dt  # nu_ext in kHz (1/ms), dt in ms -> expected spikes/step
    lam_ext_i = nu_ext_i * dt
    cdf_e = _make_poisson_cdf(lam_ext_e)
    cdf_i = _make_poisson_cdf(lam_ext_i)
    cdf_e0 = cdf_e[0]
    cdf_i0 = cdf_i[0]
    n_cdf_e = cdf_e.shape[0]
    n_cdf_i = cdf_i.shape[0]

    # Map neuron id -> sample slot (-1 if not sampled)
    n_sample = sample_idx.shape[0]
    sample_slot = np.full(n_total, -1, dtype=np.int32)
    for s_i in range(n_sample):
        sample_slot[sample_idx[s_i]] = s_i

    eval_steps = n_steps - burn_in_steps
    n_bins = max(1, eval_steps // bin_steps)
    n_pop_bins = max(1, eval_steps // pop_bin_steps)

    counts_sample = np.zeros((n_sample, n_bins), dtype=np.int32)
    pop_e_rate_bins = np.zeros(n_pop_bins, dtype=np.int32)
    last_spike_step = np.full(n_sample, -1, dtype=np.int32)
    isi_sum = np.zeros(n_sample, dtype=np.float64)
    isi_sq_sum = np.zeros(n_sample, dtype=np.float64)
    isi_n = np.zeros(n_sample, dtype=np.int32)

    total_e_spikes = 0
    total_i_spikes = 0
    aborted_runaway = False
    window_spikes = 0
    check_interval = max(1, int(round(100.0 / dt)))  # noqa: RUF046  # Numba round() returns float

    for step in range(n_steps):
        cur_slot = step % ring_len
        deliv_slot = (step + delay_steps) % ring_len

        # 1. Decay synapses & apply delayed recurrent arrivals + external Poisson input
        for i in range(n_total):
            u = np.random.random()
            if i < n_e:
                if u <= cdf_e0:
                    k_ext = 0
                else:
                    k_ext = 1
                    while k_ext < n_cdf_e - 1 and u > cdf_e[k_ext]:
                        k_ext += 1
            else:
                if u <= cdf_i0:
                    k_ext = 0
                else:
                    k_ext = 1
                    while k_ext < n_cdf_i - 1 and u > cdf_i[k_ext]:
                        k_ext += 1
            ie_val = i_exc[i] * decay_e + ring_exc[cur_slot, i] + k_ext * j_ext
            ii_val = i_inh[i] * decay_i + ring_inh[cur_slot, i]
            i_exc[i] = ie_val
            i_inh[i] = ii_val
            ring_exc[cur_slot, i] = 0.0
            ring_inh[cur_slot, i] = 0.0

            if ref_cnt[i] > 0:
                ref_cnt[i] -= 1
                v[i] = v_reset
            else:
                dt_tm = dt_over_tm_e if i < n_e else dt_over_tm_i
                v_new = v[i] + dt_tm * (-(v[i] - e_l) + ie_val - ii_val)
                if not math.isfinite(v_new):
                    aborted_runaway = True
                    return (
                        counts_sample,
                        pop_e_rate_bins,
                        isi_sum,
                        isi_sq_sum,
                        isi_n,
                        total_e_spikes,
                        total_i_spikes,
                        aborted_runaway,
                    )
                v[i] = v_new

        # 2. Spike detection, reset, and schedule delayed postsynaptic jumps
        post_burn = step >= burn_in_steps
        b_idx = (step - burn_in_steps) // bin_steps if post_burn else -1
        p_idx = (step - burn_in_steps) // pop_bin_steps if post_burn else -1

        for i in range(n_total):
            if v[i] >= v_th:
                v[i] = v_reset
                ref_cnt[i] = ref_steps
                window_spikes += 1

                if post_burn:
                    if i < n_e:
                        total_e_spikes += 1
                        if 0 <= p_idx < n_pop_bins:
                            pop_e_rate_bins[p_idx] += 1
                    else:
                        total_i_spikes += 1

                    slot = sample_slot[i]
                    if slot >= 0:
                        if 0 <= b_idx < n_bins:
                            counts_sample[slot, b_idx] += 1
                        prev = last_spike_step[slot]
                        if prev >= 0:
                            isi_ms = (step - prev) * dt
                            isi_sum[slot] += isi_ms
                            isi_sq_sum[slot] += isi_ms * isi_ms
                            isi_n[slot] += 1
                        last_spike_step[slot] = step

                start = indptr[i]
                end = indptr[i + 1]
                if i < n_e:
                    for edge in range(start, end):
                        post = indices[edge]
                        if post < n_e:
                            ring_exc[deliv_slot, post] += j_ee
                        else:
                            ring_exc[deliv_slot, post] += j_ie
                else:
                    for edge in range(start, end):
                        post = indices[edge]
                        if post < n_e:
                            ring_inh[deliv_slot, post] += j_ei
                        else:
                            ring_inh[deliv_slot, post] += j_ii

        # Early runaway abort (> 300 Hz network average over 100 ms)
        if (step + 1) % check_interval == 0:
            mean_hz_win = window_spikes / (n_total * 0.1)
            window_spikes = 0
            if mean_hz_win > 300.0:
                aborted_runaway = True
                break

    return (
        counts_sample,
        pop_e_rate_bins,
        isi_sum,
        isi_sq_sum,
        isi_n,
        total_e_spikes,
        total_i_spikes,
        aborted_runaway,
    )


def compute_snops_statistics(
    counts_sample: np.ndarray,
    bin_ms: float,
    max_fa_dim: int = 8,
    fa_cv_folds: int = 4,
    fa_seed: int = 0,
) -> tuple[dict[str, float], list[float]]:
    """Compute the 6 SNOPS population activity statistics from spike counts.

    Args:
        counts_sample: Array of shape (N_s, T) with spike counts per window.
        bin_ms: Width of each non-overlapping spike count window in ms.
        max_fa_dim: Maximum latent dimensionality searched for Factor Analysis.
        fa_cv_folds: Number of CV folds for selecting FA latent dimension.
        fa_seed: Seed for KFold splitting.

    Returns:
        stats: dict containing fr_e, ff, rsc, pct_sh, dsh, es_top1
        eigenspectrum: list of normalized eigenvalues of L L^T
    """
    n_s, n_t = counts_sample.shape
    bin_s = bin_ms / 1000.0
    x = counts_sample.astype(np.float64)

    means = x.mean(axis=1)
    variances = x.var(axis=1, ddof=1) if n_t > 1 else np.zeros(n_s)

    fr_e = float(means.mean() / bin_s)

    # Filter neurons with non-zero variance for Fano, rsc, and FA
    active = (means > 0.0) & (variances > 1e-12)
    n_active = int(active.sum())

    if n_active < 5 or n_t < 10:
        return (
            {
                "fr_e": fr_e,
                "ff": float("nan"),
                "rsc": float("nan"),
                "pct_sh": float("nan"),
                "dsh": float("nan"),
                "es_top1": float("nan"),
            },
            [],
        )

    ff = float(np.mean(variances[active] / means[active]))

    x_act = x[active]  # (n_active, T)
    corr = np.corrcoef(x_act)
    iu = np.triu_indices(n_active, k=1)
    rsc = float(np.nanmean(corr[iu]))

    # Factor Analysis on (T, n_active) centered spike counts
    x_obs = x_act.T  # (T, n_active)
    x_centered = x_obs - x_obs.mean(axis=0, keepdims=True)
    max_q = min(max_fa_dim, n_active - 1, max(1, n_t // 6))

    best_q = 1
    best_ll = -np.inf
    if max_q >= 2 and n_t >= fa_cv_folds * 3:
        kf = KFold(n_splits=fa_cv_folds, shuffle=True, random_state=fa_seed)
        for q in range(1, max_q + 1):
            fold_scores = []
            for tr_idx, va_idx in kf.split(x_centered):
                fa_cv = FactorAnalysis(n_components=q, max_iter=300, random_state=fa_seed)
                try:
                    fa_cv.fit(x_centered[tr_idx])
                    fold_scores.append(float(fa_cv.score(x_centered[va_idx])))
                except (ValueError, np.linalg.LinAlgError, RuntimeError):
                    fold_scores.append(-np.inf)
            mean_ll = float(np.mean(fold_scores))
            if mean_ll > best_ll:
                best_ll = mean_ll
                best_q = q

    fa = FactorAnalysis(n_components=best_q, max_iter=500, random_state=fa_seed)
    fa.fit(x_centered)
    # components_: (q, n_active) -> L: (n_active, q)
    loadings = fa.components_.T
    shared_cov = loadings @ loadings.T
    shared_var_diag = np.diag(shared_cov)
    total_var_diag = shared_var_diag + fa.noise_variance_
    pct_sh = float(100.0 * np.mean(shared_var_diag / np.maximum(total_var_diag, 1e-12)))

    eigvals = np.linalg.eigvalsh(shared_cov)[::-1]
    eigvals = np.maximum(eigvals[:best_q], 0.0)
    eig_sum = float(eigvals.sum())
    if eig_sum > 1e-12:
        norm_es = eigvals / eig_sum
        cum = np.cumsum(norm_es)
        dsh = float(int(np.searchsorted(cum, 0.95) + 1))
        es_top1 = float(norm_es[0])
        es_list = [float(v) for v in norm_es]
    else:
        dsh = 0.0
        es_top1 = 0.0
        es_list = []

    return (
        {
            "fr_e": fr_e,
            "ff": ff,
            "rsc": rsc,
            "pct_sh": pct_sh,
            "dsh": dsh,
            "es_top1": es_top1,
        },
        es_list,
    )


class SmallEINetworkSimulator(BaseSimulator):
    """8-10D balanced LIF spiking network supporting CBN and SBN topologies."""

    def __init__(
        self,
        n_e: int = 800,
        n_i: int = 200,
        p_conn: float = 0.1,
        topology: Literal["cbn", "sbn"] = "cbn",
        sigma_e: float = 0.10,
        sigma_i: float = 0.25,
        duration_ms: float = 10000.0,
        burn_in_ms: float = 500.0,
        dt_ms: float = 0.1,
        delay_ms: float = 1.5,
        ref_ms: float = 2.0,
        j_ext: float = 1.0,
        n_sample: int = 50,
        bin_ms: float = 100.0,
        conn_seed: int = 42,
    ) -> None:
        self.name = f"small_ei_{topology}"
        self.n_e = n_e
        self.n_i = n_i
        self.p_conn = p_conn
        self.topology = topology
        self.sigma_e = sigma_e
        self.sigma_i = sigma_i
        self.duration_ms = duration_ms
        self.burn_in_ms = burn_in_ms
        self.dt_ms = dt_ms
        self.delay_ms = delay_ms
        self.ref_ms = ref_ms
        self.j_ext = j_ext
        self.n_sample = min(n_sample, n_e)
        self.bin_ms = bin_ms
        self.conn_seed = conn_seed

        if topology == "cbn":
            self._indptr, self._indices = _build_cbn_csr(n_e, n_i, p_conn, conn_seed)
        elif topology == "sbn":
            self._indptr, self._indices = _build_sbn_csr(
                n_e, n_i, p_conn, sigma_e, sigma_i, conn_seed
            )
        else:
            raise ValueError(f"Unknown topology: {topology}")

        rng = np.random.default_rng(conn_seed + 1000)
        self._sample_idx = np.sort(rng.choice(n_e, size=self.n_sample, replace=False)).astype(
            np.int32
        )

    def parameter_specs(self) -> list[ParameterSpec]:
        return [
            ParameterSpec(
                "J_ee", "mV", 0.3, 2.5, 0.05, 8.0, "linear", 1.2, "E->E synaptic weight"
            ),
            ParameterSpec(
                "J_ie", "mV", 0.3, 3.0, 0.05, 8.0, "linear", 1.5, "E->I synaptic weight"
            ),
            ParameterSpec(
                "g_ei", "ratio", 2.0, 8.0, 0.5, 20.0, "linear", 5.0, "Relative I->E inhibition (J_ei / J_ee)"
            ),
            ParameterSpec(
                "g_ii", "ratio", 2.0, 8.0, 0.5, 20.0, "linear", 4.5, "Relative I->I inhibition (J_ii / J_ie)"
            ),
            ParameterSpec(
                "tau_e", "ms", 2.0, 12.0, 0.5, 50.0, "log", 5.0, "Excitatory AMPA synaptic decay constant"
            ),
            ParameterSpec(
                "tau_i", "ms", 4.0, 20.0, 1.0, 80.0, "log", 8.0, "Inhibitory GABA_A synaptic decay constant"
            ),
            ParameterSpec(
                "tau_m_e", "ms", 10.0, 30.0, 5.0, 60.0, "linear", 20.0, "Excitatory membrane time constant"
            ),
            ParameterSpec(
                "tau_m_i", "ms", 6.0, 20.0, 3.0, 50.0, "linear", 10.0, "Inhibitory membrane time constant"
            ),
            ParameterSpec(
                "nu_ext_e", "kHz", 2.5, 8.0, 0.5, 25.0, "linear", 5.2, "External Poisson rate to E neurons"
            ),
            ParameterSpec(
                "nu_ext_i", "kHz", 2.0, 8.0, 0.5, 25.0, "linear", 4.5, "External Poisson rate to I neurons"
            ),
        ]

    def default_params(self) -> dict[str, float]:
        return {spec.name: float(spec.nominal) for spec in self.parameter_specs() if spec.nominal is not None}

    def simulate(
        self,
        params: dict[str, float],
        seed: int = 0,
        duration_scale: float = 1.0,
    ) -> SimulationResult:
        t0 = time.perf_counter()
        p = self.default_params()
        p.update({k: float(v) for k, v in params.items()})

        j_ee = p["J_ee"]
        j_ie = p["J_ie"]
        j_ei = p["g_ei"] * j_ee
        j_ii = p["g_ii"] * j_ie
        tau_e = p["tau_e"]
        tau_i = p["tau_i"]
        tau_m_e = p["tau_m_e"]
        tau_m_i = p["tau_m_i"]
        nu_ext_e = p["nu_ext_e"]
        nu_ext_i = p["nu_ext_i"]

        # Basic physiological hard-bound check
        if min(j_ee, j_ie, j_ei, j_ii, tau_e, tau_i, tau_m_e, tau_m_i, nu_ext_e, nu_ext_i) <= 0.0:
            return SimulationResult(
                observables={},
                feasible=False,
                failure_reason="non_positive_param",
                wall_time_s=time.perf_counter() - t0,
                sim_units=float(duration_scale),
                seed=seed,
            )

        eval_ms = max(500.0, self.duration_ms * float(duration_scale))
        total_ms = self.burn_in_ms + eval_ms
        n_steps = round(total_ms / self.dt_ms)
        burn_in_steps = round(self.burn_in_ms / self.dt_ms)
        delay_steps = max(1, round(self.delay_ms / self.dt_ms))
        ref_steps = max(1, round(self.ref_ms / self.dt_ms))
        bin_steps = max(1, round(self.bin_ms / self.dt_ms))
        pop_bin_ms = 2.0  # 500 Hz sampling for population PSD up to 250 Hz
        pop_bin_steps = max(1, round(pop_bin_ms / self.dt_ms))

        (
            counts_sample,
            pop_e_rate_bins,
            isi_sum,
            isi_sq_sum,
            isi_n,
            total_e_spikes,
            total_i_spikes,
            aborted_runaway,
        ) = _simulate_lif_ei(
            self._indptr,
            self._indices,
            self.n_e,
            self.n_i,
            j_ee,
            j_ie,
            j_ei,
            j_ii,
            tau_e,
            tau_i,
            tau_m_e,
            tau_m_i,
            nu_ext_e,
            nu_ext_i,
            self.j_ext,
            self.dt_ms,
            n_steps,
            burn_in_steps,
            delay_steps,
            ref_steps,
            self._sample_idx,
            bin_steps,
            pop_bin_steps,
            int(seed),
        )

        eval_s = eval_ms / 1000.0
        fr_e_pop = float(total_e_spikes / (self.n_e * eval_s))
        fr_i_pop = float(total_i_spikes / (self.n_i * eval_s))

        if aborted_runaway or fr_e_pop > 150.0 or fr_i_pop > 250.0:
            return SimulationResult(
                observables={"fr_e": fr_e_pop, "fr_i": fr_i_pop},
                feasible=False,
                failure_reason="runaway",
                wall_time_s=time.perf_counter() - t0,
                sim_units=float(duration_scale),
                seed=seed,
                meta={"topology": self.topology, "eval_ms": eval_ms},
            )

        if fr_e_pop < 0.5:
            return SimulationResult(
                observables={"fr_e": fr_e_pop, "fr_i": fr_i_pop},
                feasible=False,
                failure_reason="silent",
                wall_time_s=time.perf_counter() - t0,
                sim_units=float(duration_scale),
                seed=seed,
                meta={"topology": self.topology, "eval_ms": eval_ms},
            )

        # Compute ISI CV across sampled E neurons with >= 3 ISIs
        valid_isi = isi_n >= 3
        if np.any(valid_isi):
            mean_isi = isi_sum[valid_isi] / isi_n[valid_isi]
            var_isi = np.maximum(
                isi_sq_sum[valid_isi] / isi_n[valid_isi] - mean_isi * mean_isi, 0.0
            )
            isi_cv = float(np.mean(np.sqrt(var_isi) / np.maximum(mean_isi, 1e-12)))
        else:
            isi_cv = float("nan")

        # Population rate PSD peak frequency in 5-120 Hz band
        pop_sig = pop_e_rate_bins.astype(np.float64)
        pop_sig = pop_sig - pop_sig.mean()
        freqs = np.fft.rfftfreq(pop_sig.size, d=pop_bin_ms / 1000.0)
        psd = np.abs(np.fft.rfft(pop_sig)) ** 2
        band = (freqs >= 5.0) & (freqs <= 120.0)
        if np.any(band) and psd[band].max() > 0.0:
            pop_psd_peak_hz = float(freqs[band][int(np.argmax(psd[band]))])
        else:
            pop_psd_peak_hz = float("nan")

        snops_stats, es_list = compute_snops_statistics(
            counts_sample, bin_ms=self.bin_ms, fa_seed=int(seed)
        )

        obs = {
            "fr_e": snops_stats["fr_e"],
            "fr_i": fr_i_pop,
            "isi_cv": isi_cv,
            "ff": snops_stats["ff"],
            "rsc": snops_stats["rsc"],
            "pct_sh": snops_stats["pct_sh"],
            "dsh": snops_stats["dsh"],
            "es_top1": snops_stats["es_top1"],
            "pop_psd_peak_hz": pop_psd_peak_hz,
        }

        feasible = all(math.isfinite(obs[k]) for k in ("fr_e", "ff", "rsc", "pct_sh", "dsh"))
        failure_reason = None if feasible else "insufficient_active_neurons"

        return SimulationResult(
            observables=obs,
            feasible=feasible,
            failure_reason=failure_reason,
            wall_time_s=time.perf_counter() - t0,
            sim_units=float(duration_scale),
            seed=seed,
            vector_observables={"es": es_list},
            meta={
                "topology": self.topology,
                "eval_ms": eval_ms,
                "n_e": self.n_e,
                "n_i": self.n_i,
                "n_edges": int(self._indices.size),
            },
        )
