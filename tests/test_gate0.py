"""Gate 0 unit tests: BudgetLedger invariants, Small E-I simulator, and FastDMF simulator."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from neurodyn_bench.models.base import SimulationResult
from neurodyn_bench.models.fastdmf_env import FastDMFSimulator
from neurodyn_bench.models.small_ei import SmallEINetworkSimulator
from neurodyn_bench.optim.budget import BudgetExhaustedError, BudgetLedger


def test_budget_ledger_hard_cap_and_cache() -> None:
    ledger = BudgetLedger(max_units=2.5)
    calls = {"n": 0}

    def fake_sim(scale: float, feasible: bool = True) -> SimulationResult:
        calls["n"] += 1
        return SimulationResult(observables={"x": 1.0}, feasible=feasible, sim_units=scale)

    # First call consumes 1.0 unit
    r1 = ledger.execute("m", {"a": 1.0}, seed=0, duration_scale=1.0, fn=lambda: fake_sim(1.0))
    assert r1.feasible
    assert ledger.used_units == pytest.approx(1.0)
    assert calls["n"] == 1

    # Exact duplicate (params, seed, scale) is a free cache hit
    _ = ledger.execute("m", {"a": 1.0}, seed=0, duration_scale=1.0, fn=lambda: fake_sim(1.0))
    assert ledger.used_units == pytest.approx(1.0)
    assert ledger.n_cache_hits == 1
    assert calls["n"] == 1

    # Failed short probe still bills 0.25 units and increments n_failed
    r_fail = ledger.execute(
        "m", {"a": 2.0}, seed=0, duration_scale=0.25, fn=lambda: fake_sim(0.25, feasible=False)
    )
    assert not r_fail.feasible
    assert ledger.used_units == pytest.approx(1.25)
    assert ledger.n_failed == 1

    # Full call consumes 1.0 unit -> 2.25 used, 0.25 remaining
    ledger.execute("m", {"a": 3.0}, seed=0, duration_scale=1.0, fn=lambda: fake_sim(1.0))
    assert ledger.remaining_units == pytest.approx(0.25)

    # Exceeding budget raises BudgetExhaustedError without running simulator
    with pytest.raises(BudgetExhaustedError):
        ledger.execute("m", {"a": 4.0}, seed=0, duration_scale=1.0, fn=lambda: fake_sim(1.0))
    assert calls["n"] == 3


def test_small_ei_determinism_and_regimes() -> None:
    sim = SmallEINetworkSimulator(topology="cbn", duration_ms=2500.0)
    nom = sim.default_params()

    r1 = sim.simulate(nom, seed=11, duration_scale=1.0)
    r2 = sim.simulate(nom, seed=11, duration_scale=1.0)
    assert r1.feasible and r2.feasible
    for key in ("fr_e", "ff", "rsc", "pct_sh", "dsh", "es_top1", "isi_cv"):
        assert np.isfinite(r1.observables[key])
        assert r1.observables[key] == pytest.approx(r2.observables[key], rel=1e-12)

    # Silent regime detection
    r_silent = sim.simulate({**nom, "nu_ext_e": 1.0, "g_ei": 9.0}, seed=11, duration_scale=0.5)
    assert not r_silent.feasible
    assert r_silent.failure_reason == "silent"

    # Runaway regime detection
    r_runaway = sim.simulate({**nom, "g_ei": 1.0, "J_ee": 2.5}, seed=11, duration_scale=0.5)
    assert not r_runaway.feasible
    assert r_runaway.failure_reason == "runaway"

    # SBN topology runs cleanly and returns finite SNOPS 6 stats
    sbn = SmallEINetworkSimulator(topology="sbn", duration_ms=2500.0)
    r_sbn = sbn.simulate(sbn.default_params(), seed=11, duration_scale=1.0)
    assert r_sbn.feasible
    assert all(np.isfinite(r_sbn.observables[k]) for k in ("fr_e", "ff", "rsc", "pct_sh", "dsh"))


def test_fastdmf_determinism_and_upstream_reference() -> None:
    # 1. Exact bit-level agreement with upstream concog/fastdmf deterministic reference CSVs
    repo_root = Path(__file__).resolve().parent.parent
    rates_ref_path = repo_root / "third_party" / "fastdmf_gitlab" / "test" / "rates_deco.csv"
    bold_ref_path = repo_root / "third_party" / "fastdmf_gitlab" / "test" / "bold_deco.csv"
    if rates_ref_path.exists() and bold_ref_path.exists():
        from neurodyn_bench.models import _fastdmf_ext

        sim_ref = FastDMFSimulator(c_scale=1.0, nb_steps=20000)
        p = sim_ref.default_params()
        p["G"] = 2.0
        p["sigma"] = 0.0
        p["fic_alpha"] = 0.75
        cpp = sim_ref._build_cpp_params(p, seed=0)
        r, b = _fastdmf_ext.run_dmf(cpp, 20000, sim_ref.n_rois, True, True)
        rates_deco = np.loadtxt(rates_ref_path, delimiter=",")
        bold_deco = np.loadtxt(bold_ref_path, delimiter=",")
        assert np.allclose(r[:, :2000], rates_deco.T, atol=1e-12)
        assert np.allclose(b, bold_deco.T, atol=1e-12)

    # 2. Stochastic determinism & runaway detection under Deco 0.2*C working point
    sim = FastDMFSimulator(nb_steps=40000, burn_in_trs=5)
    nom = sim.default_params()
    r1 = sim.simulate(nom, seed=5, duration_scale=1.0)
    r2 = sim.simulate(nom, seed=5, duration_scale=1.0)
    assert r1.feasible and r2.feasible
    assert r1.observables["mean_rate_hz"] == pytest.approx(r2.observables["mean_rate_hz"], rel=1e-12)
    assert r1.observables["fc_mean"] == pytest.approx(r2.observables["fc_mean"], rel=1e-12)

    r_bad = sim.simulate({**nom, "G": 3.5, "w": 2.2, "fic_alpha": 0.12}, seed=5, duration_scale=0.5)
    assert not r_bad.feasible
    assert r_bad.failure_reason == "rate_runaway"
