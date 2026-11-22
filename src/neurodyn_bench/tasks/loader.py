"""PEtab-style task bundle serialization, isolated loading, and 3-tier disclosure rendering."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from neurodyn_bench.models.base import BaseSimulator
from neurodyn_bench.models.fastdmf_env import FastDMFSimulator
from neurodyn_bench.models.small_ei import SmallEINetworkSimulator
from neurodyn_bench.tasks.schema import (
    BudgetSpec,
    ConditionRow,
    DisclosureTier,
    ObservableRow,
    ParameterRow,
    PrivateTaskBundle,
    PublicTaskBundle,
    SimulatorConfig,
    TaskMetadata,
)


def _write_tsv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def save_task_bundle(
    task_dir: str | Path,
    public_bundle: PublicTaskBundle,
    private_bundle: PrivateTaskBundle,
    target_fc: np.ndarray | None = None,
    target_fcd_triu: np.ndarray | None = None,
) -> Path:
    """Write a complete task bundle with physically separated public/ and private/ directories."""
    root = Path(task_dir)
    pub_dir = root / "public"
    priv_dir = root / "private"
    pub_dir.mkdir(parents=True, exist_ok=True)
    priv_dir.mkdir(parents=True, exist_ok=True)

    sim_cfg = public_bundle.simulator_config.model_copy(deep=True)
    if target_fc is not None:
        np.save(pub_dir / "target_fc.npy", target_fc)
        sim_cfg.target_fc_file = "target_fc.npy"
    if target_fcd_triu is not None:
        np.save(pub_dir / "target_fcd_triu.npy", target_fcd_triu)
        sim_cfg.target_fcd_file = "target_fcd_triu.npy"

    (pub_dir / "task.yaml").write_text(
        yaml.safe_dump(public_bundle.metadata.model_dump(), sort_keys=False),
        encoding="utf-8",
    )
    (pub_dir / "budget.yaml").write_text(
        yaml.safe_dump(public_bundle.budget.model_dump(), sort_keys=False),
        encoding="utf-8",
    )
    (pub_dir / "simulator_config.yaml").write_text(
        yaml.safe_dump(sim_cfg.model_dump(), sort_keys=False),
        encoding="utf-8",
    )

    _write_tsv(
        pub_dir / "parameters.tsv",
        [
            "name",
            "unit",
            "default_lb",
            "default_ub",
            "phys_hard_lb",
            "phys_hard_ub",
            "scale",
            "nominal",
            "active",
            "description",
        ],
        [p.model_dump() for p in public_bundle.parameters],
    )
    _write_tsv(
        pub_dir / "observables.tsv",
        ["obs_id", "condition_id", "target_mean", "target_sd", "weight", "tolerance_tau"],
        [o.model_dump() for o in public_bundle.observables],
    )
    _write_tsv(
        pub_dir / "conditions.tsv",
        ["condition_id", "param_overrides_json", "description"],
        [
            {
                "condition_id": c.condition_id,
                "param_overrides_json": json.dumps(c.param_overrides, sort_keys=True),
                "description": c.description,
            }
            for c in public_bundle.conditions
        ],
    )

    # Private bundle (evaluator-only)
    gt_payload = {
        "task_id": private_bundle.task_id,
        "task_type": private_bundle.task_type,
        "ground_truth_params": private_bundle.ground_truth_params,
        "gt_simulator_config": private_bundle.gt_simulator_config.model_dump(),
        "trap_metadata": private_bundle.trap_metadata,
    }
    (priv_dir / "ground_truth.yaml").write_text(
        yaml.safe_dump(gt_payload, sort_keys=False),
        encoding="utf-8",
    )
    _write_tsv(
        priv_dir / "heldout_conditions.tsv",
        ["condition_id", "param_overrides_json", "description"],
        [
            {
                "condition_id": c.condition_id,
                "param_overrides_json": json.dumps(c.param_overrides, sort_keys=True),
                "description": c.description,
            }
            for c in private_bundle.heldout_conditions
        ],
    )
    _write_tsv(
        priv_dir / "heldout_observables.tsv",
        ["obs_id", "condition_id", "target_mean", "target_sd", "weight", "tolerance_tau"],
        [o.model_dump() for o in private_bundle.heldout_observables],
    )
    if private_bundle.validation_cert is not None:
        (priv_dir / "validation_cert.yaml").write_text(
            yaml.safe_dump(private_bundle.validation_cert, sort_keys=False),
            encoding="utf-8",
        )
    return root


def load_public_task(
    task_dir: str | Path,
    disclosure: DisclosureTier | None = None,
) -> tuple[PublicTaskBundle, dict[str, str]]:
    """Load ONLY public/ task files and apply 3-tier disclosure rendering.

    Returns:
        (public_bundle, blind_to_real_map) where blind_to_real_map is empty unless
        disclosure == "blinded", in which case it maps anonymized keys ("p0", ...)
        back to real simulator parameter names for server-side gateway translation.
    """
    pub_dir = Path(task_dir) / "public"
    if not pub_dir.exists():
        # Allow passing public/ directory directly (e.g. inside isolated Track B sandbox)
        if (Path(task_dir) / "task.yaml").exists():
            pub_dir = Path(task_dir)
        else:
            raise FileNotFoundError(f"Public task directory not found under {task_dir}")

    meta = TaskMetadata(**yaml.safe_load((pub_dir / "task.yaml").read_text(encoding="utf-8")))
    budget = BudgetSpec(**yaml.safe_load((pub_dir / "budget.yaml").read_text(encoding="utf-8")))
    sim_cfg = SimulatorConfig(
        **yaml.safe_load((pub_dir / "simulator_config.yaml").read_text(encoding="utf-8"))
    )

    params: list[ParameterRow] = []
    for row in _read_tsv(pub_dir / "parameters.tsv"):
        params.append(
            ParameterRow(
                name=row["name"],
                unit=row["unit"],
                default_lb=float(row["default_lb"]),
                default_ub=float(row["default_ub"]),
                phys_hard_lb=float(row["phys_hard_lb"]),
                phys_hard_ub=float(row["phys_hard_ub"]),
                scale=row["scale"],
                nominal=float(row["nominal"]),
                active=row["active"].strip().lower() in ("true", "1", "yes"),
                description=row["description"],
            )
        )

    observables: list[ObservableRow] = []
    for row in _read_tsv(pub_dir / "observables.tsv"):
        observables.append(
            ObservableRow(
                obs_id=row["obs_id"],
                condition_id=row["condition_id"],
                target_mean=float(row["target_mean"]),
                target_sd=float(row["target_sd"]),
                weight=float(row["weight"]),
                tolerance_tau=float(row["tolerance_tau"]),
            )
        )

    conditions: list[ConditionRow] = []
    for row in _read_tsv(pub_dir / "conditions.tsv"):
        conditions.append(
            ConditionRow(
                condition_id=row["condition_id"],
                param_overrides=json.loads(row["param_overrides_json"])
                if row["param_overrides_json"]
                else {},
                description=row["description"],
            )
        )

    eff_disclosure: DisclosureTier = disclosure or meta.disclosure
    blind_to_real: dict[str, str] = {}

    if eff_disclosure == "names_only":
        meta = meta.model_copy(
            update={
                "disclosure": "names_only",
                "prompt": (
                    f"Calibrate fixed-structure simulator ({meta.model_type}) to match target "
                    "observables within budget, or abstain with quantitative evidence if underdetermined "
                    "(3a) or structurally unreachable (3b)."
                ),
            }
        )
        params = [p.model_copy(update={"description": ""}) for p in params]
    elif eff_disclosure == "blinded":
        meta = meta.model_copy(
            update={
                "disclosure": "blinded",
                "title": f"{meta.task_id} (Blinded)",
                "prompt": (
                    "Calibrate an anonymized black-box dynamical simulator to match target summary "
                    "statistics within budget, respecting hard bounds, or abstain with quantitative "
                    "evidence if parameters are non-identifiable (3a) or the target is unreachable (3b)."
                ),
            }
        )
        new_params: list[ParameterRow] = []
        for idx, p in enumerate(params):
            alias = f"p{idx}"
            blind_to_real[alias] = p.name
            new_params.append(
                p.model_copy(update={"name": alias, "unit": "", "description": ""})
            )
        params = new_params
        # Anonymize condition descriptions too
        conditions = [c.model_copy(update={"description": ""}) for c in conditions]

    bundle = PublicTaskBundle(
        metadata=meta,
        parameters=params,
        observables=observables,
        conditions=conditions,
        budget=budget,
        simulator_config=sim_cfg,
        public_dir=str(pub_dir),
    )
    return bundle, blind_to_real


def load_private_bundle(task_dir: str | Path) -> PrivateTaskBundle:
    """Evaluator-only loader for private/ ground truths, held-out targets, and certification."""
    priv_dir = Path(task_dir) / "private"
    if not priv_dir.exists():
        raise FileNotFoundError(f"Private task directory not found under {task_dir}")

    gt_data = yaml.safe_load((priv_dir / "ground_truth.yaml").read_text(encoding="utf-8"))
    held_conds = [
        ConditionRow(
            condition_id=row["condition_id"],
            param_overrides=json.loads(row["param_overrides_json"])
            if row["param_overrides_json"]
            else {},
            description=row["description"],
        )
        for row in _read_tsv(priv_dir / "heldout_conditions.tsv")
    ]
    held_obs = [
        ObservableRow(
            obs_id=row["obs_id"],
            condition_id=row["condition_id"],
            target_mean=float(row["target_mean"]),
            target_sd=float(row["target_sd"]),
            weight=float(row["weight"]),
            tolerance_tau=float(row["tolerance_tau"]),
        )
        for row in _read_tsv(priv_dir / "heldout_observables.tsv")
    ]
    cert_path = priv_dir / "validation_cert.yaml"
    cert = yaml.safe_load(cert_path.read_text(encoding="utf-8")) if cert_path.exists() else None

    return PrivateTaskBundle(
        task_id=gt_data["task_id"],
        task_type=gt_data["task_type"],
        ground_truth_params={k: float(v) for k, v in gt_data["ground_truth_params"].items()},
        gt_simulator_config=SimulatorConfig(**gt_data["gt_simulator_config"]),
        heldout_conditions=held_conds,
        heldout_observables=held_obs,
        trap_metadata=gt_data.get("trap_metadata", {}),
        validation_cert=cert,
    )


def instantiate_simulator(
    sim_cfg: SimulatorConfig,
    public_dir: str | Path | None = None,
) -> BaseSimulator:
    """Construct the concrete fixed-structure simulator from a SimulatorConfig."""
    kwargs = dict(sim_cfg.kwargs)
    if sim_cfg.model_type == "small_ei":
        return SmallEINetworkSimulator(**kwargs)
    if sim_cfg.model_type == "fastdmf":
        if public_dir is not None:
            pdir = Path(public_dir)
            if sim_cfg.target_fc_file and (pdir / sim_cfg.target_fc_file).exists():
                kwargs["target_fc"] = np.load(pdir / sim_cfg.target_fc_file)
            if sim_cfg.target_fcd_file and (pdir / sim_cfg.target_fcd_file).exists():
                kwargs["target_fcd_triu"] = np.load(pdir / sim_cfg.target_fcd_file)
        return FastDMFSimulator(**kwargs)
    raise ValueError(f"Unsupported simulator model_type: {sim_cfg.model_type}")


def apply_condition(params: dict[str, float], condition: ConditionRow) -> dict[str, float]:
    """Apply public or counterfactual condition overrides to a parameter dictionary.

    Supports both multiplicative modifiers (`<param>_mult: factor`) and direct
    replacements (`<param>: value`).
    """
    out = dict(params)
    for key, val in condition.param_overrides.items():
        if key.endswith("_mult"):
            base_key = key[: -len("_mult")]
            if base_key in out:
                out[base_key] = float(out[base_key]) * float(val)
        else:
            out[key] = float(val)
    return out
