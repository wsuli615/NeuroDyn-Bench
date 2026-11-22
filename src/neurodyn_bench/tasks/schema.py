"""Pydantic schemas for PEtab-style NeuroDyn-Bench task bundles and submissions."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

DisclosureTier = Literal["full", "names_only", "blinded"]
TaskType = Literal[
    "type1_routine",
    "type2_recoverable",
    "type3a_non_identifiable",
    "type3b_structure_unreachable",
]
AbstentionSubtype = Literal["3a_non_identifiable", "3b_structure_unreachable"]
ScaleType = Literal["linear", "log"]


class ParameterRow(BaseModel):
    """Single row of public/parameters.tsv with explicit default vs. physiological hard bounds."""

    name: str
    unit: str = ""
    default_lb: float
    default_ub: float
    phys_hard_lb: float
    phys_hard_ub: float
    scale: ScaleType = "linear"
    nominal: float
    active: bool = True
    description: str = ""

    @model_validator(mode="after")
    def _validate_bounds(self) -> ParameterRow:
        if self.phys_hard_lb > self.default_lb + 1e-12:
            raise ValueError(f"{self.name}: phys_hard_lb > default_lb")
        if self.default_lb >= self.default_ub:
            raise ValueError(f"{self.name}: default_lb >= default_ub")
        if self.default_ub > self.phys_hard_ub + 1e-12:
            raise ValueError(f"{self.name}: default_ub > phys_hard_ub")
        if self.scale == "log" and self.phys_hard_lb <= 0.0:
            raise ValueError(f"{self.name}: log scale requires phys_hard_lb > 0")
        return self


class ObservableRow(BaseModel):
    """Single row of public/observables.tsv or private/heldout_observables.tsv."""

    obs_id: str
    condition_id: str = "baseline"
    target_mean: float
    target_sd: float
    weight: float = 1.0
    tolerance_tau: float = 2.0


class ConditionRow(BaseModel):
    """Experimental or counterfactual condition row."""

    condition_id: str
    param_overrides: dict[str, float] = Field(default_factory=dict)
    description: str = ""


class BudgetSpec(BaseModel):
    """Hard simulation budget specification (public/budget.yaml)."""

    max_units: float
    short_probe_scale: float = 0.2
    default_n_repeats: int = 1
    sim_cost_seconds_grid: list[float] = Field(
        default_factory=lambda: [0.1, 1.0, 3.0, 10.0, 60.0, 300.0]
    )


class SimulatorConfig(BaseModel):
    """Fixed-structure simulator configuration (public/simulator_config.yaml)."""

    model_type: Literal["small_ei", "fastdmf"]
    kwargs: dict[str, Any] = Field(default_factory=dict)
    target_fc_file: str | None = None
    target_fcd_file: str | None = None


class TaskMetadata(BaseModel):
    """Public task metadata (public/task.yaml)."""

    task_id: str
    title: str
    split: Literal["dev", "test"] = "dev"
    model_type: Literal["small_ei", "fastdmf"]
    syllabus_module: str
    prompt: str
    paired_with: str | None = None
    is_clean_control: bool = False
    disclosure: DisclosureTier = "full"
    authors: list[str] = Field(default_factory=lambda: ["wsuli615"])


class PublicTaskBundle(BaseModel):
    """Complete public task view visible to evaluated controllers and agents."""

    metadata: TaskMetadata
    parameters: list[ParameterRow]
    observables: list[ObservableRow]
    conditions: list[ConditionRow]
    budget: BudgetSpec
    simulator_config: SimulatorConfig
    public_dir: str | None = None

    def active_parameters(self) -> list[ParameterRow]:
        return [p for p in self.parameters if p.active]

    def param_map(self) -> dict[str, ParameterRow]:
        return {p.name: p for p in self.parameters}

    def condition_map(self) -> dict[str, ConditionRow]:
        return {c.condition_id: c for c in self.conditions}

    def nominal_dict(self) -> dict[str, float]:
        return {p.name: p.nominal for p in self.parameters}


class PrivateTaskBundle(BaseModel):
    """Evaluator-only hidden bundle (never exposed to evaluated systems)."""

    task_id: str
    task_type: TaskType
    ground_truth_params: dict[str, float]
    gt_simulator_config: SimulatorConfig
    heldout_conditions: list[ConditionRow] = Field(default_factory=list)
    heldout_observables: list[ObservableRow] = Field(default_factory=list)
    trap_metadata: dict[str, Any] = Field(default_factory=dict)
    validation_cert: dict[str, Any] | None = None


class FinalSubmission(BaseModel):
    """Standardized final output from Track A or Track B controllers."""

    decision: Literal["submit_solution", "abstain_insufficient_evidence"]
    theta_hat: dict[str, float] | None = None
    abstention_subtype: AbstentionSubtype | None = None
    evidence_summary: dict[str, Any] = Field(default_factory=dict)
    notes: str = ""

    @model_validator(mode="after")
    def _validate_decision(self) -> FinalSubmission:
        if self.decision == "submit_solution" and not self.theta_hat:
            raise ValueError("submit_solution requires non-empty theta_hat")
        if self.decision == "abstain_insufficient_evidence" and self.abstention_subtype is None:
            raise ValueError("abstain_insufficient_evidence requires abstention_subtype")
        return self
