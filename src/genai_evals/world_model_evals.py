from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class TrajectoryStep:
    timestamp: float
    state: Tuple[float, ...]
    action: Tuple[float, ...]
    predicted_next_state: Optional[Tuple[float, ...]] = None


@dataclass(frozen=True)
class DynamicsRolloutReport:
    sequence_id: str
    horizon_length: int
    mean_squared_error: float
    max_step_error: float
    cumulative_drift_rate: float
    mean_state_cosine_similarity: float
    passed_tolerance: bool


@dataclass(frozen=True)
class ConservationReport:
    sequence_id: str
    is_physically_consistent: bool
    violations: Tuple[str, ...]
    max_observed_velocity: float


@dataclass(frozen=True)
class PlanCandidateEvaluation:
    plan_id: str
    final_distance_to_goal: float
    cumulative_action_cost: float
    composite_fitness_score: float
    predicted_success: bool


@dataclass(frozen=True)
class PlanningBenchmarkReport:
    goal_state: Tuple[float, ...]
    total_plans_evaluated: int
    optimal_plan_id: str
    optimal_plan_fitness: float
    evaluations: Tuple[PlanCandidateEvaluation, ...]


class WorldModelEvaluator:
    """Evaluator for frontier world models: prediction rollouts, conservation bounds, and planning."""

    @staticmethod
    def _euclidean_dist(a: Sequence[float], b: Sequence[float]) -> float:
        return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

    @staticmethod
    def _cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
        norm_a, norm_b = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(y * y for y in b))
        if norm_a == 0.0 and norm_b == 0.0:
            return 1.0
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        return max(-1.0, min(1.0, dot / (norm_a * norm_b)))

    @classmethod
    def evaluate_rollout_dynamics(
        cls,
        sequence_id: str,
        predicted_states: Sequence[Sequence[float]],
        ground_truth_states: Sequence[Sequence[float]],
        mse_tolerance: float = 0.50,
    ) -> DynamicsRolloutReport:
        if len(predicted_states) != len(ground_truth_states) or not predicted_states:
            raise ValueError("Trajectory states must be non-empty and of identical length")

        horizon = len(predicted_states)
        step_errors = [cls._euclidean_dist(p, gt) ** 2 for p, gt in zip(predicted_states, ground_truth_states)]
        cosine_sims = [cls._cosine_similarity(p, gt) for p, gt in zip(predicted_states, ground_truth_states)]
        mse, max_err = sum(step_errors) / horizon, max(step_errors)
        drift_rate = (step_errors[-1] - step_errors[0]) / max(1, horizon - 1)
        mean_cos = sum(cosine_sims) / horizon

        return DynamicsRolloutReport(
            sequence_id=sequence_id,
            horizon_length=horizon,
            mean_squared_error=round(mse, 4),
            max_step_error=round(max_err, 4),
            cumulative_drift_rate=round(drift_rate, 4),
            mean_state_cosine_similarity=round(mean_cos, 4),
            passed_tolerance=(mse <= mse_tolerance and mean_cos >= 0.85),
        )

    @classmethod
    def verify_conservation_constraints(
        cls,
        sequence_id: str,
        states: Sequence[Sequence[float]],
        timestamps: Sequence[float],
        max_velocity_threshold: float = 5.0,
        bounds: Tuple[float, float] = (-50.0, 50.0),
    ) -> ConservationReport:
        if len(states) != len(timestamps) or len(states) < 2:
            raise ValueError("States and timestamps must have length >= 2")

        violations: List[str] = []
        max_velocity = 0.0

        for i in range(len(states)):
            for val in states[i]:
                if val < bounds[0] or val > bounds[1]:
                    violations.append(f"Step {i}: state value {val} exceeded bounds {bounds}")
            if i > 0:
                dt = timestamps[i] - timestamps[i - 1]
                if dt <= 0:
                    violations.append(f"Non-positive delta t at step {i}: {dt}")
                    continue
                velocity = cls._euclidean_dist(states[i], states[i - 1]) / dt
                max_velocity = max(max_velocity, velocity)
                if velocity > max_velocity_threshold:
                    violations.append(f"Step {i}: velocity {round(velocity, 2)} > threshold {max_velocity_threshold}")

        return ConservationReport(
            sequence_id=sequence_id,
            is_physically_consistent=(len(violations) == 0),
            violations=tuple(violations),
            max_observed_velocity=round(max_velocity, 4),
        )

    @classmethod
    def benchmark_model_based_planning(
        cls,
        initial_state: Sequence[float],
        goal_state: Sequence[float],
        candidate_plans: Sequence[Tuple[str, Sequence[Sequence[float]]]],
        transition_model: Callable[[Sequence[float], Sequence[float]], Sequence[float]],
        success_distance_threshold: float = 1.0,
    ) -> PlanningBenchmarkReport:
        evaluations: List[PlanCandidateEvaluation] = []
        for plan_id, actions in candidate_plans:
            current = list(initial_state)
            action_cost = 0.0
            for action in actions:
                current = list(transition_model(current, action))
                action_cost += sum(a ** 2 for a in action)

            dist_to_goal = cls._euclidean_dist(current, goal_state)
            fitness = 100.0 - (dist_to_goal * 10.0) - (action_cost * 0.1)
            evaluations.append(
                PlanCandidateEvaluation(
                    plan_id=plan_id,
                    final_distance_to_goal=round(dist_to_goal, 4),
                    cumulative_action_cost=round(action_cost, 4),
                    composite_fitness_score=round(fitness, 4),
                    predicted_success=(dist_to_goal <= success_distance_threshold),
                )
            )

        best = max(evaluations, key=lambda e: e.composite_fitness_score)
        return PlanningBenchmarkReport(
            goal_state=tuple(goal_state),
            total_plans_evaluated=len(candidate_plans),
            optimal_plan_id=best.plan_id,
            optimal_plan_fitness=best.composite_fitness_score,
            evaluations=tuple(evaluations),
        )
