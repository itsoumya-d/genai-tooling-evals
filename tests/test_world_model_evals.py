import unittest
from genai_evals.world_model_evals import (
    WorldModelEvaluator,
    DynamicsRolloutReport,
    ConservationReport,
    PlanningBenchmarkReport,
)


class TestWorldModelEvaluator(unittest.TestCase):
    def test_evaluate_rollout_dynamics_passing(self):
        gt = [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 0.5],
            [2.0, 1.8, 1.0],
            [3.0, 2.5, 1.4],
        ]
        pred = [
            [0.0, 0.0, 0.0],
            [1.05, 0.95, 0.52],
            [1.95, 1.85, 0.98],
            [3.08, 2.45, 1.42],
        ]

        report = WorldModelEvaluator.evaluate_rollout_dynamics(
            sequence_id="seq_alpha_01",
            predicted_states=pred,
            ground_truth_states=gt,
            mse_tolerance=0.05,
        )

        self.assertIsInstance(report, DynamicsRolloutReport)
        self.assertEqual(report.sequence_id, "seq_alpha_01")
        self.assertEqual(report.horizon_length, 4)
        self.assertLess(report.mean_squared_error, 0.05)
        self.assertGreater(report.mean_state_cosine_similarity, 0.95)
        self.assertTrue(report.passed_tolerance)

    def test_evaluate_rollout_dynamics_divergent(self):
        gt = [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]]
        pred = [[1.0, 2.0], [5.0, 9.0], [15.0, 25.0]]

        report = WorldModelEvaluator.evaluate_rollout_dynamics(
            sequence_id="seq_divergent",
            predicted_states=pred,
            ground_truth_states=gt,
            mse_tolerance=0.20,
        )
        self.assertFalse(report.passed_tolerance)
        self.assertGreater(report.cumulative_drift_rate, 10.0)

    def test_verify_conservation_constraints_valid(self):
        states = [
            [0.0, 0.0],
            [0.5, 0.5],
            [1.0, 1.0],
            [1.5, 1.5],
        ]
        timestamps = [0.0, 1.0, 2.0, 3.0]

        report = WorldModelEvaluator.verify_conservation_constraints(
            sequence_id="seq_phys_valid",
            states=states,
            timestamps=timestamps,
            max_velocity_threshold=2.0,
            bounds=(-10.0, 10.0),
        )

        self.assertIsInstance(report, ConservationReport)
        self.assertTrue(report.is_physically_consistent)
        self.assertEqual(len(report.violations), 0)

    def test_verify_conservation_constraints_violations(self):
        # Out-of-bounds state and impossible velocity jump
        states = [
            [0.0, 0.0],
            [50.0, 50.0],  # sudden jump
            [100.0, 20.0], # exceeds bounds
        ]
        timestamps = [0.0, 0.1, 0.2]

        report = WorldModelEvaluator.verify_conservation_constraints(
            sequence_id="seq_phys_invalid",
            states=states,
            timestamps=timestamps,
            max_velocity_threshold=5.0,
            bounds=(-40.0, 40.0),
        )

        self.assertFalse(report.is_physically_consistent)
        self.assertGreater(len(report.violations), 0)

    def test_benchmark_model_based_planning(self):
        initial_state = [0.0, 0.0]
        goal_state = [10.0, 10.0]

        # Simple 2D velocity dynamics: state_{t+1} = state_t + action
        def transition_fn(s, a):
            return [s[0] + a[0], s[1] + a[1]]

        # Candidate 1: directly moves toward goal
        plan_optimal = ("plan_direct", [[5.0, 5.0], [5.0, 5.0]])
        # Candidate 2: moves away from goal
        plan_wrong = ("plan_divergent", [[-2.0, -2.0], [-3.0, -3.0]])

        report = WorldModelEvaluator.benchmark_model_based_planning(
            initial_state=initial_state,
            goal_state=goal_state,
            candidate_plans=[plan_optimal, plan_wrong],
            transition_model=transition_fn,
            success_distance_threshold=0.5,
        )

        self.assertIsInstance(report, PlanningBenchmarkReport)
        self.assertEqual(report.total_plans_evaluated, 2)
        self.assertEqual(report.optimal_plan_id, "plan_direct")
        self.assertGreater(report.optimal_plan_fitness, 0.0)

        best_eval = next(e for e in report.evaluations if e.plan_id == "plan_direct")
        self.assertTrue(best_eval.predicted_success)
        self.assertAlmostEqual(best_eval.final_distance_to_goal, 0.0, places=3)


if __name__ == "__main__":
    unittest.main()
