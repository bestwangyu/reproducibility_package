import argparse
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import reproduce_evaluations as workflow
import summarize_finalenv_fair_baselines as fair
import summarize_finalenv_robustness as robustness


class ReproductionWorkflowTests(unittest.TestCase):
    def args(self, stage="development", **extra):
        values = dict(stage=stage, device="cpu", output_root="outputs/unit_test",
                      models="frozen", smoke=False, dry_run=False)
        values.update(extra)
        return argparse.Namespace(**values)

    def test_development_paths_match_fair_and_robustness_readers(self):
        root, jobs, _ = workflow.build_plan(self.args())
        for job, (seed, evaluation) in zip(jobs, ((s, e) for s in workflow.TRAIN_SEEDS for e in workflow.EVAL_SEEDS)):
            with patch.object(fair, "require_files", side_effect=RuntimeError("stop before reading")) as required:
                with self.assertRaises(RuntimeError):
                    fair.validate_main(root, seed, evaluation)
            self.assertEqual(job["directory"], required.call_args[0][0])
            self.assertEqual(job["directory"], robustness.result_directory(root, "nominal", seed, evaluation))

    def test_independent_grid_is_complete_and_uses_only_test_data(self):
        _, jobs, summaries = workflow.build_plan(self.args("independent"))
        self.assertEqual(len(jobs), 21)
        self.assertEqual(len({j["directory"] for j in jobs}), 21)
        for job in jobs:
            cmd = job["command"]
            self.assertEqual(cmd[cmd.index("--data-dir") + 1], "data_test/1005")
            self.assertEqual(cmd[cmd.index("--offset") + 1], "0")
            self.assertEqual(cmd[cmd.index("--instances") + 1], "100")
        self.assertEqual(summaries[0][2], "summarize_p0t_frozen_test.py")

    def test_ablation_context_and_policy_remain_distinct(self):
        _, jobs, summaries = workflow.build_plan(self.args("ablations"))
        self.assertEqual((len(jobs), len(summaries)), (18, 2))
        for job in jobs:
            cmd = job["command"]
            zero = "zero_context" in job["directory"].name
            self.assertEqual(cmd[cmd.index("--stability-context-mode") + 1], "zero" if zero else "full")
            self.assertIn("checkpoints/zero_context/" if zero else "checkpoints/reward_only/", cmd[cmd.index("--policy-checkpoint") + 1])
            self.assertEqual(cmd[cmd.index("--selection-mode") + 1], "budget_first")

    def test_smoke_outputs_cannot_be_mistaken_for_formal_outputs(self):
        root, jobs, summaries = workflow.build_plan(self.args("independent", smoke=True))
        self.assertEqual(root.name, "smoke")
        self.assertEqual(len(jobs), 3)
        self.assertEqual(summaries, [])
        for job in jobs:
            self.assertEqual(job["fields"].get("instances", job["fields"].get("instances_per_rule")), 4)

    def test_archived_results_cannot_be_an_output_root(self):
        for dest in (workflow.ROOT, workflow.ROOT / "results", workflow.ROOT / "results/new"):
            with self.assertRaises(ValueError):
                workflow.build_plan(self.args(output_root=str(dest)))

    def test_dry_run_does_not_create_outputs_or_start_subprocesses(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "not_created"
            args = self.args("independent", output_root=str(dest), dry_run=True)
            with patch.object(workflow, "parse_args", return_value=args), patch.object(workflow, "verify_inputs"), patch.object(workflow, "run_logged") as run:
                with contextlib.redirect_stdout(io.StringIO()):
                    workflow.main()
            self.assertFalse(dest.exists())
            run.assert_not_called()

    def test_independent_test_rejects_retrained_models(self):
        with patch("sys.argv", ["reproduce_evaluations.py", "independent", "--models", "retrained"]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                workflow.parse_args()


class EvaluationInputTests(unittest.TestCase):
    def test_explicit_checkpoint_pair_and_original_source_interface(self):
        from evaluate_constrained_pilot import parse_args
        with patch("sys.argv", ["evaluate_constrained_pilot.py", "--source-dir", "training"]):
            self.assertEqual(parse_args().source_dir, "training")
        with patch("sys.argv", ["evaluate_constrained_pilot.py", "--policy-checkpoint", "final.pt", "--initial-checkpoint", "initial.pt"]):
            self.assertEqual(parse_args().policy_checkpoint, "final.pt")
        for options in (["--policy-checkpoint", "final.pt"],
                        ["--source-dir", "training", "--initial-checkpoint", "initial.pt"],
                        ["--source-dir", "training", "--policy-checkpoint", "final.pt"]):
            with patch("sys.argv", ["evaluate_constrained_pilot.py"] + options):
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    parse_args()


if __name__ == "__main__":
    unittest.main()
