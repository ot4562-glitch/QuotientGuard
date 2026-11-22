import copy
from contextlib import redirect_stdout, redirect_stderr
import io
import json
from pathlib import Path
import unittest

from quotientguard import audit_adamw_step, summarize_training_audits
from quotientguard.cli import main

try:
    import torch
except ImportError:
    torch = None


class AdamWExampleTests(unittest.TestCase):
    example = Path(__file__).resolve().parents[1] / "examples" / "adamw-audit.json"

    def test_reference_trace_and_boundary_summary_without_torch(self):
        trace = json.loads(self.example.read_text(encoding="utf-8"))
        optimizer = audit_adamw_step(trace)
        self.assertEqual(optimizer["status"], "MATCH", optimizer)
        window = {"schema_version": "quotientguard-objective-window-v1", "status": "SUPPORTED"}
        summary = summarize_training_audits(window=window, optimizer=optimizer)
        self.assertEqual([entry["status"] for entry in summary["boundaries"]],
                         ["NOT_PROVIDED", "NOT_PROVIDED", "SUPPORTED", "MATCH"])
        self.assertFalse(summary["pipeline_verified"])
        with self.assertRaises(ValueError):
            summarize_training_audits(objective=window)

    def test_cli_does_not_overwrite_trace(self):
        original = self.example.read_bytes()
        with redirect_stdout(io.StringIO()) as stream:
            self.assertEqual(main(["adamw-audit", str(self.example), "--json"]), 0)
        self.assertEqual(json.loads(stream.getvalue())["status"], "MATCH")
        with redirect_stderr(io.StringIO()):
            self.assertEqual(main(["adamw-audit", str(self.example), "--out", str(self.example)]), 2)
        self.assertEqual(self.example.read_bytes(), original)

    def test_malformed_values_and_counter_precision_gate(self):
        trace = json.loads(self.example.read_text(encoding="utf-8"))
        altered = copy.deepcopy(trace)
        altered["gradient"]["values"][0] = float("nan")
        self.assertEqual(audit_adamw_step(altered)["status"], "INVALID_EVIDENCE")
        for record in ("consumed", "authorized"):
            trace[record]["state"]["step"] = 2**24
        report = audit_adamw_step(trace)
        self.assertEqual(report["transition"]["reason"], "step_counter_increment_not_exact")
        self.assertEqual(report["status"], "UNSUPPORTED")


@unittest.skipIf(torch is None, "native AdamW comparisons require PyTorch")
class AdamWAuditTests(unittest.TestCase):
    def trace(self, *, warm=True, mutation=None, gradient=(0.33, -0.28), restored=False):
        p = torch.nn.Parameter(torch.tensor([0.4, -0.7], dtype=torch.float64))
        q = torch.nn.Parameter(torch.tensor([0.2, 0.1], dtype=torch.float64))
        params = [p, q]
        options = dict(lr=0.025, betas=(0.7, 0.95), eps=1e-8, weight_decay=0.13,
                       foreach=False, fused=False)
        opt = torch.optim.AdamW(params, **options)
        if warm:
            for a, values in zip(params, ([0.4, -0.1], [-0.7, 0.8])):
                a.grad = torch.tensor(values, dtype=torch.float64)
            opt.step()

        binding = dict(run_id="replay", step_id="window-2", parameter_id="actor.weight", shape=[2])

        def record(stage, source):
            return dict(stage=stage, source=source, verified=True, binding=copy.deepcopy(binding))

        def snapshot(stage, source):
            group = opt.param_groups[0]
            settings = {k: group[k] for k in ("lr", "eps", "weight_decay")}
            settings.update(zip(("beta1", "beta2"), group["betas"]))
            current = opt.state.get(p)
            state = (dict(kind="initialized", step=int(current["step"]),
                          exp_avg=current["exp_avg"].tolist(), exp_avg_sq=current["exp_avg_sq"].tolist())
                     if current else dict(kind="uninitialized"))
            return {**record(stage, source), "parameters": p.detach().tolist(),
                    "state": state, "hyperparameters": settings}

        authorized = snapshot("authorized", "retained checkpoint" if warm else "declared cold start")
        if restored:
            saved = copy.deepcopy(opt.state_dict())
            p = torch.nn.Parameter(p.detach().clone())
            q = torch.nn.Parameter(q.detach().clone())
            opt = torch.optim.AdamW([p, q], **options)
            opt.load_state_dict(saved)
        if mutation:
            mutation(opt, p, q)
        p.grad = torch.tensor(gradient, dtype=torch.float64) if gradient is not None else None
        q.grad = None
        consumed = snapshot("before_step", "native state before step")
        actual_grad = record("consumed_gradient", "native parameter gradient before step")
        actual_grad["kind"] = "dense" if p.grad is not None else "none"
        if p.grad is not None:
            actual_grad["values"] = p.grad.tolist()
        opt.step()
        after = snapshot("after_step", "native state after step")
        return {
            "binding": binding, "authorized": authorized, "consumed": consumed,
            "gradient": actual_grad, "after": after,
            "execution": dict(
                optimizer="torch.optim.AdamW", device="cpu", dtype="float64", layout="dense",
                fused=False, foreach=False, amsgrad=False, maximize=False, capturable=False,
                differentiable=False, mixed_precision=False, sharded=False, step_executed=True,
                step_dtype=str(opt.state[p]["step"].dtype).removeprefix("torch.")
                if p in opt.state else "float32",
                torch_version=str(torch.__version__)),
        }

    def test_native_cpu_cold_and_restored_steps_and_read_only(self):
        for trace in (self.trace(warm=False), self.trace(restored=True)):
            with self.subTest(step=trace["after"]["state"]["step"]):
                original = copy.deepcopy(trace)
                report = audit_adamw_step(trace)
                self.assertEqual(report["status"], "MATCH", report)
                self.assertEqual(trace, original)
                self.assertFalse(report["optimizer_step_executed_by_auditor"])
                self.assertFalse(report["pipeline_verified"])

    def test_native_lineage_mutations_do_not_blame_transition(self):
        mutations = {
            "state.exp_avg[0]": lambda opt, p, q: opt.state[p]["exp_avg"].zero_(),
            "state.exp_avg_sq[0]": lambda opt, p, q: opt.state[p]["exp_avg_sq"].add_(0.01),
            "state.step": lambda opt, p, q: opt.state[p]["step"].add_(1),
            "hyperparameters.lr": lambda opt, p, q: opt.param_groups[0].update(lr=0.03),
        }
        for field, mutation in mutations.items():
            with self.subTest(field=field):
                report = audit_adamw_step(self.trace(mutation=mutation))
                self.assertEqual(report["reason"], "state_lineage_mismatch", report)
                self.assertIn(field, {error["field"] for error in report["lineage"]["errors"]})
                self.assertEqual(report["transition"]["status"], "MATCH")

    def test_native_same_shape_wrong_parameter_state(self):
        def replace(opt, p, q):
            opt.state[p] = copy.deepcopy(opt.state[q])

        report = audit_adamw_step(self.trace(mutation=replace))
        self.assertEqual(report["reason"], "state_lineage_mismatch")
        self.assertEqual(report["transition"]["status"], "MATCH")

    def test_absent_gradient_differs_from_zero_and_weight_decay_is_decoupled(self):
        absent = self.trace(warm=False, gradient=None)
        zero = self.trace(warm=False, gradient=(0.0, 0.0))
        for trace in (absent, zero, self.trace(gradient=None), self.trace(gradient=(0.0, 0.0))):
            self.assertEqual(audit_adamw_step(trace)["status"], "MATCH")
        self.assertEqual(absent["after"]["state"], {"kind": "uninitialized"})
        self.assertEqual(zero["after"]["state"]["step"], 1)
        self.assertEqual(absent["after"]["parameters"], absent["consumed"]["parameters"])
        factor = 1 - 0.025 * 0.13
        self.assertEqual(zero["after"]["parameters"], [0.4 * factor, -0.7 * factor])

    def test_transition_mismatch_and_tolerance(self):
        trace = self.trace()
        trace["after"]["parameters"][0] += 1e-14
        self.assertEqual(audit_adamw_step(trace)["status"], "MATCH")
        trace["after"]["parameters"][0] += 1e-10
        report = audit_adamw_step(trace)
        self.assertEqual(report["reason"], "adamw_transition_mismatch")
        self.assertEqual(report["lineage"]["status"], "MATCH")
        self.assertEqual(report["transition"]["errors"][0]["field"], "parameters[0]")

    def test_evidence_and_mode_gates(self):
        trace = self.trace()
        for field in ("authorized", "consumed", "gradient", "after"):
            with self.subTest(missing=field):
                altered = copy.deepcopy(trace)
                del altered[field]
                self.assertEqual(audit_adamw_step(altered)["status"], "INCONCLUSIVE")
        for field, value in (
            ("fused", True), ("foreach", True), ("device", "cuda"), ("mixed_precision", True),
            ("sharded", True), ("dtype", "float32"), ("optimizer", "torch.optim.Adam"),
            ("step_executed", False),
        ):
            with self.subTest(mode=field):
                altered = copy.deepcopy(trace)
                altered["execution"][field] = value
                self.assertEqual(audit_adamw_step(altered)["status"], "UNSUPPORTED")
        altered = copy.deepcopy(trace)
        altered["consumed"]["hyperparameters"]["amsgrad"] = True
        self.assertEqual(audit_adamw_step(altered)["status"], "UNSUPPORTED")
        for field in ("authorized", "gradient", "after"):
            with self.subTest(wrong_join=field):
                altered = copy.deepcopy(trace)
                altered[field]["binding"]["parameter_id"] = "other.weight"
                self.assertEqual(audit_adamw_step(altered)["status"], "INCONCLUSIVE")
        trace["gradient"]["verified"] = False
        report = audit_adamw_step(trace)
        self.assertEqual(report["lineage"]["status"], "MATCH")
        self.assertEqual(report["transition"]["reason"], "gradient_provenance_unverified")


if __name__ == "__main__":
    unittest.main()
