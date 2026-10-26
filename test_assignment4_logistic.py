"""Leakage, output, AUC and reproducibility checks for the analysis."""
from pathlib import Path
import json
import sys
import time
import unittest
import numpy as np
import pandas as pd
import os
from types import ModuleType

NOTEBOOK_PATH = Path(__file__).resolve().with_name("assignment4_logistic.ipynb")
analysis = ModuleType("assignment4_notebook")
sys.modules[analysis.__name__] = analysis
notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
previous_directory = Path.cwd()
try:
    os.chdir(NOTEBOOK_PATH.parent)
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code" and "definitions" in cell.get("metadata", {}).get("tags", []):
            code = "".join(cell["source"])
            exec(compile(code, str(NOTEBOOK_PATH) + ":cell_" + str(index), "exec"), analysis.__dict__)
finally:
    os.chdir(previous_directory)


ROOT, OUT = analysis.ROOT, analysis.OUT


def rank_auc(truth, scores):
    """Independent pairwise definition of binary AUC, including ties."""
    positive = np.asarray(scores)[np.asarray(truth, dtype=bool)]
    negative = np.asarray(scores)[~np.asarray(truth, dtype=bool)]
    pairs = positive[:, None] - negative[None, :]
    return float(np.mean((pairs > 0).astype(float) + 0.5 * (pairs == 0)))


class LogisticAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folds = pd.read_csv(OUT / "evaluation_folds.csv")
        cls.pred = pd.read_csv(OUT / "predictions.csv", dtype={"truth": str, "prediction": str})
        cls.metrics = pd.read_csv(OUT / "metrics.csv")
        cls.settings = json.loads((OUT / "run_settings.json").read_text())

    def test_training_only_feature_ranking_and_scaling(self):
        train = np.array([[0., 0., 0.], [2., 0., 0.1], [4., 0., 0.], [6., 0., 0.1]])
        held_out = np.array([[1., 1000000., 0.], [3., -1000000., 0.1]])
        model = analysis.make_model(1, "feeding_group")
        analysis.fit_checked(model, train, np.array(["AL", "AL", "RF", "RF"]))
        self.assertEqual(model.named_steps["genes"].selected_indices_.tolist(), [0])
        np.testing.assert_allclose(model.named_steps["scale"].mean_, [3.])
        model.predict(held_out)
        self.assertEqual(model.named_steps["genes"].selected_indices_.tolist(), [0])
        np.testing.assert_allclose(model.named_steps["scale"].mean_, [3.])
        contaminated = analysis.MostVariableGenes(1).fit(np.vstack([train, held_out]))
        self.assertEqual(contaminated.selected_indices_.tolist(), [1])

    def test_one_fold_per_title_and_accession(self):
        self.assertFalse(self.folds["sample_id"].duplicated().any())
        self.assertEqual(self.folds.groupby("title_group")["fold"].nunique().max(), 1)
        self.assertEqual(set(self.folds["fold"]), set(range(1, 6)))
        audit = pd.read_csv(OUT / "fold_audit.csv")
        self.assertTrue(audit["title_overlap"].eq(0).all())
        self.assertEqual(audit["n_test_runs"].sum(), len(self.folds))

    def test_complete_oof_coverage_and_authoritative_cluster_truth(self):
        selected = pd.read_csv(ROOT / "results/assignment3/hierarchical/selected_clusters.csv",
                               dtype={"cluster": str}).set_index("sample_id")["cluster"]
        self.assertEqual(len(self.pred), len(self.folds) * len(analysis.GENE_COUNTS) * len(analysis.TARGETS))
        for (target, n_genes), sub in self.pred.groupby(["target", "n_genes"]):
            self.assertEqual(set(sub["sample_id"]), set(self.folds["sample_id"]))
            self.assertFalse(sub["sample_id"].duplicated().any())
            matched = sub.set_index("sample_id")
            np.testing.assert_array_equal(matched["fold"], self.folds.set_index("sample_id").loc[matched.index, "fold"])
            if target == "hierarchical_cluster":
                np.testing.assert_array_equal(matched["truth"], selected.loc[matched.index])

    def test_auc_against_independent_pairwise_calculation(self):
        for row in self.metrics.itertuples():
            sub = self.pred.loc[(self.pred["target"] == row.target) & (self.pred["n_genes"] == row.n_genes)]
            labels = ["RF"] if row.target == "feeding_group" else ["1", "2", "3"]
            aucs = [rank_auc(sub["truth"] == label, sub["score_" + label]) for label in labels]
            self.assertAlmostEqual(float(np.mean(aucs)), row.auc, places=12)

    def test_decision_labels_and_binary_probability_export(self):
        for target, sub in self.pred.groupby("target"):
            labels = ["AL", "RF"] if target == "feeding_group" else ["1", "2", "3"]
            scores = sub[["score_" + label for label in labels]].to_numpy()
            self.assertTrue(np.isfinite(scores).all())
            np.testing.assert_array_equal(np.array(labels)[scores.argmax(axis=1)], sub["prediction"].to_numpy())
            for j, label in enumerate(labels):
                probability = sub["ovr_probability_" + label].to_numpy()
                np.testing.assert_allclose(probability, 1.0 / (1.0 + np.exp(-scores[:, j])), atol=1e-14)

    def test_heatmap_contains_exact_signature_union(self):
        signatures = pd.read_csv(OUT / "final_signatures.csv")
        expected = set(signatures.loc[signatures["n_genes"] == 5000, "feature_id"])
        union = pd.read_csv(OUT / "signature_5000_union.csv")
        matrix = pd.read_csv(OUT / "signature_5000_expression_standardized.tsv", sep="\t", index_col=0)
        self.assertEqual(set(union["feature_id"]), expected)
        self.assertEqual(set(matrix.index), expected)
        self.assertEqual(set(matrix.columns), set(self.folds["sample_id"]))
        self.assertTrue(np.isfinite(matrix.to_numpy()).all())
        np.testing.assert_allclose(matrix.mean(axis=1), 0, atol=1e-10)
        self.assertTrue(signatures["cv_selection_frequency"].between(0, 1).all())

    def test_input_files_unchanged(self):
        paths = {
            "expression": ROOT / "Data/SRP064298/gene_data.tsv",
            "metadata": ROOT / "Data/SRP064298/metadata_SRP064298.tsv",
            "clusters": ROOT / "results/assignment3/hierarchical/selected_clusters.csv",
            "processing": ROOT / "Data/aggregated_metadata.json"}
        for key, path in paths.items():
            self.assertEqual(analysis.sha256(path), self.settings["input_sha256"][key])
        self.assertEqual(analysis.notebook_code_hash(), self.settings["analysis_sha256"])

    def test_fold_auc_summary_and_convergence(self):
        fold = pd.read_csv(OUT / "fold_metrics.csv")
        self.assertTrue(fold["max_iterations"].lt(analysis.PARAMETERS["max_iter"]).all())
        self.assertTrue(self.metrics["final_max_iterations"].lt(analysis.PARAMETERS["max_iter"]).all())
        for row in self.metrics.itertuples():
            values = fold.loc[(fold["target"] == row.target) & (fold["n_genes"] == row.n_genes), "auc"]
            self.assertEqual(len(values), 5)
            self.assertAlmostEqual(values.mean(), row.fold_auc_mean, places=12)
            self.assertAlmostEqual(values.std(ddof=1), row.fold_auc_std, places=12)


def execute_notebook():
    import os
    from jupyter_client import KernelManager
    notebook_path = ROOT / "assignment4_logistic.ipynb"
    nb = json.loads(notebook_path.read_text(encoding="utf-8"))
    manager = KernelManager(kernel_name="python3")
    manager.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
    manager.start_kernel(cwd=str(ROOT), env=os.environ.copy())
    client = manager.blocking_client()
    client.start_channels()
    executed = 0
    try:
        client.wait_for_ready(timeout=60)
        for cell in nb["cells"]:
            if cell["cell_type"] != "code":
                continue
            message_id = client.execute("".join(cell["source"]), stop_on_error=True)
            outputs = []
            while True:
                message = client.get_iopub_msg(timeout=120)
                if message.get("parent_header", {}).get("msg_id") != message_id:
                    continue
                kind, content = message["header"]["msg_type"], message["content"]
                if kind == "execute_input":
                    cell["execution_count"] = content["execution_count"]
                elif kind == "stream":
                    outputs.append({"output_type": kind, "name": content["name"], "text": content["text"]})
                elif kind in ("display_data", "execute_result"):
                    output = {"output_type": kind, "data": content["data"], "metadata": content.get("metadata", {})}
                    if kind == "execute_result":
                        output["execution_count"] = content["execution_count"]
                    outputs.append(output)
                elif kind == "error":
                    raise RuntimeError("Notebook error: %s: %s" % (content["ename"], content["evalue"]))
                elif kind == "status" and content["execution_state"] == "idle":
                    break
            cell["outputs"] = outputs
            executed += 1
            print("Executed notebook code cell", executed, flush=True)
        notebook_path.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    finally:
        client.stop_channels()
        manager.shutdown_kernel(now=True)
    return executed


if __name__ == "__main__":
    started = time.time()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(LogisticAnalysisTests))
    if not result.wasSuccessful():
        raise SystemExit(1)
    notebook_cells = execute_notebook() if "--notebook" in sys.argv else None
    summary = {"status": "passed", "tests_run": result.testsRun,
               "notebook_code_cells_executed": notebook_cells,
               "elapsed_seconds": round(time.time() - started, 2),
               "input_hashes_unchanged": True, "title_overlap_between_train_and_test": 0}
    (OUT / "validation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
