"""T-23: Stage-1 results table generation."""

from __future__ import annotations

import torch

from falsetto.eval.table_stage1 import evaluate_model, results_to_markdown, write_table
from falsetto.training.metrics import MetricResults


def test_results_to_markdown_structure():
    rows = {
        "MERT": MetricResults(accuracy=0.98, precision=0.97, recall=0.99, f1=0.98, auc=0.995, specificity=0.96),
        "FXencoder": MetricResults(accuracy=0.90, precision=0.9, recall=0.9, f1=0.9, auc=0.95, specificity=0.9),
    }
    md = results_to_markdown(rows, title="Stage-1")
    assert "| Extractor |" in md
    assert "| MERT |" in md and "| FXencoder |" in md
    assert "0.9950" in md  # AUC formatted to 4 dp
    # header + separator + 2 data rows + title/blank
    assert md.count("\n") >= 5


def test_write_table(tmp_path):
    rows = {"MERT": MetricResults(accuracy=0.99)}
    out = write_table(rows, tmp_path / "table.md", title="t")
    assert out.exists()
    assert "MERT" in out.read_text()


def test_evaluate_model_on_toy_head():
    # A tiny head that returns the mean feature as a logit; perfectly separable data.
    class ToyHead(torch.nn.Module):
        def forward(self, waveforms, return_embedding=False):
            return waveforms.mean(dim=(1, 2)) * 10  # [B]

    feats = torch.stack([torch.full((1, 4), -1.0)] * 3 + [torch.full((1, 4), 1.0)] * 3)
    labels = torch.tensor([0, 0, 0, 1, 1, 1]).float()
    loader = [(feats, labels)]
    res = evaluate_model(ToyHead(), loader, torch.device("cpu"))
    assert res.accuracy == 1.0
    assert res.auc == 1.0
