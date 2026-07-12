"""Demo analysis pipeline: real audio -> structure -> verdict.

Runs the genuine pipeline on any waveform — MERT embeddings of beat-aligned 4-bar
segments, an adaptive-scale self-similarity matrix, and (if a model is attached)
the Fusion Segment Transformer's verdict + per-segment gate. Everything here is
real; only the classifier's *weights* come from the self-supervised demo task.

The SSM uses a data-adaptive scale (mean pairwise distance) rather than the
library default ``d`` so the matrix spans a useful dynamic range for both the
model's SSM stream and the visualization.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..data.beat import BeatTracker
from ..data.segment_fixed import segment_fixed
from ..data.segment_sequence import pad_crop_sequence
from ..utils.audio_io import resample

SR = 24000
SEQ_LEN = 48
SEG_SECONDS = 1.5  # embedding-segment length (finer than 4-bar so short clips give a rich SSM)


def adaptive_ssm(embeddings: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """``exp(-||e_i-e_j||^2 / s)`` with ``s`` = mean off-diagonal distance (per track)."""
    E = embeddings
    d2 = torch.cdist(E, E) ** 2
    n = E.size(0)
    off = d2[~torch.eye(n, dtype=torch.bool, device=E.device)]
    scale = off.mean().clamp_min(1e-6)
    ssm = torch.exp(-d2 / scale)
    if mask is not None:
        keep = (~mask).to(ssm.dtype)
        ssm = ssm * keep.unsqueeze(0) * keep.unsqueeze(1)
    return ssm


@dataclass
class DemoFeatures:
    embeddings: torch.Tensor  # [SEQ_LEN, 768]
    mask: torch.Tensor  # [SEQ_LEN] bool, True = pad
    ssm: torch.Tensor  # [SEQ_LEN, SEQ_LEN]
    n_segments: int
    boundaries_sec: list[tuple[float, float]]
    downbeats: list[float]
    segmentation: str
    duration_sec: float


@dataclass
class DemoResult:
    p_incoherent: float  # raw model output (illustrative; proxy-trained)
    coherence: float  # 1 - p_incoherent, presented as a structure estimate
    band: str  # descriptive structure band
    gate_per_segment: list[float]  # length n_segments
    mean_gate: float
    features: DemoFeatures


class DemoPipeline:
    """MERT-based structure extractor (no trained head needed — mean-pooled MERT)."""

    def __init__(self, mert, device: torch.device, seq_len: int = SEQ_LEN,
                 seg_seconds: float = SEG_SECONDS) -> None:
        self.mert = mert
        self.device = device
        self.seq_len = seq_len
        self.seg_seconds = seg_seconds
        self.beat_tracker = BeatTracker(device=str(device))

    @torch.no_grad()
    def features(self, waveform: torch.Tensor, sr: int) -> DemoFeatures:
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.size(0) > 1:
            waveform = waveform.mean(0, keepdim=True)
        if sr != SR:
            waveform = resample(waveform, sr, SR)
        duration = waveform.size(1) / SR

        # Real downbeat detection (shown on the waveform), and fine fixed-window
        # segments for the embedding sequence so even short clips yield a rich SSM.
        downbeats = self.beat_tracker.track(waveform, SR).downbeats
        clips = segment_fixed(waveform, SR, self.seg_seconds)  # [N, 1, S]
        clips = clips[: self.seq_len]
        embs = [self.mert.pooled(clips[i].to(self.device)).cpu() for i in range(clips.size(0))]
        E_full = torch.stack(embs)  # [M, 768]
        boundaries = [(i * self.seg_seconds, (i + 1) * self.seg_seconds) for i in range(len(embs))]

        E, mask = pad_crop_sequence(E_full, self.seq_len)
        ssm = adaptive_ssm(E, mask)
        n = int((~mask).sum().item())
        return DemoFeatures(
            embeddings=E, mask=mask, ssm=ssm, n_segments=n,
            boundaries_sec=boundaries, downbeats=downbeats,
            segmentation=f"{self.seg_seconds:g}s windows", duration_sec=duration,
        )


class DemoAnalyzer:
    """Full analysis: structure features + Fusion model verdict + gate."""

    def __init__(self, mert, model, device: torch.device, seq_len: int = SEQ_LEN) -> None:
        self.pipeline = DemoPipeline(mert, device, seq_len)
        self.model = model.to(device).eval()
        self.device = device

    @torch.no_grad()
    def analyze(self, waveform: torch.Tensor, sr: int) -> DemoResult:
        feats = self.pipeline.features(waveform, sr)
        E = feats.embeddings.unsqueeze(0).to(self.device)
        ssm = feats.ssm.unsqueeze(0).to(self.device)
        mask = feats.mask.unsqueeze(0).to(self.device)

        logit, gate = self.model(E, ssm=ssm, key_padding_mask=mask, return_gate=True)
        p = torch.sigmoid(logit.reshape(-1)[0]).item()

        n = max(feats.n_segments, 1)
        gate_seg = gate.squeeze(0).mean(dim=-1)[:n].cpu().tolist()  # per-segment scalar
        mean_gate = float(sum(gate_seg) / len(gate_seg)) if gate_seg else 0.5

        coherence = 1.0 - p
        if coherence >= 0.66:
            band = "Strongly structured"
        elif coherence >= 0.4:
            band = "Moderately structured"
        else:
            band = "Weak / drifting structure"
        return DemoResult(
            p_incoherent=p, coherence=coherence, band=band,
            gate_per_segment=gate_seg, mean_gate=mean_gate, features=feats,
        )
