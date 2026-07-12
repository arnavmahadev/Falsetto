"""Typed configuration schema.

A single :class:`Config` object bundles the five sections referenced throughout
the papers/tasks — ``data``, ``extractor``, ``model``, ``train``, ``eval`` —
plus a few run-level fields (seed, device, experiment tracker). Every field has
a sensible default so a bare ``Config()`` is valid and partial YAML overrides only
what it names. See :mod:`falsetto.config.io` for load/save + round-trip.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DataConfig:
    """Dataset + dataloader settings (Phase 1)."""

    dataset: str = "fakemusiccaps"  # fakemusiccaps | sonics | aime
    root: str = "data/raw/fakemusiccaps"  # where audio lives on disk (not committed)
    manifest: str = "data/manifests/fakemusiccaps.csv"
    clip_seconds: float = 10.0  # Stage-1 fixed-window length
    segment_length: int = 48  # Stage-2 sequence length (4-bar segments), pad/crop to this
    batch_size: int = 8
    num_workers: int = 4
    pin_memory: bool = True
    augment: bool = False  # SSL-only pitch/time aug, training split only (T-09)
    aug_prob: float = 0.5
    pitch_shift_semitones: float = 2.0
    time_stretch_min: float = 0.8
    time_stretch_max: float = 1.25
    # Stratified 8:1:1 split with no track leakage (T-06)
    split_train: float = 0.8
    split_val: float = 0.1
    split_test: float = 0.1


@dataclass
class ExtractorConfig:
    """Stage-1 feature-extractor settings (Phase 2)."""

    name: str = "mert"  # mert | wav2vec2 | music2vec | fxencoder | muffin
    pretrained: str = "m-a-p/MERT-v1-95M"
    sample_rate: int = 24000  # 16k w2v/music2vec, 24k MERT/muffin, 44.1k stereo FXencoder
    embed_dim: int = 768
    returns_sequence: bool = True
    # Layer strategy: "last" hidden state, or "weighted" learnable sum of all layers (T-12).
    layer_strategy: str = "last"
    freeze: bool = True
    trust_remote_code: bool = True
    stereo: bool = False
    cache_dir: str = "data/cache"  # per-clip .pt embedding cache (T-11)


@dataclass
class ModelConfig:
    """Model architecture settings (Phases 3-5)."""

    name: str = "audiocat"  # audiocat | fx_segment | segment_transformer | fusion_segment_transformer
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    ffn_dim: int = 1024
    dropout: float = 0.1
    num_classes: int = 1  # single logit, BCE-with-logits
    # AudioCAT cross-attention decoder
    num_latents: int = 8
    # Fusion Segment Transformer stream depths (Paper 2)
    emb_stream_layers: int = 6
    ssm_stream_layers: int = 2
    fusion: str = "gmu"  # "gmu" (gated) | "mean" (plain cross-attn, ablation T-38)


@dataclass
class TrainConfig:
    """Optimization + checkpointing settings (Phase 3-5)."""

    epochs: int = 50
    lr: float = 1e-5
    weight_decay: float = 1e-6
    optimizer: str = "adam"  # adam | adamw | fused_adam
    loss: str = "bce"  # bce | focal
    focal_gamma: float = 2.0
    focal_alpha: float = 0.25
    amp: bool = True
    grad_clip: float = 1.0
    early_stopping: bool = True
    patience: int = 8  # epochs without val improvement before stopping
    monitor: str = "auc"  # metric to select the best checkpoint (T-22)
    monitor_mode: str = "max"  # max | min
    ckpt_dir: str = "checkpoints"
    log_every: int = 20  # steps


@dataclass
class EvalConfig:
    """Evaluation + reporting settings (Phase 6)."""

    split: str = "test"
    metrics: list[str] = field(
        default_factory=lambda: [
            "accuracy",
            "precision",
            "recall",
            "f1",
            "auc",
            "specificity",
        ]
    )
    out_dir: str = "results"


@dataclass
class Config:
    """Top-level experiment config."""

    name: str = "default"
    seed: int = 42
    device: str = "auto"  # auto | cpu | cuda | mps
    tracker: str = "tensorboard"  # tensorboard | wandb | none
    output_dir: str = "runs"
    deterministic: bool = True

    data: DataConfig = field(default_factory=DataConfig)
    extractor: ExtractorConfig = field(default_factory=ExtractorConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
