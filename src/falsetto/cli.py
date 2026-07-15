"""FALSETTO command-line entry point (TASKS.md T-42).

Wraps the main workflows over the package API:

    falsetto extract  --config <cfg> [--split S] [--limit N]
    falsetto train    --config <cfg> --stage 1
    falsetto train    --config <cfg> --stage 2 --seqcache <dir>
    falsetto eval     --config <cfg> --stage1-ckpt <ckpt> [--split test]
    falsetto predict  <audio> --stage1-config <c1> --stage1-ckpt <k1> \\
                              --stage2-config <c2> --stage2-ckpt <k2>
"""

from __future__ import annotations

import argparse
import sys

from . import __version__


def _cmd_extract(args) -> int:
    from tqdm import tqdm

    from .config import load_config
    from .data.audio import load_for_extractor
    from .data.cache import EmbeddingCache
    from .data.manifests import load_manifest
    from .extractors import build_extractor
    from .utils.device import select_device
    from .utils.logging import get_logger

    log = get_logger("cli.extract")
    cfg = load_config(args.config)
    device = select_device(args.device or cfg.device)
    df = load_manifest(cfg.data.manifest)
    if args.split:
        df = df[df["split"] == args.split]
    if args.limit:
        df = df.head(args.limit)
    extractor = build_extractor(cfg.extractor).to(device)
    extractor.freeze()
    cache = EmbeddingCache(cfg.extractor.cache_dir)
    n = 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc="extract"):
        clip_id = str(row["filepath"])
        if cache.has(cfg.extractor.name, clip_id):
            continue
        wav, _ = load_for_extractor(row["filepath"], cfg.extractor.name)
        cache.save(cfg.extractor.name, clip_id, extractor.extract(wav.to(device)))
        n += 1
    log.info("cached %d embeddings -> %s", n, cfg.extractor.cache_dir)
    return 0


def _cmd_train(args) -> int:
    from .config import load_config
    from .utils.logging import get_logger

    log = get_logger("cli.train")
    cfg = load_config(args.config)
    if args.epochs:
        cfg.train.epochs = args.epochs
    if args.device:
        cfg.device = args.device

    if args.stage == 1:
        from .training.train_stage1 import train_stage1_from_config

        ckpt = train_stage1_from_config(cfg, resume=args.resume)
    else:
        if not args.seqcache:
            log.error("stage 2 needs --seqcache <dir> of precomputed sequences")
            return 2
        from pathlib import Path

        from .training.train_stage2 import Stage2SequenceDataset, train_stage2_from_sequences

        cache = Path(args.seqcache)
        train_items = Stage2SequenceDataset.from_cache_dir(cache / "train").items
        val_items = Stage2SequenceDataset.from_cache_dir(cache / "val").items
        embed_dim = train_items[0]["embeddings"].shape[-1]
        ckpt = train_stage2_from_sequences(cfg, train_items, val_items, embed_dim, cfg.data.segment_length)
    log.info("best checkpoint: %s", ckpt)
    return 0


def _cmd_eval(args) -> int:
    from .config import load_config
    from .eval.table_stage1 import evaluate_checkpoint, results_to_markdown

    cfg = load_config(args.config)
    res = evaluate_checkpoint(cfg, args.stage1_ckpt, split=args.split)
    print(results_to_markdown({cfg.extractor.name: res}, title=f"Stage-1 ({args.split})"))
    return 0


def _cmd_predict(args) -> int:
    from .config import load_config
    from .inference.predict import Predictor

    predictor = Predictor.from_configs(
        load_config(args.stage1_config), args.stage1_ckpt,
        load_config(args.stage2_config), args.stage2_ckpt,
        device=args.device or "auto",
    )
    pred = predictor.predict_file(args.audio)
    print(f"P(AI) = {pred.p_ai:.4f}  ->  {pred.label}  "
          f"({pred.num_segments} segments, {pred.segmentation})")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="falsetto", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", action="version", version=f"falsetto {__version__}")
    sub = parser.add_subparsers(dest="command")

    p_ex = sub.add_parser("extract", help="extract + cache embeddings for a manifest (T-17)")
    p_ex.add_argument("--config", required=True)
    p_ex.add_argument("--split")
    p_ex.add_argument("--limit", type=int)
    p_ex.add_argument("--device")
    p_ex.set_defaults(func=_cmd_extract)

    p_tr = sub.add_parser("train", help="train a Stage-1 or Stage-2 model (T-22/T-29/T-35)")
    p_tr.add_argument("--config", required=True)
    p_tr.add_argument("--stage", type=int, choices=(1, 2), default=1)
    p_tr.add_argument("--seqcache", help="stage-2 sequence cache dir")
    p_tr.add_argument("--resume", help="stage-1 checkpoint to continue from (e.g. after a dropped session)")
    p_tr.add_argument("--epochs", type=int)
    p_tr.add_argument("--device")
    p_tr.set_defaults(func=_cmd_train)

    p_ev = sub.add_parser("eval", help="evaluate a Stage-1 checkpoint -> table (T-23/T-36)")
    p_ev.add_argument("--config", required=True)
    p_ev.add_argument("--stage1-ckpt", required=True)
    p_ev.add_argument("--split", default="test")
    p_ev.set_defaults(func=_cmd_eval)

    p_pr = sub.add_parser("predict", help="end-to-end real/AI prediction (T-41)")
    p_pr.add_argument("audio")
    p_pr.add_argument("--stage1-config", required=True)
    p_pr.add_argument("--stage1-ckpt", required=True)
    p_pr.add_argument("--stage2-config", required=True)
    p_pr.add_argument("--stage2-ckpt", required=True)
    p_pr.add_argument("--device")
    p_pr.set_defaults(func=_cmd_predict)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
