"""FALSETTO Studio — the interactive demo.

A Gradio app that runs the real pipeline on any uploaded (or bundled) track and
shows what the Segment Transformer sees: the self-similarity matrix, beat-tracked
segments, the fusion gate, and a verdict. Clean, academic styling — flat, no
gradients, one teal accent.

Honest framing: the classifier is trained on a self-supervised *structural
coherence* task (intact music vs. structure-shuffled music) over real recordings
and synthetic clips, because the labeled AI-music datasets aren't bundled. Every
visualization is computed for real on your audio via MERT.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..config.schema import ExtractorConfig
from ..extractors import build_extractor
from ..utils.audio_io import load_audio
from ..utils.device import select_device
from ..utils.logging import get_logger
from .assets import META_NAME, build_demo_assets, load_demo_model
from .pipeline import DemoAnalyzer, DemoResult
from .plots import TEAL, gate_figure, ssm_figure, waveform_figure

_log = get_logger("demo.studio")

_SERIF = '"Iowan Old Style", "Palatino Linotype", "Book Antiqua", Palatino, Georgia, serif'
_CSS = f"""
.gradio-container {{ max-width: 1000px !important; margin: 0 auto !important; }}
#fs-nav {{ display: flex; align-items: center; gap: 12px; padding: 8px 2px 16px;
    border-bottom: 1px solid #E4DFD4; margin-bottom: 24px; }}
#fs-nav .brand {{ font-family: {_SERIF}; font-weight: 700; font-size: 20px; letter-spacing: .01em; color: #1C1A16; }}
#fs-nav .brand .dot {{ color: #56658A; }}
#fs-nav .links {{ margin-left: auto; display: flex; gap: 8px; }}
#fs-nav .links a {{ font-size: 13px; color: #6A675E; border: 1px solid #E4DFD4;
    padding: 6px 13px; border-radius: 7px; text-decoration: none; }}
#fs-head h1 {{ font-family: {_SERIF}; font-weight: 700; font-size: 33px; line-height: 1.1;
    letter-spacing: -.01em; margin: 4px 0 10px; color: #1C1A16; }}
#fs-head h1 em {{ font-style: italic; font-weight: 500; color: #56658A; }}
#fs-head .sub {{ font-family: {_SERIF}; font-size: 18px; color: #6A675E; line-height: 1.5; margin: 0; max-width: 60ch; }}
#fs-head .meta {{ font-size: 13px; color: #9A968B; margin-top: 12px; }}
button.primary, .primary button, .gr-button {{ background-image: none !important; box-shadow: none !important; }}
footer {{ display: none !important; }}
"""


def load_analyzer(assets_dir: str | Path, device: str = "auto"):
    assets_dir = Path(assets_dir)
    ckpt = assets_dir / "fusion_demo.pt"
    if not ckpt.exists():
        _log.info("no demo assets found — building them first...")
        build_demo_assets(assets_dir, device=device)
    dev = select_device(device)
    mert = build_extractor(ExtractorConfig(name="mert")).to(dev)
    mert.freeze()
    model = load_demo_model(ckpt, dev)
    meta = json.loads((assets_dir / META_NAME).read_text())
    return DemoAnalyzer(mert, model, dev), meta


def _verdict_html(result: DemoResult) -> str:
    pct = int(round(result.coherence * 100))
    f = result.features
    chips = "".join(
        f'<span style="margin-right:16px">{v}&nbsp;<span style="opacity:.6">{k}</span></span>'
        for k, v in [
            ("segments", f.n_segments), ("downbeats", len(f.downbeats)),
            ("analyzed", f"{f.duration_sec:.0f}s"), ("mean gate", f"{result.mean_gate:.2f}"),
        ]
    )
    return f"""
    <div style="border:1px solid #E4DFD4;border-radius:10px;padding:20px 22px;background:#FBFAF5;
                font-family:-apple-system,'Segoe UI',system-ui,sans-serif;">
      <div style="font-family:{_SERIF};font-size:12px;letter-spacing:.09em;text-transform:uppercase;color:#9A968B;">Structure profile</div>
      <div style="font-family:{_SERIF};font-size:26px;font-weight:700;color:#1C1A16;margin:3px 0 4px;">{result.band}</div>
      <div style="font-size:14px;color:#6A675E;margin-bottom:12px;">structural-coherence estimate
        <span style="background:#E7E9F0;color:#56658A;font-weight:700;padding:1px 9px;border-radius:20px;margin-left:2px;">{pct}%</span>
      </div>
      <div style="background:#EAE6DC;border-radius:20px;height:9px;overflow:hidden;margin-bottom:6px;">
        <div style="width:{pct}%;height:100%;background:#56658A;border-radius:20px;"></div>
      </div>
      <div style="font-size:11.5px;color:#9A968B;margin-bottom:14px;">
        demo model, proxy-trained — illustrative, not a validated AI-vs-real verdict
      </div>
      <div style="font-family:ui-monospace,Menlo,monospace;font-size:12.5px;color:#6A675E;">{chips}</div>
    </div>
    """


_ABOUT = """
### How to read this

- **Self-similarity matrix** — segment *i* vs *j* via `exp(−‖eᵢ−eⱼ‖²/s)` on real MERT embeddings.
  Bright off-diagonals repeating at a fixed lag mean the music repeats (verses, choruses);
  a diffuse block that drifts apart means it doesn't. This structure is the paper's core signal.
- **Waveform + downbeats** — beats are detected (Beat&nbsp;This! → librosa fallback) and grouped into
  segments (shaded).
- **Fusion gate** — per segment, the gated multimodal unit's balance between the *content* stream and
  the *structure* (SSM) stream — the headline mechanism of the second paper.

**About the structure estimate.** The full detector classifies real vs. AI from these structural
features. Training it needs the labeled datasets (FakeMusicCaps / SONICS / AIME — tens of GB to TB,
not bundled), so the number shown here comes from a small **demo model trained on a proxy**
(structured music vs. drifting/unstructured audio) and is **illustrative only**. The visualizations,
though, are computed for real on your audio — point `falsetto train` at the datasets to get a
validated detector.
"""


def build_interface(analyzer: DemoAnalyzer, meta: dict, examples_dir: Path):
    import gradio as gr

    def analyze(audio_path):
        if not audio_path:
            return "<i style='opacity:.6'>Upload or choose a track to analyze.</i>", None, None, None
        waveform, sr = load_audio(audio_path, mono=True)
        result = analyzer.analyze(waveform, sr)
        return (_verdict_html(result), waveform_figure(waveform, sr, result.features),
                ssm_figure(result.features), gate_figure(result))

    ex_entries = meta.get("examples", [])
    ex_paths = [[str(examples_dir / e["file"])] for e in ex_entries if (examples_dir / e["file"]).exists()]
    ex_labels = [f"{e['title']}  ·  expect {e['expected']}" for e in ex_entries
                 if (examples_dir / e["file"]).exists()]

    with gr.Blocks(title="FALSETTO Studio", analytics_enabled=False) as demo:
        gr.HTML(
            f"""<div id="fs-nav">
              <span class="brand">FALSETTO<span class="dot">.</span></span>
              <span class="links">
                <a href="https://arxiv.org/abs/2509.08283" target="_blank">Paper 1</a>
                <a href="https://arxiv.org/abs/2601.13647" target="_blank">Paper 2</a>
              </span>
            </div>
            <div id="fs-head">
              <h1>Musical structure <em>explorer</em></h1>
              <p class="sub">See what the AI-music detector sees — the self-similarity structure of a track,
              computed for real with MERT.</p>
              <p class="meta">Reproduction of Kim &amp; Go (Segment Transformer). Demo model trained on a
              structural-coherence proxy ({meta.get('n_real', 0)} real recordings + synthetic).</p>
            </div>"""
        )
        with gr.Row(equal_height=False):
            with gr.Column(scale=5):
                audio = gr.Audio(type="filepath", label="Track")
                btn = gr.Button("Analyze structure", variant="primary")
            with gr.Column(scale=6):
                verdict = gr.HTML()
        if ex_paths:
            gr.Examples(examples=ex_paths, inputs=[audio], example_labels=ex_labels or None,
                        label="Examples — real recordings & synthetic clips")
        wave_plot = gr.Plot(label="Waveform + downbeats")
        with gr.Row():
            ssm_plot = gr.Plot(label="Self-similarity matrix")
            gate_plot = gr.Plot(label="Fusion gate per segment")
        gr.Markdown(_ABOUT)

        outs = [verdict, wave_plot, ssm_plot, gate_plot]
        btn.click(analyze, inputs=[audio], outputs=outs)
        audio.change(analyze, inputs=[audio], outputs=outs)
    return demo


def _theme():
    import gradio as gr

    # Committed warm-paper look: force the cream palette in both light and dark so the
    # demo always reads like the editorial (Papers-With-Code) style.
    cream, card, ink, muted, line, slate = "#F6F4EE", "#FBFAF5", "#1C1A16", "#6A675E", "#E4DFD4", "#56658A"
    return gr.themes.Base(
        primary_hue=gr.themes.colors.indigo,
        neutral_hue=gr.themes.colors.stone,
        font=("-apple-system", "Segoe UI", "system-ui", "Roboto", "sans-serif"),
        font_mono=("ui-monospace", "SF Mono", "Menlo", "monospace"),
    ).set(
        body_background_fill=cream, body_background_fill_dark=cream,
        block_background_fill=card, block_background_fill_dark=card,
        body_text_color=ink, body_text_color_dark=ink,
        body_text_color_subdued=muted, body_text_color_subdued_dark=muted,
        border_color_primary=line, border_color_primary_dark=line,
        block_shadow="none", block_border_width="1px",
        button_primary_background_fill=slate, button_primary_background_fill_dark=slate,
        button_primary_background_fill_hover="#45527A", button_primary_background_fill_hover_dark="#45527A",
        button_primary_text_color="#FFFFFF", button_primary_text_color_dark="#FFFFFF",
    )


def launch(assets_dir: str | Path = "demo_assets", device: str = "auto", share: bool = False):
    analyzer, meta = load_analyzer(assets_dir, device)
    demo = build_interface(analyzer, meta, Path(assets_dir) / "examples")
    demo.launch(share=share, theme=_theme(), css=_CSS)
