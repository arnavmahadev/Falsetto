"""FALSETTO Studio: a results showcase.

A Gradio app that presents the reproduced pipeline's output on a fixed set of
example tracks: pick a track and see what the Segment Transformer sees, i.e. the
self-similarity matrix, beat-tracked segments, the fusion gate, and a verdict.
It is a way to look at the results, not a tool for uploading and analyzing your
own audio. Clean, academic styling: flat, no gradients, one slate-blue accent,
matching the results site (docs/index.html).

Honest framing: the classifier is trained on a self-supervised *structural
coherence* task (intact music vs. structure-shuffled music) over real recordings
and synthetic clips, because the labeled AI-music datasets aren't bundled. Every
visualization is computed for real, on each example track, via MERT.
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

# Type stacks mirror the results site. EB Garamond (the site's embedded display
# face) leads the serif stack and falls back to Iowan/Palatino/Georgia when it
# isn't loaded, which is what actually renders here.
_SERIF = '"EB Garamond", "Iowan Old Style", "Palatino Linotype", Palatino, Georgia, serif'
_SANS = '-apple-system, "Segoe UI", system-ui, Roboto, sans-serif'
_MONO = 'ui-monospace, "SF Mono", "SFMono-Regular", Menlo, Consolas, monospace'

# Palette tokens are the canonical results-site values (docs/index.html, light).
# Exposed as CSS custom properties so both the CSS and the inline verdict HTML
# reference one source, the same way the site is built.
_CSS = f"""
:root {{
  --paper:#F4F1E9; --surface:#FCFAF4; --surface-2:#EDE8DB;
  --ink:#1B1915; --muted:#6A6459; --faint:#98917F;
  --line:#E1DBCC; --rule:#C9C1AE;
  --accent:#47567C; --accent-soft:#E7E9F1; --pending:#946517;
}}
.gradio-container {{ max-width: 1000px !important; margin: 0 auto !important; }}
#fs-nav {{ display: flex; align-items: center; gap: 12px; padding: 8px 2px 16px;
    border-bottom: 1px solid var(--line); margin-bottom: 24px; }}
#fs-nav .brand {{ font-family: {_SERIF}; font-weight: 700; font-size: 20px; letter-spacing: .01em; color: var(--ink); }}
#fs-nav .brand .dot {{ color: var(--accent); }}
#fs-nav .links {{ margin-left: auto; display: flex; gap: 8px; }}
#fs-nav .links a {{ font-family: {_MONO}; font-size: 12px; color: var(--muted); border: 1px solid var(--line);
    padding: 6px 13px; border-radius: 7px; text-decoration: none; }}
#fs-nav .links a:hover {{ border-color: var(--rule); color: var(--accent); }}
#fs-head h1 {{ font-family: {_SERIF}; font-weight: 700; font-size: 33px; line-height: 1.1;
    letter-spacing: -.01em; margin: 4px 0 10px; color: var(--ink); }}
#fs-head h1 em {{ font-style: italic; font-weight: 500; color: var(--accent); }}
#fs-head .sub {{ font-family: {_SERIF}; font-size: 18px; color: var(--muted); line-height: 1.5; margin: 0; max-width: 60ch; }}
#fs-head .meta {{ font-family: {_MONO}; font-size: 11px; letter-spacing: .04em; color: var(--faint); margin-top: 12px; }}
button.primary, .primary button, .gr-button {{ background-image: none !important; box-shadow: none !important; }}
footer {{ display: none !important; }}
"""

# The demo commits to the warm-paper light look (and the server-rendered
# matplotlib plots are baked light), so pin the theme to light regardless of the
# viewer's OS/browser preference. Without this, a dark-mode viewer gets the cream
# background but Gradio's dark-mode *text* tokens, i.e. light text on light paper.
# Attached via `demo.load(js=...)` in build_interface (the canonical on-load hook,
# which is baked into the Blocks config and runs on every page load for the local
# launch *and* the Modal/Space mounts alike). Passing js to launch()/mount only
# stores it in the config without running it, so the load event is what fires:
# it redirects once to ?__theme=light, which the frontend reads to force light.
_FORCE_LIGHT_JS = """
() => {
  const u = new URL(window.location.href);
  if (u.searchParams.get('__theme') !== 'light') {
    u.searchParams.set('__theme', 'light');
    window.location.replace(u.href);
  }
}
"""


def load_analyzer(assets_dir: str | Path, device: str = "auto"):
    assets_dir = Path(assets_dir)
    ckpt = assets_dir / "fusion_demo.pt"
    if not ckpt.exists():
        _log.info("no demo assets found, building them first...")
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
    <div style="border:1px solid var(--line);border-radius:10px;padding:20px 22px;background:var(--surface);
                font-family:{_SANS};">
      <div style="font-family:{_MONO};font-size:11px;letter-spacing:.12em;text-transform:uppercase;font-weight:600;color:var(--faint);">Structure profile</div>
      <div style="font-family:{_SERIF};font-size:26px;font-weight:700;color:var(--ink);margin:4px 0 4px;">{result.band}</div>
      <div style="font-size:14px;color:var(--muted);margin-bottom:12px;">structural-coherence estimate
        <span style="background:var(--accent-soft);color:var(--accent);font-weight:700;padding:1px 9px;border-radius:20px;margin-left:2px;">{pct}%</span>
      </div>
      <div style="background:var(--surface-2);border-radius:20px;height:9px;overflow:hidden;margin-bottom:6px;">
        <div style="width:{pct}%;height:100%;background:var(--accent);border-radius:20px;"></div>
      </div>
      <div style="font-size:11.5px;color:var(--faint);margin-bottom:14px;">
        demo model, proxy-trained and illustrative, not a validated AI-vs-real verdict
      </div>
      <div style="font-family:{_MONO};font-size:12.5px;color:var(--muted);">{chips}</div>
    </div>
    """


_ABOUT = """
### How to read this

- **Self-similarity matrix:** segment *i* vs *j* via `exp(−‖eᵢ−eⱼ‖²/s)` on real MERT embeddings.
  Bright off-diagonals repeating at a fixed lag mean the music repeats (verses, choruses);
  a diffuse block that drifts apart means it doesn't. This structure is the paper's core signal.
- **Waveform + downbeats:** beats are detected (Beat&nbsp;This! → librosa fallback) and grouped into
  segments (shaded).
- **Fusion gate:** per segment, the gated multimodal unit's balance between the *content* stream and
  the *structure* (SSM) stream. This is the headline mechanism of the second paper.

**About the structure estimate.** The full detector classifies real vs. AI from these structural
features. Training it needs the labeled datasets (FakeMusicCaps / SONICS / AIME, tens of GB to TB,
not bundled), so the number shown here comes from a small **demo model trained on a proxy**
(structured music vs. drifting/unstructured audio) and is **illustrative only**. The visualizations,
though, are computed for real on each track. Point `falsetto train` at the datasets to get a
validated detector.
"""


def build_interface(analyzer: DemoAnalyzer, meta: dict, examples_dir: Path, analyze_fn=None):
    import gradio as gr

    # `analyze_fn(waveform, sr) -> DemoResult` lets a host wrap just the compute
    # step (e.g. ZeroGPU's @spaces.GPU, which attaches a GPU only while it runs)
    # without this module having to import `spaces`. Defaults to the analyzer.
    run = analyze_fn if analyze_fn is not None else analyzer.analyze

    # This is a results showcase, not an upload tool: viewers choose from a fixed
    # set of example tracks and see the analysis the reproduced pipeline produced
    # for each. There is deliberately no audio-upload input.
    ex_entries = [e for e in meta.get("examples", []) if (examples_dir / e["file"]).exists()]
    choice_to_path = {f"{e['title']}  ·  expect {e['expected']}": str(examples_dir / e["file"])
                      for e in ex_entries}
    choices = list(choice_to_path)
    default_choice = choices[0] if choices else None
    _empty = "<i style='opacity:.6'>Select an example track to see its analysis.</i>"

    def show(choice):
        path = choice_to_path.get(choice)
        if not path:
            return _empty, None, None, None, None
        waveform, sr = load_audio(path, mono=True)
        result = run(waveform, sr)
        return (_verdict_html(result), path,
                waveform_figure(waveform, sr, result.features),
                ssm_figure(result.features), gate_figure(result))

    # In Gradio 6 the theme and CSS are applied by the *caller*, not the Blocks
    # constructor (passing them here is a silent no-op): launch(theme=, css=) for
    # a served app, or mount_gradio_app(theme=, css=) when the Blocks is mounted
    # into FastAPI (e.g. on Modal). So callers own the styling; see `launch`
    # below and deploy/modal/app.py.
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
              <h1>Structure analysis <em>results</em></h1>
              <p class="sub">The reproduced Segment Transformer pipeline, run on a set of example tracks.
              Pick one to see what it found: the self-similarity structure, segmentation, and fusion
              gate, all computed for real with MERT.</p>
              <p class="meta">Reproduction of Kim &amp; Go (Segment Transformer). Demo model trained on a
              structural-coherence proxy ({meta.get('n_real', 0)} real recordings + synthetic).</p>
            </div>"""
        )
        with gr.Row(equal_height=False):
            with gr.Column(scale=5):
                track = gr.Radio(choices=choices, value=default_choice, label="Example tracks")
                player = gr.Audio(value=choice_to_path.get(default_choice), interactive=False,
                                  label="Listen")
            with gr.Column(scale=6):
                verdict = gr.HTML(value=_empty)
        wave_plot = gr.Plot(label="Waveform + downbeats")
        with gr.Row():
            ssm_plot = gr.Plot(label="Self-similarity matrix")
            gate_plot = gr.Plot(label="Fusion gate per segment")
        gr.Markdown(_ABOUT)

        outs = [verdict, player, wave_plot, ssm_plot, gate_plot]
        track.change(show, inputs=[track], outputs=outs)
        # Land on the first track's results already rendered, so the page reads as
        # a showcase rather than an empty form. Computed on load (not at build time)
        # so it also works on ZeroGPU, where GPU calls must run inside a request.
        demo.load(show, inputs=[track], outputs=outs)
        # Force the light theme on load (see _FORCE_LIGHT_JS). Registered on the
        # Blocks so it works for every caller (local launch and FastAPI mounts).
        demo.load(fn=None, inputs=None, outputs=None, js=_FORCE_LIGHT_JS)
    return demo


def _theme():
    import gradio as gr

    # Committed warm-paper look: force the cream palette in both light and dark so the
    # demo always reads like the editorial (Papers-With-Code) style. Values are the
    # canonical results-site tokens (docs/index.html, light).
    cream, card, ink, muted, line, slate = "#F4F1E9", "#FCFAF4", "#1B1915", "#6A6459", "#E1DBCC", "#47567C"
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
        button_primary_background_fill_hover="#38446A", button_primary_background_fill_hover_dark="#38446A",
        button_primary_text_color="#FFFFFF", button_primary_text_color_dark="#FFFFFF",
    )


def launch(assets_dir: str | Path = "demo_assets", device: str = "auto", share: bool = False):
    analyzer, meta = load_analyzer(assets_dir, device)
    demo = build_interface(analyzer, meta, Path(assets_dir) / "examples")
    demo.launch(share=share, theme=_theme(), css=_CSS)
