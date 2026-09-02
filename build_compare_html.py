#!/usr/bin/env python3
"""Build a self-contained HTML page comparing HIGH vs LOW thinking pilot runs."""

from __future__ import annotations

import base64
import html
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).parent
OUT = ROOT / "results" / "review-v15-200.html"
MP3_CACHE = ROOT / "inputs" / "cache" / "mp3"


def audio_data_uri(flac_path: Path) -> str:
    """Base64 data URI of a 64kbps mono MP3 transcode (cached)."""
    MP3_CACHE.mkdir(parents=True, exist_ok=True)
    mp3_path = MP3_CACHE / (flac_path.stem + ".mp3")
    if not mp3_path.is_file() or mp3_path.stat().st_size == 0:
        subprocess.run(
            ["ffmpeg", "-loglevel", "error", "-y", "-i", str(flac_path),
             "-ac", "1", "-b:a", "64k", str(mp3_path)],
            check=True,
        )
    data = base64.b64encode(mp3_path.read_bytes()).decode("ascii")
    return f"data:audio/mpeg;base64,{data}"


def read_jsonl(path: Path) -> dict[str, dict]:
    rows = [json.loads(line) for line in path.open(encoding="utf-8")]
    return {row["id"]: row for row in rows if row.get("status") == "ok"}


def tagged_html(transcript: str, annotations: list[dict]) -> str:
    at: dict[int, list[dict]] = {}
    for item in annotations:
        at.setdefault(item["insert_char_index"], []).append(item)
    parts: list[str] = []
    for index in range(len(transcript) + 1):
        for item in at.get(index, []):
            kind = item["type"]
            bits = [f"confidence {item['confidence']:.2f}",
                    f"placement {item['placement_confidence']:.2f}"]
            if "intensity" in item:
                bits.insert(0, f"intensity {item['intensity']:.2f}")
            title = html.escape(", ".join(bits))
            intensity_html = (
                f'<span class="inten">int {item["intensity"]:.2f}</span>'
                if "intensity" in item else ""
            )
            parts.append(
                f'<span class="tag {kind}" title="{title}">{html.escape(item["label"])}'
                f'<span class="conf">{item["confidence"]:.2f}</span>{intensity_html}</span>'
            )
            for alt in item.get("alternatives") or []:
                parts.append(
                    f'<span class="tag alt {kind}" title="alternative">'
                    f'{html.escape(alt["label"])}'
                    f'<span class="conf">{alt["confidence"]:.2f}</span></span>'
                )
        if index < len(transcript):
            parts.append(html.escape(transcript[index]))
    return "".join(parts)


def usage(row: dict) -> tuple[int, int, int, int]:
    meta = row["usage_metadata"]
    return (
        int(meta.get("promptTokenCount") or 0),
        int(meta.get("thoughtsTokenCount") or 0),
        int(meta.get("candidatesTokenCount") or 0),
        int(meta.get("cachedContentTokenCount") or 0),
    )


def cost(prompt: int, thoughts: int, out: int, cached: int = 0) -> float:
    return (prompt - cached) * 0.75e-6 + cached * 0.075e-6 + (thoughts + out) * 3.75e-6


def stage1_html(markers: list[dict]) -> str:
    parts = []
    for item in markers:
        kind = item["type"]
        extra = f" int {item['intensity']:.2f}" if "intensity" in item else ""
        parts.append(
            f'<span class="tag {kind}" title="audio-only{extra}">'
            f'{html.escape(item["label"])}'
            f'<span class="conf">{item["approx_start_seconds"]:.1f}s · {item["confidence"]:.2f}</span></span>'
        )
    return " ".join(parts)


def main() -> None:
    runs = {
        "v15": read_jsonl(ROOT / "results" / "pilot200-v15.jsonl"),
    }
    css_class = {"v15": "vlatest"}
    manifest = {row["id"]: row for row in map(json.loads, (ROOT / "inputs" / "manifest.jsonl").open(encoding="utf-8"))}
    ids = [row_id for row_id in manifest if all(row_id in rows for rows in runs.values())]

    totals = {name: [0, 0, 0, 0] for name in runs}
    cards: list[str] = []
    for position, row_id in enumerate(ids, start=1):
        audio_src = audio_data_uri(ROOT / "inputs" / manifest[row_id]["audio"])
        first = next(iter(runs.values()))[row_id]
        rows_html = ""
        for name, rows in runs.items():
            row = rows[row_id]
            use = usage(row)
            for i in range(4):
                totals[name][i] += use[i]
            stage1_block = ""
            if row.get("stage1_markers"):
                stage1_block = (
                    '<div class="stage1"><span class="stage1label">第一阶段（纯音频）</span>'
                    f'{stage1_html(row["stage1_markers"])}</div>'
                )
            thought_block = ""
            if row.get("thought_summary"):
                thought_block = (
                    '<details class="thoughts"><summary>思考摘要</summary>'
                    f'<pre>{html.escape(row["thought_summary"])}</pre></details>'
                )
            rows_html += f"""
            <div class="run">
              <div class="runhead"><span class="level {css_class[name]}">MEDIUM {name}</span>
                <span class="tok">prompt {use[0]:,}{f" (缓存 {use[3]:,})" if use[3] else ""} · 思考 {use[1]:,} · 输出 {use[2]:,} tok · ${cost(*use):.4f}</span></div>
              {stage1_block}
              <div class="tagged">{tagged_html(row["official_transcript"], row["annotations"])}</div>
              {thought_block}
            </div>"""
        cards.append(f"""
        <div class="card">
          <div class="cardhead">
            <span class="idx">#{position}</span>
            <span class="lang">{html.escape(str(manifest[row_id]["language"]))}</span>
            <code class="rid">{html.escape(row_id)}</code>
            <span class="dur">wer/cer {manifest[row_id]["asr_error_rate"]:.2f} · {first["duration"]:.1f}s</span>
          </div>
          <audio controls preload="none" src="{audio_src}"></audio>
          <div class="transcript">{html.escape(first["official_transcript"])}</div>
          {rows_html}
        </div>""")

    def fmt_total(name: str) -> str:
        p, t, o, c = totals[name]
        cached = f"（其中缓存 {c:,}）" if c else ""
        return (f"prompt {p:,}{cached} · 思考 {t:,} · 输出 {o:,} · 合计 {p + t + o:,} tok · "
                f"${cost(p, t, o, c):.4f}")

    page = f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>v15 标注（200 条）</title>
<style>
  :root {{ color-scheme: light; }}
  body {{ font-family: -apple-system, "PingFang SC", "Helvetica Neue", sans-serif;
         margin: 0; background: #f4f5f7; color: #1c1e21; }}
  header {{ background: #fff; border-bottom: 1px solid #e3e5e8; padding: 20px 28px; }}
  h1 {{ font-size: 20px; margin: 0 0 10px; }}
  .summary {{ font-size: 13px; color: #444; line-height: 1.8; }}
  .summary b {{ display: inline-block; width: 68px; }}
  main {{ max-width: 980px; margin: 24px auto; padding: 0 16px; }}
  .card {{ background: #fff; border: 1px solid #e3e5e8; border-radius: 10px;
          padding: 16px 20px; margin-bottom: 18px; }}
  .cardhead {{ display: flex; align-items: center; gap: 10px; margin-bottom: 10px; }}
  .idx {{ font-weight: 700; color: #555; }}
  .lang {{ font-size: 11px; font-weight: 700; text-transform: uppercase; background: #eef1f4;
          color: #556; padding: 2px 7px; border-radius: 999px; }}
  .rid {{ font-size: 11px; color: #888; overflow-wrap: anywhere; }}
  .dur {{ margin-left: auto; font-size: 12px; color: #666; white-space: nowrap; }}
  audio {{ width: 100%; height: 34px; margin-bottom: 10px; }}
  .transcript {{ font-size: 13px; color: #999; margin-bottom: 12px; line-height: 1.5; }}
  .run {{ border-top: 1px dashed #e3e5e8; padding: 10px 0 4px; }}
  .runhead {{ display: flex; align-items: center; gap: 10px; margin-bottom: 6px; }}
  .level {{ font-size: 11px; font-weight: 700; padding: 2px 8px; border-radius: 999px; }}
  .level.medium {{ background: #fdeee0; color: #c2570c; }}
  .level.twostage {{ background: #e2f4ea; color: #14804a; }}
  .level.vlatest {{ background: #e8e5fb; color: #5b3fc4; }}
  .stage1 {{ font-size: 12px; line-height: 2; margin-bottom: 4px; opacity: 0.85; }}
  .stage1label {{ font-size: 11px; color: #999; margin-right: 6px; }}
  .tok {{ font-size: 11px; color: #999; }}
  .tagged {{ font-size: 15px; line-height: 2.1; }}
  .tag {{ font-size: 11px; font-weight: 600; padding: 2px 7px; border-radius: 999px;
         margin: 0 2px; white-space: nowrap; cursor: default; }}
  .tag.emotion {{ background: #e8f0fe; color: #1a56c4; border: 1px solid #c6d8fb; }}
  .tag.event {{ background: #fef3e2; color: #b45d09; border: 1px solid #f9ddb2; }}
  .tag .conf {{ font-weight: 400; opacity: 0.65; margin-left: 4px; font-size: 10px; }}
  .tag.alt {{ opacity: 0.55; border-style: dashed; font-weight: 500; }}
  .tag .inten {{ font-weight: 400; margin-left: 5px; padding-left: 5px; font-size: 10px;
                border-left: 1px solid currentColor; opacity: 0.8; }}
  .thoughts {{ margin-top: 8px; font-size: 12px; color: #666; }}
  .thoughts summary {{ cursor: pointer; color: #888; font-size: 11px; }}
  .thoughts pre {{ white-space: pre-wrap; background: #fafbfc; border: 1px solid #eef0f2;
                  border-radius: 6px; padding: 10px 12px; line-height: 1.6; }}
</style>
</head>
<body>
<header>
  <h1>v15 标注（全部 200 条，前 100 条与此前 pilot 相同，MEDIUM thinking）</h1>
  <div class="summary">
    <div><b>v15</b>{fmt_total("v15")}</div>
    <div style="margin-top:6px">闭集词表 + 置信度校准 + 备选标签（虚线框，主标签与备选置信度总和 ≤ 1）+ 括号等脚本约定不作为声学证据 + 不切词。
      每张卡片底部可展开思考摘要。
      标签内的小数字是 confidence；悬停可看 placement_confidence。
      <span class="tag emotion">情绪<span class="conf">0.92</span></span>
      <span class="tag event">事件<span class="conf">0.88</span></span></div>
  </div>
</header>
<main>{"".join(cards)}</main>
</body>
</html>"""
    OUT.write_text(page, encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB, {len(ids)} items)")


if __name__ == "__main__":
    main()
