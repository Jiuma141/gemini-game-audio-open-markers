#!/usr/bin/env python3
"""Run a resumable 100-item Gemini open emotion/event marker pilot."""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import getpass
import hashlib
import json
import mimetypes
import os
import random
import re
import threading
import time
import traceback
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any


MODEL = "gemini-3.7-flash"
PROMPT_VERSION = "game-audio-open-marker-sequence-v5"
THINKING_LEVEL = "LOW"
TEMPERATURE = 0.0
LABEL_RE = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")

BASE_PROMPT = """You are an expert annotator of acted game voice audio. Listen to the complete audio. Use vocal delivery as the primary evidence and the official transcript as the exact text/index reference.

Return a JSON array of emotion-state markers and audible human paralinguistic-event markers, in transcript order.

Emotion markers:
- The emotion taxonomy is open: do not force a fixed seven-class taxonomy and do not limit the number of possible emotion label types.
- Use a concise, reusable lowercase English snake_case label, such as happy, weary_resignation, restrained_anger, nervous_excitement, or tender_sadness. These examples are not an exhaustive taxonomy.
- The first emotion marker must have insert_char_index 0.
- An emotion marker starts a state that continues until the next emotion marker or the end of the utterance.
- Add another emotion marker only when the dominant audible emotion clearly changes. Do not split merely because a new clause begins, and do not repeat consecutive identical labels.
- intensity is the audible strength of the emotion in [0,1]. confidence is confidence that the emotion label is correct. placement_confidence is confidence that the emotion transition begins at that transcript boundary.

Event markers:
- The event taxonomy is open: do not force a fixed event list and do not limit the number of possible event label types.
- Detect only clearly audible human vocal or respiratory paralinguistic events. Do not label ordinary speech, inaudible/ordinary breathing, background music, ambience, or non-human sound effects.
- Use a concise, reusable lowercase English snake_case label, such as sighing, laughter, sobbing, voice_break, gasping, coughing, whispering, or throat_clearing. These examples are not an exhaustive taxonomy.
- confidence is confidence that the event exists and its label is correct. placement_confidence is confidence that its tag belongs at that transcript boundary.

Indexing and output rules:
- insert_char_index is a zero-based Unicode character boundary in the ORIGINAL official transcript: 0 is before its first character and transcript length is after its final character.
- Multiple markers may share a boundary. At the same boundary, place emotion markers before event markers.
- Do not return timestamps, descriptions, keywords, valence/arousal/dominance, the transcript, explanations, or any fields outside the schema.
- Return JSON only."""

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def dump(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", exclude_none=True)
    return str(value)


def numbered_characters(text: str) -> str:
    return " | ".join(
        f"{index}={json.dumps(char, ensure_ascii=False)}" for index, char in enumerate(text)
    ) + f" | {len(text)}=<END>"


def build_prompt(row: dict[str, Any]) -> str:
    payload = {
        "language": row["language"],
        "duration_seconds": row["duration"],
        "official_transcript": row["text"],
        "character_index_table": numbered_characters(row["text"]),
    }
    return BASE_PROMPT + "\n\nINPUT:\n" + json.dumps(
        payload, ensure_ascii=False, separators=(",", ":")
    )


def response_schema() -> dict[str, Any]:
    return {
        "type": "array",
        "minItems": 1,
        "items": {
            "type": "object",
            "properties": {
                "type": {"type": "string", "enum": ["emotion", "event"]},
                "label": {"type": "string"},
                "intensity": {"type": "number", "minimum": 0, "maximum": 1},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "insert_char_index": {"type": "integer", "minimum": 0},
                "placement_confidence": {"type": "number", "minimum": 0, "maximum": 1},
            },
            "required": [
                "type", "label", "confidence", "insert_char_index", "placement_confidence"
            ],
            "additionalProperties": False,
        },
    }


def config_hash() -> str:
    raw = json.dumps(
        {
            "model": MODEL,
            "prompt_version": PROMPT_VERSION,
            "prompt": BASE_PROMPT,
            "schema": response_schema(),
            "thinking_level": THINKING_LEVEL,
            "temperature": TEMPERATURE,
        },
        ensure_ascii=False,
        sort_keys=True,
    ).encode()
    return hashlib.sha256(raw).hexdigest()[:16]


def load_rows(manifest: Path, text_manifest: Path) -> list[dict[str, Any]]:
    texts = {str(row["id"]): str(row["text_plain"]) for row in read_jsonl(text_manifest)}
    rows = []
    for source in read_jsonl(manifest):
        row = dict(source)
        row_id = str(row["id"])
        if row_id not in texts or not texts[row_id]:
            raise ValueError(f"missing reference text for {row_id}")
        audio_path = Path(str(row["audio"]))
        if not audio_path.is_absolute():
            audio_path = (manifest.parent / audio_path).resolve()
        if not audio_path.is_file() or audio_path.stat().st_size == 0:
            raise FileNotFoundError(audio_path)
        row["audio_path_local"] = str(audio_path)
        row["text"] = texts[row_id]
        rows.append(row)
    if len(rows) != 100 or len({str(row["id"]) for row in rows}) != 100:
        raise ValueError(f"expected exactly 100 unique rows, got {len(rows)}")
    return rows


def post_gemini(body: dict[str, Any]) -> dict[str, Any]:
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY is not set")
    request = urllib.request.Request(
        f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent",
        data=json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": key,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=900) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Gemini HTTP {exc.code}: {detail[:4000]}") from exc


def validate_payload(payload: Any, transcript: str) -> list[dict[str, Any]]:
    if not isinstance(payload, list) or not payload:
        raise ValueError("response must be a non-empty JSON array")
    allowed = {
        "type", "label", "intensity", "confidence",
        "insert_char_index", "placement_confidence",
    }
    cleaned: list[dict[str, Any]] = []
    for position, source in enumerate(payload):
        if not isinstance(source, dict):
            raise ValueError(f"item {position} is not an object")
        extras = set(source) - allowed
        if extras:
            raise ValueError(f"item {position} has extra fields: {sorted(extras)}")
        marker_type = str(source.get("type") or "")
        if marker_type not in {"emotion", "event"}:
            raise ValueError(f"item {position} has invalid type: {marker_type!r}")
        label = str(source.get("label") or "").strip()
        if not LABEL_RE.fullmatch(label):
            raise ValueError(f"item {position} has invalid open label: {label!r}")
        char_index = source.get("insert_char_index")
        if isinstance(char_index, bool) or not isinstance(char_index, int):
            raise ValueError(f"item {position} has non-integer index: {char_index!r}")
        if not 0 <= char_index <= len(transcript):
            raise ValueError(
                f"item {position} index {char_index} outside transcript length {len(transcript)}"
            )
        confidence = float(source["confidence"])
        placement_confidence = float(source["placement_confidence"])
        if not 0 <= confidence <= 1 or not 0 <= placement_confidence <= 1:
            raise ValueError(f"item {position} confidence outside [0,1]")
        item: dict[str, Any] = {
            "type": marker_type,
            "label": label,
        }
        if marker_type == "emotion":
            if "intensity" not in source:
                raise ValueError(f"emotion item {position} is missing intensity")
            intensity = float(source["intensity"])
            if not 0 <= intensity <= 1:
                raise ValueError(f"emotion item {position} intensity outside [0,1]")
            item["intensity"] = intensity
        elif "intensity" in source:
            raise ValueError(f"event item {position} must not contain intensity")
        item.update({
            "confidence": confidence,
            "insert_char_index": char_index,
            "placement_confidence": placement_confidence,
        })
        cleaned.append(item)

    type_order = {"emotion": 0, "event": 1}
    cleaned.sort(key=lambda item: (item["insert_char_index"], type_order[item["type"]]))
    emotions = [item for item in cleaned if item["type"] == "emotion"]
    if not emotions or emotions[0]["insert_char_index"] != 0:
        raise ValueError("the first emotion marker must begin at character index 0")
    if len({item["insert_char_index"] for item in emotions}) != len(emotions):
        raise ValueError("multiple emotion markers share one character boundary")
    for previous, current in zip(emotions, emotions[1:]):
        if previous["label"] == current["label"]:
            raise ValueError("consecutive emotion markers have identical labels")
    return cleaned


def render_text(transcript: str, annotations: list[dict[str, Any]]) -> str:
    at: dict[int, list[dict[str, Any]]] = {}
    for item in annotations:
        at.setdefault(item["insert_char_index"], []).append(item)
    parts: list[str] = []
    for index in range(len(transcript) + 1):
        for item in at.get(index, []):
            parts.append(f"[{item['label']}]")
        if index < len(transcript):
            parts.append(transcript[index])
    return "".join(parts)


def retryable(exc: Exception) -> bool:
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if code in {408, 409, 429, 500, 502, 503, 504}:
        return True
    message = str(exc).lower()
    return isinstance(exc, (ValueError, json.JSONDecodeError)) or any(
        token in message
        for token in ("429", "quota", "timeout", "temporar", "unavailable", "reset")
    )


def infer(row: dict[str, Any], max_attempts: int) -> dict[str, Any]:
    started = time.monotonic()
    wrapper: dict[str, Any] = {
        "schema_version": "game_audio_open_marker_result_v1",
        "id": str(row["id"]),
        "dataset": "Genshin",
        "language": row["language"],
        "speaker_id": row.get("speaker_id"),
        "audio": row["audio"],
        "duration": row["duration"],
        "official_transcript": row["text"],
        "model": MODEL,
        "prompt_version": PROMPT_VERSION,
        "config_hash": config_hash(),
        "thinking_level": THINKING_LEVEL,
        "temperature": TEMPERATURE,
        "status": "error",
    }
    audio_path = Path(row["audio_path_local"])
    audio_bytes = audio_path.read_bytes()
    mime_type = str(
        row.get("mime_type") or mimetypes.guess_type(audio_path.name)[0] or "audio/flac"
    )
    prompt = build_prompt(row)
    history = []
    for attempt in range(1, max_attempts + 1):
        try:
            api_response = post_gemini(
                {
                    "contents": [{
                        "role": "user",
                        "parts": [
                            {
                                "inlineData": {
                                    "mimeType": mime_type,
                                    "data": base64.b64encode(audio_bytes).decode("ascii"),
                                }
                            },
                            {"text": prompt},
                        ],
                    }],
                    "generationConfig": {
                        "temperature": TEMPERATURE,
                        "responseMimeType": "application/json",
                        "responseJsonSchema": response_schema(),
                        "thinkingConfig": {"thinkingLevel": THINKING_LEVEL},
                    },
                }
            )
            candidates = api_response.get("candidates") or []
            if not candidates:
                raise ValueError(f"response has no candidates: {api_response}")
            parts = ((candidates[0].get("content") or {}).get("parts") or [])
            raw_response_text = "".join(str(part.get("text") or "") for part in parts)
            if not raw_response_text:
                raise ValueError(f"response has no JSON text: {api_response}")
            payload = json.loads(raw_response_text)
            annotations = validate_payload(payload, row["text"])
            wrapper.update({
                "status": "ok",
                "raw_response_text": raw_response_text,
                "raw_model_output": payload,
                "annotations": annotations,
                "tagged_text": render_text(row["text"], annotations),
                "usage_metadata": api_response.get("usageMetadata"),
                "response_id": api_response.get("responseId"),
                "api_model_version": api_response.get("modelVersion"),
                "finish_reason": candidates[0].get("finishReason"),
                "attempts": attempt,
                "elapsed_seconds": time.monotonic() - started,
                "completed_at_unix": time.time(),
            })
            return wrapper
        except Exception as exc:
            history.append({
                "attempt": attempt,
                "type": type(exc).__name__,
                "message": str(exc)[:2000],
            })
            if attempt >= max_attempts or not retryable(exc):
                break
            time.sleep(min(30.0, 2 ** (attempt - 1) + random.random()))
    wrapper.update({
        "attempts": len(history),
        "attempt_history": history,
        "error": history[-1] if history else {"message": "unknown error"},
        "traceback": traceback.format_exc(limit=8),
        "elapsed_seconds": time.monotonic() - started,
        "completed_at_unix": time.time(),
    })
    return wrapper


def read_completed(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {
        str(row["id"])
        for row in read_jsonl(path)
        if row.get("status") == "ok" and row.get("config_hash") == config_hash()
    }


def write_report(output: Path, report_path: Path, total_rows: int) -> None:
    rows = read_jsonl(output) if output.exists() else []
    current = [row for row in rows if row.get("config_hash") == config_hash()]
    by_id = {str(row["id"]): row for row in current}
    final_rows = list(by_id.values())
    ok = [row for row in final_rows if row.get("status") == "ok"]
    errors = [row for row in final_rows if row.get("status") != "ok"]
    prompt_tokens = candidate_tokens = thought_tokens = total_tokens = 0
    for row in ok:
        usage = row.get("usage_metadata") or {}
        prompt_tokens += int(usage.get("prompt_token_count") or usage.get("promptTokenCount") or 0)
        candidate_tokens += int(
            usage.get("candidates_token_count") or usage.get("candidatesTokenCount") or 0
        )
        thought_tokens += int(
            usage.get("thoughts_token_count") or usage.get("thoughtsTokenCount") or 0
        )
        total_tokens += int(usage.get("total_token_count") or usage.get("totalTokenCount") or 0)
    emotions = [
        marker
        for row in ok
        for marker in row["annotations"]
        if marker["type"] == "emotion"
    ]
    events = [
        marker
        for row in ok
        for marker in row["annotations"]
        if marker["type"] == "event"
    ]
    estimated_standard_cost = prompt_tokens * 0.75e-6 + (
        candidate_tokens + thought_tokens
    ) * 3.75e-6
    report = {
        "schema_version": "game_audio_open_marker_report_v1",
        "model": MODEL,
        "prompt_version": PROMPT_VERSION,
        "config_hash": config_hash(),
        "thinking_level": THINKING_LEVEL,
        "temperature": TEMPERATURE,
        "target_rows": total_rows,
        "result_rows": len(final_rows),
        "successful_rows": len(ok),
        "error_rows": len(errors),
        "prompt_tokens": prompt_tokens,
        "candidate_tokens": candidate_tokens,
        "thought_tokens": thought_tokens,
        "total_tokens_reported": total_tokens,
        "estimated_standard_cost_usd": estimated_standard_cost,
        "emotion_markers": len(emotions),
        "event_markers": len(events),
        "multi_emotion_rows": sum(
            sum(marker["type"] == "emotion" for marker in row["annotations"]) > 1
            for row in ok
        ),
        "emotion_label_counts": Counter(marker["label"] for marker in emotions),
        "event_label_counts": Counter(marker["label"] for marker in events),
        "generated_at_unix": time.time(),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--text-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-attempts", type=int, default=4)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    if not os.environ.get("GEMINI_API_KEY"):
        os.environ["GEMINI_API_KEY"] = getpass.getpass("Gemini API key: ")
    if not os.environ["GEMINI_API_KEY"]:
        raise RuntimeError("empty Gemini API key")

    rows = load_rows(args.manifest.resolve(), args.text_manifest.resolve())
    if args.limit is not None:
        rows = rows[: args.limit]
    completed = read_completed(args.output)
    pending = [row for row in rows if str(row["id"]) not in completed]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(json.dumps({
        "config_hash": config_hash(),
        "selected": len(rows),
        "completed": len(completed & {str(row['id']) for row in rows}),
        "pending": len(pending),
    }), flush=True)

    ok = errors = 0
    with args.output.open("a", encoding="utf-8", buffering=1) as handle:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(infer, row, args.max_attempts): row for row in pending}
            for index, future in enumerate(concurrent.futures.as_completed(futures), start=1):
                result = future.result()
                handle.write(json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n")
                handle.flush()
                if result["status"] == "ok":
                    ok += 1
                else:
                    errors += 1
                print(json.dumps({
                    "done_this_run": index,
                    "ok_this_run": ok,
                    "errors_this_run": errors,
                    "id": result["id"],
                    "status": result["status"],
                }, ensure_ascii=False), flush=True)
    write_report(args.output, args.report, len(rows))
    print(json.dumps({"finished": True, "ok": ok, "errors": errors}), flush=True)


if __name__ == "__main__":
    main()
