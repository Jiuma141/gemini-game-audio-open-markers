#!/usr/bin/env python3
"""Run a resumable Gemini closed-vocabulary emotion/event marker pilot."""

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
import threading
import time
import traceback
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any


MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.7-flash")
PROMPT_VERSION = "game-audio-closed-marker-sequence-v17"
THINKING_LEVEL = os.environ.get("GEMINI_THINKING_LEVEL", "MEDIUM")
TEMPERATURE = 0.0
API_ROOT = "https://generativelanguage.googleapis.com/v1beta"
UPLOAD_ROOT = "https://generativelanguage.googleapis.com/upload/v1beta"
DOWNLOAD_ROOT = "https://generativelanguage.googleapis.com/download/v1beta"
# Explicit context cache holding BASE_PROMPT; every request then only sends the
# per-clip INPUT block and the audio. Storage is billed per token-hour. Sync
# runs finish in minutes; batch jobs may take up to 24h, so their cache must
# outlive the job's turnaround SLO.
CACHE_TTL_SECONDS_SYNC = int(os.environ.get("GEMINI_CACHE_TTL_SECONDS", "7200"))
CACHE_TTL_SECONDS_BATCH = int(os.environ.get("GEMINI_CACHE_TTL_SECONDS", str(30 * 3600)))
# Pricing assumptions (USD per token) used for the report's cost estimate.
PRICE_INPUT = 0.75e-6
PRICE_CACHED_INPUT = 0.075e-6
PRICE_OUTPUT = 3.75e-6
PRICE_CACHE_STORAGE_PER_TOKEN_HOUR = 1.0e-6
# Batch API bills uncached input and output at half price; cached tokens keep
# the (larger) caching discount instead of stacking.
BATCH_DISCOUNT = 0.5

# Closed label vocabulary: the "english" column of tts_bracket_emotion_enword_180d.csv,
# split into emotion/delivery-state labels and paralinguistic-event labels.
EMOTION_LABELS = (
    "happy", "calm", "angry", "excited", "sad", "horny", "fear", "serious",
    "confused", "surprised", "whisper", "shouting", "gentle", "neutral",
    "breathy", "thoughtful", "screaming", "love", "pain", "sensual",
    "tired", "seductive", "frustrated", "energetic", "flirty", "anxious",
    "annoyed", "confident", "playful", "curious", "worried", "soft", "shy",
    "proud", "nervous", "dramatic", "smug", "cold", "sarcastic",
    "determined", "urgent", "funny", "warm", "sleepy", "mocking",
    "begging", "embarrassed", "authoritative", "teasing", "hope", "weak",
    "disappointed", "desperate", "moved", "mysterious", "quiet",
    "romantic", "passionate", "caring", "intense", "grateful",
    "empathetic", "hesitant", "disgusted", "encouraging", "threatening",
    "skeptical", "relieved", "sincere", "crazy", "bored", "evil",
    "reassuring", "nostalgic", "guilty", "drunk", "cute", "sweet",
    "satisfied", "polite", "envious", "obsessed", "tender", "sick",
    "tsundere", "sassy", "apologetic",
)
EVENT_LABELS = (
    "pause", "moaning", "laugh", "crying", "sigh", "giggle", "chuckles",
    "panting", "gasp", "cough", "clear_throat", "groan", "whimpering",
    "grunt", "yawn", "stutter", "sniffles", "growl",
)
LABEL_SETS = {"emotion": frozenset(EMOTION_LABELS), "event": frozenset(EVENT_LABELS)}
MAX_ALTERNATIVES = 3
# Primary + alternative confidences are shares of one probability mass; allow a
# little rounding slack before treating the response as invalid.
PROBABILITY_MASS_TOLERANCE = 0.05

BASE_PROMPT = """You are an expert annotator of acted game voice audio. Listen to the complete audio. Vocal delivery is the ONLY evidence for labels; the official transcript is ONLY an exact text/index reference for positioning markers.

Return a JSON array of emotion-state markers and audible vocal-event markers, in transcript order. One utterance may contain any number of markers, and a marker may sit at any character boundary of the transcript.

Closed vocabulary (hard constraint):
- Every emotion label MUST be copied verbatim from ALLOWED_EMOTION_LABELS, and every event label MUST be copied verbatim from ALLOWED_EVENT_LABELS.
- Never invent, translate, pluralize, combine, or reword labels, and never use an emotion label on an event marker or vice versa.
- First decide from the audio what you actually hear, then map it to the closest allowed label. If more than one allowed label is genuinely plausible, give the best one as label and list the others in alternatives (see calibration rules below).

Do not let the text bias you:
- The transcript's wording and punctuation are NOT evidence of emotions, delivery, or events; only the acoustics are.
- Delivery labels such as whisper, shouting, screaming, breathy, soft, quiet must be decided ONLY from loudness, phonation, and breathiness that you hear, never from the content of the speech (e.g. "Shh" or exclamation marks do not imply whispering or shouting).
- Script conventions are not acoustic evidence either: parentheses or brackets marking an inner monologue or aside do NOT imply whispering, softness, or any particular emotion. A parenthesized line spoken at normal voice with a smug or sarcastic tone is smug or sarcastic, not whisper.

Emotion markers:
- Emotion labels cover both emotions (e.g. happy, angry, nervous) and audible vocal-delivery states (e.g. whisper, shouting, breathy, soft).
- Choose the label that best matches the audibly dominant emotion or delivery. If delivery is plain and no clear emotion is audible, use "neutral".
- Attitude labels (e.g. sarcastic, mocking, teasing, playful, smug, threatening, evil, skeptical, flirty) require the attitude to be audible in the voice itself: exaggerated or sing-song intonation, drawn-out syllables, a sneer or laugh in the voice, menacing low pitch, and so on. Words alone never justify them. Test: if the same voice read unrelated neutral words, would you still hear that attitude? If not, label the plain delivery you actually hear (serious, calm, neutral, polite, surprised, hesitant, disappointed…) even when the sentence itself is sarcastic, threatening, or doubtful.
- The first emotion marker must have insert_char_index 0.
- An emotion marker starts a state that continues until the next emotion marker or the end of the utterance.
- Add another emotion marker only when the dominant audible emotion or delivery clearly changes. Do not split merely because a new clause begins, and do not repeat consecutive identical labels.
- confidence and placement_confidence follow the calibration rules below.

Event markers:
- Detect only clearly audible human vocal or respiratory paralinguistic events. Do not label ordinary speech, inaudible/ordinary breathing, background music, ambience, or non-human sound effects.
- "pause" is ONLY for a long, deliberate silent gap in speech of roughly 0.5 seconds or more. Short beats between words or clauses, ordinary breaths, and brief gaps at punctuation are NOT pauses. Punctuation is not evidence: commas, periods, ellipses, and dashes often carry no audible pause, and real pauses can occur mid-clause without punctuation. When unsure whether a gap is long enough, omit the pause marker.
- Breathing: plain breaths (inhales or exhales between phrases, before speaking, or at the end) are NOT events and have no label; leave them unmarked. Only expressive breathing counts, and only when it is unmistakable: "sigh" is a long, expressive (often voiced) exhale; "panting" is repeated rapid breathing; "gasp" is a sudden sharp intake caused by shock or fear. Sustained breathy delivery is the emotion label "breathy", not an event.
- Vocal events such as laugh, chuckles, giggle, grunt, groan, and whimpering must likewise be audible as sounds in their own right. Do not add them because the wording is funny, angry, or frightened.
- If a real event has no close match in ALLOWED_EVENT_LABELS, omit that event rather than forcing a wrong label.
- Place each event marker at the transcript boundary where the event is heard; events may repeat wherever they actually occur.
- confidence is the probability that the event exists AND its label is correct; placement_confidence follows the calibration rules below.

Confidence and alternatives:
- confidence is a PROBABILITY: the chance that a careful expert listener would choose this exact label for this marker. It is not a quality score and must NOT default to 0.8-0.9.
- Calibrate against these anchors: 0.9 or higher only when no other allowed label is plausible; 0.6-0.8 when one alternative is also plausible; 0.4-0.6 when two or more alternatives are comparably plausible; below 0.4 when you are mostly guessing (for events, also when you are unsure the event exists at all).
- Whenever another allowed label of the same type could reasonably describe the same marker, list it in alternatives (up to 3, most likely first), each with its own confidence. Never silently pick one label when the audio is ambiguous.
- The primary label and its alternatives share ONE probability mass: their confidences must sum to at most 1, and the primary confidence must be the highest. Leave alternatives empty only when the primary label is clearly the sole fit.
- placement_confidence is likewise the probability that the marker belongs at exactly that boundary rather than a neighboring one; lower it when a transition or event straddles several characters.
- Across a whole utterance some markers should normally fall well below 0.8; an output in which every confidence is 0.8 or higher is almost certainly overconfident.

Indexing and output rules:
- insert_char_index is a zero-based Unicode character boundary in the ORIGINAL official transcript: 0 is before its first character and transcript length is after its final character.
- In space-delimited scripts (e.g. English), a marker must NEVER split a word: place it at a word boundary, next to whitespace or punctuation. In unspaced scripts (e.g. Chinese, Japanese), any character boundary is acceptable.
- Multiple markers may share a boundary. At the same boundary, place emotion markers before event markers.
- Do not return timestamps, descriptions, keywords, valence/arousal/dominance, the transcript, explanations, or any fields outside the schema.
- Before answering, verify every label appears verbatim in its allowed list.
- Return JSON only.

ALLOWED_EMOTION_LABELS: """ + ", ".join(EMOTION_LABELS) + """
ALLOWED_EVENT_LABELS: """ + ", ".join(EVENT_LABELS)

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


def build_input_block(row: dict[str, Any]) -> str:
    payload = {
        "language": row["language"],
        "duration_seconds": row["duration"],
        "official_transcript": row["text"],
        "character_index_table": numbered_characters(row["text"]),
    }
    return "INPUT:\n" + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def build_prompt(row: dict[str, Any]) -> str:
    return BASE_PROMPT + "\n\n" + build_input_block(row)


def response_schema() -> dict[str, Any]:
    # The Gemini API rejects schemas whose enums exceed ~88 total values, so the
    # emotion labels cannot be enum-constrained; they are enforced by the
    # prompt vocabulary plus validate_payload (invalid labels trigger a retry).
    # The event labels fit and stay enum-constrained.
    unit = {"type": "number", "minimum": 0, "maximum": 1}

    def alternatives(label_schema: dict[str, Any]) -> dict[str, Any]:
        return {
            "type": "array",
            "maxItems": MAX_ALTERNATIVES,
            "items": {
                "type": "object",
                "properties": {"label": label_schema, "confidence": unit},
                "required": ["label", "confidence"],
                "additionalProperties": False,
            },
        }

    event_label = {"type": "string", "enum": list(EVENT_LABELS)}
    return {
        "type": "array",
        "minItems": 1,
        "items": {
            "anyOf": [
                {
                    "type": "object",
                    "properties": {
                        "type": {"type": "string", "enum": ["emotion"]},
                        "label": {"type": "string"},
                        "alternatives": alternatives({"type": "string"}),
                        "confidence": unit,
                        "insert_char_index": {"type": "integer", "minimum": 0},
                        "placement_confidence": unit,
                    },
                    "required": [
                        "type", "label", "alternatives", "confidence",
                        "insert_char_index", "placement_confidence",
                    ],
                    "additionalProperties": False,
                },
                {
                    "type": "object",
                    "properties": {
                        "type": {"type": "string", "enum": ["event"]},
                        "label": event_label,
                        "alternatives": alternatives(event_label),
                        "confidence": unit,
                        "insert_char_index": {"type": "integer", "minimum": 0},
                        "placement_confidence": unit,
                    },
                    "required": [
                        "type", "label", "alternatives", "confidence",
                        "insert_char_index", "placement_confidence",
                    ],
                    "additionalProperties": False,
                },
            ],
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
    if not rows:
        raise ValueError("manifest contains no rows")
    if len({str(row["id"]) for row in rows}) != len(rows):
        raise ValueError("manifest contains duplicate ids")
    return rows


class GeminiHTTPError(RuntimeError):
    def __init__(self, code: int, detail: str):
        super().__init__(f"Gemini HTTP {code}: {detail[:4000]}")
        self.code = code
        self.detail = detail


def gemini_request(method: str, path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY is not set")
    data = None
    if body is not None:
        data = json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    request = urllib.request.Request(
        f"{API_ROOT}/{path}",
        data=data,
        headers={"Content-Type": "application/json", "x-goog-api-key": key},
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=900) as response:
            raw = response.read().decode("utf-8")
            return json.loads(raw) if raw.strip() else {}
    except urllib.error.HTTPError as exc:
        raise GeminiHTTPError(exc.code, exc.read().decode("utf-8", errors="replace")) from exc


def post_gemini(body: dict[str, Any]) -> dict[str, Any]:
    return gemini_request("POST", f"models/{MODEL}:generateContent", body)


class PromptCache:
    """Explicit Gemini context cache for the fixed BASE_PROMPT prefix.

    The cache is created lazily on first use and recreated if the API reports
    it missing (expired). `enabled=False` makes every request inline the full
    prompt instead.
    """

    def __init__(self, enabled: bool, ttl_seconds: int = CACHE_TTL_SECONDS_SYNC):
        self.enabled = enabled
        self.ttl_seconds = ttl_seconds
        self._name: str | None = None
        self._lock = threading.Lock()
        self.created_tokens = 0
        self.creations = 0

    def adopt(self, name: str) -> None:
        """Reuse a cache created by an earlier (interrupted) run."""
        with self._lock:
            self._name = name

    def name(self) -> str | None:
        if not self.enabled:
            return None
        with self._lock:
            if self._name is None:
                self._name = self._create()
            return self._name

    def _create(self) -> str:
        response = gemini_request("POST", "cachedContents", {
            "model": f"models/{MODEL}",
            "displayName": f"{PROMPT_VERSION}-{config_hash()}",
            "contents": [{"role": "user", "parts": [{"text": BASE_PROMPT}]}],
            "ttl": f"{self.ttl_seconds}s",
        })
        self.created_tokens = int((response.get("usageMetadata") or {}).get("totalTokenCount") or 0)
        self.creations += 1
        print(json.dumps({
            "cache_created": response["name"],
            "cached_tokens": self.created_tokens,
            "ttl_seconds": self.ttl_seconds,
        }), flush=True)
        return str(response["name"])

    def invalidate(self, name: str) -> None:
        with self._lock:
            if self._name == name:
                self._name = None

    def delete(self) -> None:
        with self._lock:
            if self._name is None:
                return
            try:
                gemini_request("DELETE", self._name)
            except Exception as exc:  # best effort: the TTL will reclaim it anyway
                print(json.dumps({"cache_delete_failed": str(exc)[:300]}), flush=True)
            self._name = None


PROMPT_CACHE = PromptCache(enabled=True)


def cache_missing(exc: Exception) -> bool:
    if not isinstance(exc, GeminiHTTPError):
        return False
    detail = exc.detail.lower()
    return exc.code in {400, 403, 404} and "cachedcontent" in detail.replace(" ", "")


def splits_word(transcript: str, index: int) -> bool:
    """True when index falls inside a word of a space-delimited script.

    Only alphabetic characters below the CJK ranges (Latin/Greek/Cyrillic…)
    count: unspaced scripts such as Chinese or Japanese allow any boundary.
    """
    if not 0 < index < len(transcript):
        return False
    before, after = transcript[index - 1], transcript[index]
    return all(c.isalpha() and ord(c) < 0x2E80 for c in (before, after))


def validate_alternatives(
    raw: Any, marker_type: str, primary_label: str, primary_confidence: float, position: int
) -> list[dict[str, Any]]:
    if raw is None:
        return []
    if not isinstance(raw, list) or len(raw) > MAX_ALTERNATIVES:
        raise ValueError(f"item {position} alternatives must be a list of at most {MAX_ALTERNATIVES}")
    seen = {primary_label}
    cleaned: list[dict[str, Any]] = []
    for alt in raw:
        if not isinstance(alt, dict) or set(alt) - {"label", "confidence"}:
            raise ValueError(f"item {position} has a malformed alternative: {alt!r}")
        alt_label = str(alt.get("label") or "").strip()
        if alt_label not in LABEL_SETS[marker_type]:
            raise ValueError(
                f"item {position} has out-of-vocabulary {marker_type} alternative: {alt_label!r}"
            )
        if alt_label in seen:
            raise ValueError(f"item {position} repeats label {alt_label!r} in alternatives")
        seen.add(alt_label)
        alt_confidence = float(alt["confidence"])
        if not 0 <= alt_confidence <= 1:
            raise ValueError(f"item {position} alternative confidence outside [0,1]")
        if alt_confidence > primary_confidence:
            raise ValueError(
                f"item {position} alternative {alt_label!r} outranks the primary label"
            )
        cleaned.append({"label": alt_label, "confidence": alt_confidence})
    total = primary_confidence + sum(alt["confidence"] for alt in cleaned)
    if total > 1 + PROBABILITY_MASS_TOLERANCE:
        raise ValueError(
            f"item {position} confidences sum to {total:.2f}; primary plus alternatives must not exceed 1"
        )
    cleaned.sort(key=lambda alt: -alt["confidence"])
    return cleaned


def validate_payload(payload: Any, transcript: str) -> list[dict[str, Any]]:
    if not isinstance(payload, list) or not payload:
        raise ValueError("response must be a non-empty JSON array")
    allowed = {
        "type", "label", "alternatives", "confidence",
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
        if label not in LABEL_SETS[marker_type]:
            raise ValueError(
                f"item {position} has out-of-vocabulary {marker_type} label: {label!r}"
            )
        char_index = source.get("insert_char_index")
        if isinstance(char_index, bool) or not isinstance(char_index, int):
            raise ValueError(f"item {position} has non-integer index: {char_index!r}")
        if not 0 <= char_index <= len(transcript):
            raise ValueError(
                f"item {position} index {char_index} outside transcript length {len(transcript)}"
            )
        if splits_word(transcript, char_index):
            raise ValueError(
                f"item {position} index {char_index} splits a word in a "
                "space-delimited script"
            )
        confidence = float(source["confidence"])
        placement_confidence = float(source["placement_confidence"])
        if not 0 <= confidence <= 1 or not 0 <= placement_confidence <= 1:
            raise ValueError(f"item {position} confidence outside [0,1]")
        alternatives = validate_alternatives(
            source.get("alternatives"), marker_type, label, confidence, position
        )
        cleaned.append({
            "type": marker_type,
            "label": label,
            "alternatives": alternatives,
            "confidence": confidence,
            "insert_char_index": char_index,
            "placement_confidence": placement_confidence,
        })

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


def load_audio(row: dict[str, Any]) -> tuple[bytes, str]:
    audio_path = Path(row["audio_path_local"])
    mime_type = str(
        row.get("mime_type") or mimetypes.guess_type(audio_path.name)[0] or "audio/flac"
    )
    return audio_path.read_bytes(), mime_type


def build_request(
    row: dict[str, Any], audio_bytes: bytes, mime_type: str, cache_name: str | None
) -> dict[str, Any]:
    """GenerateContentRequest body shared by the sync and batch transports."""
    audio_part = {
        "inlineData": {
            "mimeType": mime_type,
            "data": base64.b64encode(audio_bytes).decode("ascii"),
        }
    }
    generation_config = {
        "temperature": TEMPERATURE,
        "responseMimeType": "application/json",
        "responseJsonSchema": response_schema(),
        "thinkingConfig": {
            "thinkingLevel": THINKING_LEVEL,
            "includeThoughts": True,
        },
    }
    if cache_name is None:
        # Text first so requests share an identical prefix for implicit caching.
        return {
            "contents": [{"role": "user", "parts": [{"text": build_prompt(row)}, audio_part]}],
            "generationConfig": generation_config,
        }
    # BASE_PROMPT comes from the cache as the prompt prefix; only the per-clip
    # INPUT block and audio travel with the request.
    return {
        "cachedContent": cache_name,
        "contents": [{"role": "user", "parts": [{"text": build_input_block(row)}, audio_part]}],
        "generationConfig": generation_config,
    }


def split_response(api_response: dict[str, Any]) -> tuple[str, str]:
    """Return (raw JSON text, thought summary) from a GenerateContentResponse."""
    candidates = api_response.get("candidates") or []
    if not candidates:
        raise ValueError(f"response has no candidates: {api_response}")
    parts = ((candidates[0].get("content") or {}).get("parts") or [])
    thought_summary = "".join(
        str(part.get("text") or "") for part in parts if part.get("thought")
    )
    raw_response_text = "".join(
        str(part.get("text") or "") for part in parts if not part.get("thought")
    )
    if not raw_response_text:
        raise ValueError(f"response has no JSON text: {api_response}")
    return raw_response_text, thought_summary


def call_model(
    audio_bytes: bytes, mime_type: str, row: dict[str, Any]
) -> tuple[dict[str, Any], str, str]:
    cache_name = PROMPT_CACHE.name()
    try:
        api_response = post_gemini(build_request(row, audio_bytes, mime_type, cache_name))
    except GeminiHTTPError as exc:
        if cache_name is not None and cache_missing(exc):
            PROMPT_CACHE.invalidate(cache_name)
            return call_model(audio_bytes, mime_type, row)
        raise
    raw_response_text, thought_summary = split_response(api_response)
    return api_response, raw_response_text, thought_summary


def retryable(exc: Exception) -> bool:
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if code in {408, 409, 429, 500, 502, 503, 504}:
        return True
    if cache_missing(exc):
        return True
    message = str(exc).lower()
    return isinstance(exc, (ValueError, json.JSONDecodeError)) or any(
        token in message
        for token in ("429", "quota", "timeout", "temporar", "unavailable", "reset")
    )


def new_wrapper(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "game_audio_open_marker_result_v2",
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


def accept_response(
    wrapper: dict[str, Any], row: dict[str, Any], api_response: dict[str, Any], transport: str
) -> None:
    """Validate a model response and record it on the wrapper; raises if invalid."""
    raw_response_text, thought_summary = split_response(api_response)
    payload = json.loads(raw_response_text)
    annotations = validate_payload(payload, row["text"])
    candidates = api_response.get("candidates") or []
    wrapper.update({
        "status": "ok",
        "transport": transport,
        "raw_response_text": raw_response_text,
        "thought_summary": thought_summary or None,
        "raw_model_output": payload,
        "annotations": annotations,
        "tagged_text": render_text(row["text"], annotations),
        "usage_metadata": api_response.get("usageMetadata"),
        "context_cache": PROMPT_CACHE.enabled,
        "response_id": api_response.get("responseId"),
        "api_model_version": api_response.get("modelVersion"),
        "finish_reason": candidates[0].get("finishReason"),
        "completed_at_unix": time.time(),
    })


def infer(
    row: dict[str, Any], max_attempts: int, history: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    """Annotate one row over the synchronous API with retries.

    `history` carries earlier failed attempts (e.g. from a batch job) so the
    result records the full story.
    """
    started = time.monotonic()
    wrapper = new_wrapper(row)
    audio_bytes, mime_type = load_audio(row)
    history = list(history or [])
    first_attempt = len(history) + 1
    for attempt in range(first_attempt, first_attempt + max_attempts):
        try:
            api_response, _, _ = call_model(audio_bytes, mime_type, row)
            accept_response(wrapper, row, api_response, "sync")
            wrapper.update({
                "attempts": attempt,
                "attempt_history": history or None,
                "elapsed_seconds": time.monotonic() - started,
            })
            return wrapper
        except Exception as exc:
            history.append({
                "attempt": attempt,
                "transport": "sync",
                "type": type(exc).__name__,
                "message": str(exc)[:2000],
            })
            if attempt >= first_attempt + max_attempts - 1 or not retryable(exc):
                break
            time.sleep(min(30.0, 2 ** (attempt - first_attempt) + random.random()))
    wrapper.update({
        "attempts": len(history),
        "attempt_history": history,
        "error": history[-1] if history else {"message": "unknown error"},
        "traceback": traceback.format_exc(limit=8),
        "elapsed_seconds": time.monotonic() - started,
        "completed_at_unix": time.time(),
    })
    return wrapper


def upload_file(data: bytes, mime_type: str, display_name: str) -> str:
    """Upload bytes through the File API (resumable, single chunk); returns files/…"""
    key = os.environ["GEMINI_API_KEY"]
    start = urllib.request.Request(
        f"{UPLOAD_ROOT}/files",
        data=json.dumps({"file": {"display_name": display_name}}).encode("utf-8"),
        headers={
            "x-goog-api-key": key,
            "X-Goog-Upload-Protocol": "resumable",
            "X-Goog-Upload-Command": "start",
            "X-Goog-Upload-Header-Content-Length": str(len(data)),
            "X-Goog-Upload-Header-Content-Type": mime_type,
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(start, timeout=120) as response:
            upload_url = response.headers.get("X-Goog-Upload-URL")
    except urllib.error.HTTPError as exc:
        raise GeminiHTTPError(exc.code, exc.read().decode("utf-8", errors="replace")) from exc
    if not upload_url:
        raise RuntimeError("file upload start did not return X-Goog-Upload-URL")
    finish = urllib.request.Request(
        upload_url,
        data=data,
        headers={
            "Content-Length": str(len(data)),
            "X-Goog-Upload-Offset": "0",
            "X-Goog-Upload-Command": "upload, finalize",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(finish, timeout=1800) as response:
            info = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise GeminiHTTPError(exc.code, exc.read().decode("utf-8", errors="replace")) from exc
    return str(info["file"]["name"])


def download_file(file_name: str) -> bytes:
    request = urllib.request.Request(
        f"{DOWNLOAD_ROOT}/{file_name}:download?alt=media",
        headers={"x-goog-api-key": os.environ["GEMINI_API_KEY"]},
    )
    try:
        with urllib.request.urlopen(request, timeout=1800) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        raise GeminiHTTPError(exc.code, exc.read().decode("utf-8", errors="replace")) from exc


def batch_state_path(output: Path) -> Path:
    return output.with_suffix(output.suffix + ".batch.json")


def submit_batch(pending: list[dict[str, Any]], output: Path) -> dict[str, Any]:
    """Upload one request per pending row and start a batch job; persist its state."""
    cache_name = PROMPT_CACHE.name()
    lines = []
    for row in pending:
        audio_bytes, mime_type = load_audio(row)
        lines.append(json.dumps({
            "key": str(row["id"]),
            "request": build_request(row, audio_bytes, mime_type, cache_name),
        }, ensure_ascii=False, separators=(",", ":")))
    data = ("\n".join(lines) + "\n").encode("utf-8")
    display_name = f"{PROMPT_VERSION}-{config_hash()}-{int(time.time())}"
    file_name = upload_file(data, "application/jsonl", display_name)
    print(json.dumps({
        "batch_input_uploaded": file_name, "requests": len(pending),
        "bytes": len(data),
    }), flush=True)
    operation = gemini_request("POST", f"models/{MODEL}:batchGenerateContent", {
        "batch": {"display_name": display_name, "input_config": {"file_name": file_name}},
    })
    state = {
        "job": operation["name"],
        "config_hash": config_hash(),
        "cache_name": cache_name,
        "input_file": file_name,
        "ids": [str(row["id"]) for row in pending],
        "submitted_at_unix": time.time(),
    }
    batch_state_path(output).write_text(json.dumps(state, indent=2), encoding="utf-8")
    print(json.dumps({"batch_submitted": operation["name"], "requests": len(pending)}), flush=True)
    return state


def load_batch_state(output: Path, pending_ids: set[str]) -> dict[str, Any] | None:
    """Return a saved, still-relevant batch job for this output, if any."""
    path = batch_state_path(output)
    if not path.is_file():
        return None
    state = json.loads(path.read_text(encoding="utf-8"))
    if state.get("config_hash") != config_hash() or not (set(state.get("ids", [])) & pending_ids):
        path.unlink()
        return None
    return state


def batch_state_name(operation: dict[str, Any]) -> str:
    """Normalise BATCH_STATE_* / JOB_STATE_* to the bare suffix, e.g. SUCCEEDED."""
    state = str((operation.get("metadata") or {}).get("state") or "")
    return state.rsplit("_", 1)[-1]


def wait_for_batch(job: str, poll_seconds: float) -> dict[str, Any]:
    terminal = {"SUCCEEDED", "FAILED", "CANCELLED", "EXPIRED"}
    last_seen = None
    while True:
        operation = gemini_request("GET", job)
        metadata = operation.get("metadata") or {}
        stats = metadata.get("batchStats") or {}
        seen = json.dumps({"batch_state": metadata.get("state"), "batch_stats": stats}, sort_keys=True)
        if seen != last_seen:
            print(seen, flush=True)
            last_seen = seen
        if operation.get("done") or batch_state_name(operation) in terminal:
            return operation
        time.sleep(poll_seconds)


def batch_results(operation: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Map request key -> {'response': …} or {'error': …} from a finished job."""
    response = operation.get("response") or {}
    if response.get("inlinedResponses"):
        items = response["inlinedResponses"]
    elif response.get("responsesFile"):
        raw = download_file(str(response["responsesFile"])).decode("utf-8")
        items = [json.loads(line) for line in raw.splitlines() if line.strip()]
    else:
        raise RuntimeError(f"batch job finished without results: {operation}")
    results: dict[str, dict[str, Any]] = {}
    for item in items:
        key = str(item.get("key") or (item.get("metadata") or {}).get("key") or "")
        results[key] = item
    return results


def run_batch(
    args: argparse.Namespace, pending: list[dict[str, Any]], handle: Any
) -> tuple[int, int]:
    """Annotate pending rows through the Batch API; fall back to sync for failures."""
    by_id = {str(row["id"]): row for row in pending}
    state = load_batch_state(args.output, set(by_id))
    if state is None:
        state = submit_batch(pending, args.output)
    else:
        print(json.dumps({"batch_resumed": state["job"], "requests": len(state["ids"])}), flush=True)
        if state.get("cache_name"):
            PROMPT_CACHE.adopt(state["cache_name"])
    operation = wait_for_batch(state["job"], args.batch_poll_seconds)
    if batch_state_name(operation) != "SUCCEEDED":
        # Nothing to collect; drop the state so a rerun submits a fresh job.
        batch_state_path(args.output).unlink(missing_ok=True)
        raise RuntimeError(
            f"batch job {state['job']} ended in "
            f"{(operation.get('metadata') or {}).get('state')}: {operation.get('error')}"
        )
    results = batch_results(operation)

    ok = errors = 0
    fallback: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    for row_id in state["ids"]:
        row = by_id.get(row_id)
        if row is None:  # already completed by an earlier run
            continue
        item = results.get(row_id)
        failure: dict[str, Any] | None = None
        if item is None:
            failure = {"type": "MissingBatchResult", "message": "no result for key"}
        elif item.get("error"):
            failure = {"type": "BatchError", "message": json.dumps(item["error"])[:2000]}
        else:
            wrapper = new_wrapper(row)
            try:
                accept_response(wrapper, row, item["response"], "batch")
                wrapper.update({"attempts": 1, "batch_job": state["job"], "elapsed_seconds": None})
                handle.write(json.dumps(wrapper, ensure_ascii=False, separators=(",", ":")) + "\n")
                ok += 1
                continue
            except Exception as exc:
                failure = {"type": type(exc).__name__, "message": str(exc)[:2000]}
        failure.update({"attempt": 1, "transport": "batch", "batch_job": state["job"]})
        fallback.append((row, [failure]))
    handle.flush()
    print(json.dumps({"batch_ok": ok, "batch_fallback_to_sync": len(fallback)}), flush=True)

    if fallback:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(infer, row, args.max_attempts, history) for row, history in fallback]
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                handle.write(json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n")
                handle.flush()
                if result["status"] == "ok":
                    ok += 1
                else:
                    errors += 1
                print(json.dumps({"sync_fallback": result["id"], "status": result["status"]}), flush=True)
    batch_state_path(args.output).unlink(missing_ok=True)
    return ok, errors


def read_completed(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {
        str(row["id"])
        for row in read_jsonl(path)
        if row.get("status") == "ok" and row.get("config_hash") == config_hash()
    }


def confidence_histogram(markers: list[dict[str, Any]]) -> dict[str, int]:
    """Count marker confidences in 0.1-wide bins keyed by the bin's lower edge."""
    counts = Counter(f"{min(int(marker['confidence'] * 10), 9) / 10:.1f}" for marker in markers)
    return dict(sorted(counts.items()))


def write_report(
    output: Path, report_path: Path, total_rows: int, cache_started_unix: float | None = None
) -> None:
    rows = read_jsonl(output) if output.exists() else []
    current = [row for row in rows if row.get("config_hash") == config_hash()]
    by_id = {str(row["id"]): row for row in current}
    final_rows = list(by_id.values())
    ok = [row for row in final_rows if row.get("status") == "ok"]
    errors = [row for row in final_rows if row.get("status") != "ok"]
    prompt_tokens = cached_tokens = candidate_tokens = thought_tokens = total_tokens = 0
    token_cost = 0.0
    transports: Counter[str] = Counter()
    for row in ok:
        usage = row.get("usage_metadata") or {}
        prompt = int(usage.get("prompt_token_count") or usage.get("promptTokenCount") or 0)
        cached = int(
            usage.get("cached_content_token_count") or usage.get("cachedContentTokenCount") or 0
        )
        candidates = int(
            usage.get("candidates_token_count") or usage.get("candidatesTokenCount") or 0
        )
        thoughts = int(usage.get("thoughts_token_count") or usage.get("thoughtsTokenCount") or 0)
        prompt_tokens += prompt
        cached_tokens += cached
        candidate_tokens += candidates
        thought_tokens += thoughts
        total_tokens += int(usage.get("total_token_count") or usage.get("totalTokenCount") or 0)
        transport = str(row.get("transport") or "sync")
        transports[transport] += 1
        discount = BATCH_DISCOUNT if transport == "batch" else 1.0
        token_cost += (
            (prompt - cached) * PRICE_INPUT * discount
            + cached * PRICE_CACHED_INPUT
            + (candidates + thoughts) * PRICE_OUTPUT * discount
        )
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
    # promptTokenCount includes the cached prefix; only the remainder is billed
    # at the full input rate. Cache storage is approximated by the run's span
    # (batch jobs: from submission to collection).
    cache_hours = 0.0
    if cached_tokens and ok:
        stamps = [float(row.get("completed_at_unix") or 0) for row in ok]
        stamps.append(float(cache_started_unix or stamps[0]))
        cache_hours = max(max(stamps) - min(stamps), 60.0) / 3600
    cache_storage_cost = (
        PROMPT_CACHE.created_tokens or (cached_tokens // len(ok) if ok else 0)
    ) * cache_hours * PRICE_CACHE_STORAGE_PER_TOKEN_HOUR
    estimated_standard_cost = token_cost + cache_storage_cost
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
        "rows_by_transport": dict(transports),
        "prompt_tokens": prompt_tokens,
        "cached_prompt_tokens": cached_tokens,
        "candidate_tokens": candidate_tokens,
        "thought_tokens": thought_tokens,
        "total_tokens_reported": total_tokens,
        "estimated_standard_cost_usd": estimated_standard_cost,
        "estimated_cache_storage_cost_usd": cache_storage_cost,
        "pricing_assumptions_usd_per_million": {
            "input": PRICE_INPUT * 1e6,
            "cached_input": PRICE_CACHED_INPUT * 1e6,
            "output_and_thoughts": PRICE_OUTPUT * 1e6,
            "cache_storage_per_hour": PRICE_CACHE_STORAGE_PER_TOKEN_HOUR * 1e6,
            "batch_multiplier_on_uncached": BATCH_DISCOUNT,
        },
        "emotion_markers": len(emotions),
        "event_markers": len(events),
        "multi_emotion_rows": sum(
            sum(marker["type"] == "emotion" for marker in row["annotations"]) > 1
            for row in ok
        ),
        "emotion_label_counts": Counter(marker["label"] for marker in emotions),
        "event_label_counts": Counter(marker["label"] for marker in events),
        "confidence_histogram": {
            "emotion": confidence_histogram(emotions),
            "event": confidence_histogram(events),
        },
        "markers_with_alternatives": sum(
            bool(marker.get("alternatives")) for marker in emotions + events
        ),
        "alternative_markers_total": sum(
            len(marker.get("alternatives") or []) for marker in emotions + events
        ),
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
    parser.add_argument("--skip-first", type=int, default=0,
                        help="drop the first N manifest rows before selection")
    parser.add_argument("--sample", type=int,
                        help="randomly sample N rows (after --skip-first)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ids", help="comma-separated id substrings; keep only matching rows")
    parser.add_argument("--no-cache", action="store_true",
                        help="send the full prompt with every request instead of an explicit context cache")
    parser.add_argument("--sync", action="store_true",
                        help="call generateContent directly instead of submitting a Batch API job")
    parser.add_argument("--batch-poll-seconds", type=float, default=30.0)
    args = parser.parse_args()
    PROMPT_CACHE.enabled = not args.no_cache
    PROMPT_CACHE.ttl_seconds = CACHE_TTL_SECONDS_SYNC if args.sync else CACHE_TTL_SECONDS_BATCH

    if not os.environ.get("GEMINI_API_KEY"):
        os.environ["GEMINI_API_KEY"] = getpass.getpass("Gemini API key: ")
    if not os.environ["GEMINI_API_KEY"]:
        raise RuntimeError("empty Gemini API key")

    rows = load_rows(args.manifest.resolve(), args.text_manifest.resolve())
    if args.ids:
        needles = [needle.strip() for needle in args.ids.split(",") if needle.strip()]
        rows = [row for row in rows if any(needle in str(row["id"]) for needle in needles)]
        if not rows:
            raise ValueError(f"no manifest rows match --ids {args.ids!r}")
    if args.skip_first:
        rows = rows[args.skip_first:]
    if args.sample is not None:
        rows = random.Random(args.seed).sample(rows, args.sample)
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

    cache_started = time.time()
    ok = errors = 0
    if pending:
        try:
            with args.output.open("a", encoding="utf-8", buffering=1) as handle:
                if args.sync:
                    ok, errors = run_sync(args, pending, handle)
                else:
                    ok, errors = run_batch(args, pending, handle)
        finally:
            # A batch job that is still pending (interrupted run) keeps its
            # cache alive so the queued requests can still reference it.
            if not args.sync and batch_state_path(args.output).is_file():
                print(json.dumps({
                    "batch_job_left_running": True,
                    "note": "rerun the same command to resume polling and collect results",
                }), flush=True)
            else:
                PROMPT_CACHE.delete()
    write_report(args.output, args.report, len(rows), cache_started)
    print(json.dumps({"finished": True, "ok": ok, "errors": errors}), flush=True)


def run_sync(
    args: argparse.Namespace, pending: list[dict[str, Any]], handle: Any
) -> tuple[int, int]:
    ok = errors = 0
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
    return ok, errors


if __name__ == "__main__":
    main()
