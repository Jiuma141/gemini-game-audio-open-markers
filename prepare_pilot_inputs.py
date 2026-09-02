#!/usr/bin/env python3
"""Build a diverse 200-row pilot manifest from five Genshin GCS shards.

Picks 40 rows per shard in two phases so the manifest's first 100 rows are
byte-identical to the original 100-item pilot: phase 1 takes 20 rows per shard
with max 2 per speaker, phase 2 rescans the shard for 20 more with the cap
raised to 3. All rows are 3-15s, non-empty text, ASR error rate < 0.3. The
manifest is ordered round-robin across shards so any prefix mixes languages and
speakers. Streams each audio tar from GCS and stops once all selected members
of that shard are extracted.
"""

from __future__ import annotations

import json
import subprocess
import tarfile
from pathlib import Path

SHARDS = ("Genshin-000000", "Genshin-000080", "Genshin-000161", "Genshin-000242", "Genshin-000321")
GCS_AUDIO = "gs://noiz-taiwan-audio-data/preprocessed/Genshin/audio"
ROOT = Path(__file__).parent
INPUTS = ROOT / "inputs"
AUDIO_DIR = INPUTS / "audio"
METADATA_DIR = INPUTS / "cache" / "sample_shards"
ASR_DIR = INPUTS / "cache" / "asr"
# (rows to reach, max rows per speaker) per selection phase. Phase 1 must stay
# unchanged so the first 100 manifest rows match the earlier pilot runs.
PHASES = ((20, 2), (40, 3))
ROWS_PER_SHARD = PHASES[-1][0]
MIN_DURATION = 3.0
MAX_DURATION = 15.0
MAX_ASR_ERROR_RATE = 0.3


def load_asr_error_rates(shard: str) -> dict[str, float]:
    """id -> WER/CER from features/asr_error_rate_v1; unscored rows are absent."""
    rates: dict[str, float] = {}
    with (ASR_DIR / f"{shard}.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("scored") and row.get("error_rate") is not None:
                rates[str(row["id"])] = float(row["error_rate"])
    return rates


def select_rows(shard: str) -> list[dict]:
    error_rates = load_asr_error_rates(shard)
    with (METADATA_DIR / f"{shard}.jsonl").open(encoding="utf-8") as handle:
        candidates = [json.loads(line) for line in handle]
    per_speaker: dict[str, int] = {}
    selected: list[dict] = []
    chosen: set[str] = set()
    for target, max_per_speaker in PHASES:
        for row in candidates:
            if len(selected) == target:
                break
            speaker = str(row["speaker_id"])
            error_rate = error_rates.get(str(row["id"]))
            if (
                str(row["id"]) in chosen
                or not str(row.get("text") or "").strip()
                or not MIN_DURATION <= float(row.get("duration") or 0) <= MAX_DURATION
                or per_speaker.get(speaker, 0) >= max_per_speaker
                or not str(row["audio_path"]).startswith(f"audio/{shard}.tar/")
                or error_rate is None
                or error_rate >= MAX_ASR_ERROR_RATE
            ):
                continue
            row["asr_error_rate"] = error_rate
            per_speaker[speaker] = per_speaker.get(speaker, 0) + 1
            selected.append(row)
            chosen.add(str(row["id"]))
        if len(selected) != target:
            raise SystemExit(f"{shard}: only found {len(selected)} of {target} qualifying rows")
    return selected


def extract(shard: str, rows: list[dict]) -> None:
    remaining = {str(row["audio_path"]).split(".tar/", 1)[1] for row in rows}
    remaining = {
        name for name in remaining
        if not (AUDIO_DIR / Path(name).name).is_file()
        or (AUDIO_DIR / Path(name).name).stat().st_size == 0
    }
    if not remaining:
        print(f"{shard}: all files already extracted", flush=True)
        return
    process = subprocess.Popen(
        ["gcloud", "storage", "cat", f"{GCS_AUDIO}/{shard}.tar"],
        stdout=subprocess.PIPE,
    )
    assert process.stdout is not None
    try:
        with tarfile.open(fileobj=process.stdout, mode="r|") as archive:
            for member in archive:
                name = member.name.lstrip("./")
                if name not in remaining:
                    continue
                source = archive.extractfile(member)
                assert source is not None
                (AUDIO_DIR / Path(name).name).write_bytes(source.read())
                remaining.discard(name)
                if not remaining:
                    break
    finally:
        process.terminate()
        process.wait()
    if remaining:
        raise SystemExit(f"{shard}: missing members: {sorted(remaining)[:5]}")
    print(f"{shard}: extracted {len(rows)} files", flush=True)


def main() -> None:
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    per_shard = {shard: select_rows(shard) for shard in SHARDS}
    for shard, rows in per_shard.items():
        langs: dict[str, int] = {}
        for row in rows:
            langs[row["language"]] = langs.get(row["language"], 0) + 1
        print(f"{shard}: {langs}, speakers={len({r['speaker_id'] for r in rows})}")
        extract(shard, rows)

    ordered = [
        per_shard[shard][index]
        for index in range(ROWS_PER_SHARD)
        for shard in SHARDS
    ]
    with (INPUTS / "manifest.jsonl").open("w", encoding="utf-8") as manifest:
        with (INPUTS / "texts.jsonl").open("w", encoding="utf-8") as texts:
            for row in ordered:
                member = str(row["audio_path"]).split(".tar/", 1)[1]
                manifest.write(json.dumps({
                    "id": row["id"],
                    "audio": f"audio/{Path(member).name}",
                    "language": row["language"],
                    "speaker_id": row["speaker_id"],
                    "duration": row["duration"],
                    "asr_error_rate": row["asr_error_rate"],
                    "mime_type": "audio/flac",
                }, ensure_ascii=False) + "\n")
                texts.write(json.dumps(
                    {"id": row["id"], "text_plain": row["text"]}, ensure_ascii=False
                ) + "\n")
    print(f"wrote {len(ordered)} rows to inputs/manifest.jsonl and inputs/texts.jsonl")


if __name__ == "__main__":
    main()
