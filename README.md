# Gemini game-audio closed-marker annotator

This repository contains the annotation runner used for a frozen multilingual
Genshin pilot: 200 clips (Chinese, English, Japanese, and Korean) drawn from
five GCS shards by `prepare_pilot_inputs.py`, whose first 100 rows are the
original 100-item pilot. Audio and annotation results are intentionally not
included.

The annotator sends each audio clip together with its official transcript to
Gemini and requests an ordered sequence of emotion-state and human
paralinguistic-event markers drawn from a closed vocabulary. Markers are placed
at Unicode character boundaries in the original transcript, so long utterances
can contain multiple emotion states.

## Experiment configuration

- Model: `gemini-3.7-flash`
- Prompt version: `game-audio-closed-marker-sequence-v15`
- Temperature: `0`
- Thinking level: `MEDIUM` (override with `GEMINI_THINKING_LEVEL`); thought
  summaries are requested and stored with each result
- Emotion labels: 87 closed `snake_case` labels covering emotions and
  vocal-delivery states (e.g. `whisper`, `shouting`, `breathy`)
- Event labels: 20 closed `snake_case` labels (e.g. `pause`, `laugh`, `sigh`,
  `inhale`)
- Alternatives: each marker may list up to 3 alternative labels; primary and
  alternative confidences share one probability mass (sum ≤ 1)
- Requests: synchronous `generateContent` calls fanned out over a thread pool
  (`--workers`); the batch API is not used
- Context cache: the fixed rule/vocabulary prefix (~1.4k tokens) is stored in
  an explicit Gemini context cache for the duration of a run and referenced by
  every request, so only the per-clip input block and audio are sent in full
- Output: resumable JSONL plus a summary report

The vocabulary is the `english` column of
`tts_bracket_emotion_enword_180d.csv`, split into emotion/delivery labels and
event labels. The full prompt, label lists, and JSON response schema are
embedded in `run_open_markers.py` so the experiment can be audited and
reproduced.

Prompt rules worth knowing when reading results:

- Only the acoustics count as evidence. Transcript wording, punctuation, and
  script conventions (e.g. parentheses marking an aside) must not influence
  labels.
- `pause` is reserved for deliberate silent gaps of roughly 0.5 s or more.
- Confidence is a calibrated probability with explicit anchors (≥0.9 only when
  no other label is plausible; 0.6–0.8 with one plausible alternative; and so
  on), not a default quality score.
- In space-delimited scripts a marker must never split a word.

## Requirements

- Python 3.9 or newer
- A Gemini API key with access to the configured model
- Local audio files referenced by the manifest

The runner uses only the Python standard library.

## Input files

Each input set may contain any number of rows as long as IDs are unique;
`prepare_pilot_inputs.py` writes the 200-row pilot set to `inputs/`.

`manifest.jsonl` contains one JSON object per audio clip:

```json
{"id":"clip-001","audio":"audio/clip-001.flac","language":"zh","duration":3.42,"speaker_id":"speaker-01"}
```

Required fields are `id`, `audio`, `language`, and `duration`.
`speaker_id` and `mime_type` are optional. Relative audio paths are resolved
from the directory containing the manifest.

`text_manifest.jsonl` supplies the official reference text:

```json
{"id":"clip-001","text_plain":"示例台词。"}
```

Every ID in the audio manifest must have a non-empty `text_plain` entry.

## Run

Set the API key in the environment, or omit it and enter the key at the hidden
interactive prompt:

```bash
export GEMINI_API_KEY="your-api-key"
python3 run_open_markers.py \
  --manifest manifest.jsonl \
  --text-manifest text_manifest.jsonl \
  --output results/annotations.jsonl \
  --report results/report.json \
  --workers 4
```

Useful options:

- `--limit N` runs only the first `N` items of the manifest.
- `--skip-first N` drops the first `N` rows before selection, and
  `--sample N --seed S` picks `N` random rows from what remains; combine them
  to evaluate on a held-out random subset (e.g. `--skip-first 10 --sample 10`).
- `--ids a,b` keeps only rows whose ID contains one of the given substrings,
  for spot-checking individual clips.
- `--max-attempts N` controls retry attempts per item; the default is 4.
  Responses that fail validation (out-of-vocabulary label, mid-word index,
  overconfident alternatives, …) are retried.
- `--no-cache` sends the full prompt with every request instead of creating an
  explicit context cache. `GEMINI_CACHE_TTL_SECONDS` (default 7200) bounds the
  cache lifetime; the cache is deleted when the run finishes.
- Re-running the same command resumes completed items whose configuration hash
  matches the current model, prompt, schema, and generation settings.

## Marker output

Gemini returns a JSON array such as:

```json
[
  {
    "type": "emotion",
    "label": "tired",
    "confidence": 0.62,
    "alternatives": [{"label": "sad", "confidence": 0.25}],
    "insert_char_index": 0,
    "placement_confidence": 0.97
  },
  {
    "type": "event",
    "label": "sigh",
    "confidence": 0.91,
    "alternatives": [],
    "insert_char_index": 0,
    "placement_confidence": 0.96
  }
]
```

Validation enforces the closed vocabulary, index bounds and word boundaries,
a first emotion marker at index 0, no consecutive duplicate emotion labels, and
the probability-mass rule for alternatives.

Each successful JSONL row retains the exact response text in
`raw_response_text`, the model's thought summary in `thought_summary`, the
parsed response in `raw_model_output`, and the validated, sorted markers in
`annotations`. It also records model metadata, token usage, attempts, elapsed
time, and a convenience `tagged_text` rendering such as
`[tired][sigh]我们的合作课题，几天下来都没什么进展。`.

The report summarises token usage (prompt, cached prompt, output, and thought
tokens) with a cost estimate that applies the cached-input discount and
approximate cache storage, marker counts, per-label counts, confidence histograms, and how
many markers carry alternatives.

## Security and data handling

- The API key is read from `GEMINI_API_KEY` and is never written to output.
- Do not commit `.env` files, audio, manifests containing private metadata, or
  generated annotation results unless you have permission to publish them.
- The cost estimate in the report reflects the pricing assumptions encoded at
  the time of this pilot; verify current Gemini pricing before budgeting a new
  run.
