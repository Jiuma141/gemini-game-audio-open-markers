# Gemini game-audio open-marker annotator

This repository contains the exact annotation runner used for a frozen
100-item multilingual Genshin pilot (25 clips each in Chinese, English,
Japanese, and Korean). Audio and annotation results are intentionally not
included.

The annotator sends each audio clip together with its official transcript to
Gemini and requests an ordered sequence of open-vocabulary emotion-state and
human paralinguistic-event markers. Markers are placed at Unicode character
boundaries in the original transcript, so long utterances can contain multiple
emotion states.

## Experiment configuration

- Model: `gemini-3.7-flash`
- Prompt version: `game-audio-open-marker-sequence-v5`
- Temperature: `0`
- Thinking level: `LOW`
- Emotion labels: open English `snake_case` vocabulary
- Event labels: open English `snake_case` vocabulary
- Concurrency: configurable with `--workers`
- Output: resumable JSONL plus a summary report

The full prompt and JSON response schema are embedded in
`run_open_markers.py` so the experiment can be audited and reproduced.

## Requirements

- Python 3.9 or newer
- A Gemini API key with access to the configured model
- Local audio files referenced by the manifest

The runner uses only the Python standard library.

## Input files

The pilot expects exactly 100 unique rows in each input set.

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

- `--limit N` runs only the first `N` items from the validated 100-item input.
- `--max-attempts N` controls retry attempts per item; the default is 4.
- Re-running the same command resumes completed items whose configuration hash
  matches the current model, prompt, schema, and generation settings.

## Marker output

Gemini returns a JSON array such as:

```json
[
  {
    "type": "emotion",
    "label": "weary_resignation",
    "intensity": 0.61,
    "confidence": 0.88,
    "insert_char_index": 0,
    "placement_confidence": 0.97
  },
  {
    "type": "event",
    "label": "sighing",
    "confidence": 0.94,
    "insert_char_index": 0,
    "placement_confidence": 0.96
  }
]
```

Each successful JSONL row retains the exact response text in
`raw_response_text`, the parsed response in `raw_model_output`, and the
validated, sorted markers in `annotations`. It also records model metadata,
token usage, attempts, elapsed time, and a convenience `tagged_text` rendering.

## Security and data handling

- The API key is read from `GEMINI_API_KEY` and is never written to output.
- Do not commit `.env` files, audio, manifests containing private metadata, or
  generated annotation results unless you have permission to publish them.
- The cost estimate in the report reflects the pricing assumptions encoded at
  the time of this pilot; verify current Gemini pricing before budgeting a new
  run.
