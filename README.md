# OmniVoice 🌍

<p align="center">
  <img width="200" height="200" alt="OmniVoice" src="https://zhu-han.github.io/omnivoice/pics/omnivoice.jpg" />
</p>

<p align="center">
  <a href="https://huggingface.co/k2-fsa/OmniVoice"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-FFD21E" alt="Hugging Face Model"></a>
  &nbsp;
  <a href="https://huggingface.co/spaces/k2-fsa/OmniVoice"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Space-blue" alt="Hugging Face Space"></a>
  &nbsp;
  <a href="https://arxiv.org/abs/2604.00688"><img src="https://img.shields.io/badge/arXiv-Paper-B31B1B.svg"></a>
  &nbsp;
  <a href="https://zhu-han.github.io/omnivoice"><img src="https://img.shields.io/badge/GitHub.io-Demo_Page-blue?logo=GitHub&style=flat-square"></a>
  &nbsp;
  <a href="https://colab.research.google.com/github/k2-fsa/OmniVoice/blob/master/docs/OmniVoice.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
</p>

OmniVoice is a state-of-the-art massively multilingual zero-shot text-to-speech (TTS) model supporting over 600 languages. Built on a novel diffusion language model-style architecture, it generates high-quality speech with superior inference speed, supporting voice cloning and voice design.

**Contents**: [Key Features](#key-features) | [Installation](#installation) | [Quick Start](#quick-start) | [Python API](#python-api) | [Command-Line Tools](#command-line-tools) | [Docker & API](#docker--api) | [Training & Evaluation](#training--evaluation) | [Discussion](#discussion--communication) | [Citation](#citation)

## Key Features

- **600+ Languages Supported**: The broadest language coverage among zero-shot TTS models ([full list](docs/languages.md)).
- **Voice Cloning**: State-of-the-art voice cloning quality.
- **Voice Design**: Control voices via assigned speaker attributes (gender, age, pitch, dialect/accent, whisper, etc.).
- **Fine-grained Control**: Non-verbal symbols (e.g., `[laughter]`) and pronunciation correction via pinyin or phonemes.
- **Fast Inference**: RTF as low as 0.025 (40x faster than real-time).
- **Diffusion Language Model-style Architecture**: A clean, streamlined, and scalable design that delivers both quality and speed.

---

## Installation

Choose **one** of the following methods: **pip** or **uv**.

### pip

> We recommend using a fresh virtual environment (e.g., `conda`, `venv`, etc.) to avoid conflicts.

**Step 1**: Install PyTorch

<details>
<summary>NVIDIA GPU</summary>

```bash
# Install pytorch with your CUDA version, e.g.
pip install torch==2.8.0+cu128 torchaudio==2.8.0+cu128 --extra-index-url https://download.pytorch.org/whl/cu128
```
> See [PyTorch official site](https://pytorch.org/get-started/locally/) for other versions installation.

</details>

<details>
<summary>Apple Silicon</summary>

```bash
pip install torch==2.8.0 torchaudio==2.8.0
```

</details>

**Step 2**: Install OmniVoice (choose one)

```bash
# From PyPI (stable release)
pip install omnivoice

# From the latest source on GitHub (no need to clone)
pip install git+https://github.com/k2-fsa/OmniVoice.git

# For development (clone first, editable install)
git clone https://github.com/k2-fsa/OmniVoice.git
cd OmniVoice
pip install -e .
```

### uv

Clone the repository and sync dependencies:

```bash
git clone https://github.com/k2-fsa/OmniVoice.git
cd OmniVoice
uv sync
```

> **Tip**: Can use mirror with `uv sync --default-index "https://mirrors.aliyun.com/pypi/simple"`

---

## Quick Start

Try OmniVoice without coding:

- Launch the local web UI: `omnivoice-demo --ip 0.0.0.0 --port 8001`

- Or try it directly on [HuggingFace Space](https://huggingface.co/spaces/k2-fsa/OmniVoice)

- Or run it in Google Colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/k2-fsa/OmniVoice/blob/master/docs/OmniVoice.ipynb)

> If you have trouble connecting to HuggingFace when downloading the pre-trained models, set `export HF_ENDPOINT="https://hf-mirror.com"` before running.

For full usage, see the [Python API](#python-api) and [Command-Line Tools](#command-line-tools) sections below.

---

## Python API

OmniVoice supports three generation modes. All features in this section are also available via [command-line tools](#command-line-tools).

### Voice Cloning

Clone a voice from a short reference audio. Provide `ref_audio` and `ref_text`:

```python
from omnivoice import OmniVoice
import soundfile as sf
import torch

model = OmniVoice.from_pretrained(
    "k2-fsa/OmniVoice",
    device_map="cuda:0",
    dtype=torch.float16
)
# Apple Silicon users: use device_map="mps" instead

audio = model.generate(
    text="Hello, this is a test of zero-shot voice cloning.",
    ref_audio="ref.wav",
    ref_text="Transcription of the reference audio.",
) # audio is a list of `np.ndarray` with shape (T,) at 24 kHz.

# If you don't want to input `ref_text` manually, you can directly omit the `ref_text`.
# The model will use Whisper ASR to auto-transcribe it.

sf.write("out.wav", audio[0], 24000)
```

> **Tips**
>
> - Use a 3–10 seconds reference audio clip. Longer audio slows down inference and may degrade cloning quality.
> - For standard pronunciation, use a reference audio in the **same language** as the target speech. In cross-lingual voice cloning (i.e., the reference audio and target speech are in different languages), the generated speech will carry an accent from the reference audio's language.
> - For better results with Arabic numerals, normalize them to words first (e.g., "123" → "one hundred twenty-three") with text normalization tools (e.g., [WeTextProcessing](https://github.com/wenet-e2e/WeTextProcessing)).
>
> For more tips, see [docs/tips.md](docs/tips.md).

### Voice Design

Describe the desired voice with speaker attributes — no reference audio needed.
Supported attributes: **gender** (male/female), **age** (child to elderly),
**pitch** (very low to very high), **style** (whisper), **English accent**
(American, British, etc.), and **Chinese dialect** (四川话, 陕西话, etc.).
Attributes are comma-separated and freely combinable across categories.

```python
audio = model.generate(
    text="Hello, this is a test of zero-shot voice design.",
    instruct="female, low pitch, british accent",
)
```

> **Note**: Voice design was trained on Chinese and English data only. It can generalize to other languages, but results can be unstable for some low-resource languages.

See [docs/voice-design.md](docs/voice-design.md) for the full attribute
reference, Chinese equivalents, and usage tips.

### Auto Voice

Let the model choose a voice automatically:

```python
audio = model.generate(text="This is a sentence without any voice prompt.")
```

### Generation Parameters

All above three modes share the same `model.generate()` API. You can further control the generation behavior via keyword arguments:

```python
audio = model.generate(
    text="...",
    num_step=32,  # diffusion steps (or 16 for faster inference)
    speed=1.0,     # speed factor (>1.0 faster, <1.0 slower)
    duration=10.0, # fixed output duration in seconds (overrides speed)
    # ... more options
)
```
See more detailed control in [docs/generation-parameters.md](docs/generation-parameters.md).

### Non-Verbal & Pronunciation Control

OmniVoice supports inline **non-verbal symbols** and **pronunciation correction** within the input text.

**Non-verbal symbols**: Insert tags like `[laughter]` directly in the text to add expressive non-verbal sounds.

```python
audio = model.generate(text="[laughter] You really got me. I didn't see that coming at all.")
```

Supported tags: `[laughter]`, `[sigh]`, `[confirmation-en]`, `[question-en]`, `[question-ah]`, `[question-oh]`, `[question-ei]`, `[question-yi]`, `[surprise-ah]`, `[surprise-oh]`, `[surprise-wa]`, `[surprise-yo]`, `[dissatisfaction-hnn]`.

**Pronunciation control (Chinese)**: Use pinyin with tone numbers to correct specific character pronunciations.

```python
audio = model.generate(text="这批货物打ZHE2出售后他严重SHE2本了，再也经不起ZHE1腾了。")
```

**Pronunciation control (English)**: Use [CMU pronunciation dictionary](https://svn.code.sf.net/p/cmusphinx/code/trunk/cmudict/cmudict.0.7a)  (uppercase, in brackets) to override default English pronunciations.

```python
audio = model.generate(text="He plays the [B EY1 S] guitar while catching a [B AE1 S] fish.")
```

---

## Command-Line Tools

Three CLI entry points are provided. The CLI tools support all features available in the Python API (voice cloning, voice design, auto voice, generation parameters, etc.) — all controlled via command-line arguments.

| Command | Description | Source |
|---|---|---|
| `omnivoice-demo` | Interactive Gradio web demo | [omnivoice/cli/demo.py](omnivoice/cli/demo.py) |
| `omnivoice-infer` | Single-item inference | [omnivoice/cli/infer.py](omnivoice/cli/infer.py) |
| `omnivoice-infer-batch` | Batch inference across multiple GPUs | [omnivoice/cli/infer_batch.py](omnivoice/cli/infer_batch.py) |

### Demo

```bash
omnivoice-demo --ip 0.0.0.0 --port 8001
```

Provides a web UI for voice cloning and voice design. See `omnivoice-demo --help` for all options.

### Single Inference

```bash
# Voice Cloning
# ref_text can be omitted (Whisper will auto-transcribe ref_audio to get it).
omnivoice-infer \
    --model k2-fsa/OmniVoice \
    --text "This is a test for text to speech." \
    --ref_audio ref.wav \
    --ref_text "Transcription of the reference audio." \
    --output hello.wav

# Voice Design
omnivoice-infer --model k2-fsa/OmniVoice \
    --text "This is a test for text to speech." \
    --instruct "male, British accent" \
    --output hello.wav

# Auto Voice
omnivoice-infer \
    --model k2-fsa/OmniVoice \
    --text "This is a test for text to speech."\
    --output hello.wav
```

### Batch Inference

`omnivoice-infer-batch` can distribute batch inference across multiple GPUs, designed for large-scale TTS tasks.

```bash
omnivoice-infer-batch \
    --model k2-fsa/OmniVoice \
    --test_list test.jsonl \
    --res_dir results/
```

The test list is a JSONL file where each line is a JSON object:
```json
{"id": "sample_001", "text": "Hello world", "ref_audio": "/path/to/ref.wav", "ref_text": "Reference transcript", "instruct": "female, british accent", "language_id": "en", "duration": 10.0, "speed": 1.0}
```
Only `id` and `text` are mandatory fields. `ref_audio` and `ref_text` are used in voice cloning mode. `instruct` is used in voice design mode. If no reference audio or instruct are provided, the model will generate text in a random voice.

`language_id`, `duration`, and `speed` are optional. `duration` (in seconds) fixes the output length; `speed` controls the speaking rate. If `duration` and `speed` are both provided, `speed` will be ignored.

---

## Docker & API

Deploy OmniVoice with Docker Compose — includes a Gradio demo and an OpenAI-compatible REST API.

### Quick Start

```bash
# Create .env file with your HuggingFace token
echo "HF_TOKEN=your_token_here" > .env

# Build and start
docker compose up --build
```

Two services are available:

| Service | Port | Description |
|---|---|---|
| `omnivoice-demo` | 7860 | Gradio web UI |
| `omnivoice-server` | 8000 | OpenAI-compatible REST API |

### REST API

The API is compatible with the [OpenAI Speech endpoint](https://platform.openai.com/docs/api-reference/audio/createSpeech).

#### Create Speech

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "omnivoice",
    "input": "Hello world",
    "voice": "default",
    "response_format": "wav",
    "speed": 1.0
  }' \
  --output speech.wav
```

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | string | `"omnivoice"` | Model identifier |
| `input` | string | — | Text to synthesize |
| `voice` | string | `"default"` | Voice ID (use `default` for auto-selected first voice or system default) |
| `response_format` | string | `"wav"` | Output format: `wav`, `mp3`, `opus`, `flac`, `aac` |
| `speed` | float | `1.0` | Speech speed (`>1.0` faster, `<1.0` slower) |
| `instructions` | string | — | Voice design instructions (overrides voice metadata) |
| `stream_format` | string | `"audio"` | Response format: `"audio"` (binary) or `"sse"` (Server-Sent Events) |
| `seed` | integer | — | Random seed for reproducibility (same seed = same voice) |

**Default Voice:**

When `voice` is `"default"`:
- If any voices exist, the first one in the list is used
- If no voices exist, a system-generated default voice (female, middle-aged) is created automatically

**Reproducibility:**

Use the `seed` parameter to get consistent voice output across calls:

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello world", "seed": 42}' \
  --output speech.wav
```

**SSE Streaming:**

When `stream_format` is `"sse"`, the response is sent as Server-Sent Events with hex-encoded audio chunks:

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Hello world",
    "stream_format": "sse"
  }'
```

Response:
```
data: 52494646...
data: 0a0b0c0d...
data: [DONE]
```

#### Create Voice

**Clone a voice from reference audio** (voice_type defaults to "clone" when audio_sample is provided):

```bash
curl -X POST http://localhost:8000/v1/audio/voices \
  -F "name=my-voice" \
  -F "audio_sample=@reference.wav" \
  -F "ref_text=Transcription of reference audio" \
  -F "consent=I have permission to use this voice"
```

**Create a designed voice** (voice_type defaults to "design" when instructions is provided):

```bash
curl -X POST http://localhost:8000/v1/audio/voices \
  -F "name=custom-voice" \
  -F "instructions=Male / 男, Young Adult / 青年, British Accent / 英国口音"
```

**Explicit voice type:**

```bash
curl -X POST http://localhost:8000/v1/audio/voices \
  -F "name=my-voice" \
  -F "voice_type=clone" \
  -F "audio_sample=@reference.wav"
```

#### List Voices

```bash
curl http://localhost:8000/v1/audio/voices
```

#### Get Voice Details

```bash
curl http://localhost:8000/v1/audio/voices/<voice_id>
```

#### Delete Voice

```bash
curl -X DELETE http://localhost:8000/v1/audio/voices/<voice_id>
```

#### List Models

```bash
curl http://localhost:8000/v1/models
```

### Voice Storage

Voices are saved to `./voices/` (configurable via `VOICES_DIR` env var). Each voice directory contains:

- `metadata.json` — voice settings and config
- `ref.wav` — reference audio (clone type only)

The folder is mounted to `/app/voices` in the container for persistence.

### Model Caching

- **Main models** (`./cache/`) — OmniVoice and HuggingFace models, mounted to `/root/.cache/huggingface`
- **ASR models** (`./asr-cache/`) — Whisper ASR model for auto-transcription, mounted to `/root/.cache/asr`

---

## Training & Evaluation

See [examples/](examples/) for the complete pipeline — from data preparation to training, evaluation, and finetuning.

---

## Discussion & Communication

You can directly discuss on [GitHub Issues](https://github.com/k2-fsa/OmniVoice/issues).

You can also scan the QR code to join our wechat group or follow our wechat official account.

| Wechat Group | Wechat Official Account |
| ------------ | ----------------------- |
|![wechat](https://k2-fsa.org/zh-CN/assets/pic/wechat_group.jpg) |![wechat](https://k2-fsa.org/zh-CN/assets/pic/wechat_account.jpg) |

---

## Community Projects

OmniVoice is supported by a growing ecosystem of community projects.
Explore them in [Community Projects](docs/community-projects.md).

---

## Citation

```bibtex
@article{zhu2026omnivoice,
      title={OmniVoice: Towards Omnilingual Zero-Shot Text-to-Speech with Diffusion Language Models},
      author={Zhu, Han and Ye, Lingxuan and Kang, Wei and Yao, Zengwei and Guo, Liyong and Kuang, Fangjun and Han, Zhifeng and Zhuang, Weiji and Lin, Long and Povey, Daniel},
      journal={arXiv preprint arXiv:2604.00688},
      year={2026}
}
```

---

## Disclaimer

This project is intended only for academic research purposes. Users are strictly prohibited from using this model for unauthorized voice cloning, voice impersonation, fraud, scams, or any other illegal or unethical activities. All users shall ensure full compliance with applicable local laws, regulations, and ethical standards. The developers assume no liability for any misuse of this model and advocate for responsible AI development and use, encouraging the community to uphold safety and ethical principles in AI research and applications.
