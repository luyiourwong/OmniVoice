#!/usr/bin/env python3

import io
import json
import logging
import os
import random
import shutil
import uuid

import torch

import soundfile as sf
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from omnivoice import OmniVoiceGenerationConfig
from omnivoice.shared import ASR_MODEL, get_model

DEBUG_LOGGING = os.environ.get("DEBUG_LOGGING", "false").lower() == "true"
DETERMINISTIC_MODE = os.environ.get("DETERMINISTIC_MODE", "true").lower() == "true"
DEFAULT_SEED = 42
log_level = logging.DEBUG if DEBUG_LOGGING else logging.INFO


def _normalize_seed(seed: int | None) -> int:
    if seed is None or seed <= 0:
        return random.randint(1, 2**31 - 1)
    return seed
logging.basicConfig(level=log_level)
logger = logging.getLogger(__name__)

app = FastAPI(title="OmniVoice TTS API")

VOICES_DIR = os.environ.get("VOICES_DIR", "./voices")
SUPPORTED_FORMATS = {"wav", "mp3", "opus", "flac", "aac"}
DEFAULT_FORMAT = "wav"
DEFAULT_MODEL = "omnivoice"
DEFAULT_VOICE = "default"
DEFAULT_SPEED = 1.0

os.makedirs(VOICES_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class VoiceCreateClone(BaseModel):
    name: str
    ref_audio_url: str | None = None
    ref_text: str | None = None
    language: str | None = None
    generation_config: dict = {}


class VoiceCreateDesign(BaseModel):
    name: str
    instruct: str
    language: str | None = None
    generation_config: dict = {}


class VoiceResponse(BaseModel):
    id: str
    name: str
    type: str  # "clone" or "design"
    language: str | None = None
    instructions: str | None = None
    consent: str | None = None
    seed: int | None = None
    created_at: str


class SpeechRequest(BaseModel):
    model: str = DEFAULT_MODEL
    input: str
    voice: str | None = None
    response_format: str = DEFAULT_FORMAT
    speed: float = DEFAULT_SPEED
    instructions: str | None = None
    stream_format: str = "audio"
    # Random seed for this request; overrides the voice's seed (same seed = same output).
    seed: int | None = None


# Generation settings for a one-off designed voice (same as newly created voices).
DEFAULT_DESIGN_CONFIG = {"num_step": 32, "guidance_scale": 2.0, "denoise": True}


def _set_seed(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)


# ---------------------------------------------------------------------------
# Voice storage helpers
# ---------------------------------------------------------------------------


def _list_voice_dirs() -> list[str]:
    if not os.path.isdir(VOICES_DIR):
        return []
    return [
        d
        for d in os.listdir(VOICES_DIR)
        if os.path.isdir(os.path.join(VOICES_DIR, d))
    ]


def _load_voice_meta(voice_id: str) -> dict | None:
    meta_path = os.path.join(VOICES_DIR, voice_id, "metadata.json")
    if not os.path.isfile(meta_path):
        return None
    with open(meta_path, "r") as f:
        return json.load(f)


def _save_voice_meta(voice_id: str, meta: dict) -> None:
    voice_dir = os.path.join(VOICES_DIR, voice_id)
    os.makedirs(voice_dir, exist_ok=True)
    meta_path = os.path.join(voice_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)


def _delete_voice(voice_id: str) -> bool:
    voice_dir = os.path.join(VOICES_DIR, voice_id)
    if os.path.isdir(voice_dir):
        shutil.rmtree(voice_dir)
        return True
    return False


def _find_voice_by_name(name: str) -> str | None:
    for vid in _list_voice_dirs():
        meta = _load_voice_meta(vid)
        if meta and meta.get("name") == name:
            return vid
    return None


# ---------------------------------------------------------------------------
# Voice endpoints
# ---------------------------------------------------------------------------


@app.post("/v1/audio/voices", status_code=201)
async def create_voice(
    name: str = Form(...),
    voice_type: str = Form(None),
    audio_sample: UploadFile = File(None),
    ref_text: str | None = Form(None),
    language: str | None = Form(None),
    instructions: str | None = Form(None),
    num_step: int = Form(32),
    guidance_scale: float = Form(2.0),
    denoise: bool = Form(True),
    speed: float = Form(1.0),
    consent: str | None = Form(None),
):
    """Create a new voice — either by cloning from reference audio or by design."""
    logger.debug("create_voice: name=%s, voice_type=%s, language=%s", name, voice_type, language)
    if voice_type is None:
        if audio_sample:
            voice_type = "clone"
        elif instructions:
            voice_type = "design"
        else:
            raise HTTPException(400, "Either audio_sample or instructions must be provided")

    if voice_type not in ("clone", "design"):
        raise HTTPException(400, "voice_type must be 'clone' or 'design'")

    if voice_type == "clone":
        if not audio_sample:
            raise HTTPException(400, "audio_sample is required for clone type")
        content = await audio_sample.read()
    else:
        if not instructions:
            raise HTTPException(400, "instructions is required for design type")
        content = None

    voice_id = str(uuid.uuid4())[:8]
    voice_seed = _normalize_seed(None)
    meta = {
        "id": voice_id,
        "name": name,
        "type": voice_type,
        "language": language,
        "instructions": instructions,
        "ref_text": ref_text,
        "consent": consent,
        "seed": voice_seed,
        "generation_config": {
            "num_step": num_step,
            "guidance_scale": guidance_scale,
            "denoise": denoise,
        },
    }

    voice_dir = os.path.join(VOICES_DIR, voice_id)
    os.makedirs(voice_dir, exist_ok=True)

    if content:
        audio_ext = "wav"
        audio_path = os.path.join(VOICES_DIR, voice_id, f"ref.{audio_ext}")
        with open(audio_path, "wb") as f:
            f.write(content)
        meta["ref_audio_file"] = f"ref.{audio_ext}"

        if not ref_text:
            m = get_model()
            if m._asr_pipe is None:
                m.load_asr_model(model_name=ASR_MODEL)
            wav_buf = io.BytesIO(content)
            waveform, sr = sf.read(wav_buf)
            if waveform.ndim > 1:
                waveform = waveform.mean(axis=1)
            transcript = m.transcribe((waveform, sr))
            meta["ref_text"] = transcript

    _save_voice_meta(voice_id, meta)
    logger.debug("create_voice: created voice_id=%s, seed=%s", voice_id, voice_seed)

    return VoiceResponse(
        id=voice_id,
        name=name,
        type=voice_type,
        language=language,
        instructions=instructions,
        created_at=str(int(__import__("time").time())),
    )


@app.get("/v1/audio/voices")
async def list_voices():
    voice_ids = _list_voice_dirs()
    logger.debug("list_voices: found %d voice dirs", len(voice_ids))
    voices = []
    for vid in voice_ids:
        meta = _load_voice_meta(vid)
        if meta:
            voices.append(
                VoiceResponse(
                    id=meta["id"],
                    name=meta["name"],
                    type=meta["type"],
                    language=meta.get("language"),
                    instructions=meta.get("instructions"),
                    consent=meta.get("consent"),
                    seed=meta.get("seed"),
                    created_at=meta.get("created_at", ""),
                )
            )
    return {"voices": voices}


@app.get("/v1/audio/voices/{voice_id}")
async def get_voice(voice_id: str):
    meta = _load_voice_meta(voice_id)
    logger.debug("get_voice: voice_id=%s, found=%s", voice_id, meta is not None)
    if not meta:
        raise HTTPException(404, f"Voice '{voice_id}' not found")
    return VoiceResponse(
        id=meta["id"],
        name=meta["name"],
        type=meta["type"],
        language=meta.get("language"),
        instructions=meta.get("instructions"),
        consent=meta.get("consent"),
        seed=meta.get("seed"),
        created_at=meta.get("created_at", ""),
    )


class VoiceUpdate(BaseModel):
    name: str | None = None
    language: str | None = None
    instructions: str | None = None
    num_step: int | None = None
    guidance_scale: float | None = None
    denoise: bool | None = None
    seed: int | None = None


@app.patch("/v1/audio/voices/{voice_id}")
async def update_voice(voice_id: str, updates: VoiceUpdate):
    """Update settings of an existing voice."""
    meta = _load_voice_meta(voice_id)
    if not meta:
        raise HTTPException(404, f"Voice '{voice_id}' not found")
    logger.debug("update_voice: voice_id=%s, updates=%s", voice_id, updates.model_dump(exclude_unset=True))
    updated_fields = updates.model_fields_set
    if "name" in updated_fields:
        meta["name"] = updates.name
    if "language" in updated_fields:
        meta["language"] = updates.language
    if "instructions" in updated_fields:
        meta["instructions"] = updates.instructions
    if "num_step" in updated_fields or "guidance_scale" in updated_fields or "denoise" in updated_fields:
        meta.setdefault("generation_config", {})
        if "num_step" in updated_fields:
            meta["generation_config"]["num_step"] = updates.num_step
        if "guidance_scale" in updated_fields:
            meta["generation_config"]["guidance_scale"] = updates.guidance_scale
        if "denoise" in updated_fields:
            meta["generation_config"]["denoise"] = updates.denoise
    if "seed" in updated_fields:
        meta["seed"] = _normalize_seed(updates.seed)
    _save_voice_meta(voice_id, meta)
    return VoiceResponse(
        id=meta["id"],
        name=meta["name"],
        type=meta["type"],
        language=meta.get("language"),
        instructions=meta.get("instructions"),
        consent=meta.get("consent"),
        seed=meta.get("seed"),
        created_at=meta.get("created_at", ""),
    )


@app.delete("/v1/audio/voices/{voice_id}")
async def delete_voice(voice_id: str):
    logger.debug("delete_voice: voice_id=%s", voice_id)
    if not _delete_voice(voice_id):
        raise HTTPException(404, f"Voice '{voice_id}' not found")
    return {"deleted": voice_id}


# ---------------------------------------------------------------------------
# Speech endpoint
# ---------------------------------------------------------------------------


@app.post("/v1/audio/speech")
async def create_speech(req: SpeechRequest):
    logger.debug("create_speech: voice=%s, text_len=%d, format=%s, speed=%s", req.voice, len(req.input), req.response_format, req.speed)
    if req.response_format not in SUPPORTED_FORMATS:
        raise HTTPException(400, f"Unsupported format: {req.response_format}")
    if not req.input or not req.input.strip():
        raise HTTPException(400, "Input text is empty")

    m = get_model()

    gen_kwargs = {"speed": req.speed}

    # No voice but instructions: design a voice for this request only, instead of falling
    # back to the first saved voice (which ignored the instructions when it was a clone).
    if req.instructions and req.voice in (None, "", DEFAULT_VOICE):
        gen_kwargs.update(DEFAULT_DESIGN_CONFIG)
        gen_kwargs["instruct"] = req.instructions
        seed = req.seed if req.seed is not None else DEFAULT_SEED
        logger.debug("create_speech: one-off design voice, instruct=%s, seed=%s", req.instructions, seed)
        return _speech_response(req, m, gen_kwargs, seed)

    voice_id = _find_voice_by_name(req.voice) or req.voice
    logger.debug("create_speech: resolved voice_id=%s (from=%s)", voice_id, req.voice)
    if not voice_id or voice_id == "default":
        voice_ids = sorted(_list_voice_dirs())
        if voice_ids:
            for vid in voice_ids:
                meta = _load_voice_meta(vid)
                if meta and meta.get("type") in ("clone", "design"):
                    voice_id = vid
                    break
            else:
                voice_id = voice_ids[0]
        else:
            default_voice_id = "default-female-middle"
            default_seed = _normalize_seed(None)
            default_meta = {
                "id": default_voice_id,
                "name": "default-female-middle",
                "type": "design",
                "language": None,
                "instructions": "female, middle-aged",
                "consent": "system-generated-default",
                "seed": default_seed,
                "generation_config": {
                    "num_step": 32,
                    "guidance_scale": 2.0,
                    "denoise": True,
                },
            }
            _save_voice_meta(default_voice_id, default_meta)
            voice_id = default_voice_id

    voice_meta = _load_voice_meta(voice_id)
    if voice_meta:
        if "seed" not in voice_meta or voice_meta["seed"] is None:
            voice_meta["seed"] = DEFAULT_SEED
            _save_voice_meta(voice_id, voice_meta)
            logger.debug("create_speech: added default seed=%s to voice %s", voice_meta["seed"], voice_id)
        if "name" not in voice_meta or not voice_meta["name"]:
            voice_meta["name"] = voice_id
            _save_voice_meta(voice_id, voice_meta)
    elif os.path.isdir(os.path.join(VOICES_DIR, voice_id)):
        logger.debug("create_speech: creating metadata for legacy voice %s", voice_id)
        ref_audio_path = os.path.join(VOICES_DIR, voice_id, "ref.wav")
        is_clone = os.path.isfile(ref_audio_path)
        voice_meta = {
            "id": voice_id,
            "name": voice_id,
            "type": "clone" if is_clone else "design",
            "language": None,
            "instructions": None,
            "consent": None,
            "seed": DEFAULT_SEED,
            "generation_config": {
                "num_step": 32,
                "guidance_scale": 2.0,
                "denoise": True,
            },
        }
        if is_clone:
            voice_meta["ref_audio_file"] = "ref.wav"
        _save_voice_meta(voice_id, voice_meta)
    seed = voice_meta.get("seed") if voice_meta else DEFAULT_SEED
    if req.seed is not None:
        seed = req.seed  # the request's seed wins over the voice's
    if not voice_meta and req.voice not in (None, "", "default"):
        available = sorted(_list_voice_dirs())
        raise HTTPException(
            404,
            f"Voice '{req.voice}' not found. Available voices: {available}",
        )

    logger.debug("create_speech: final voice_id=%s, voice_name=%s, seed=%s", voice_id, voice_meta.get("name") if voice_meta else None, seed)

    if voice_meta:
        gen_cfg = voice_meta.get("generation_config", {})
        gen_kwargs.update(gen_cfg)
        if "language" in voice_meta and voice_meta["language"]:
            gen_kwargs["language"] = voice_meta["language"]
        if voice_meta["type"] == "clone":
            ref_audio_path = os.path.join(VOICES_DIR, voice_id, voice_meta.get("ref_audio_file", "ref.wav"))
            if os.path.isfile(ref_audio_path):
                gen_kwargs["ref_audio"] = ref_audio_path
                gen_kwargs["ref_text"] = voice_meta.get("ref_text")
        elif voice_meta["type"] == "design":
            instruct = req.instructions or voice_meta.get("instructions")
            if instruct:
                gen_kwargs["instruct"] = instruct

    return _speech_response(req, m, gen_kwargs, seed)


def _speech_response(req: SpeechRequest, m, gen_kwargs: dict, seed: int | None):
    """Generate the speech and return it in the requested format (or as SSE)."""
    try:
        _set_seed(seed)
        logger.debug("create_speech: deterministic_mode=%s, using seed=%s", DETERMINISTIC_MODE, seed)
        audio = m.generate(text=req.input, generation_config=OmniVoiceGenerationConfig(), **gen_kwargs)[0]
        logger.debug("create_speech: generated audio_len=%d, seed=%s, gen_kwargs=%s", len(audio), seed, gen_kwargs)
    except Exception as e:
        logger.exception("Generation failed")
        raise HTTPException(500, str(e))

    if req.stream_format == "sse":
        async def audio_sse_generator():
            chunk_size = 16000  # ~100ms at 16kHz
            for i in range(0, len(audio), chunk_size):
                chunk = audio[i : i + chunk_size]
                chunk_buffer = io.BytesIO()
                sf.write(chunk_buffer, chunk, m.sampling_rate, format="wav")
                chunk_data = chunk_buffer.getvalue()
                yield f"data: {chunk_data.hex()}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            audio_sse_generator(),
            media_type="text/event-stream",
            headers={
                "X-Accel-Buffering": "no",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )

    buffer = io.BytesIO()

    fmt = req.response_format
    if fmt == "wav":
        sf.write(buffer, audio, m.sampling_rate, format="wav")
    elif fmt == "flac":
        sf.write(buffer, audio, m.sampling_rate, format="flac")
    elif fmt in ("mp3", "opus", "aac"):
        tmp = io.BytesIO()
        sf.write(tmp, audio, m.sampling_rate, format="wav")
        tmp.seek(0)
        try:
            from pydub import AudioSegment

            seg = AudioSegment.from_wav(tmp)
            seg.export(buffer, format=fmt)
        except Exception:
            buffer = tmp

    buffer.seek(0)
    content_type = f"audio/{fmt}"
    return StreamingResponse(buffer, media_type=content_type)


@app.get("/v1/models")
async def list_models():
    return {"data": [{"id": DEFAULT_MODEL, "object": "model", "created": 0, "owned_by": "omnivoice"}]}


def main():
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
