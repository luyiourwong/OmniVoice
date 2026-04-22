import os

import torch

from omnivoice import OmniVoice

MODEL_PATH = os.environ.get("MODEL_PATH", "k2-fsa/OmniVoice")
DEVICE = os.environ.get("DEVICE", None)
ASR_MODEL = os.environ.get("ASR_MODEL", "openai/whisper-large-v3-turbo")

_model = None


def get_model(load_asr: bool = False) -> OmniVoice:
    global _model
    if _model is None:
        device = DEVICE or ("cuda" if torch.cuda.is_available() else "cpu")
        _model = OmniVoice.from_pretrained(
            MODEL_PATH,
            device_map=device,
            dtype=torch.float16 if device == "cuda" else torch.float32,
            load_asr=False,
        )
    elif load_asr and _model._asr_pipe is None:
        _model.load_asr_model(model_name=ASR_MODEL)
        _model._asr_pipe.model.to("cpu")
    return _model
