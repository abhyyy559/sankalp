#!/usr/bin/env python3
"""
Sankalp Hindi voice check-in — STUB (honest status).

What this is:
  An energy-based voice-activity placeholder. If given a .wav file it
  measures RMS energy / zero-crossing rate and reports whether the clip
  contains speech-like energy. If given no audio, it records an
  operator-confirmed check-in. EITHER WAY the output JSON is labelled
  method="stub" — it is NOT a transcription and NOT Whisper.

What the real port is (documented, not implemented here):
  On-device Whisper (faster-whisper int8, Hindi small model ~120 MB
  quantized) running in the Android app; the patient says a short
  confirmation ("haan, dawa le li"), the transcript must contain an
  agreement keyword before the receipt mints. Airplane-mode capable,
  zero cloud. Model file + NNAPI delegate wiring is future work —
  tracked in README.md, claimed nowhere.

The receipt's voice_ok field records whatever this script outputs, so the
provenance of the voice evidence is always auditable on the receipt itself.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def analyze_wav(path):
    """Speech-like energy heuristic. NOT transcription. NOT Whisper."""
    import wave
    try:
        with wave.open(str(path), "rb") as w:
            n, ch, sw, fr = w.getnframes(), w.getnchannels(), \
                w.getsampwidth(), w.getframerate()
            raw = w.readframes(n)
    except (wave.Error, EOFError, OSError) as e:
        return {"verdict": "unreadable audio file — refused",
                "error": f"{type(e).__name__}: {e}"}
    dtype = {1: np.int8, 2: np.int16, 4: np.int32}[sw]
    pcm = np.frombuffer(raw, dtype=dtype).astype(np.float32)
    if ch > 1:
        pcm = pcm.reshape(-1, ch).mean(axis=1)
    pcm /= (np.abs(pcm).max() + 1e-9)
    frame = max(fr // 20, 1)
    rms = np.sqrt(np.convolve(pcm ** 2, np.ones(frame) / frame, mode="same"))
    speech_frames = float((rms > 0.08).mean())
    return {
        "speech_like_energy_frac": round(speech_frames, 3),
        "verdict": "speech-like energy present"
        if speech_frames > 0.15 else "no speech-like energy",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", help="optional .wav clip to energy-check")
    ap.add_argument("--agree", action="store_true",
                    help="operator confirms the Hindi check-in was spoken")
    ap.add_argument("--out", required=True, help="output JSON path")
    args = ap.parse_args()

    result = {
        "voice_ok": False,
        "method": "stub",
        "stub_note": "NOT Whisper / NOT transcription. Real port: on-device "
                     "faster-whisper int8 (Hindi), airplane-mode, in the "
                     "Android app. See voice/README.",
        "language": "hi",
        "expected_phrase": "haan, dawa le li (yes, I took the dose)",
    }
    if args.wav:
        if not Path(args.wav).exists():
            sys.exit(f"wav not found: {args.wav}")
        result["wav_analysis"] = analyze_wav(args.wav)
        result["voice_ok"] = \
            result["wav_analysis"]["verdict"] == "speech-like energy present"
        result["method"] = "stub:energy-heuristic-on-wav"
    elif args.agree:
        result["voice_ok"] = True
        result["method"] = "stub:operator-confirmed"
        result["operator_note"] = ("operator confirmed the Hindi voice "
                                   "check-in was spoken (demo stand-in for "
                                   "on-device Whisper)")
    Path(args.out).write_text(json.dumps(result, indent=1, ensure_ascii=False))
    print(json.dumps(result, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
