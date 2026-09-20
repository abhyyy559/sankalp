# Sankalp voice check-in — STUB

## Honest status

**This is not Whisper. It is not transcription.** `checkin.py` is a stub with
two modes:

| Mode | What it does |
|---|---|
| `--agree` | Operator confirms the Hindi check-in was spoken. Output JSON says `method: "stub:operator-confirmed"`. |
| `--wav clip.wav` | Measures RMS energy / zero-crossing-style speech-activity on the clip. Output says `method: "stub:energy-heuristic-on-wav"`. Detects *that someone spoke*, not *what they said*. |

Every output carries `stub_note: "NOT Whisper / NOT transcription"`, and the
receipt's `voice_method` field records the provenance, so a verifier can
always see exactly what kind of voice evidence backed a receipt.

## The real port (documented, not implemented)

The Android app (`android/`) will run on-device **faster-whisper int8**
(Hindi small model, ~120 MB quantized) in airplane mode: the patient says
*"haan, dawa le li"* ("yes, I took the dose"), the transcript must contain
an agreement keyword before the receipt mints. No audio leaves the phone.
Model file + NNAPI delegate wiring is future work — tracked here, claimed
nowhere.

## Usage

```bash
python3 voice/checkin.py --agree --out /tmp/voice.json
python3 voice/checkin.py --wav clip.wav --out /tmp/voice.json
```
