# Sankalp — Android skeleton (prototype)

On-device, offline TB dose-evidence app. This is a **code skeleton**: coherent,
reviewable, but not yet built, run, or measured on a device.

## Honest status

This skeleton has **NOT** been compiled into an APK here; delegates
(NNAPI/NPU) have **NOT** run; the counting heuristic is ported from the
Python reference which measured its accuracy on CPU (see ../../README.md).
No accuracy or latency numbers are claimed for the Android path.

| Area | Status |
|---|---|
| Data model (`SlotResult.kt`) | Written |
| Heuristic foil counter (`FoilCounter.kt`) — grayscale, CLAHE, quad detect, warp, grid, per-slot heuristic, confidence gate (refuse if uncertain > 12%) | Written, not compiled, not run |
| TFLite classifier path (`TfliteCounter.kt`) — GPU → NNAPI → CPU delegate order | Written, **model not trained/exported, path NOT wired to run** |
| CameraX capture + count + purge (`CameraActivity.kt`) — raw bitmap purged after counting, never saved | Written, not compiled, not run |
| Receipt signing (`ReceiptSigner.kt`) — SHA-256 chain, Ed25519 (Tink preferred, JCA fallback API 33+), app-private storage only | Written, not compiled, not run |
| Hindi voice check-in (`VoiceCheckin.kt`) — planned on-device faster-whisper int8 | **STUB only** |
| Build scripts (`build.gradle.kts`, `app/build.gradle.kts`) | Written, not built |
| APK | Does not exist |
| NNAPI / NPU execution | Has not happened; do not claim it |
| Measured accuracy / latency (Android) | None; do not claim any |

## Build notes

- minSdk 28, namespace `org.sankalp`.
- OpenCV Android SDK AAR is required; add it as a local module and
  uncomment the `openCVLibrary` line in `app/build.gradle.kts`, then call
  `OpenCVLoader.initLocal()` before first use of `FoilCounter`.
- The `.tflite` slot classifier does not exist yet; `TfliteCounter.create()`
  returns null until a model file is placed in the app files dir.

## Files

- `app/src/main/java/org/sankalp/SlotResult.kt` — slot/result data classes
- `app/src/main/java/org/sankalp/FoilCounter.kt` — heuristic counting pipeline
- `app/src/main/java/org/sankalp/TfliteCounter.kt` — planned learned classifier (not wired)
- `app/src/main/java/org/sankalp/CameraActivity.kt` — capture → count → purge → sign
- `app/src/main/java/org/sankalp/ReceiptSigner.kt` — chained, signed receipts
- `app/src/main/java/org/sankalp/VoiceCheckin.kt` — STUB
