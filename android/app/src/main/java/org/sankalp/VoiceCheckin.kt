package org.sankalp

/**
 * Voice check-in STUB.
 *
 * HONEST STATUS: this is NOT implemented. The real port is an on-device
 * Hindi voice check-in built on faster-whisper (int8) listening for a
 * confirmation keyword/phrase (e.g. "हो गया" / "दवाई ले ली") that must
 * agree with the foil count before a receipt mints.
 *
 * Nothing here records audio, runs Whisper, or understands Hindi.
 * Until the port lands, [voiceAgrees] returns a STUB status so the rest of
 * the pipeline (count -> gate -> receipt) can be exercised end-to-end.
 * The stub never fabricates an agreement.
 */
object VoiceCheckin {

    enum class VoiceStatus { AGREED, DISAGREED, STUB, ERROR }

    /** Outcome of the voice check-in. */
    data class VoiceResult(
        val status: VoiceStatus,
        /** Transcript text when available; null for the stub. */
        val transcript: String?,
        /** True only when status == AGREED (real check-in agreed). */
        val agreed: Boolean,
    )

    /**
     * Real behaviour (planned):
     *  1. Record a short utterance on-device (nothing leaves the phone).
     *  2. Run on-device faster-whisper int8, Hindi language.
     *  3. Keyword/confirmation detection vs. the counted dose.
     *  4. Return AGREED only if the patient confirmed taking the dose.
     *
     * Current behaviour: returns STUB — agreed=false — so the pipeline
     * treats the check-in as not yet performed. Callers (see
     * [CameraActivity]) must decide how to handle the STUB state during
     * development (e.g. skip voice for demo, never minting silently).
     */
    fun voiceAgrees(): VoiceResult =
        VoiceResult(status = VoiceStatus.STUB, transcript = null, agreed = false)
}
