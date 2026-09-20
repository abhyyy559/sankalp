package org.sankalp

/**
 * Data model for the offline dose-counting pipeline.
 *
 * A medicine blister strip is photographed, and the on-device model counts
 * doses by classifying each slot as FULL (pill present), PRESSED (pill
 * taken, foil pushed through) or UNCERTAIN (cannot tell reliably).
 */
enum class SlotStatus { FULL, PRESSED, UNCERTAIN }

enum class GateStatus { ACCEPTED, REFUSED }

/** One detected slot in the blister-strip grid. */
data class Slot(
    val id: Int,
    /** Normalized center of the slot in warped-strip coordinates [0, 1]. */
    val cx: Float,
    val cy: Float,
    val status: SlotStatus,
    /** Classifier confidence in [0, 1]. */
    val confidence: Float,
)

/** Outcome of counting one strip photo. */
data class StripCountResult(
    val slots: List<Slot>,
    /** Fraction of slots classified UNCERTAIN. */
    val uncertainFrac: Float,
    val status: GateStatus,
    val latencyMs: Long,
    /** Which delegate would run / ran inference: "CPU-HEURISTIC", "CPU", "NNAPI", "GPU". */
    val delegate: String,
) {
    /** Remaining doses = slots classified FULL. Only valid when [status] is ACCEPTED. */
    val doseCount: Int get() = slots.count { it.status == SlotStatus.FULL }
}
