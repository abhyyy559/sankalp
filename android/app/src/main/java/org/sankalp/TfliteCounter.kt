package org.sankalp

import android.content.Context
import android.graphics.Bitmap
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.nnapi.NnApiDelegate
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Planned learned slot-classifier replacing [FoilCounter]'s heuristic.
 *
 * HONEST STATUS (read before touching this file):
 *  - The quantized .tflite slot-classifier model has NOT been trained or
 *    exported. This path is NOT wired to run anywhere in the app.
 *  - NNAPI / NPU / GPU delegates have NOT run on any device. Do NOT claim
 *    NPU execution, GPU execution, or any measured latency for this path.
 *  - When a model exists, call [load] with a File pointing at it, then use
 *    [classifySlotPatch] to score per-slot crops and feed the results into
 *    the same confidence gate used by [FoilCounter].
 *
 * Delegate selection order once a model ships: GPU -> NNAPI -> CPU (XNNPACK).
 */
class TfliteCounter private constructor(
    private val interpreter: Interpreter,
    private val delegates: List<AutoCloseable>,
    /** Which delegate was actually selected, e.g. "GPU", "NNAPI", "CPU". */
    val delegate: String,
) : AutoCloseable {

    companion object {
        private const val MODEL_FILE_NAME = "slot_classifier_int8.tflite"
        private const val INPUT_SIZE = 64
        private const val NUM_CLASSES = 3 // FULL, PRESSED, UNCERTAIN

        /**
         * Creates the classifier. Returns null if the model file is missing —
         * callers must fall back to [FoilCounter] (the heuristic path).
         */
        fun create(context: Context): TfliteCounter? {
            val modelFile = File(context.filesDir, MODEL_FILE_NAME)
            if (!modelFile.exists()) return null // model not yet exported

            val options = Interpreter.Options()
            var delegateName = "CPU"
            var gpuDelegate: GpuDelegate? = null
            var nnapiDelegate: NnApiDelegate? = null

            // Preferred: GPU delegate.
            try {
                gpuDelegate = GpuDelegate()
                options.addDelegate(gpuDelegate)
                delegateName = "GPU"
            } catch (_: Throwable) {
                // fall through to NNAPI
            }
            // Next: NNAPI.
            if (gpuDelegate == null) {
                try {
                    nnapiDelegate = NnApiDelegate()
                    options.addDelegate(nnapiDelegate)
                    delegateName = "NNAPI"
                } catch (_: Throwable) {
                    // fall through to CPU
                }
            }
            // CPU fallback with XNNPACK acceleration.
            if (gpuDelegate == null && nnapiDelegate == null) {
                options.setUseXNNPACK(true)
                options.setNumThreads(Runtime.getRuntime().availableProcessors().coerceAtMost(4))
                delegateName = "CPU"
            }

            val buffer = loadModelFile(modelFile)
            val interpreter = Interpreter(buffer, options)
            val delegates = listOfNotNull(gpuDelegate, nnapiDelegate)
            return TfliteCounter(interpreter, delegates, delegateName)
        }

        private fun loadModelFile(file: File): ByteBuffer {
            val bytes = file.readBytes()
            return ByteBuffer.allocateDirect(bytes.size)
                .order(ByteOrder.nativeOrder())
                .apply {
                    put(bytes)
                    rewind()
                }
        }
    }

    /**
     * Classifies one 64x64 slot patch. Returns a probability triple
     * (pFull, pPressed, pUncertain). NOT called in production yet.
     */
    fun classifySlotPatch(patch: Bitmap): FloatArray {
        val input = ByteBuffer.allocateDirect(4 * INPUT_SIZE * INPUT_SIZE * 3)
            .order(ByteOrder.nativeOrder())
        val pixels = IntArray(INPUT_SIZE * INPUT_SIZE)
        val scaled = Bitmap.createScaledBitmap(patch, INPUT_SIZE, INPUT_SIZE, true)
        scaled.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)
        for (p in pixels) {
            input.putFloat(((p shr 16) and 0xFF) / 255f)
            input.putFloat(((p shr 8) and 0xFF) / 255f)
            input.putFloat((p and 0xFF) / 255f)
        }
        input.rewind()
        val output = Array(1) { FloatArray(NUM_CLASSES) }
        interpreter.run(input, output)
        return output[0]
    }

    override fun close() {
        interpreter.close()
        delegates.forEach { it.close() }
    }
}
