package org.sankalp

import android.graphics.Bitmap
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc

/**
 * Heuristic foil-strip dose counter.
 *
 * Pipeline: grayscale -> CLAHE contrast enhancement -> strip quad detection ->
 * perspective warp -> slot grid detection -> per-slot heuristic classification
 * -> confidence gate.
 *
 * IMPORTANT: this file is part of the prototype skeleton. It has NOT been
 * compiled into an APK, and no accuracy or latency numbers are claimed for
 * this Android path. The heuristic logic here is a port of the Python
 * reference implementation (see ../../README.md), which measured its accuracy
 * on desktop CPU only.
 *
 * Requires OpenCV Android SDK (org.opencv.*). Add the SDK AAR and call
 * OpenCVLoader.initLocal() before first use.
 */
class FoilCounter {

    companion object {
        /** Reject the count when more than this fraction of slots is uncertain. */
        private const val UNCERTAIN_REFUSAL_THRESHOLD = 0.12f

        /** Claimed accuracy of this heuristic on Android: none. Not measured. */
        const val MEASURED_ACCURACY: String? = null
    }

    /**
     * Counts remaining doses on a blister strip photographed by [bitmap].
     *
     * @param expectedRows/expectedCols grid geometry of this strip format.
     */
    fun countStrip(bitmap: Bitmap, expectedRows: Int, expectedCols: Int): StripCountResult {
        val t0 = System.nanoTime()

        // 1) Bitmap -> Mat, grayscale.
        val rgba = Mat()
        Utils.bitmapToMat(bitmap, rgba)
        val gray = Mat()
        Imgproc.cvtColor(rgba, gray, Imgproc.COLOR_RGBA2GRAY)

        // 2) CLAHE (Contrast Limited Adaptive Histogram Equalization) to even
        //    out uneven lighting before thresholding.
        val clahe = Imgproc.createCLAHE(2.0, Size(8.0, 8.0))
        val equalized = Mat()
        clahe.apply(gray, equalized)

        // 3) Strip quad detection: blur, adaptive threshold, find largest
        //    4-point contour.
        val blurred = Mat()
        Imgproc.GaussianBlur(equalized, blurred, Size(5.0, 5.0), 0.0)
        val binary = Mat()
        Imgproc.adaptiveThreshold(
            blurred, binary, 255.0,
            Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY_INV, 51, 8.0
        )
        val quad = findLargestQuad(binary)
        val warped = if (quad != null) {
            warpQuad(equalized, quad, warpedWidth = 480, warpedHeight = 960)
        } else {
            // No quad found: fall back to the whole frame (weak, likely to
            // fail the confidence gate below, which is the correct behaviour).
            equalized.clone()
        }

        // 4) Slot grid detection: divide the warped strip into
        //    expectedRows x expectedCols cells.
        val cells = gridCells(warped, expectedRows, expectedCols)

        // 5) Per-slot classification with confidence.
        val slots = cells.mapIndexed { i, cell ->
            classifySlot(warped, cell, i)
        }

        // 6) Confidence gate.
        val uncertainFrac =
            slots.count { it.status == SlotStatus.UNCERTAIN }.toFloat() / slots.size.coerceAtLeast(1)
        val status = if (uncertainFrac > UNCERTAIN_REFUSAL_THRESHOLD) GateStatus.REFUSED
        else GateStatus.ACCEPTED

        val latencyMs = (System.nanoTime() - t0) / 1_000_000L
        return StripCountResult(
            slots = slots,
            uncertainFrac = uncertainFrac,
            status = status,
            latencyMs = latencyMs,
            delegate = "CPU-HEURISTIC",
        )
    }

    // ------------------------------------------------------------------
    // Internal pipeline stages.
    // ------------------------------------------------------------------

    private data class Cell(val x: Int, val y: Int, val w: Int, val h: Int)

    private fun findLargestQuad(binary: Mat): MatOfPoint2f? {
        val contours = mutableListOf<MatOfPoint>()
        val hierarchy = Mat()
        Imgproc.findContours(binary, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)
        var best: MatOfPoint2f? = null
        var bestArea = 0.0
        for (c in contours) {
            val area = Imgproc.contourArea(c)
            if (area < 10_000.0) continue // ignore small noise
            val approx = MatOfPoint2f()
            val curve = MatOfPoint2f(*c.toArray())
            Imgproc.approxPolyDP(curve, approx, 0.02 * Imgproc.arcLength(curve, true), true)
            if (approx.toArray().size == 4 && area > bestArea) {
                bestArea = area
                best = approx
            }
        }
        return best?.let { orderQuad(it) }
    }

    private fun orderQuad(quad: MatOfPoint2f): MatOfPoint2f {
        val pts = quad.toArray().sortedBy { it.y }
        val top = pts.take(2).sortedBy { it.x }
        val bottom = pts.takeLast(2).sortedBy { it.x }
        return MatOfPoint2f(top[0], top[1], bottom[1], bottom[0])
    }

    private fun warpQuad(src: Mat, quad: MatOfPoint2f, warpedWidth: Int, warpedHeight: Int): Mat {
        val dstPts = MatOfPoint2f(
            Point(0.0, 0.0),
            Point(warpedWidth.toDouble(), 0.0),
            Point(warpedWidth.toDouble(), warpedHeight.toDouble()),
            Point(0.0, warpedHeight.toDouble()),
        )
        val m = Imgproc.getPerspectiveTransform(quad, dstPts)
        val out = Mat()
        Imgproc.warpPerspective(src, out, m, Size(warpedWidth.toDouble(), warpedHeight.toDouble()))
        return out
    }

    private fun gridCells(warped: Mat, rows: Int, cols: Int): List<Cell> {
        val cellW = warped.cols() / cols
        val cellH = warped.rows() / rows
        return buildList {
            for (r in 0 until rows) for (c in 0 until cols) {
                add(Cell(c * cellW, r * cellH, cellW, cellH))
            }
        }
    }

    /**
     * Heuristic slot classifier — same spirit as the Python reference:
     *  - high dark-pill fraction in the center  -> FULL
     *  - bright, flat (low-variance) center    -> PRESSED (empty)
     *  - otherwise                             -> UNCERTAIN, confidence = margin
     *
     * No learned weights run here; see [TfliteCounter] for the planned
     * learned-classifier replacement.
     */
    private fun classifySlot(warped: Mat, cell: Cell, id: Int): Slot {
        // Center crop (60%) to avoid slot borders.
        val cx0 = cell.x + (cell.w * 0.2).toInt()
        val cy0 = cell.y + (cell.h * 0.2).toInt()
        val cw = (cell.w * 0.6).toInt()
        val ch = (cell.h * 0.6).toInt()
        val roi = warped.submat(cy0.coerceIn(0, warped.rows() - 1),
            (cy0 + ch).coerceIn(1, warped.rows()),
            cx0.coerceIn(0, warped.cols() - 1),
            (cx0 + cw).coerceIn(1, warped.cols()))

        // Dark-pill fraction: pixels darker than threshold relative to the ROI.
        val meanScalar = Core.mean(roi)
        val mean = meanScalar.`val`[0]
        val darkThresh = mean * 0.75
        val dark = Mat()
        Imgproc.threshold(roi, dark, darkThresh, 255.0, Imgproc.THRESH_BINARY_INV)
        val darkFrac = Core.countNonZero(dark).toDouble() / (roi.rows() * roi.cols()).coerceAtLeast(1)

        // Brightness and flatness (variance) of the center.
        val stddevMat = MatOfDouble(), meanMat = MatOfDouble()
        Core.meanStdDev(roi, meanMat, stddevMat)
        val std = stddevMat.toArray()[0]
        val centerMean = meanMat.toArray()[0]

        val fullScore = darkFrac                       // ~0.05..0.45 typical
        val pressedScore = (centerMean / 255.0) * (1.0 - (std / 80.0).coerceIn(0.0, 1.0))

        val (status, confidence) = when {
            fullScore > 0.28 && fullScore - pressedScore > 0.08 ->
                SlotStatus.FULL to (0.5f + (fullScore - pressedScore).toFloat() / 2f)
            pressedScore > 0.55 && pressedScore - fullScore > 0.08 ->
                SlotStatus.PRESSED to (0.5f + (pressedScore - fullScore).toFloat() / 2f)
            else ->
                // UNCERTAIN: confidence is how close the two scores are (the
                // *lack* of a decisive margin).
                SlotStatus.UNCERTAIN to
                    (1.0f - kotlin.math.abs(fullScore - pressedScore).toFloat()).coerceIn(0f, 1f)
        }
        return Slot(
            id = id,
            cx = (cell.x + cell.w / 2f) / warped.cols(),
            cy = (cell.y + cell.h / 2f) / warped.rows(),
            status = status,
            confidence = confidence.coerceIn(0f, 1f),
        )
    }
}
