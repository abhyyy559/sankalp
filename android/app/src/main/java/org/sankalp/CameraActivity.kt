package org.sankalp

import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

/**
 * Capture flow: CameraX -> FoilCounter (background) -> raw purge ->
 * voice check-in -> receipt signing.
 *
 * Privacy rule: the raw photo is NEVER written to disk. It lives only in
 * memory long enough to count, then is purged before any receipt is minted.
 */
class CameraActivity : AppCompatActivity() {

    private var imageCapture: ImageCapture? = null
    private lateinit var cameraExecutor: ExecutorService
    private val foilCounter = FoilCounter()
    private var receiptSigner: ReceiptSigner? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        cameraExecutor = Executors.newSingleThreadExecutor()
        receiptSigner = ReceiptSigner(this)

        // Bind preview + capture use cases.
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val provider = cameraProviderFuture.get()
            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(findViewById<PreviewView>(R.id.previewView).surfaceProvider)
            }
            imageCapture = ImageCapture.Builder()
                .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
                .build()
            provider.unbindAll()
            provider.bindToLifecycle(
                this, CameraSelector.DEFAULT_BACK_CAMERA, preview, imageCapture
            )
        }, ContextCompat.getMainExecutor(this))
    }

    /**
     * Captures one strip photo and runs the full dose-evidence pipeline on it.
     *
     * @param expectedRows/expectedCols grid geometry for this strip format.
     */
    fun captureStrip(expectedRows: Int, expectedCols: Int) {
        val capture = imageCapture ?: return
        capture.takePicture(cameraExecutor, object : ImageCapture.OnImageCapturedCallback() {
            override fun onCaptureSuccess(image: ImageProxy) {
                var bitmap = imageProxyToBitmap(image)
                image.close()
                try {
                    // 1) Count on a background executor.
                    val result = foilCounter.countStrip(bitmap, expectedRows, expectedCols)

                    // 2) PURGE the raw photo: drop the only reference and hint
                    //    the runtime to reclaim the native pixel buffer now.
                    bitmap = null
                    @Suppress("ExplicitGarbageCollectionCall")
                    System.gc()

                    if (result.status == GateStatus.REFUSED) {
                        onRefused(result)
                        return
                    }

                    // 3) Voice check-in must agree before a receipt mints.
                    val voice = VoiceCheckin.voiceAgrees()
                    if (!voice.agreed) {
                        onVoiceRejected(result, voice)
                        return
                    }

                    // 4) Mint + store the signed receipt (metadata only).
                    val receiptJson = receiptSigner!!.mintReceipt(result, voiceOk = true)
                    onReceiptMinted(receiptJson)
                } catch (t: Throwable) {
                    onError(t)
                } finally {
                    // Belt-and-braces: ensure the raw frame cannot linger.
                    bitmap = null
                }
            }

            override fun onError(exception: ImageCaptureException) {
                onError(exception)
            }
        })
    }

    /**
     * Converts a YUV_420_888 ImageProxy to a Bitmap.
     * Standard CameraX YUV->RGB conversion (RenderScript-free version);
     * implement with the usual yuvToRgb helper before building.
     */
    private fun imageProxyToBitmap(image: ImageProxy): android.graphics.Bitmap {
        // TODO: implement YUV_420_888 -> ARGB_8888 conversion (standard snippet).
        throw UnsupportedOperationException("YUV conversion not yet implemented")
    }

    private fun onRefused(result: StripCountResult) {
        // TODO: show "couldn't count reliably — retake the photo" UI.
    }

    private fun onVoiceRejected(result: StripCountResult, voice: VoiceResult) {
        // TODO: show voice check-in failed UI; no receipt is minted.
    }

    private fun onReceiptMinted(receiptJson: String) {
        // TODO: hand receipt to the sync / demo-laptop export UI.
    }

    private fun onError(t: Throwable) {
        // TODO: error UI / logging.
    }

    override fun onDestroy() {
        cameraExecutor.shutdown()
        super.onDestroy()
    }
}
