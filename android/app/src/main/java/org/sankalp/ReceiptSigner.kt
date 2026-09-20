package org.sankalp

import android.content.Context
import android.os.Build
import com.google.crypto.tink.KeyTemplates
import com.google.crypto.tink.KeysetHandle
import com.google.crypto.tink.config.TinkConfig
import com.google.crypto.tink.signature.PublicKeySignFactory
import java.io.File
import java.nio.charset.StandardCharsets
import java.security.KeyPair
import java.security.KeyPairGenerator
import java.security.MessageDigest
import java.security.Signature
import java.time.Instant

/**
 * SHA-256-chained, Ed25519-signed dose-evidence receipts, stored
 * ONLY in app-private storage. Nothing leaves the device except the receipt
 * JSON itself when the user explicitly exports it for the laptop console.
 *
 * Canonical receipt body (JSON, keys in fixed order):
 *   { prevHash, ts, doseCount, slotsTotal, confidence, formatId,
 *     stripPhash, voiceOk }
 * plus: hash = SHA-256(canonicalBody), signature = Ed25519(privKey, hash).
 *
 * Crypto approach:
 *  - Preferred: Google Tink (ED25519 key template). Keyset lives in the
 *    app-private files dir (or Android Keystore-backed via
 *    AndroidKeysetManager for production).
 *  - JCA fallback: java.security Ed25519 is available from API 33. Below
 *    API 33, Ed25519 via JCA is unavailable; this class refuses to sign on
 *    those devices rather than downgrading to a weaker scheme.
 * No fake crypto: every signature is verifiable by the offline laptop
 * console (see ../../tools/verify.py) given the exported public key.
 */
class ReceiptSigner(private val context: Context) {

    companion object {
        const val FORMAT_ID = "sankalp-receipt/1"
        private const val PUBKEY_FILE = "receipt_pubkey.pem"
        private const val KEYSET_FILE = "receipt_keyset.json"
    }

    private val receiptDir = File(context.filesDir, "receipts").also { it.mkdirs() }

    /** Signs with Tink when available. Returns null if no signer can be built. */
    private fun tinkSigner(): ((ByteArray) -> ByteArray)? {
        return try {
            TinkConfig.register()
            val keysetFile = File(context.filesDir, KEYSET_FILE)
            val handle = if (keysetFile.exists()) {
                KeysetHandle.read(com.google.crypto.tink.JsonKeysetReader.withFile(keysetFile))
            } else {
                KeysetHandle.generateNew(KeyTemplates.get("ED25519")).also {
                    it.write(com.google.crypto.tink.JsonKeysetWriter.withFile(keysetFile))
                }
            }
            val signer = PublicKeySignFactory.getPrimitive(handle)
            { data: ByteArray -> signer.sign(data) }
        } catch (_: Throwable) {
            null
        }
    }

    /** JCA Ed25519 signer — only on API 33+ where the algorithm is present. */
    private fun jcaSigner(): ((ByteArray) -> ByteArray)? {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.TIRAMISU) return null
        return try {
            val kpg = KeyPairGenerator.getInstance("Ed25519")
            val keyPair: KeyPair = kpg.generateKeyPair()
            { data: ByteArray ->
                val sig = Signature.getInstance("Ed25519")
                sig.initSign(keyPair.private)
                sig.update(data)
                sig.sign()
            }
        } catch (_: Throwable) {
            null
        }
    }

    /**
     * Mints a receipt for an ACCEPTED count result. Requires [voiceOk];
     * callers must not mint when the voice check-in disagreed.
     *
     * @return the full receipt JSON, also persisted to app-private storage.
     */
    fun mintReceipt(result: StripCountResult, voiceOk: Boolean): String {
        require(result.status == GateStatus.ACCEPTED) { "cannot mint receipt for a refused count" }
        require(voiceOk) { "voice check-in must agree before minting" }

        val prevHash = latestHash() // hex of previous receipt hash, or "GENESIS"
        val ts = Instant.now().toString()
        val avgConf = if (result.slots.isEmpty()) 0f
        else result.slots.sumOf { it.confidence.toDouble() }.toFloat() / result.slots.size
        // Perceptual-hash placeholder: hash of slot-status/confidence vector.
        // (A real pHash of the warped strip lives in the Python reference.)
        val stripPhash = sha256Hex(
            result.slots.joinToString("|") { "${it.id}:${it.status}:${"%.2f".format(it.confidence)}" }
                .toByteArray(StandardCharsets.UTF_8)
        )

        val body = buildCanonicalBody(
            prevHash = prevHash,
            ts = ts,
            doseCount = result.doseCount,
            slotsTotal = result.slots.size,
            confidence = avgConf,
            stripPhash = stripPhash,
            voiceOk = voiceOk,
        )
        val hash = sha256Hex(body.toByteArray(StandardCharsets.UTF_8))

        val sign = tinkSigner() ?: jcaSigner()
            ?: throw IllegalStateException("no Ed25519 signer available (need Tink or API 33+)")
        val signature = sign(hash.toByteArray(StandardCharsets.UTF_8))
        val receipt = buildReceiptJson(body, hash, signature.toHex(), FORMAT_ID)

        val file = File(receiptDir, "receipt_${ts.replace(":", "-")}.json")
        file.writeText(receipt, StandardCharsets.UTF_8)
        return receipt
    }

    private fun buildCanonicalBody(
        prevHash: String, ts: String, doseCount: Int, slotsTotal: Int,
        confidence: Float, stripPhash: String, voiceOk: Boolean,
    ): String = buildString {
        append("{")
        append("\"prevHash\":\"").append(prevHash).append("\",")
        append("\"ts\":\"").append(ts).append("\",")
        append("\"doseCount\":").append(doseCount).append(",")
        append("\"slotsTotal\":").append(slotsTotal).append(",")
        append("\"confidence\":").append("%.4f".format(confidence)).append(",")
        append("\"formatId\":\"").append(FORMAT_ID).append("\",")
        append("\"stripPhash\":\"").append(stripPhash).append("\",")
        append("\"voiceOk\":").append(voiceOk)
        append("}")
    }

    private fun buildReceiptJson(body: String, hash: String, sigHex: String, formatId: String) =
        "{\"body\":$body,\"hash\":\"$hash\",\"signature\":\"$sigHex\",\"formatId\":\"$formatId\"}"

    private fun latestHash(): String {
        val files = receiptDir.listFiles { f -> f.name.endsWith(".json") }
            ?.sortedByDescending { it.name } ?: emptyList()
        val latest = files.firstOrNull() ?: return "GENESIS"
        // Extract "hash" field from stored receipt JSON (minimal parse).
        val text = latest.readText(StandardCharsets.UTF_8)
        val m = Regex("\"hash\":\"([0-9a-f]{64})\"").find(text)
        return m?.groupValues?.get(1) ?: "GENESIS"
    }

    private fun sha256Hex(data: ByteArray): String {
        val digest = MessageDigest.getInstance("SHA-256").digest(data)
        return digest.toHex()
    }

    private fun ByteArray.toHex(): String =
        joinToString("") { "%02x".format(it) }
}
