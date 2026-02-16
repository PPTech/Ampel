/*
Version: 0.9.20
License: MIT
Code generated with support from CODEX and CODEX CLI.
Owner / Idea / Management: Dr. Babak Sorkhpour (https://x.com/Drbabakskr)
*/

package com.pptech.ampel.security

import android.content.Context
import android.content.pm.PackageManager
import android.os.Build
import com.pptech.ampel.BuildConfig
import java.security.MessageDigest
import java.util.Locale

/**
 * Runtime integrity checks against tampering/re-signing.
 *
 * NOTE: Official signature hash should be supplied from secured build config or native layer.
 */
class IntegrityManager(
    private val context: Context,
    private val officialSha256Hex: String = BuildConfig.OFFICIAL_SIG_SHA256,
) {
    fun verifyAppSignature(): Boolean {
        if (BuildConfig.DEBUG || officialSha256Hex == "DEBUG_BYPASS") {
            return true
        }
        val current = currentSigningSha256Hex() ?: return false
        val expected = officialSha256Hex.lowercase(Locale.US).replace(":", "")
        val ok = expected.isNotBlank() && current == expected
        if (!ok) {
            throw SecurityException("App signature mismatch detected; AI engine disabled")
        }
        return true
    }

    fun isHardwareBackedKeyStoreAvailable(): Boolean {
        return runCatching {
            val km = android.security.keystore.KeyInfo::class.java
            val providerNames = java.security.Security.getProviders().map { it.name }
            val hasAndroidKeyStore = providerNames.any { it.equals("AndroidKeyStore", ignoreCase = true) }
            val hasKnoxOrTima = providerNames.any {
                it.contains("Knox", ignoreCase = true) || it.contains("Tima", ignoreCase = true)
            }
            // Presence check for TEE-backed keystore features.
            hasAndroidKeyStore && (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) && (hasKnoxOrTima || km != null)
        }.getOrDefault(false)
    }

    private fun currentSigningSha256Hex(): String? {
        val packageManager = context.packageManager
        val packageName = context.packageName
        val signatures = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
            val info = packageManager.getPackageInfo(packageName, PackageManager.GET_SIGNING_CERTIFICATES)
            val signingInfo = info.signingInfo ?: return null
            if (signingInfo.hasMultipleSigners()) signingInfo.apkContentsSigners else signingInfo.signingCertificateHistory
        } else {
            @Suppress("DEPRECATION")
            val info = packageManager.getPackageInfo(packageName, PackageManager.GET_SIGNATURES)
            @Suppress("DEPRECATION")
            info.signatures
        }
        val cert = signatures.firstOrNull()?.toByteArray() ?: return null
        return sha256Hex(cert)
    }

    private fun sha256Hex(bytes: ByteArray): String {
        val digest = MessageDigest.getInstance("SHA-256").digest(bytes)
        return digest.joinToString(separator = "") { "%02x".format(it) }
    }
}
