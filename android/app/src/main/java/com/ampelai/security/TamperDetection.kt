// Version: 0.9.5
// License: MIT
// Code generated with support from CODEX and CODEX CLI.
// Owner / Idea / Management: Dr. Babak Sorkhpour (https://x.com/Drbabakskr)
// Author: Dr. Babak Sorkhpour with support from ChatGPT

package com.ampelai.security

import android.content.Context
import java.security.MessageDigest

object TamperDetection {
    private const val EXPECTED_SHA256 = "REPLACE_WITH_RELEASE_CERT_SHA256"

    fun verifySignatureOrBlock(context: Context): Boolean {
        val sig = context.packageManager.getPackageInfo(context.packageName, 0).signatures.firstOrNull()
            ?: return false
        val digest = MessageDigest.getInstance("SHA-256").digest(sig.toByteArray())
        val hex = digest.joinToString("") { "%02x".format(it) }
        return hex.equals(EXPECTED_SHA256, ignoreCase = true)
    }
}
