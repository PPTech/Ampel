/*
Version: 0.9.20
License: MIT
Code generated with support from CODEX and CODEX CLI.
Owner / Idea / Management: Dr. Babak Sorkhpour (https://x.com/Drbabakskr)
*/

package com.pptech.ampel.privacy

import android.content.Context
import com.pptech.ampel.data.SecureStorage
import java.io.File

/**
 * GDPR/CCPA privacy controls.
 */
class PrivacyManager(
    private val context: Context,
    private val secureStorage: SecureStorage,
    private val allowedDomains: Set<String> = setOf("localhost", "127.0.0.1"),
) {
    fun nukeUserData() {
        // Secure preferences
        secureStorage.clearAll()

        // Shared preferences fallback cleanup
        context.filesDir?.parentFile?.let { appRoot ->
            File(appRoot, "shared_prefs").listFiles()?.forEach { it.delete() }
            File(appRoot, "databases").listFiles()?.forEach { it.deleteRecursively() }
            File(appRoot, "cache").deleteRecursively()
            File(appRoot, "code_cache").deleteRecursively()
            File(appRoot, "no_backup").listFiles()?.forEach { it.delete() }
        }

        // Local log files
        context.filesDir?.listFiles()?.forEach { file ->
            if (file.name.contains("log", ignoreCase = true) || file.name.endsWith(".trace")) {
                file.deleteRecursively()
            }
        }
    }

    fun enforceEdgeOnly(url: String, isRawMediaPayload: Boolean = false) {
        val host = runCatching { java.net.URI(url).host?.lowercase() }.getOrNull()
            ?: throw SecurityException("Invalid URL blocked by privacy policy")
        if (!EdgeOnlyPolicy.isAllowed(host, allowedDomains, isRawMediaPayload)) {
            throw SecurityException("Edge-only policy violation: network request blocked")
        }
    }

    object EdgeOnlyPolicy {
        fun isAllowed(host: String, allowedDomains: Set<String>, isRawMediaPayload: Boolean): Boolean {
            if (isRawMediaPayload) return false
            return allowedDomains.any { domain -> host == domain || host.endsWith(".$domain") }
        }
    }
}
