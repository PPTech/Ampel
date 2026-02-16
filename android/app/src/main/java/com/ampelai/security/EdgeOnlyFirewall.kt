// Version: 0.9.5
// License: MIT
// Code generated with support from CODEX and CODEX CLI.
// Owner / Idea / Management: Dr. Babak Sorkhpour (https://x.com/Drbabakskr)
// Author: Dr. Babak Sorkhpour with support from ChatGPT

package com.ampelai.security

object EdgeOnlyFirewall {
    fun allowUpload(contentType: String): Boolean {
        val lower = contentType.lowercase()
        if (lower.contains("video") || lower.contains("image")) return false
        return true
    }
}
