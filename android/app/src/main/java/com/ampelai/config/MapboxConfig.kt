// Version: 0.9.2
// License: MIT
// Code generated with support from CODEX and CODEX CLI.
// Owner / Idea / Management: Dr. Babak Sorkhpour (https://x.com/Drbabakskr)

package com.ampelai.config

import com.ampelai.BuildConfig

object MapboxConfig {
    fun accessToken(): String {
        val token = BuildConfig.MAPBOX_ACCESS_TOKEN
        require(token.isNotBlank()) { "Missing MAPBOX_ACCESS_TOKEN in local.properties" }
        return token
    }
}
