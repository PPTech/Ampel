// Version: 0.9.5
// License: MIT
// Code generated with support from CODEX and CODEX CLI.
// Owner / Idea / Management: Dr. Babak Sorkhpour (https://x.com/Drbabakskr)
// Author: Dr. Babak Sorkhpour with support from ChatGPT

package com.ampelai.security

import okhttp3.CertificatePinner

object CertificatePinning {
    fun build(): CertificatePinner {
        return CertificatePinner.Builder()
            .add("api.ampel.local", "sha256/REPLACE_WITH_BASE64_PIN")
            .build()
    }
}
