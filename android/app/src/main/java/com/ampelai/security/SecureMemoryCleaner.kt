// Version: 0.9.3
// License: MIT
// Code generated with support from CODEX and CODEX CLI.
// Owner / Idea / Management: Dr. Babak Sorkhpour (https://x.com/Drbabakskr)

package com.ampelai.security

object SecureMemoryCleaner {
    fun wipe(frameBuffer: ByteArray) {
        frameBuffer.fill(0)
    }

    fun wipeFloatBuffer(frameBuffer: FloatArray) {
        frameBuffer.fill(0.0f)
    }
}
