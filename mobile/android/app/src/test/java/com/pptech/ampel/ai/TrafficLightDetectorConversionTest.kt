/*
Version: 0.9.20
License: MIT
Code generated with support from CODEX and CODEX CLI.
Owner / Idea / Management: Dr. Babak Sorkhpour (https://x.com/Drbabakskr)
*/

package com.pptech.ampel.ai

import org.junit.Assert.assertTrue
import org.junit.Test

class TrafficLightDetectorConversionTest {
    @Test
    fun syntheticYuvConvertsToNonUniformRgb() {
        val width = 4
        val height = 4
        val yPlane = byteArrayOf(
            16, 48, 80, 112,
            24, 56, 88, 120,
            32, 64, 96, 127,
            40, 72, 104, 125,
        )
        val uPlane = ByteArray(width * height / 4) { 90.toByte() }
        val vPlane = ByteArray(width * height / 4) { 180.toByte() }

        val rgb = TrafficLightDetector.yuv420ToRgbIntArray(
            width = width,
            height = height,
            yPlane = yPlane,
            uPlane = uPlane,
            vPlane = vPlane,
            yRowStride = width,
            yPixelStride = 1,
            uvRowStride = width / 2,
            uvPixelStride = 1,
        )

        assertTrue(rgb.isNotEmpty())
        assertTrue(rgb.distinct().size > 1)
    }
}
