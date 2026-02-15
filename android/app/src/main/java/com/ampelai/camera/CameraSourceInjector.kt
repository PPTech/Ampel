// Version: 0.9.3
// License: MIT
// Code generated with support from CODEX and CODEX CLI.
// Owner / Idea / Management: Dr. Babak Sorkhpour (https://x.com/Drbabakskr)

package com.ampelai.camera

object CameraSourceInjector {
    fun provider(mockMode: Boolean): CameraProvider {
        return if (mockMode) {
            VideoProvider(assetPath = "assets/test_drive_red_light.mp4", gpxAssetPath = "assets/sample_trace.gpx")
        } else {
            PhysicalCameraProvider()
        }
    }
}

interface CameraProvider
class PhysicalCameraProvider : CameraProvider
class VideoProvider(val assetPath: String, val gpxAssetPath: String) : CameraProvider
