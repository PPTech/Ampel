// Version: 0.9.3
// License: MIT
// Code generated with support from CODEX and CODEX CLI.
// Owner / Idea / Management: Dr. Babak Sorkhpour (https://x.com/Drbabakskr)

import AVFoundation
import Foundation

enum CameraSourceInjector {
    static func provider(mockMode: Bool) -> CameraProviderProtocol {
        if mockMode {
            return VideoFileProvider(videoName: "test_drive_red_light", fileExtension: "mp4", gpxName: "sample_trace")
        }
        return PhysicalCameraProvider()
    }
}

protocol CameraProviderProtocol {}
struct PhysicalCameraProvider: CameraProviderProtocol {}
struct VideoFileProvider: CameraProviderProtocol {
    let videoName: String
    let fileExtension: String
    let gpxName: String
}
