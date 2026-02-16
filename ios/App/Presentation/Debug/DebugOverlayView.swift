// Version: 0.9.3
// License: MIT
// Code generated with support from CODEX and CODEX CLI.
// Owner / Idea / Management: Dr. Babak Sorkhpour (https://x.com/Drbabakskr)

import SwiftUI

struct DebugOverlayView: View {
    let inferenceMs: Double
    let detectedClass: String
    let confidence: Double
    let bufferState: String
    let batteryTempC: Double

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Inference: \(Int(inferenceMs))ms")
                .foregroundColor(inferenceMs > 50 ? .red : .green)
            Text("Detected: \(detectedClass) \(Int(confidence * 100))%")
            Text("Safety: \(bufferState)")
            Text(String(format: "Battery Temp: %.1f°C", batteryTempC))
        }
        .font(.system(size: 14, weight: .semibold, design: .rounded))
        .padding(10)
        .background(.black.opacity(0.45))
        .foregroundColor(.white)
        .cornerRadius(10)
    }
}
