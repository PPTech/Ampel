// Version: 0.9.3
// License: MIT
// Code generated with support from CODEX and CODEX CLI.
// Owner / Idea / Management: Dr. Babak Sorkhpour (https://x.com/Drbabakskr)

package com.ampelai.debug

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp

@Composable
fun DebugOverlay(
    inferenceMs: Int,
    detectedClass: String,
    confidencePercent: Int,
    bufferState: String,
    batteryTempC: Float,
) {
    Column(
        modifier = Modifier
            .background(Color.Black.copy(alpha = 0.45f))
            .padding(10.dp)
    ) {
        Text("Inference: ${inferenceMs}ms", color = if (inferenceMs > 50) Color.Red else Color.Green)
        Text("Detected: $detectedClass: $confidencePercent%", color = Color.White)
        Text("Safety: $bufferState", color = Color.White)
        Text("Battery Temp: ${"%.1f".format(batteryTempC)}°C", color = Color.White)
    }
}
