/*
Version: 0.9.10
License: MIT
Code generated with support from CODEX and CODEX CLI.
Owner / Idea / Management: Dr. Babak Sorkhpour (https://x.com/Drbabakskr)
*/

package com.pptech.ampel.ui

import androidx.camera.view.PreviewView
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import com.pptech.ampel.ai.TrafficLightDetector
import com.pptech.ampel.camera.CameraManager

@Composable
fun MainScreen() {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val detections = remember { mutableStateListOf<TrafficLightDetector.Detection>() }

    val detector = remember { TrafficLightDetector(context) }
    val cameraManager = remember {
        CameraManager(
            context = context,
            lifecycleOwner = lifecycleOwner,
            detector = detector,
            onDetections = { incoming ->
                detections.clear()
                detections.addAll(incoming)
            },
        )
    }

    Box(modifier = Modifier.fillMaxSize()) {
        AndroidView(
            factory = { ctx -> PreviewView(ctx) },
            modifier = Modifier.fillMaxSize(),
            update = { previewView -> cameraManager.bind(previewView) },
        )

        OverlayView(detections = detections)

        StatusBanner(detections = detections)
    }

    DisposableEffect(Unit) {
        onDispose { cameraManager.shutdown() }
    }
}

@Composable
private fun OverlayView(detections: List<TrafficLightDetector.Detection>) {
    Canvas(modifier = Modifier.fillMaxSize()) {
        detections.forEach { detection ->
            val box = detection.bbox ?: return@forEach
            val color = when (detection.state) {
                "RED" -> androidx.compose.ui.graphics.Color.Red
                "GREEN" -> androidx.compose.ui.graphics.Color.Green
                else -> androidx.compose.ui.graphics.Color.Yellow
            }
            drawRect(
                color = color.copy(alpha = 0.75f),
                topLeft = androidx.compose.ui.geometry.Offset(box[0], box[1]),
                size = androidx.compose.ui.geometry.Size(box[2], box[3]),
                style = Stroke(width = 4f),
            )
        }
    }
}

@Composable
private fun StatusBanner(detections: List<TrafficLightDetector.Detection>) {
    val top = detections.firstOrNull()
    val text = when (top?.state) {
        "RED" -> "RED LIGHT - STOP"
        "GREEN" -> "GREEN"
        "YELLOW" -> "YELLOW - CAUTION"
        else -> "DETECTING..."
    }

    Text(
        text = text,
        color = androidx.compose.ui.graphics.Color.White,
        modifier = Modifier
            .fillMaxWidth()
            .background(androidx.compose.ui.graphics.Color.Black.copy(alpha = 0.55f))
            .padding(horizontal = 16.dp, vertical = 10.dp),
    )
}
