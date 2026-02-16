/*
Version: 0.9.20
License: MIT
Code generated with support from CODEX and CODEX CLI.
Owner / Idea / Management: Dr. Babak Sorkhpour (https://x.com/Drbabakskr)
*/

package com.pptech.ampel.ui

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
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
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import com.pptech.ampel.ai.TrafficLightDetector
import com.pptech.ampel.camera.CameraManager

@Composable
fun MainScreen() {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val detections = remember { mutableStateListOf<TrafficLightDetector.Detection>() }
    var lux by remember { mutableFloatStateOf(100f) }

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

    DisposableEffect(context) {
        val sensorManager = context.getSystemService(Context.SENSOR_SERVICE) as SensorManager
        val lightSensor = sensorManager.getDefaultSensor(Sensor.TYPE_LIGHT)
        val listener = object : SensorEventListener {
            override fun onSensorChanged(event: SensorEvent?) {
                lux = event?.values?.firstOrNull() ?: lux
            }

            override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) = Unit
        }
        if (lightSensor != null) {
            sensorManager.registerListener(listener, lightSensor, SensorManager.SENSOR_DELAY_NORMAL)
        }
        onDispose {
            sensorManager.unregisterListener(listener)
            cameraManager.shutdown()
        }
    }

    val isNight = lux < 10f
    val isHighContrast = lux > 10_000f
    val baseBg = if (isNight) Color.Black else Color(0xFF050A14)
    val bannerBg = if (isHighContrast) Color.White else Color.Black.copy(alpha = 0.55f)
    val bannerText = if (isHighContrast) Color.Black else Color.White

    Box(modifier = Modifier.fillMaxSize().background(baseBg)) {
        AndroidView(
            factory = { ctx -> PreviewView(ctx) },
            modifier = Modifier.fillMaxSize(),
            update = { previewView -> cameraManager.bind(previewView) },
        )

        OverlayView(detections = detections)
        StatusBanner(detections = detections, background = bannerBg, textColor = bannerText, highContrast = isHighContrast)
    }
}

@Composable
private fun OverlayView(detections: List<TrafficLightDetector.Detection>) {
    Canvas(modifier = Modifier.fillMaxSize()) {
        detections.forEach { detection ->
            val box = detection.bbox ?: return@forEach
            val color = when (detection.state) {
                "RED" -> Color.Red
                "GREEN" -> Color.Green
                else -> Color.Yellow
            }
            drawRect(
                color = color.copy(alpha = 0.82f),
                topLeft = androidx.compose.ui.geometry.Offset(box[0].toFloat(), box[1].toFloat()),
                size = androidx.compose.ui.geometry.Size(box[2].toFloat(), box[3].toFloat()),
                style = Stroke(width = 5f),
            )
        }
    }
}

@Composable
private fun StatusBanner(
    detections: List<TrafficLightDetector.Detection>,
    background: Color,
    textColor: Color,
    highContrast: Boolean,
) {
    val top = detections.firstOrNull()
    val text = when (top?.state) {
        "RED" -> "RED LIGHT - STOP"
        "GREEN" -> "GREEN - GO"
        "YELLOW" -> "YELLOW - CAUTION"
        else -> "DETECTING..."
    }

    Text(
        text = text,
        color = textColor,
        fontSize = if (highContrast) 24.sp else 18.sp,
        modifier = Modifier
            .fillMaxWidth()
            .background(background)
            .padding(horizontal = 16.dp, vertical = 10.dp),
    )
}
