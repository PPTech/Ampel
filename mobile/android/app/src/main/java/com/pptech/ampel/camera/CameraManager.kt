/*
Version: 0.9.10
License: MIT
Code generated with support from CODEX and CODEX CLI.
Owner / Idea / Management: Dr. Babak Sorkhpour (https://x.com/Drbabakskr)
*/

package com.pptech.ampel.camera

import android.content.Context
import android.util.Size
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import com.pptech.ampel.ai.TrafficLightDetector
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class CameraManager(
    private val context: Context,
    private val lifecycleOwner: LifecycleOwner,
    private val detector: TrafficLightDetector,
    private val onDetections: (List<TrafficLightDetector.Detection>) -> Unit,
) : ImageAnalysis.Analyzer {

    private val analysisExecutor: ExecutorService = Executors.newSingleThreadExecutor()

    fun bind(previewView: PreviewView) {
        val providerFuture = ProcessCameraProvider.getInstance(context)
        providerFuture.addListener(
            {
                val cameraProvider = providerFuture.get()
                val preview = Preview.Builder()
                    .setTargetResolution(Size(640, 480)) // Thermal/battery safety baseline (VGA)
                    .build()
                    .also { it.setSurfaceProvider(previewView.surfaceProvider) }

                val analysis = ImageAnalysis.Builder()
                    .setTargetResolution(Size(640, 480)) // Thermal safety target
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .build()
                    .also { it.setAnalyzer(analysisExecutor, this) }

                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(
                    lifecycleOwner,
                    CameraSelector.DEFAULT_BACK_CAMERA,
                    preview,
                    analysis,
                )
            },
            ContextCompat.getMainExecutor(context),
        )
    }

    override fun analyze(image: ImageProxy) {
        try {
            val detections = detector.detect(image)
            onDetections(detections)
        } finally {
            image.close()
        }
    }

    fun shutdown() {
        analysisExecutor.shutdown()
        detector.close()
    }
}
