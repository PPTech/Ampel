/*
Version: 0.9.10
License: MIT
Code generated with support from CODEX and CODEX CLI.
Owner / Idea / Management: Dr. Babak Sorkhpour (https://x.com/Drbabakskr)
*/

package com.pptech.ampel.ai

import android.content.Context
import android.graphics.Bitmap
import android.graphics.ImageFormat
import androidx.camera.core.ImageProxy
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.nnapi.NnApiDelegate
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Mobile-first TFLite wrapper with Samsung-friendly delegate selection.
 * Output records are compatible with shared TrafficEvent schema mapping.
 */
class TrafficLightDetector(
    context: Context,
    modelAssetPath: String = "models/traffic_light.tflite",
) {
    data class Detection(
        val state: String,
        val confidence: Float,
        val bbox: FloatArray? = null,
        val laneId: Int? = null,
    )

    private val nnApiDelegate: NnApiDelegate? = runCatching { NnApiDelegate() }.getOrNull()
    private val gpuDelegate: GpuDelegate? = if (nnApiDelegate == null) {
        runCatching { GpuDelegate() }.getOrNull()
    } else {
        null
    }

    private val interpreter: Interpreter = Interpreter(
        loadModelFile(context, modelAssetPath),
        Interpreter.Options().apply {
            setNumThreads(2)
            nnApiDelegate?.let { addDelegate(it) }
            gpuDelegate?.let { addDelegate(it) }
        },
    )

    private val labels = listOf("RED", "YELLOW", "GREEN", "UNKNOWN")

    fun detect(imageProxy: ImageProxy): List<Detection> {
        val bitmap = imageProxy.toRgbBitmap() ?: return emptyList()
        val input = bitmapToInputBuffer(bitmap, 224, 224)

        val outputScores = Array(1) { FloatArray(labels.size) }
        interpreter.run(input, outputScores)

        val best = outputScores[0].withIndex().maxByOrNull { it.value } ?: return emptyList()
        val state = labels.getOrElse(best.index) { "UNKNOWN" }

        // MVP bbox: whole frame region until model with detection head is integrated.
        val box = floatArrayOf(0f, 0f, bitmap.width.toFloat(), bitmap.height.toFloat())
        return listOf(Detection(state = state, confidence = best.value.coerceIn(0f, 1f), bbox = box, laneId = 0))
    }

    fun close() {
        interpreter.close()
        nnApiDelegate?.close()
        gpuDelegate?.close()
    }

    private fun loadModelFile(context: Context, assetPath: String): ByteBuffer {
        val bytes = context.assets.open(assetPath).use { it.readBytes() }
        return ByteBuffer.allocateDirect(bytes.size).apply {
            order(ByteOrder.nativeOrder())
            put(bytes)
            rewind()
        }
    }

    private fun bitmapToInputBuffer(bitmap: Bitmap, width: Int, height: Int): ByteBuffer {
        val scaled = Bitmap.createScaledBitmap(bitmap, width, height, true)
        val input = ByteBuffer.allocateDirect(1 * width * height * 3 * 4).apply {
            order(ByteOrder.nativeOrder())
        }
        val pixels = IntArray(width * height)
        scaled.getPixels(pixels, 0, width, 0, 0, width, height)
        for (pixel in pixels) {
            input.putFloat(((pixel shr 16) and 0xFF) / 255f)
            input.putFloat(((pixel shr 8) and 0xFF) / 255f)
            input.putFloat((pixel and 0xFF) / 255f)
        }
        input.rewind()
        return input
    }

    private fun ImageProxy.toRgbBitmap(): Bitmap? {
        if (format != ImageFormat.YUV_420_888) return null
        val yBuffer = planes[0].buffer
        val ySize = yBuffer.remaining()
        val nv21 = ByteArray(ySize)
        yBuffer.get(nv21)
        return Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888).apply {
            eraseColor(android.graphics.Color.BLACK)
        }
    }
}
