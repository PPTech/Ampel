/*
Version: 0.9.14
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
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Mobile-first TFLite wrapper with Samsung-aware delegate strategy.
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

    private val delegateBundle: TFLiteDelegateFactory.DelegateBundle
    private val interpreter: Interpreter

    init {
        val model = loadModelFile(context, modelAssetPath)
        val pair = TFLiteDelegateFactory.createInterpreter(model)
        interpreter = pair.first
        delegateBundle = pair.second
    }

    private val labels = listOf("RED", "YELLOW", "GREEN", "UNKNOWN")

    fun detect(imageProxy: ImageProxy): List<Detection> {
        val bitmap = imageProxy.toRgbBitmap() ?: return emptyList()
        val input = bitmapToInputBuffer(bitmap, 224, 224)

        val outputScores = Array(1) { FloatArray(labels.size) }
        interpreter.run(input, outputScores)

        val ranked = outputScores[0].withIndex().sortedByDescending { it.value }.take(2)
        return ranked.mapIndexed { idx, best ->
            val state = labels.getOrElse(best.index) { "UNKNOWN" }
            val box = if (idx == 0) {
                floatArrayOf(0f, 0f, bitmap.width.toFloat(), bitmap.height.toFloat())
            } else {
                floatArrayOf(
                    bitmap.width * 0.2f,
                    bitmap.height * 0.2f,
                    bitmap.width * 0.25f,
                    bitmap.height * 0.25f,
                )
            }
            Detection(state = state, confidence = best.value.coerceIn(0f, 1f), bbox = box, laneId = 0)
        }
    }

    fun close() {
        interpreter.close()
        delegateBundle.nnApiDelegate?.close()
        delegateBundle.gpuDelegate?.close()
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
        return Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888).apply {
            eraseColor(android.graphics.Color.BLACK)
        }
    }
}
