/*
Version: 0.9.14
License: MIT
Code generated with support from CODEX and CODEX CLI.
Owner / Idea / Management: Dr. Babak Sorkhpour (https://x.com/Drbabakskr)
*/

package com.pptech.ampel.ai

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.YuvImage
import androidx.camera.core.ImageProxy
import org.tensorflow.lite.Interpreter
import java.io.ByteArrayOutputStream
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
        val nv21 = yuv420ToNv21(this)
        val yuvImage = YuvImage(nv21, ImageFormat.NV21, width, height, null)
        val output = ByteArrayOutputStream()
        if (!yuvImage.compressToJpeg(Rect(0, 0, width, height), 100, output)) {
            return null
        }
        val jpegBytes = output.toByteArray()
        return BitmapFactory.decodeByteArray(jpegBytes, 0, jpegBytes.size)
    }

    private fun yuv420ToNv21(image: ImageProxy): ByteArray {
        val yPlane = image.planes[0]
        val uPlane = image.planes[1]
        val vPlane = image.planes[2]

        val width = image.width
        val height = image.height
        val ySize = width * height
        val nv21 = ByteArray(ySize + (width * height / 2))

        var outputOffset = 0
        val yBuffer = yPlane.buffer
        val yRowStride = yPlane.rowStride
        for (row in 0 until height) {
            val rowStart = row * yRowStride
            for (col in 0 until width) {
                nv21[outputOffset++] = yBuffer.get(rowStart + col)
            }
        }

        val uBuffer = uPlane.buffer
        val vBuffer = vPlane.buffer
        val chromaHeight = height / 2
        val chromaWidth = width / 2
        val uRowStride = uPlane.rowStride
        val vRowStride = vPlane.rowStride
        val uPixelStride = uPlane.pixelStride
        val vPixelStride = vPlane.pixelStride

        for (row in 0 until chromaHeight) {
            val uRowStart = row * uRowStride
            val vRowStart = row * vRowStride
            for (col in 0 until chromaWidth) {
                val uIndex = uRowStart + col * uPixelStride
                val vIndex = vRowStart + col * vPixelStride
                nv21[outputOffset++] = vBuffer.get(vIndex)
                nv21[outputOffset++] = uBuffer.get(uIndex)
            }
        }

        return nv21
    }
}
