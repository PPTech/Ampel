/*
Version: 0.9.20
License: MIT
Code generated with support from CODEX and CODEX CLI.
Owner / Idea / Management: Dr. Babak Sorkhpour (https://x.com/Drbabakskr)
*/

package com.pptech.ampel.ai

import android.content.Context
import android.graphics.Bitmap
import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.YuvImage
import android.util.Log
import androidx.camera.core.ImageProxy
import org.tensorflow.lite.Interpreter
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Mobile-first TFLite wrapper with Samsung-aware delegate strategy.
 *
 * Edge-only privacy policy: no raw frames leave device.
 */
class TrafficLightDetector(
    context: Context,
    modelAssetPath: String = "models/traffic_light.tflite",
) {
    data class Detection(
        val state: String,
        val confidence: Float,
        val bbox: IntArray? = null,
        val laneId: Int? = null,
        val timestamp: Long = System.currentTimeMillis(),
    )

    private val delegateBundle: TFLiteDelegateFactory.DelegateBundle
    private val interpreter: Interpreter
    private val temporalBuffer = TemporalBuffer(size = 5)

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

        val best = outputScores[0].withIndex().maxByOrNull { it.value } ?: return emptyList()
        val rawState = labels.getOrElse(best.index) { "UNKNOWN" }
        val smoothed = temporalBuffer.push(rawState)
        Log.d("TrafficLightDetector", "state_raw=$rawState state_smoothed=$smoothed conf=${best.value}")

        return listOf(
            Detection(
                state = smoothed,
                confidence = best.value.coerceIn(0f, 1f),
                bbox = null, // classifier-only model: no bbox output available
                laneId = null,
            ),
        )
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

        val nv21 = yuv420888ToNv21(this)
        val yuvImage = YuvImage(nv21, ImageFormat.NV21, width, height, null)
        val stream = ByteArrayOutputStream()
        yuvImage.compressToJpeg(Rect(0, 0, width, height), 92, stream)
        val jpegBytes = stream.toByteArray()
        return android.graphics.BitmapFactory.decodeByteArray(jpegBytes, 0, jpegBytes.size)
    }

    private class TemporalBuffer(private val size: Int) {
        private val states = ArrayDeque<String>()

        fun push(state: String): String {
            states.addLast(state)
            while (states.size > size) {
                states.removeFirst()
            }
            val freq = states.groupingBy { it }.eachCount()
            return freq.maxByOrNull { it.value }?.key ?: state
        }
    }

    companion object {
        /**
         * Visible for deterministic unit tests and analyzer conversion path.
         */
        fun yuv420ToRgbIntArray(
            width: Int,
            height: Int,
            yPlane: ByteArray,
            uPlane: ByteArray,
            vPlane: ByteArray,
            yRowStride: Int,
            yPixelStride: Int,
            uvRowStride: Int,
            uvPixelStride: Int,
        ): IntArray {
            val out = IntArray(width * height)
            for (y in 0 until height) {
                val yRow = yRowStride * y
                val uvRow = uvRowStride * (y / 2)
                for (x in 0 until width) {
                    val yIndex = yRow + x * yPixelStride
                    val uvIndex = uvRow + (x / 2) * uvPixelStride

                    val yValue = yPlane.getOrElse(yIndex) { 0 }.toInt() and 0xFF
                    val uValue = (uPlane.getOrElse(uvIndex) { 128 }.toInt() and 0xFF) - 128
                    val vValue = (vPlane.getOrElse(uvIndex) { 128 }.toInt() and 0xFF) - 128

                    val r = (yValue + 1.370705f * vValue).toInt().coerceIn(0, 255)
                    val g = (yValue - 0.337633f * uValue - 0.698001f * vValue).toInt().coerceIn(0, 255)
                    val b = (yValue + 1.732446f * uValue).toInt().coerceIn(0, 255)

                    out[y * width + x] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
                }
            }
            return out
        }

        private fun yuv420888ToNv21(image: ImageProxy): ByteArray {
            val yBuffer = image.planes[0].buffer
            val uBuffer = image.planes[1].buffer
            val vBuffer = image.planes[2].buffer

            val yBytes = ByteArray(yBuffer.remaining()).also { yBuffer.get(it) }
            val uBytes = ByteArray(uBuffer.remaining()).also { uBuffer.get(it) }
            val vBytes = ByteArray(vBuffer.remaining()).also { vBuffer.get(it) }

            val out = ByteArray(image.width * image.height * 3 / 2)
            // copy Y
            var outOffset = 0
            for (row in 0 until image.height) {
                val yRow = row * image.planes[0].rowStride
                for (col in 0 until image.width) {
                    val yIndex = yRow + col * image.planes[0].pixelStride
                    out[outOffset++] = yBytes.getOrElse(yIndex) { 0 }
                }
            }

            // interleave VU for NV21
            val chromaHeight = image.height / 2
            val chromaWidth = image.width / 2
            for (row in 0 until chromaHeight) {
                val uvRow = row * image.planes[1].rowStride
                for (col in 0 until chromaWidth) {
                    val uvIndex = uvRow + col * image.planes[1].pixelStride
                    out[outOffset++] = vBytes.getOrElse(uvIndex) { 0 }
                    out[outOffset++] = uBytes.getOrElse(uvIndex) { 0 }
                }
            }
            return out
        }
    }
}
