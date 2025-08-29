package com.marusys.c2w1a2

import android.app.Activity
import android.os.Bundle
import android.graphics.Color
import android.util.Log
import android.view.KeyEvent
import com.github.mikephil.charting.charts.ScatterChart
import com.github.mikephil.charting.data.Entry
import com.github.mikephil.charting.data.ScatterData
import com.github.mikephil.charting.data.ScatterDataSet
import com.github.mikephil.charting.interfaces.datasets.IScatterDataSet
import org.json.JSONObject
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer

class MainActivity : Activity() {
    companion object {
        private const val TAG = "C2W1A2"
    }

    private lateinit var chart: ScatterChart
    private var mode: Int = 0

    override fun onKeyDown(keyCode: Int, event: KeyEvent?): Boolean {
        mode = (++mode)%6
        plotData(mode)
        return true;
    }

    private lateinit var model: MappedByteBuffer
    private lateinit var interpreter: Interpreter

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        chart = findViewById(R.id.scatterChart)
        plotData(mode)
    }


    private fun classify(x: Float, y: Float) : Float {
        val inputFeatures = floatArrayOf(x, y)

        val inputTensor = interpreter.getInputTensor(0)
        val inputShape = inputTensor.shape()
        val inputSizeInBytes = inputShape.size * Float.SIZE_BYTES
        val inputBuffer = ByteBuffer.allocateDirect(inputSizeInBytes).apply {
            order(ByteOrder.nativeOrder())
        }
        inputBuffer.rewind()
        inputBuffer.asFloatBuffer().put(inputFeatures)

        val outputTensor = interpreter.getOutputTensor(0)
        val outputShape = outputTensor.shape()
        val outputBuffer = Array(outputShape[0]) { FloatArray(outputShape[1]) }

        interpreter.run(inputBuffer, outputBuffer)

        val c = outputBuffer[0][0]
        Log.i(TAG, "classify(" + x + "," + y + ") = " + c)
        return c
    }

    private fun model_init(mode: Int) {
        when(mode) {
            1 -> model = FileUtil.loadMappedFile(this, "model_baseline.tflite")
            2 -> model = FileUtil.loadMappedFile(this, "model_dropout.tflite")
            3 -> model = FileUtil.loadMappedFile(this, "model_l2_regularized.tflite")
        }
        interpreter = Interpreter(model)
    }
    private fun plotData(mode: Int) {
        val jsonString = loadJsonFromAsset("2D_dataset.json")
        if (jsonString != null) {
            val rootObject = JSONObject(jsonString)

            val trainXArray = rootObject.getJSONArray("train_X")
            val xValuesArray = trainXArray.getJSONArray(0)
            val yValuesArray = trainXArray.getJSONArray(1)

            val trainYArray = rootObject.getJSONArray("train_Y").getJSONArray(0)

            val entries0 = ArrayList<Entry>()
            val entries1 = ArrayList<Entry>()

            when(mode) {
                1,2,3 -> model_init(mode)
            }

            for (i in 0 until xValuesArray.length()) {
                val x = xValuesArray.getDouble(i).toFloat()
                val y = yValuesArray.getDouble(i).toFloat()

                when(mode) {
                    0 -> {
                        val c = trainYArray.getInt(i)
                        if (c == 0) entries0.add(Entry(x, y))
                        else entries1.add(Entry(x, y))
                    }
                    1,2,3-> {
                        val pred = classify(x, y)
                        if (pred < 0.5) entries0.add(Entry(x, y))
                        else entries1.add(Entry(x, y))
                    }
                }
            }

            val dataSet0 = ScatterDataSet(entries0, "class0").apply {
                setScatterShape(ScatterChart.ScatterShape.CIRCLE)
                color = Color.rgb(255, 0, 0)
                scatterShapeSize = 15f
            }

            val dataSet1 = ScatterDataSet(entries1, "class1").apply {
                setScatterShape(ScatterChart.ScatterShape.CIRCLE)
                color = Color.rgb(0, 0, 255)
                scatterShapeSize = 15f
            }

            val dataSets = arrayListOf<IScatterDataSet>(dataSet0, dataSet1)
            chart.data = ScatterData(dataSets)
            chart.invalidate()
        }
    }


    private fun loadJsonFromAsset(fileName: String): String? {
        return try {
            val inputStream = assets.open(fileName)
            val size = inputStream.available()
            val buffer = ByteArray(size)
            inputStream.read(buffer)
            inputStream.close()
            String(buffer, Charsets.UTF_8)
        } catch (ex: IOException) {
            ex.printStackTrace()
            null
        }
    }
}