package com.example.lungnoduleapp

import android.content.Context
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import org.pytorch.Module
import org.pytorch.IValue
import org.pytorch.Tensor
import java.io.File
import java.io.FileOutputStream

class MainActivity : ComponentActivity() {

    lateinit var model: Module

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // 🔥 Load model
        try {
            model = Module.load(assetFilePath(this, "deep_mlp_mobile.pt"))
        } catch (e: Exception) {
            e.printStackTrace()
        }

        setContent {
            LungAppUI(model)
        }
    }
}

@Composable
fun LungAppUI(model: Module) {

    val numFeatures = 8

    // ⚠️ change if needed
    val featureNames = listOf(
        "Total Nodules",
        "Nodules >= 3mm",
        "Nodules < 3mm",
        "Diagnosis Method (0=Unknown, 1=Stable, 2=Biopsy, 3=Surgery, 4=Progression)",
        "Primary Tumor Site (0=Head & Neck, 1=Lung, 2=Uterine, 3=NSCLC)",
        "Nodule 1 Diagnosis (0=Unknown, 1=Benign, 2=Primary Cancer, 3=Metastatic)",
        "Nodule 1 Method (0=Unknown, 1=Stable, 2=Biopsy, 3=Surgery, 4=Progression)",
        "Nodule 2 Diagnosis (0=Unknown, 1=Benign, 2=Primary Cancer, 3=Metastatic)"
    )

    var inputs by remember { mutableStateOf(List(numFeatures) { "" }) }
    var result by remember { mutableStateOf("Result will appear here") }

    Column(modifier = Modifier.padding(16.dp)) {

        Text("Lung Nodule Prediction", style = MaterialTheme.typography.titleLarge)

        Spacer(modifier = Modifier.height(16.dp))

        // 🔷 Input fields
        for (i in 0 until numFeatures) {
            OutlinedTextField(
                value = inputs[i],
                onValueChange = {
                    val temp = inputs.toMutableList()
                    temp[i] = it
                    inputs = temp
                },
                label = { Text(featureNames[i])  },
                modifier = Modifier.fillMaxWidth()
            )
            Spacer(modifier = Modifier.height(8.dp))
        }

        // 🔷 Predict button
        Button(onClick = {
            try {
                val inputArray = inputs.map {
                    if (it.isEmpty()) 0f else it.toFloat()
                }.toFloatArray()

                // 🔥 PASTE YOUR VALUES HERE (VERY IMPORTANT)
                val mean = floatArrayOf(
                    5.48734177f, 2.50632911f, 2.98101266f, 1.92405063f,
                    1.34810127f, 1.55696203f, 0.18987342f, 0.26582278f
                )

                val std = floatArrayOf(
                    5.8663785f, 2.5821854f, 4.336293f, 1.24037716f,
                    1.17961169f, 1.50313458f, 0.6764754f, 0.94412052f
                )

                // 🔥 Scaling
                for (i in inputArray.indices) {
                    inputArray[i] = (inputArray[i] - mean[i]) / std[i]
                }

                // 🔥 Tensor (FIXED)
                val tensor = Tensor.fromBlob(
                    inputArray,
                    longArrayOf(1, numFeatures.toLong())
                )

                val output = model.forward(IValue.from(tensor)).toTensor()
                val scores = output.dataAsFloatArray

                result = if (scores[1] > scores[0]) "Malignant" else "Benign"

            } catch (e: Exception) {
                result = "Invalid input"
            }
        }) {
            Text("Predict")
        }

        Spacer(modifier = Modifier.height(16.dp))

        Text("Result: $result")
    }
}

// 🔥 REQUIRED FUNCTION (corrected formatting)
fun assetFilePath(context: Context, assetName: String): String {
    val file = File(context.filesDir, assetName)

    if (file.exists() && file.length() > 0) {
        return file.absolutePath
    }

    context.assets.open(assetName).use { inputStream ->
        FileOutputStream(file).use { outputStream ->
            val buffer = ByteArray(4 * 1024)
            while (true) {
                val read = inputStream.read(buffer)
                if (read == -1) break
                outputStream.write(buffer, 0, read)
            }
            outputStream.flush()
        }
    }

    return file.absolutePath
}