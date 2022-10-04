package com.example.speech_app

import android.Manifest
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.view.View
import android.widget.Button
import android.widget.TextView
import java.util.*
import kotlin.concurrent.scheduleAtFixedRate
import org.tensorflow.lite.task.audio.classifier.AudioClassifier


class MainActivity : AppCompatActivity() {

    var TAG = "MainActivity"
    lateinit var textView: TextView
    lateinit var btnRecording: Button

    var modelPath = "speech_app.tflite"

    //    var modelPath="voice.tflite"
    var probabilityThreshold: Float = 0.5f
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        val REQUEST_RECORD_AUDIO = 1337
        requestPermissions(
            arrayOf(Manifest.permission.RECORD_AUDIO),
            REQUEST_RECORD_AUDIO
        )
        textView = findViewById<TextView>(R.id.output)
        btnRecording = findViewById<Button>(R.id.btn_record)
        val recorderSpecsTextView = findViewById<TextView>(R.id.textViewAudioRecorderSpecs)

        // TODO 2.3: Loading the model from the assets folder
        val classifier = AudioClassifier.createFromFile(this, modelPath)

        // TODO 3.1: Creating an audio recorder
        val tensor = classifier.createInputTensorAudio()

        // TODO 3.2: showing the audio recorder specification
        val format = classifier.requiredTensorAudioFormat
        val recorderSpecs =
            "Number Of Channels: ${format.channels}\n" + "Sample Rate: ${format.sampleRate}"
        recorderSpecsTextView.text = recorderSpecs

        // TODO 3.3: Creating
        val record = classifier.createAudioRecord()
        btnRecording.setOnClickListener(View.OnClickListener {
            record.startRecording()
        })


        Timer().scheduleAtFixedRate(1, 500) {

            // TODO 4.1: Classifing audio data
            tensor.load(record)
            val output = classifier.classify(tensor)

            // TODO 4.2: Filtering out classifications with low probability
            val filteredModelOutput = output[0].categories.filter {
                it.score > probabilityThreshold
            }

            // TODO 4.3: Creating a multiline string with the filtered results
            val outputStr = filteredModelOutput.sortedBy { -it.score }
                .joinToString(separator = "\n") { "${it.label} " }
//            record.stop()

            // TODO 4.4: Updating the UI
            if (outputStr.isNotEmpty())
                runOnUiThread {
                    textView.text = outputStr
                }
        }
    }
}