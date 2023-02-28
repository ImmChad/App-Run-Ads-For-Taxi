package com.example.runads

import android.annotation.SuppressLint
import android.content.Context
import android.content.pm.PackageManager
import android.content.res.Configuration
import android.hardware.Camera
import android.hardware.camera2.CameraManager
import android.os.Bundle
import android.view.Surface
import android.view.SurfaceView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.core.content.getSystemService
import org.opencv.android.*
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame
import org.opencv.core.*
import org.opencv.dnn.Dnn
import org.opencv.imgproc.Imgproc
import org.opencv.objdetect.CascadeClassifier
import java.io.File
import java.io.FileOutputStream
import java.io.IOException


class MainActivity : AppCompatActivity(), CameraBridgeViewBase.CvCameraViewListener2 {


    private lateinit var javaCameraView: CameraBridgeViewBase
    private lateinit var tvDisplayStatus: TextView
    var caseFile: File? = null
    var faceDetector: CascadeClassifier? = null
    var mRgba: Mat? = null
    var mGrey: Mat? = null

//    val net = Dnn.readNetFromTensorflow("E:/000_MyProJect/RunAdsForTaxi/Library/HAR-model-PaddleDection/PaddleDetection/configs/face_detection/blazeface_1000e.yml")




    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)


        // request divice open camera
        val cameraPermission = android.Manifest.permission.CAMERA
        val permission = ContextCompat.checkSelfPermission(this, cameraPermission)
        if (permission != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, arrayOf(cameraPermission), 1)
        }

        javaCameraView = findViewById(R.id.javaCameraView)
        tvDisplayStatus = findViewById(R.id.tv_display_status)


        javaCameraView.setCameraPermissionGranted()
        javaCameraView.visibility = SurfaceView.VISIBLE
        javaCameraView.setCameraIndex(CameraBridgeViewBase.CAMERA_ID_FRONT)
        javaCameraView.setCvCameraViewListener(this)
    }

    @SuppressLint("ServiceCast")
    override fun onConfigurationChanged(newConfig: Configuration) {
        super.onConfigurationChanged(newConfig)



        val rotation = windowManager.defaultDisplay.rotation
        var degrees = 0
        when (rotation) {
            Surface.ROTATION_0 -> degrees = 0
            Surface.ROTATION_90 -> degrees = 90
            Surface.ROTATION_180 -> degrees = 180
            Surface.ROTATION_270 -> degrees = 270
        }
        tvDisplayStatus.text = "Screen is rotated $degrees"


    }

    override fun onResume() {
        super.onResume()

        if (!OpenCVLoader.initDebug()) {
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_4_0, this, baseCallback)
        } else {
            baseCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS)
        }
    }

    override fun onPause() {
        super.onPause()
        javaCameraView?.disableView()
    }

    override fun onDestroy() {
        super.onDestroy()
        javaCameraView?.disableView()
    }

    override fun onCameraViewStarted(width: Int, height: Int) {
        mRgba = Mat(width, height, CvType.CV_8UC4)
        mGrey = Mat(width, height, CvType.CV_8UC1)
    }

    override fun onCameraViewStopped() {
        mRgba?.release()
        mGrey?.release()
    }

    override fun onCameraFrame(inputFrame: CvCameraViewFrame): Mat? {

        // NEW SOURCE CODE
//        val frame = inputFrame?.rgba() ?: return Mat()
//
//        // Tiền xử lý ảnh
//        val preprocessedFrame = preprocess(frame)
//
//        // Đưa ảnh qua mô hình để dự đoán hành động của con người
//        val predictions = predict(preprocessedFrame)
//
//        // Hiển thị các dự đoán lên ảnh
//        drawPredictions(frame, predictions)
//
//        return frame



        // OLD SOURCE CODE
        mRgba = inputFrame.rgba()
        mGrey = inputFrame.gray()
        //detect Face
        val facedetections = MatOfRect()
        faceDetector!!.detectMultiScale(mRgba, facedetections)
        if(facedetections.toArray().isEmpty()) {
            tvDisplayStatus.text = "Can't Ditect"
        }
        else {
            for (react in facedetections.toArray()) {
                tvDisplayStatus.text = "Ditected"
                Imgproc.rectangle(
                    mRgba, Point(react.x.toDouble(), react.y.toDouble()),
                    Point((react.x + react.width).toDouble(), (react.y + react.height).toDouble()),
                    Scalar(255.0, 0.0, 0.0)

                )
            }
        }
        return mRgba

    }

//    fun preprocess(frame: Mat): Mat {
//        // Thực hiện các bước tiền xử lý trên frame
//        val resizedFrame = Mat()
//        Imgproc.resize(frame, resizedFrame, Size(224.0, 224.0))
//        val blob = Dnn.blobFromImage(resizedFrame, 1.0, Size(224.0, 224.0), Scalar(0.0, 0.0, 0.0), true, false)
//        return blob
//    }
//
//    fun predict(frame: Mat): Int {
//
//        // Lấy tên của tất cả các output layer
//        val layerNames = net.layerNames
//        // Duyệt qua từng tên layer và lấy số lượng channels của các output layer
//        var numClasses = 0
//        for (name in layerNames) {
//            numClasses = numClasses + 1
//        }
//
//
//        val prediction = numClasses
//
//
//
//
//
//        // Thực hiện dự đoán trên frame đã được tiền xử lý
//        val inputBlob = preprocess(frame)
//        net.setInput(inputBlob)
//        val outputBlob = net.forward()
//        // Xử lý outputBlob để lấy ra kết quả dự đoán
//        return prediction
//    }
//
//    fun drawPredictions(frame: Mat, prediction: Int) {
//        // Thực hiện vẽ kết quả dự đoán lên frame
//        val label = "Action: ${prediction}"
//        Imgproc.putText(frame, label, Point(10.0, 30.0), Imgproc.FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0.0, 255.0, 0.0), 2)
//    }


    private val baseCallback: BaseLoaderCallback = object : BaseLoaderCallback(this) {
        @Throws(IOException::class)
        override fun onManagerConnected(status: Int) {
            when (status) {
                SUCCESS -> {
                    val `is` = resources.openRawResource(R.raw.haarcascade_frontalface_alt)
                    val cascadeDir = getDir("cascade", Context.MODE_PRIVATE)
                    caseFile = File(cascadeDir, "haarcascade_frontalface_alt.xml")
                    val fos = FileOutputStream(caseFile)
                    val buffer = ByteArray(4096)
                    var bytesRead: Int
                    while (`is`.read(buffer).also { bytesRead = it } != -1) {
                        fos.write(buffer, 0, bytesRead)
                    }
                    `is`.close()
                    fos.close()
                    faceDetector = CascadeClassifier(caseFile!!.absolutePath)
                    if (faceDetector!!.empty()) {
                        faceDetector = null
                    } else {
                        cascadeDir.delete()
                    }
                    javaCameraView!!.enableView()
                }
                else -> super.onManagerConnected(status)
            }
        }
    }


}