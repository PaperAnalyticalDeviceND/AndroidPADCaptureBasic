package edu.nd.crc.paddetection;

import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.core.Rect;
import org.opencv.core.Size;
import org.opencv.highgui.Highgui;
import org.opencv.imgproc.Imgproc;

import android.app.Activity;
import android.app.ProgressDialog;
import android.content.Context;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.hardware.Camera;
import android.os.AsyncTask;
import android.os.Bundle;
import android.os.Environment;
import android.os.SystemClock;
import android.util.Log;
import android.view.View;
import android.view.WindowManager;
import android.widget.Toast;

import com.sh1r0.caffe_android_lib.CaffeMobile;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStreamWriter;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.Collections;
import java.util.Date;
import java.util.List;
import java.util.Scanner;
import java.util.Vector;

public class AndroidCameraExample extends Activity implements CvCameraViewListener2, CNNListener {
	private JavaCamResView mOpenCvCameraView;

    static {
        System.loadLibrary("caffe");
        System.loadLibrary("caffe_jni");
        System.loadLibrary("opencv_java");
    }

    private Mat mRgba, mTemplate;
    private String LOG_TAG = "PAD";
    private static String[] IMAGENET_CLASSES;
    private CaffeMobile caffeMobile;
    private ProgressDialog dialog;

    @Override
	public void onCreate(Bundle savedInstanceState) {
		super.onCreate(savedInstanceState);
		getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

		setContentView(R.layout.activity_main);

		mOpenCvCameraView = (JavaCamResView) findViewById(R.id.activity_surface_view);
		mOpenCvCameraView.setCvCameraViewListener(this);
        mOpenCvCameraView.enableFpsMeter();

        mOpenCvCameraView.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                dialog = ProgressDialog.show(AndroidCameraExample.this, "Predicting...", "Cropping Image", true);

                CNNTask cnnTask = new CNNTask(AndroidCameraExample.this);
                cnnTask.execute(mRgba, mTemplate);

                mOpenCvCameraView.togglePreview();
            }
        });

        // Parse input template file
        Bitmap tBM = BitmapFactory.decodeStream(this.getClass().getResourceAsStream("/template.png"));

        // Convert to OpenCV matrix
        Mat tMat = new Mat();
        Utils.bitmapToMat(tBM, tMat);

        mTemplate = new Mat();
        Imgproc.cvtColor(tMat, mTemplate, Imgproc.COLOR_BGRA2GRAY);

        caffeMobile = new CaffeMobile();
        caffeMobile.setNumThreads(4);
        File sdcard_path = Environment.getExternalStorageDirectory();
        caffeMobile.loadModel(sdcard_path+"/caffe_mobile/bvlc_reference_caffenet/deploy.prototxt", sdcard_path+"/caffe_mobile/bvlc_reference_caffenet/Sandipan1_Full_26Drugs_iter_90000.caffemodel");

        float[] meanValues = {104, 117, 123};
        caffeMobile.setMean(meanValues);

        /*Camera mCamera = Camera.open();
        Camera.Parameters params = mCamera.getParameters();
        List<Camera.Size> sizes = params.getSupportedPictureSizes();
        for (Camera.Size size : sizes) {
            Log.i("Camera", "Available resolution: "+size.width+" "+size.height);
        }*/

        AssetManager am = this.getAssets();
        try {
            InputStream is = am.open("drug_names.txt");
            Scanner sc = new Scanner(is);
            List<String> lines = new ArrayList<String>();
            while (sc.hasNextLine()) {
                lines.add(sc.nextLine());
            }
            IMAGENET_CLASSES = lines.toArray(new String[0]);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

	@Override
	public void onPause() {
		super.onPause();
		if (mOpenCvCameraView != null) {
            mOpenCvCameraView.disableView();
        }
	}

	@Override
	public void onResume() {
		super.onResume();
        mOpenCvCameraView.enableView();
	}

	public void onDestroy() {
		super.onDestroy();
		if (mOpenCvCameraView != null) {
            mOpenCvCameraView.disableView();
        }
	}

    public void onCameraViewStarted(int width, int height) {
        mOpenCvCameraView.Setup();
	}

	public void onCameraViewStopped() {

	}

	public Mat onCameraFrame(CvCameraViewFrame inputFrame) {
        mRgba = inputFrame.rgba();
		return inputFrame.rgba();
	}

    public class PredictionGuess implements Comparable<PredictionGuess> {
        public int Index;
        public float Confidence;

        public PredictionGuess(int i, float c) {
            this.Index = i;
            this.Confidence = c;
        }

        @Override
        public int compareTo(PredictionGuess that) {
            if( this.Confidence > that.Confidence ){
                return -1;
            }else{
                return 1;
            }
        }
    }

    private class CNNTask extends AsyncTask<Mat, Void, Vector<PredictionGuess>> {
        private CNNListener listener;
        private long startTime;

        public CNNTask(CNNListener listener) {
            this.listener = listener;
        }

        @Override
        protected Vector<PredictionGuess> doInBackground(Mat... input) {
            startTime = SystemClock.uptimeMillis();

            DateFormat df = new SimpleDateFormat("yyyy-MM-dd-HH-mm-ss");
            Date today = Calendar.getInstance().getTime();

            File SDlocation = Environment.getExternalStorageDirectory();
            File padImageDirectory = new File(SDlocation + "/PAD/" + df.format(today));
            padImageDirectory.mkdirs();

            Mat mTemp = new Mat();
            mTemp = input[0];
            Imgproc.resize(mTemp, mTemp, new Size(960, 720));

            Mat result = new Mat(mTemp, new Rect(105, 120, mTemp.width()-172, mTemp.height()-240));
            Core.flip(result.t(), result, 1);

            File outputFile = new File(padImageDirectory, "capture.jpeg");
            Imgproc.cvtColor(result, mTemp, Imgproc.COLOR_BGRA2RGBA);
            Highgui.imwrite(outputFile.getPath(), mTemp);

            runOnUiThread(new Runnable() {
                  @Override
                  public void run() {
                      dialog.setMessage("Rectifying Image");
                  }
            });

            // Run white balance
            Mat cropped = ContourDetection.RectifyImage(mTemp, input[1]);

            File cFile = new File(padImageDirectory, "rectified.jpeg");
            Imgproc.cvtColor(cropped, mTemp, Imgproc.COLOR_BGRA2RGBA);
            Highgui.imwrite(cFile.getPath(), mTemp);

            Mat cResult = cropped.submat(359, 849, 70, 710);

            File crFile = new File(padImageDirectory, "cropped.jpeg");
            Imgproc.cvtColor(cResult, mTemp, Imgproc.COLOR_BGRA2RGBA);
            Highgui.imwrite(crFile.getPath(), mTemp);

            File resFile = new File(padImageDirectory, "resized.jpeg");
            Imgproc.resize(mTemp, mTemp, new Size(227, 227));
            Highgui.imwrite(resFile.getPath(), mTemp);

            runOnUiThread(new Runnable() {
                @Override
                public void run() {
                    dialog.setMessage("Predicting Drug");
                }
            });

            float[] scores = caffeMobile.getConfidenceScore(resFile.getPath());

            Vector<PredictionGuess> guesses = new Vector<>();
            for( int i = 0; i < scores.length; i++){
                guesses.add(new PredictionGuess(i, scores[i]));
            }
            Collections.sort(guesses);

            try {
                File oFile = new File(padImageDirectory, "guesses.txt");
                oFile.createNewFile();
                FileOutputStream fOut = new FileOutputStream(oFile);
                OutputStreamWriter myOutWriter = new OutputStreamWriter(fOut);
                for( int i = 0; i < 10; i++ ){
                    myOutWriter.append(String.format("%s - %f%%\n", IMAGENET_CLASSES[guesses.get(i).Index], guesses.get(i).Confidence * 100.0));
                }
                myOutWriter.close();
                fOut.close();
            } catch (Exception e) {
                Log.e(LOG_TAG, "Failed to write guess file: " + e.getMessage());
            }

            Vector<PredictionGuess> top = new Vector<>();
            for( int i = 0; i < 3; i++) {
                top.add(guesses.get(i));
            }
            return top;
        }

        @Override
        protected void onPostExecute(Vector<PredictionGuess> guess) {
            Log.i(LOG_TAG, String.format("elapsed wall time: %d ms", SystemClock.uptimeMillis() - startTime));
            for( int i = 0; i < guess.size(); i++){
                Log.i(LOG_TAG, String.format("Guess[%f]: %s", guess.get(i).Confidence, IMAGENET_CLASSES[guess.get(i).Index]));
            }
            listener.onTaskCompleted(guess);
            super.onPostExecute(guess);
        }
    }

    @Override
    public void onTaskCompleted(Vector<PredictionGuess> result) {
        Context context = getApplicationContext();
        Toast.makeText(context, String.format("Predicted Drugs\n %s - %f%%\n %s - %f%%\n %s - %f%%", IMAGENET_CLASSES[result.get(0).Index], result.get(0).Confidence * 100.0, IMAGENET_CLASSES[result.get(1).Index], result.get(1).Confidence * 100.0, IMAGENET_CLASSES[result.get(2).Index], result.get(2).Confidence * 100.0), Toast.LENGTH_LONG).show();

        mOpenCvCameraView.togglePreview();
        if (dialog != null) {
            dialog.dismiss();
        }
    }
}
