package edu.nd.crc.paddetection;

import org.json.JSONArray;
import org.json.JSONObject;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.core.CvType;
import org.opencv.core.MatOfPoint;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.highgui.Highgui;
import org.opencv.imgproc.Imgproc;
import org.opencv.imgproc.Moments;

import android.Manifest;
import android.app.Activity;
import android.app.ProgressDialog;
import android.content.Context;
import android.content.pm.PackageManager;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.hardware.Camera;
import android.os.AsyncTask;
import android.os.Bundle;
import android.os.Environment;
import android.os.SystemClock;
import android.support.design.widget.FloatingActionButton;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.util.Log;
import android.view.View;
import android.view.ViewDebug;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.Toast;
import android.app.AlertDialog;
import android.content.DialogInterface;
import android.provider.MediaStore;

import com.sh1r0.caffe_android_lib.CaffeMobile;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
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
import android.os.Handler;
import android.os.Looper;

public class AndroidCameraExample extends Activity implements CvCameraViewListener2, CNNListener {
	private JavaCamResView mOpenCvCameraView;

    static {
        System.loadLibrary("caffe");
        System.loadLibrary("caffe_jni");
        System.loadLibrary("opencv_java");
    }

    public Mat mRgba = new Mat(), mRgbaTemp = new Mat();
    private Mat mTemplate;
    private String LOG_TAG = "PAD";
    private static ArrayList<ArrayList<String>> IMAGENET_CLASSES = new ArrayList<>();
    private static ArrayList<String> IMAGENET_WEIGHTS = new ArrayList<>();
    private static ArrayList<String> IMAGENET_DEPLOY = new ArrayList<>();
    private static ArrayList<String> IMAGENET_MEAN = new ArrayList<>();
    private static ArrayList<Float> IMAGENET_BRIGHTNESS = new ArrayList<>();
    private static ArrayList<String> IMAGENET_DESCRIPTION = new ArrayList<>();
    private static ArrayList<String> IMAGENET_EXCLUDE = new ArrayList<>();
    private static ArrayList<Integer> IMAGENET_CONTINUATION = new ArrayList<>();
    private ArrayList<CaffeMobile> caffeMobile = new ArrayList<>();
    private ProgressDialog dialog, progdialog;
    private Mat mRgbaModified;
    private static int IMAGE_WIDTH = 720;

    //saved contour results
    private boolean markersDetected = false;
    private Mat points = new Mat(4, 2,CvType.CV_32F);
    private Mat checks = new Mat(3, 1,CvType.CV_64F);
    private Mat testMat = new Mat();
    private Mat cropped = new Mat();

    //UI
    private FloatingActionButton analyzeButton;

    public void doAnalysis(View view) {
        // Kabloey
        if (markersDetected) {
            dialog = ProgressDialog.show(AndroidCameraExample.this, "Predicting...", "Cropping Image", true);

            CNNTask cnnTask = new CNNTask(AndroidCameraExample.this);
            cnnTask.execute(mRgba, mTemplate);

            //stop processing images
            mOpenCvCameraView.togglePreview();

            //flag that we have used this Green Circle image
            markersDetected = false;

            //disable button until acquired
            runOnUiThread(new Runnable() {
                @Override
                public void run() {
                    analyzeButton.setEnabled(false);
                }
            });


        }else{
            AlertDialog.Builder alert = new AlertDialog.Builder(AndroidCameraExample.this);
            alert.setTitle("Fiducials not aquired!");
            alert.setMessage("Please align fiducials with screen markers.\nCircles will turn green when aligned.");
            alert.setPositiveButton("OK",null);
            alert.show();
            //Context context = getApplicationContext();
            //Toast.makeText(context, "Fiducials not aquired!\nPlease align fiducials with screen markers.\nCircles will turn green when aligned.", Toast.LENGTH_LONG).show();
        }
    }

    @Override
	public void onCreate(Bundle savedInstanceState) {
		super.onCreate(savedInstanceState);
		getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

		setContentView(R.layout.activity_main);

        ActivityCompat.requestPermissions(this,
                new String[]{android.Manifest.permission.CAMERA,
                        Manifest.permission.READ_EXTERNAL_STORAGE,
                        Manifest.permission.WRITE_EXTERNAL_STORAGE}, 91);

		mOpenCvCameraView = (JavaCamResView) findViewById(R.id.activity_surface_view);
		mOpenCvCameraView.setCvCameraViewListener(this);
        mOpenCvCameraView.enableFpsMeter();

        analyzeButton = (FloatingActionButton) findViewById(R.id.floatingAnalyze);

        // Parse input template file
        Bitmap tBM = BitmapFactory.decodeStream(this.getClass().getResourceAsStream("/template.png"));

        // Convert to OpenCV matrix
        Mat tMat = new Mat();
        Utils.bitmapToMat(tBM, tMat);

        // Parse input test file
        Bitmap tBM2 = BitmapFactory.decodeStream(this.getClass().getResourceAsStream("/test42401.png"));
        // Convert to OpenCV matrix
        Utils.bitmapToMat(tBM2, testMat);

        mTemplate = new Mat();
        Imgproc.cvtColor(tMat, mTemplate, Imgproc.COLOR_BGRA2GRAY);

        /*Camera mCamera = Camera.open();
        Camera.Parameters params = mCamera.getParameters();
        List<Camera.Size> sizes = params.getSupportedPictureSizes();
        for (Camera.Size size : sizes) {
            Log.i("Camera", "Available resolution: "+size.width+" "+size.height);
        }*/

        //get sd cards path
        File sdcard_path = Environment.getExternalStorageDirectory();

        //get the JSON file
        File JSONfile = new File(sdcard_path,"/neural_networks/neural_networks.json");

        //read drug names and nn filenames
        try {
            BufferedReader bufferedReader = new BufferedReader(new FileReader(JSONfile));
            StringBuilder sb = new StringBuilder();

            //load file into string
            String line = bufferedReader.readLine();

            while (line != null) {
                sb.append(line);
                sb.append("\n");
                line = bufferedReader.readLine();
            }

            //convert to JSON
            JSONObject obj = new JSONObject(sb.toString());

            //get first net
            JSONArray nets = obj.getJSONArray("nets");

            //loop over nets
            for(int j=0; j<nets.length(); j++) {
                Log.i("ContoursOut","Loading Net "+ Integer.toString(j));

                JSONObject mynet = nets.getJSONObject(j);

                //get drugs list
                JSONArray drugs = mynet.getJSONArray("DRUGS");

                ArrayList<String> temp_imagenet_classes = new ArrayList<String>();

                for (int i = 0; i < drugs.length(); i++) {
                    temp_imagenet_classes.add(drugs.getString(i));
                }

                IMAGENET_CLASSES.add(temp_imagenet_classes);

                //weights
                IMAGENET_WEIGHTS.add("/neural_networks/" + mynet.getString("WEIGHTS"));

                //deploy
                IMAGENET_DEPLOY.add("/neural_networks/" + mynet.getString("DEPLOY"));

                //mean
                IMAGENET_MEAN.add("/neural_networks/" + mynet.getString("IMAGENET"));

                //brightness
                IMAGENET_BRIGHTNESS.add((float) mynet.getDouble("BRIGHTNESS"));

                //exclusions
                IMAGENET_EXCLUDE.add(mynet.getString("LANES"));

                //description
                IMAGENET_DESCRIPTION.add(mynet.getString("DESCRIPTION"));

                //continuation
                IMAGENET_CONTINUATION.add(mynet.getInt("CONTINUATION"));

                //for each net loaded create a caffemobile instance
                caffeMobile.add(new CaffeMobile());

                //diagnostics
                Log.i("ContoursOut", "Loaded JSON NN ("+j+") file,"+ IMAGENET_CLASSES.get(j).get(0)+","+IMAGENET_CLASSES.get(j).get(1)+","+
                        IMAGENET_CLASSES.get(j).size()+","+IMAGENET_WEIGHTS.get(j)+","+IMAGENET_DESCRIPTION.get(j)+","+
                        IMAGENET_CONTINUATION.get(j).toString());
            }

        } catch (Exception e) {
            e.printStackTrace();
            Log.i("ContoursOut", "Did not load JSON NN file: "+e.toString());
        }

        //throw up progress dialog
        //progdialog = ProgressDialog.show(AndroidCameraExample.this, "Loading Neural Network: " + IMAGENET_DESCRIPTION.get(0), "Loading weights", true);

        //load weights on separate task
        //LoadCaffeModelTask loadCaffeModelTask = new LoadCaffeModelTask();
        //loadCaffeModelTask.execute();

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

    private class DataPoint implements Comparable<DataPoint>  {
        public int i;
        public float Distance, Diameter;
        public Point Center;
        public Boolean valid = true;

        public DataPoint(int ii, float id, float idi, Point imc){
            i = ii;
            Distance = id;
            Diameter = idi;
            Center = imc;
        }

        public int compareTo(DataPoint other) {
            return new Float(Distance).compareTo(new Float(other.Distance));
        }
    };

    public Mat onCameraFrame(CvCameraViewFrame inputFrame) {
        mRgbaModified = inputFrame.rgba();
        mRgbaModified.copyTo(mRgbaTemp);

        boolean portrait = true;
        Mat work = new Mat();
        if(mRgbaModified.size().height >  mRgbaModified.size().width) {
            Imgproc.resize(inputFrame.gray(), work, new Size(IMAGE_WIDTH, (mRgbaModified.size().height * IMAGE_WIDTH) / mRgbaModified.size().width), 0, 0, Imgproc.INTER_LINEAR);
        }else{
            portrait = false;
            Imgproc.resize(inputFrame.gray(), work, new Size((mRgbaModified.size().width * IMAGE_WIDTH) / mRgbaModified.size().height, IMAGE_WIDTH), 0, 0, Imgproc.INTER_LINEAR);
            Core.transpose(work, work);
            Core.flip(work, work, 1);
        }

        boolean fiducialsAcquired = ContourDetection.GetFudicialLocations(mRgbaModified, work, points, checks, portrait);

        //auto analyze?
        if (fiducialsAcquired) {

            Log.i("ContoursOut", String.format("Points (%f, %f),(%f, %f) %f, %f.",
                    points.get(0,0)[0], points.get(0,1)[0], points.get(1,0)[0], points.get(1,1)[0],points.size().height, points.size().width));
            //Log.i("ContoursOut", String.format("Points (%f, %f, %f).",
              //      checks.get(0,0)[0], checks.get(0,1)[0], checks.get(0,2)[0]));

            //save successful frame
            mRgbaTemp.copyTo(mRgba);

            //flag saved
            markersDetected = true;

            //enable button once acquired
            runOnUiThread(new Runnable() {
                @Override
                public void run() {
                    analyzeButton.setEnabled(true);
                }
            });

            // rectify image, include QR/Fiducial points
            //Note: sending color corrected image to rectifyer
            //Mat cropped = new Mat();
            boolean transformedOk = ContourDetection.RectifyImage(mRgba, mTemplate, points, cropped, checks);

            //error?
            if(transformedOk){
                Log.i("ContoursOut", String.format("Got here 2"));

                new Handler(Looper.getMainLooper()).post(new Runnable() {
                    @Override
                    public void run() {
                        Log.d("UI thread", "I am the UI thread");

                        AlertDialog.Builder alert = new AlertDialog.Builder(AndroidCameraExample.this);
                        alert.setTitle("Fiducials aquired!");
                        alert.setMessage("Store PAD image?");
                        alert.setPositiveButton("OK",
                                new DialogInterface.OnClickListener() {
                                    public void onClick(DialogInterface dialog, int which) {
                                        DateFormat df = new SimpleDateFormat("yyyy-MM-dd-HH-mm-ss");
                                        Date today = Calendar.getInstance().getTime();

                                        File SDlocation = Environment.getExternalStorageDirectory();
                                        File padImageDirectory = new File(SDlocation + "/PAD/" + df.format(today));
                                        padImageDirectory.mkdirs();

                                        //save rectified image
                                        File cFile = new File(padImageDirectory, "rectified.jpeg");
                                        Imgproc.cvtColor(cropped, cropped, Imgproc.COLOR_BGRA2RGBA);
                                        Highgui.imwrite(cFile.getPath(), cropped);

                                        //gallery?
                                        try {
                                            MediaStore.Images.Media.insertImage(getContentResolver(), cFile.getPath(),
                                                    df.format(today) , "Rectified Image");
                                        } catch(Exception e) {
                                            Log.i("ContoursOut", "Cannot save to gallery" + e.toString());
                                        }

                                        //start preview
                                        mOpenCvCameraView.StartPreview();

                                        dialog.dismiss();
                                    }
                                }
                        );
                        alert.setNegativeButton("Cancel",
                                new DialogInterface.OnClickListener() {
                                    public void onClick(DialogInterface dialog, int which) {
                                        //start preview
                                        mOpenCvCameraView.StartPreview();

                                        dialog.dismiss();
                                    }
                                }
                        );
                        alert.show();
                    }
                });
                //new Thread(new Task()).start();

                //stop preview
                mOpenCvCameraView.StopPreview();
            }

            //mOpenCvCameraView.StopPreview();

        }

        return mRgbaModified;
	}

    public class PredictionGuess implements Comparable<PredictionGuess> {
        public int Index;
        public float Confidence;
        public int NetIndex;

        public PredictionGuess(int i, float c, int j) {
            this.Index = i;
            this.Confidence = c;
            this.NetIndex = j;
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

            Log.i("ContoursOut", String.format("Got here 1"));
            //create top prediction list
            Vector<PredictionGuess> top = new Vector<>();

            DateFormat df = new SimpleDateFormat("yyyy-MM-dd-HH-mm-ss");
            Date today = Calendar.getInstance().getTime();

            File SDlocation = Environment.getExternalStorageDirectory();
            File padImageDirectory = new File(SDlocation + "/PAD/" + df.format(today));
            padImageDirectory.mkdirs();

            Mat mTemp = new Mat();
            Mat result = new Mat();
            mTemp = input[0];
            /*Imgproc.resize(mTemp, result, new Size(1280, 720)); //should already be this size

            //Mat result = new Mat(Imgproc); //new Mat(mTemp, new Rect(105, 120, mTemp.width()-172, mTemp.height()-240));
            Core.flip(result.t(), result, 1);

            //get stats/brighness
            Scalar brightnessScalar = Core.mean(result);

            double brightness = Math.sqrt((brightnessScalar.val[0] * brightnessScalar.val[0]) * 0.577 +
                    (brightnessScalar.val[1] * brightnessScalar.val[1]) * 0.577 +
                    (brightnessScalar.val[2] * brightnessScalar.val[2]) * 0.577);

            //Log.i("ContoursOut", String.format("Mean %s, %f.",brightnessScalar.toString(), brightness));

            //correct brightness
            float image_brightness = IMAGENET_BRIGHTNESS.get(0);
            Scalar brightnessRatio = new Scalar(image_brightness / brightness, image_brightness / brightness, image_brightness / brightness, 1);

            Core.multiply(result, brightnessRatio, result);*/
            /*brightnessScalar = Core.mean(result);
            brightness = Math.sqrt((brightnessScalar.val[0] * brightnessScalar.val[0]) * 0.577 +
                    (brightnessScalar.val[1] * brightnessScalar.val[1]) * 0.577 +
                    (brightnessScalar.val[2] * brightnessScalar.val[2]) * 0.577);
            Log.i("ContoursOut", String.format("Mean %s, %f.",brightnessScalar.toString(), brightness));*/

            File outputFile = new File(padImageDirectory, "capture.jpeg");
            Imgproc.cvtColor(result, mTemp, Imgproc.COLOR_BGRA2RGBA);
            Highgui.imwrite(outputFile.getPath(), mTemp);

            runOnUiThread(new Runnable() {
                  @Override
                  public void run() {
                      dialog.setMessage("Rectifying Image");
                  }
            });

            // rectify image, include QR/Fiducial points
            //Note: sending color corrected image to rectifyer
            Mat cropped = new Mat();
            boolean transformedOk = ContourDetection.RectifyImage(input[0], input[1], points, cropped, checks);

            //error?
            if(!transformedOk){
                return top;
            }

            //save rectified image
            File cFile = new File(padImageDirectory, "rectified.jpeg");
            //Imgproc.cvtColor(cropped, mTemp, Imgproc.COLOR_BGRA2RGBA);
            Highgui.imwrite(cFile.getPath(), cropped);

            //gallery?
            try {
                MediaStore.Images.Media.insertImage(getContentResolver(), cFile.getPath(),
                        df.format(today) , "Rectified Image");
            } catch(Exception e) {
                Log.i("ContoursOut", "Cannot save to gallery" + e.toString());
            }

            //crop out results area
            Mat cResult = cropped.submat(359, 849, 71, 707);

            //~~~~loop over nets~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            //flag predicting
            runOnUiThread(new Runnable() {
                @Override
                public void run() {
                    dialog.setMessage("Predicting Drug");
                }
            });

            for(int netindex=0; netindex<caffeMobile.size(); netindex++) {
                //clear top prediction list
                top = new Vector<>();

                //remove non trained lanes (AJ for this example, with 52*10 for width)
                Mat imgout = cropped.submat(359, 849, 71, 71 + 53 * (12 - IMAGENET_EXCLUDE.get(netindex).length()));

                //loop over lanes
                for (int i = 0, j = 0; i < 12; i++) {
                    if (!IMAGENET_EXCLUDE.get(netindex).contains(String.valueOf((char) (i + 65)))) { //remove A and J
                        //if(i != 0 && i != 9) { //remove A and J
                        cResult.submat(0, 490, 53 * i, 53 * (i + 1)).copyTo(imgout.submat(0, 490, 53 * j, 53 * (j + 1)));
                        j++;
                    } else {
                        Log.i("ContoursOut", "Excluded " + String.valueOf((char) (i + 65)));
                    }
                }

                //save it
                File crFile = new File(padImageDirectory, "cropped-"+netindex+".jpeg");
                //Imgproc.cvtColor(cResult, mTemp, Imgproc.COLOR_BGRA2RGBA);
                //Imgproc.cvtColor(imgout, mTemp, Imgproc.COLOR_BGRA2RGBA);
                imgout.copyTo(mTemp);
                Highgui.imwrite(crFile.getPath(), mTemp);

                File resFile = new File(padImageDirectory, "resized-"+netindex+".jpeg");
                Imgproc.resize(mTemp, mTemp, new Size(227, 227));
                //test
                //Imgproc.cvtColor(testMat, mTemp, Imgproc.COLOR_BGRA2RGBA);
                Highgui.imwrite(resFile.getPath(), mTemp);

                Log.i("ContoursOut", String.format("Catagorized image at %s, net %d.", resFile.getPath(), netindex));

                float[] scores = caffeMobile.get(netindex).getConfidenceScore(resFile.getPath());

                Vector<PredictionGuess> guesses = new Vector<>();
                for (int i = 0; i < scores.length; i++) {
                    guesses.add(new PredictionGuess(i, scores[i], netindex));
                }
                Collections.sort(guesses);

                try {
                    File oFile = new File(padImageDirectory, "guesses-"+netindex+".txt");
                    oFile.createNewFile();
                    FileOutputStream fOut = new FileOutputStream(oFile);
                    OutputStreamWriter myOutWriter = new OutputStreamWriter(fOut);
                    for (int i = 0; i < scores.length; i++) {
                        myOutWriter.append(String.format("%s - %f%%\n", IMAGENET_CLASSES.get(netindex).get(guesses.get(i).Index),
                                guesses.get(i).Confidence * 100.0));
                    }
                    myOutWriter.close();
                    fOut.close();
                } catch (Exception e) {
                    Log.e(LOG_TAG, "Failed to write guess file: " + e.getMessage());
                }

                //fill PredictionGuess vector
                //get max of number of classes or 3 for top predictions
                int numPredictions = Math.min(3, IMAGENET_CLASSES.get(netindex).size());
                //loop over top predictions
                for (int i = 0; i < numPredictions; i++) {
                    top.add(guesses.get(i));
                }

                //test continuation? Loop if continuation equal to first guess, set to -1 if unused so allways breaks.
                if(guesses.get(0).Index != IMAGENET_CONTINUATION.get(netindex)) break;
            }

            //return data
            return top;
        }

        @Override
        protected void onPostExecute(Vector<PredictionGuess> guess) {
            Log.i(LOG_TAG, String.format("elapsed wall time: %d ms", SystemClock.uptimeMillis() - startTime));
            for( int i = 0; i < guess.size(); i++){
                Log.i(LOG_TAG, String.format("Guess[%f]: %s", guess.get(i).Confidence, IMAGENET_CLASSES.get(guess.get(i).NetIndex).get(guess.get(i).Index)));
            }
            listener.onTaskCompleted(guess);
            super.onPostExecute(guess);
        }
    }

    @Override
    public void onTaskCompleted(Vector<PredictionGuess> result) {
        //get rid of predict dialog
        if (dialog != null) {
            dialog.dismiss();
        }

        //show results if OK
        AlertDialog.Builder alert = new AlertDialog.Builder(AndroidCameraExample.this);

        if(result.size() < 1){
            alert.setTitle("Error rectifying image!");
            alert.setMessage(String.format("Please re-acquire image."));
        }else{
            alert.setTitle("Predicted Drug");
            String alert_message = new String();

            //loop
            for(int i=0; i<result.size(); i++){
                alert_message += String.format(" %s - %2.1f%%\n", IMAGENET_CLASSES.get(result.get(i).NetIndex).get(result.get(i).Index),
                        result.get(i).Confidence * 100.0);
            }

            alert.setMessage(alert_message);
        }
        alert.setPositiveButton("OK",null);
        alert.show();

        //Context context = getApplicationContext();
        //Toast.makeText(context, String.format("Predicted Drugs\n %s - %f%%\n %s - %f%%\n %s - %f%%", IMAGENET_CLASSES[result.get(0).Index], result.get(0).Confidence * 100.0, IMAGENET_CLASSES[result.get(1).Index], result.get(1).Confidence * 100.0, IMAGENET_CLASSES[result.get(2).Index], result.get(2).Confidence * 100.0), Toast.LENGTH_LONG).show();

        mOpenCvCameraView.togglePreview();
    }

    private class LoadCaffeModelTask extends AsyncTask<Void, Void, Void> {

        public LoadCaffeModelTask() {
        }

        @Override
        protected Void doInBackground(Void... params) {
            //load Caffe models
            for(int i=0; i<caffeMobile.size(); i++) {
                //flag which is being loaded?

                //load
                caffeMobile.get(i).setNumThreads(4);
                File sdcard_path = Environment.getExternalStorageDirectory();
                caffeMobile.get(i).loadModel(sdcard_path + IMAGENET_DEPLOY.get(i), sdcard_path + IMAGENET_WEIGHTS.get(i));

                //file
                caffeMobile.get(i).setMean(sdcard_path + IMAGENET_MEAN.get(i));
            }
            return null;
        }

        @Override
        protected void onPostExecute(Void result) {
            //remove progress dialog
            progdialog.dismiss();

            super.onPostExecute(result);
        }
    }

}
