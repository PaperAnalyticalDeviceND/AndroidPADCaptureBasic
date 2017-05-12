package edu.nd.crc.paddetection;

import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
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
    private Mat mRgbaModified;
    private static int IMAGE_WIDTH = 600;

    //saved contour results
    private boolean markersDetected = false;
    private List<Point> points;
    private List<Point> checks;

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

                if (markersDetected) {
                    dialog = ProgressDialog.show(AndroidCameraExample.this, "Predicting...", "Cropping Image", true);

                    CNNTask cnnTask = new CNNTask(AndroidCameraExample.this);
                    cnnTask.execute(mRgba, mTemplate);

                    mOpenCvCameraView.togglePreview();
                }
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
        //mRgba = inputFrame.rgba();
        mRgbaModified = inputFrame.rgba();

        float ratio = (float)mRgbaModified.size().width / (float)IMAGE_WIDTH;

        Mat work = new Mat();
        Imgproc.resize(inputFrame.gray(), work, new Size(IMAGE_WIDTH, (mRgbaModified.size().height * IMAGE_WIDTH) / mRgbaModified.size().width), 0, 0, Imgproc.INTER_LINEAR );

        //inputFrame.gray().copyTo(work);

        Mat work_blur = new Mat();
        Imgproc.blur(work, work_blur, new Size(2, 2));


        Mat edges = work.clone();
        Imgproc.Canny(edges, edges, 40 , 150, 3, true);

        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(edges, contours, hierarchy, Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE);

        if( contours.size() > 0 ) {
            int iBuff[] = new int[ (int) (hierarchy.total() * hierarchy.channels()) ];
            hierarchy.get(0, 0, iBuff);

            Vector<Integer> Markers = new Vector<>();
            for (int i = 0; i < contours.size(); i++) {
                int k = i;
                int c = 0;

                while (iBuff[k*4+2] != -1) {
                    k = iBuff[k*4+2];
                    c = c + 1;
                }

                if (iBuff[k*4+2] != -1) {
                    c = c + 1;
                }

                if (c >= 5) {
                    Markers.add(i);
                }
            }

            List<DataPoint> order = new Vector<>();
            List<Float> diameter = new Vector<>();
            for( int i=0; i < Markers.size(); i++){
                Imgproc.drawContours(mRgbaModified, contours, Markers.get(i), new Scalar(255, 200, 0), 2, 8, hierarchy, 0, new Point(0, 0));

                Moments mum = Imgproc.moments(contours.get(Markers.get(i)), false);
                Point mc = new Point( mum.get_m10()/mum.get_m00() , mum.get_m01()/mum.get_m00() );

                //calculate distance to nearest edge
                float dist = Math.min(Math.min(Math.min((float)mc.x, (float)(IMAGE_WIDTH - mc.x)), (float)mc.y), (float)(inputFrame.rgba().size().height - mc.y));

                Rect box = Imgproc.boundingRect(contours.get(Markers.get(i)));

                float dia = Math.max((float)box.width, (float)box.height) * 0.5f;

                //only add it if sensible
                if(dia < 20 && dia > 5){
                    order.add(new DataPoint(i, dist, dia, mc));
                    diameter.add(dia);
                }
            }

            for( int i=0; i < order.size(); i++){
                if(order.get(i).valid){
                    for( int j=i+1; j < order.size(); j++){
                        if(order.get(j).valid){
                            double ix = order.get(i).Center.x;
                            double iy = order.get(i).Center.y;
                            double jx = order.get(j).Center.x;
                            double jy = order.get(j).Center.y;

                            if(Math.abs(ix - jx) < 5 && Math.abs(iy - jy) < 5){
                                if(order.get(i).Diameter < order.get(j).Diameter){
                                    order.get(i).valid = false;
                                    break;
                                }else{
                                    order.get(j).valid = false;
                                }
                            }
                        }
                    }
                }
            }

            Collections.sort(order);

            //get center of mass
            Point com = new Point(0.0, 0.0);
            float pcountf = 0.0f;

            for( int i=0; i<order.size(); i++){
                if(order.get(i).valid){
                    com.x += order.get(i).Center.x;
                    com.y += order.get(i).Center.y;
                    pcountf += 1.0;
                }
            }

            com.x /= pcountf;
            com.y /= pcountf;

            //scale back to image size
            Point comDisplay = new Point(com.x * ratio, com.y * ratio);
            Point c_o_m = new Point(517, 429); //with offsets, subtract y from width of 768

            double distance = Math.sqrt((com.x * ratio - 517) * (com.x * ratio - 517) + (com.y * ratio - 429) * (com.y * ratio - 429));

            Core.circle(mRgbaModified, comDisplay, 10, new Scalar(0, 255, 255), 2, 8, 0);
            Core.circle(mRgbaModified, c_o_m, 12, new Scalar(255, 0, 0), 2, 8, 0);

            //reasonable COM?
            Scalar markerColor;
            Boolean markersOK;

            List<Point> QR = new Vector<>();
            List<Point> Fiducial = new Vector<>();

            if(distance < 50 && pcountf > 5) {
            //if(com.x < 280 && com.x > 200 && com.y < 200 && com.y > 120 && pcountf > 5) {
                markerColor = new Scalar(0, 255, 0);
                markersOK = true;
                //QR = new Vector<>();
                //Fiducial = new Vector<>();
            }else{
                markerColor = new Scalar(255, 0, 0);
                markersOK = false;
            }

            //count points
            int pcount = 0;

            //loop
            for (int j = 0; j < order.size(); j++) {
                if (order.get(j).valid) {
                    float dia;
                    String targetType;
                    Point mcd = order.get(j).Center;

                    //if top LHS then QR code marker
                    if (mcd.y > (com.y - 30) && mcd.x < com.x) {
                        dia = 30;
                        targetType = "QR";
                        if(markersOK){
                            QR.add(new Point(768 - (mcd.y * ratio), mcd.x * ratio));
                        }
                    } else {
                        dia = 15;
                        targetType = "Fiducial";
                        if(markersOK){
                            Fiducial.add(new Point(768 - (mcd.y * ratio), mcd.x * ratio));
                        }
                    }

                    if (markersOK) Log.i("Contours", String.format("Tyep: %s, Point %d: %d, %d, %f, %f, %f, %f", targetType, j, (int) (mcd.x * ratio + 0.5),
                            (int) (mcd.y * ratio + 0.5), diameter.get(j), com.x * ratio, com.y * ratio, distance));

                    //scale back to image size
                    mcd.x *= ratio;
                    mcd.y *= ratio;
                    Core.circle(mRgbaModified, mcd, (int) dia, markerColor, 2, 8, 0);

                    if (pcount++ >= 5) break;
                }
            }

            //auto analyze?
            if( markersOK ) {
                //save successful frame
                mRgba = inputFrame.rgba();

                //sort points
                int qrxhigh = -1, qryhigh = -1, qr1 = -1, fudxlow = -1, fudylow = -1, fud4 = -1;
                double qrxmax = 0, qrymax = 0, fudxmin = 768, fudymin = 1280;
                for(int i=0; i<3; i++){
                    if(QR.get(i).x > qrxmax){
                        qrxhigh = i;
                        qrxmax = QR.get(i).x;
                    }
                    if(QR.get(i).y > qrymax){
                        qryhigh = i;
                        qrymax = QR.get(i).y;
                    }
                    if(Fiducial.get(i).x < fudxmin){
                        fudxlow = i;
                        fudxmin = Fiducial.get(i).x;
                    }
                    if(Fiducial.get(i).y < fudymin){
                        fudylow = i;
                        fudymin = Fiducial.get(i).y;
                    }
                }
                for(int i=0; i<3; i++){
                    if(i != qrxhigh && i != qryhigh) qr1 = i;
                    if(i != fudxlow && i != fudylow) fud4 = i;
                }

                //create points
                points = new Vector<>();
                points.add(QR.get(qr1)); points.add(QR.get(qryhigh));
                points.add(Fiducial.get(fudxlow)); points.add(Fiducial.get(fud4)); points.add(Fiducial.get(fudylow));
                checks = new Vector<>();
                checks.add(QR.get(qrxhigh));

                //flag saved
                markersDetected = true;
                //mOpenCvCameraView.StopPreview();

                //SwitchVisable = false;
                //SaveVisable = RejectVisable = true;
            }

        }

        return mRgbaModified;
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
            Mat result = new Mat();
            mTemp = input[0];
            Imgproc.resize(mTemp, result, new Size(1280, 768)); //should already be this size
            //Point a = QR.get(0);

            //Mat result = new Mat(Imgproc); //new Mat(mTemp, new Rect(105, 120, mTemp.width()-172, mTemp.height()-240));
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

            // rectify image
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
