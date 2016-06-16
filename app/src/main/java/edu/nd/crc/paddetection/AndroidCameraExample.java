package edu.nd.crc.paddetection;

import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.highgui.Highgui;
import org.opencv.imgproc.Imgproc;
import org.opencv.imgproc.Moments;

import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.net.Uri;
import android.os.Bundle;
import android.os.Environment;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.View;
import android.view.WindowManager;
import android.widget.Toast;

import java.io.File;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.Collections;
import java.util.Date;
import java.util.List;
import java.util.Vector;

public class AndroidCameraExample extends Activity implements CvCameraViewListener2 {
	private JavaCamResView mOpenCvCameraView;

    static {
        System.loadLibrary("opencv_java");
    }

    private Mat mRgba;
    private double IMAGE_WIDTH = 600;

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
                Log.d("PictureDemo", "Saved Image");
                File SDlocation = Environment.getExternalStorageDirectory();
                File padImageDirectory = new File(SDlocation + "/images/");
                padImageDirectory.mkdirs();

                DateFormat df = new SimpleDateFormat("yyyy-MM-dd-HH-mm-ss");
                Date today = Calendar.getInstance().getTime();

                Mat mTemp = new Mat();
                Mat result = new Mat(mRgba, new Rect(105, 120, mRgba.width()-172, mRgba.height()-240));
                Core.flip(result.t(), result, 1);

                boolean Success = true;

                File outputFile = new File(padImageDirectory, df.format(today) + ".jpeg");
                Imgproc.cvtColor(result, mTemp, Imgproc.COLOR_BGRA2RGBA);
                if(Highgui.imwrite(outputFile.getPath(), mTemp) ) {
                    Intent intentA = new Intent(Intent.ACTION_MEDIA_SCANNER_SCAN_FILE);
                    intentA.setData(Uri.fromFile(outputFile));
                    sendBroadcast(intentA);
                }else {
                    Success = false;
                }

                Context context = getApplicationContext();
                if( Success ) {
                    Toast.makeText(context, "Save Succeeded", Toast.LENGTH_SHORT).show();
                }else{
                    Toast.makeText(context, "Save Failed", Toast.LENGTH_SHORT).show();
                }

                RectifyImage(outputFile.getPath());
            }
        });
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

    public void RectifyImage(String path){
        Mat input = Highgui.imread(path);
        Mat gray = new Mat();

        WhiteBalance.InPlace(input);

        Imgproc.cvtColor(input, gray, Imgproc.COLOR_RGB2GRAY);

        float ratio = (float)input.size().width / (float)IMAGE_WIDTH;

        Mat work = new Mat();
        Imgproc.resize(gray, work, new Size(IMAGE_WIDTH, (input.size().height * IMAGE_WIDTH) / input.size().width), 0, 0, Imgproc.INTER_LINEAR );
        Mat output = work.clone();


        Mat work_blur = new Mat();
        Imgproc.blur(work, work_blur, new Size(2, 2));

        Mat edges = work.clone();
        Imgproc.Canny(edges, edges, 40, 150, 3, true);

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

                if (c >= 3) {
                    Markers.add(i);
                }
            }

            List<DataPoint> order = new Vector<>();
            for( int i=0; i < Markers.size(); i++){
                Imgproc.drawContours(output, contours, Markers.get(i), new Scalar(255, 200, 0), 2, 8, hierarchy, 0, new Point(0, 0));

                Moments mum = Imgproc.moments(contours.get(Markers.get(i)), false);
                Point mc = new Point( mum.get_m10()/mum.get_m00() , mum.get_m01()/mum.get_m00() );

                //calculate distance to nearest edge
                float dist = Math.min(Math.min(Math.min((float)mc.x, (float)(IMAGE_WIDTH - mc.x)), (float)mc.y), (float)(input.size().height - mc.y));

                Rect box = Imgproc.boundingRect(contours.get(Markers.get(i)));

                float dia = Math.max((float)box.width, (float)box.height) * 0.5f;

                //only add it if sensible
                if(dia < 30 && dia > 15){
                    order.add(new DataPoint(i, dist, dia, mc));
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
            Core.circle(output, com, 10, new Scalar(0, 255, 255), 1, 8, 0);

            //count points
            int pcount = 0;

            //loop
            for( int j=0; j<order.size(); j++){
                if(order.get(j).valid){
                    float dia = 0.0f;
                    Point mcd = order.get(j).Center;

                    //if top LHS then QR code marker
                    if(mcd.x < (com.x + 30) && mcd.y < com.y){
                        dia = 27;
                    }else{
                        dia = 20;
                    }

                    Core.circle(output, mcd, (int)dia, new Scalar(0, 0, 255), 1, 8, 0);

                    Log.i("Contours", String.format("Point: %d, %d, %d", (int)(mcd.y * ratio + 0.5), (int)(mcd.x * ratio  + 0.5), (int)(dia * ratio + 0.5)));
                    if(pcount++ >= 5) break;
                }
            }

            File SDlocation = Environment.getExternalStorageDirectory();
            File padImageDirectory = new File(SDlocation + "/images/");
            padImageDirectory.mkdirs();

            DateFormat df = new SimpleDateFormat("yyyy-MM-dd-HH-mm-ss");
            Date today = Calendar.getInstance().getTime();

            File outputFile = new File(padImageDirectory, df.format(today) + "-rectify.jpeg");
            if(Highgui.imwrite(outputFile.getPath(), output) ) {
                Intent intentA = new Intent(Intent.ACTION_MEDIA_SCANNER_SCAN_FILE);
                intentA.setData(Uri.fromFile(outputFile));
                sendBroadcast(intentA);
            }
        }
    }
}
