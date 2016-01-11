package edu.nd.crc.paddetection;

import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.highgui.Highgui;
import org.opencv.imgproc.Imgproc;
import org.opencv.imgproc.Moments;

import android.app.Activity;
import android.os.Bundle;
import android.os.Environment;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.WindowManager;

import java.io.File;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.Date;
import java.util.List;
import java.util.Vector;

public class AndroidCameraExample extends Activity implements CvCameraViewListener2 {
	private Mat                    mRgba;
	private JavaCamResView mOpenCvCameraView;

    static {
        System.loadLibrary("opencv_java");
    }

    private Menu mMenu;
    private MenuItem mItemSwitchCamera;
    private MenuItem mItemSaveImage;
    private MenuItem mItemRejectImage;
    private Mat mRgbaModified;
    private boolean SwitchVisable;
    private boolean SaveVisable;
    private boolean RejectVisable;

    @Override
	public void onCreate(Bundle savedInstanceState) {
		super.onCreate(savedInstanceState);
		getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

		setContentView(R.layout.activity_main);

		mOpenCvCameraView = (JavaCamResView) findViewById(R.id.activity_surface_view);
		mOpenCvCameraView.setCvCameraViewListener(this);
        mOpenCvCameraView.enableFpsMeter();

        SwitchVisable = true;
        SaveVisable = RejectVisable = false;
    }

	@Override
	public void onPause()
	{
		super.onPause();
		if (mOpenCvCameraView != null) {
            mOpenCvCameraView.disableView();
        }
	}

	@Override
	public void onResume()
	{
		super.onResume();
        mOpenCvCameraView.enableView();
	}

	public void onDestroy() {
		super.onDestroy();
		if (mOpenCvCameraView != null) {
            mOpenCvCameraView.disableView();
        }
	}

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        mItemSwitchCamera = menu.add("Toggle Camera");
        mItemSaveImage = menu.add("Save Image");
        mItemRejectImage = menu.add("Reject Image");
        return true;
    }

    @Override
    public boolean onPrepareOptionsMenu(Menu menu) {
        mItemSwitchCamera.setVisible(SwitchVisable);
        mItemSaveImage.setVisible(SaveVisable);
        mItemRejectImage.setVisible(RejectVisable);

        return super.onPrepareOptionsMenu(menu);
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {

        if (item == mItemSwitchCamera) {
            mOpenCvCameraView.togglePreview();
        } else if ( item == mItemRejectImage ){
            mOpenCvCameraView.togglePreview();

            SwitchVisable = true;
            SaveVisable = RejectVisable = false;
        } else if ( item == mItemSaveImage ){
            Log.d("PictureDemo", "Saved Image");
            File SDlocation = Environment.getExternalStorageDirectory();
            File padImageDirectory = new File(SDlocation + "/images/");
            padImageDirectory.mkdirs();

            DateFormat df = new SimpleDateFormat("yyyy-MM-dd-HH-mm-ss");
            Date today = Calendar.getInstance().getTime();

            Mat mTemp = new Mat();

            File outputFile = new File(padImageDirectory, df.format(today) + ".jpeg");
            Imgproc.cvtColor(mRgba, mTemp, Imgproc.COLOR_BGRA2RGBA);
            Highgui.imwrite(outputFile.getPath(), mTemp);

            File outputFileM = new File(padImageDirectory, df.format(today) + "-contours.jpeg");
            Imgproc.cvtColor(mRgbaModified, mTemp, Imgproc.COLOR_BGRA2RGBA);
            Highgui.imwrite(outputFileM.getPath(), mTemp);

            SwitchVisable = true;
            SaveVisable = RejectVisable = false;
        }

        return true;
    }


    public void onCameraViewStarted(int width, int height) {
		mRgba = new Mat(height, width, CvType.CV_8UC4);
        mRgbaModified = new Mat(height, width, CvType.CV_8UC4);
        
        mOpenCvCameraView.Setup();

	}

	public void onCameraViewStopped() {
		mRgba.release();
	}

	public Mat onCameraFrame(CvCameraViewFrame inputFrame) {
        mRgba = inputFrame.rgba();
        mRgbaModified = inputFrame.rgba();

        Mat edges = inputFrame.gray().clone();
        Imgproc.Canny(edges, edges, 100, 200, 3, true);

        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(edges, contours, hierarchy, Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE);

        if( contours.size() > 0 ) {
            Vector<Moments> mu = new Vector<>();
            Vector<Point> mc = new Vector<>();

            for (int i = 0; i < contours.size(); i++) {
                mu.add(Imgproc.moments(contours.get(i), false));
                mc.add(new Point(mu.get(i).get_m10() / mu.get(i).get_m00(), mu.get(i).get_m01() / mu.get(i).get_m00()));
            }

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

            for (int i = 0; i < Markers.size(); i++) {
                Imgproc.drawContours(mRgbaModified, contours, Markers.get(i), new Scalar(255, 200, 0), 2, 8, hierarchy, 0, new Point(0, 0));
            }

            if( Markers.size() >= 6 ) {
                mOpenCvCameraView.StopPreview();

                SwitchVisable = false;
                SaveVisable = RejectVisable = true;
            }
        }

		return mRgbaModified;
	}
}
