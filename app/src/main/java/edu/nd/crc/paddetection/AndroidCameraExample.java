package edu.nd.crc.paddetection;

import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.core.Mat;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.core.Rect;
import org.opencv.highgui.Highgui;
import org.opencv.imgproc.Imgproc;

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
import java.util.Calendar;
import java.util.Date;

public class AndroidCameraExample extends Activity implements CvCameraViewListener2 {
	private JavaCamResView mOpenCvCameraView;

    static {
        System.loadLibrary("opencv_java");
    }

    private Mat mRgba;

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
}
