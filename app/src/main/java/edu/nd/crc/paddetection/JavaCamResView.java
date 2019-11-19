package edu.nd.crc.paddetection;

import java.util.List;

import org.opencv.android.JavaCameraView;

import android.content.Context;
import android.hardware.Camera;
import android.util.AttributeSet;
import android.util.DisplayMetrics;
import android.util.Log;
import android.view.Display;

import static android.hardware.Camera.Parameters.FLASH_MODE_TORCH;
import static android.hardware.Camera.Parameters.FOCUS_MODE_CONTINUOUS_VIDEO;

public class JavaCamResView extends JavaCameraView {
    private boolean mPreviewShowing;

    public JavaCamResView(Context context, AttributeSet attrs) {
        super(context, attrs);

        mPreviewShowing = true;
    }

    public void Setup(){
        StopPreview();
        disconnectCamera();

        //connectCamera(960, 720); 1920x1080, 3840x2160, 1280x720
        //connectCamera(1920, 1080);
        Log.d("Preview", "Width" + getWidth() + ", " + getHeight());
        connectCamera(getWidth(), getHeight());

        Camera.Parameters params = this.mCamera.getParameters();
        params.setFlashMode(FLASH_MODE_TORCH);
        params.setFocusMode(FOCUS_MODE_CONTINUOUS_VIDEO);
        //params.setPreviewSize(3840, 2160);
        this.mCamera.setParameters(params);
        StartPreview();
    }

    public void togglePreview(){
        if(mPreviewShowing){
            Log.d("Preview", "Stop");
            StopPreview();
        }else{
            Log.d("Preview", "Start");
            StartPreview();
        }
    }

    public void StopPreview(){
        mCamera.stopPreview();
        mCamera.setPreviewCallback(null);
        mPreviewShowing = false;
    }

    public void StartPreview(){
        mCamera.startPreview();
        mCamera.setPreviewCallback(this);
        mPreviewShowing = true;
    }
}