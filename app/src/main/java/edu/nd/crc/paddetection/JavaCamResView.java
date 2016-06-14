package edu.nd.crc.paddetection;

import java.util.List;

import org.opencv.android.JavaCameraView;

import android.content.Context;
import android.hardware.Camera;
import android.util.AttributeSet;
import android.util.Log;

public class JavaCamResView extends JavaCameraView {
    private boolean mPreviewShowing;

    public JavaCamResView(Context context, AttributeSet attrs) {
        super(context, attrs);

        mPreviewShowing = true;
    }

    public void Setup(){
        StopPreview();
        disconnectCamera();
        connectCamera(960, 720);
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