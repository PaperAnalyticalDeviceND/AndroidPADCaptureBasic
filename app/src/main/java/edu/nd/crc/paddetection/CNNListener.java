package edu.nd.crc.paddetection;

import java.util.Vector;

public interface CNNListener {
    void onTaskCompleted(Vector<AndroidCameraExample.PredictionGuess> result);
}