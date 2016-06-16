package edu.nd.crc.paddetection;

import org.opencv.core.Mat;

public class WhiteBalance {
    public static int[][] CalculateHistogram(Mat mat){
        int[][] hists = new int[3][256];
        for( int i = 0; i < 3; i++ ){
            for( int j = 0; j < 256; j++) {
                hists[i][j] = 0;
            }
        }

        for (int y = 0; y < mat.rows(); y++) {
            for (int x = 0; x < mat.cols(); x++) {
                byte buffs[] = new byte[3];
                mat.get(y, x, buffs);
                for (int j = 0; j < 3; ++j) {
                    hists[j][buffs[j]& 0xFF] += 1;
                }
            }
        }

        return hists;
    }

    public static void InPlace(Mat mat) {
        double discard_ratio = 0.05;
        int[][] hists = CalculateHistogram(mat);

        // cumulative hist
        int total = mat.cols()*mat.rows();
        int[] vmin = new int[3];
        int[] vmax = new int[3];
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 255; ++j) {
                hists[i][j + 1] += hists[i][j];
            }
            vmin[i] = 0;
            vmax[i] = 255;
            while (vmin[i] < 255 && hists[i][vmin[i]] < discard_ratio * total) {
                vmin[i] += 1;
            }
            while (vmin[i] > 0 && hists[i][vmax[i]] > (1 - discard_ratio) * total) {
                vmax[i] -= 1;
            }
            if (vmax[i] < 255 - 1) vmax[i] += 1;
        }


        for (int y = 0; y < mat.rows(); ++y) {
            for (int x = 0; x < mat.cols(); ++x) {
                byte buff[] = new byte[mat.cols() * mat.channels()];
                mat.get(y, 0, buff);

                for (int j = 0; j < 3; ++j) {
                    int val = (int)mat.get(y,x)[j];
                    if (val < vmin[j]) val = vmin[j];
                    if (val > vmax[j]) val = vmax[j];
                    buff[x * 3 + j] = (byte)((val - vmin[j]) * 255.0 / (vmax[j] - vmin[j]));
                }
                mat.put(y, 0, buff);
            }
        }
    }
}
