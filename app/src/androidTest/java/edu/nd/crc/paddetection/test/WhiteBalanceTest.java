package edu.nd.crc.paddetection.test;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.support.test.filters.SmallTest;
import android.support.test.runner.AndroidJUnit4;
import android.test.AndroidTestCase;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

import java.util.Scanner;

import edu.nd.crc.paddetection.WhiteBalance;

import static org.hamcrest.CoreMatchers.is;
import static org.hamcrest.CoreMatchers.notNullValue;
import static org.hamcrest.MatcherAssert.assertThat;

@RunWith(AndroidJUnit4.class)
@SmallTest
public class WhiteBalanceTest extends AndroidTestCase {
    static{
        System.loadLibrary("opencv_java");
    }

    @Test
    public void test_calculate_histogram(){
        // Parse input image file
        Bitmap aBM = BitmapFactory.decodeStream(this.getClass().getResourceAsStream("/wbinput.png"));
        assertThat(aBM, is(notNullValue()));

        // Convert to OpenCV matrix
        Mat aMat = new Mat();
        Utils.bitmapToMat(aBM, aMat);

        Mat input = new Mat();
        Imgproc.cvtColor(aMat, input, Imgproc.COLOR_BGRA2RGB);

        // Run histogram calculation
        int[][] actual = WhiteBalance.CalculateHistogram(input);

        // Load expected data
        int[][] expected = new int[3][256];

        Scanner sc = new Scanner(this.getClass().getResourceAsStream("/hists.txt"));
        int x = 0;
        while (sc.hasNextLine()) {
            int y = 0;

            String line = sc.nextLine();
            for( String val : line.split(", ")) {
                expected[x][y] = Integer.parseInt(val);
                y++;
            }
            x++;
        }

        // Check sizes of histogram
        assertThat(actual.length, is(expected.length));
        for( int i = 0; i < actual.length; i++ ){
            assertThat(actual[i].length, is(expected[i].length));
        }

        // Check histogram values
        for( int i = 0; i < actual.length; i++ ){
            for( int j = 0; j < actual[i].length; j++ ) {
                assertThat(String.format("Value[%d,%d] Matches", i, j), actual[i][j], is(expected[i][j]));
            }
        }
    }

    //@Test
    public void test_in_place() {
        // Parse input image file
        Bitmap aBM = BitmapFactory.decodeStream(this.getClass().getResourceAsStream("/wbinput.png"));
        assertThat(aBM, is(notNullValue()));

        // Convert to OpenCV matrix
        Mat aMat = new Mat();
        Utils.bitmapToMat(aBM, aMat);

        Mat actual = new Mat();
        Imgproc.cvtColor(aMat, actual, Imgproc.COLOR_BGRA2RGB);

        // Run white balance
        WhiteBalance.InPlace(actual);

        // Parse expected image file
        Bitmap eBM = BitmapFactory.decodeStream(this.getClass().getResourceAsStream("/wbexpected.png"));
        assertThat(eBM, is(notNullValue()));

        // Convert expected to OpenCV matrix
        Mat eMat = new Mat();
        Utils.bitmapToMat(eBM, eMat);

        Mat expected = new Mat();
        Imgproc.cvtColor(eMat, expected, Imgproc.COLOR_BGRA2RGB);

        // Check that actual matrix matches expected matrix
        for( int i = 0; i < actual.rows(); i++ ){
            for( int j = 0; j < actual.cols(); j++ ) {
                assertThat(String.format("Value[%d,%d] Matches", i, j), actual.get(i, j), is(expected.get(i, j)));
            }
        }
    }
}