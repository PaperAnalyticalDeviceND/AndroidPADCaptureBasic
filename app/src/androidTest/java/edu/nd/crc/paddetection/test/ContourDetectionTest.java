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
import org.opencv.core.Point;
import org.opencv.core.Point3;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

import edu.nd.crc.paddetection.ContourDetection;

import static org.hamcrest.CoreMatchers.is;
import static org.hamcrest.CoreMatchers.notNullValue;
import static org.hamcrest.MatcherAssert.assertThat;

@RunWith(AndroidJUnit4.class)
@SmallTest
public class ContourDetectionTest extends AndroidTestCase {
    static{
        System.loadLibrary("opencv_java");
    }

    @Test
    public void test_find_markers(){
        // Parse input image file
        Bitmap aBM = BitmapFactory.decodeStream(this.getClass().getResourceAsStream("/wbinput.png"));
        assertThat(aBM, is(notNullValue()));

        // Convert to OpenCV matrix
        Mat aMat = new Mat();
        Utils.bitmapToMat(aBM, aMat);

        Mat input = new Mat();
        Imgproc.cvtColor(aMat, input, Imgproc.COLOR_BGRA2RGB);

        // Find Markers
        List<Point3> actual = ContourDetection.GetMarkers(input);

        // Load Expected
        List<Point3> expected = new ArrayList<>();
        expected.add(new Point3(721, 438, 16));
        expected.add(new Point3(58, 430, 16));
        expected.add(new Point3(59, 167, 22));
        expected.add(new Point3(61, 72, 22));
        expected.add(new Point3(156, 72, 22));

        for( Point3 e : expected ){
            boolean found = false;
            for( Point3 a : actual ){
                if( Math.abs(a.x - e.x) < 1e-5 && Math.abs(a.y - e.y) < 1e-5 && Math.abs(a.z - e.z) < 1e-5 ) {
                    found = true;
                    break;
                }
            }
            assertTrue(String.format("Expected %s to match %s", expected.toString(), actual.toString()), found);
        }
    }

    @Test
    public void test_marker_seperation(){
        // Load Input
        List<Point3> input = new ArrayList<>();
        input.add(new Point3(721, 438, 16));
        input.add(new Point3(58, 430, 16));
        input.add(new Point3(59, 167, 22));
        input.add(new Point3(61, 72, 22));
        input.add(new Point3(156, 72, 22));


        List<Point> QR = new ArrayList<>(), Other = new ArrayList<>();
        ContourDetection.SeperateMarkers(input, QR, Other);

        // Test QR
        List<Point> QRExpected = new ArrayList<>();
        QRExpected.add( new Point(72, 61) );
        QRExpected.add( new Point(72, 156) );
        QRExpected.add( new Point(167, 59) );

        for( Point e : QRExpected ){
            boolean found = false;
            for( Point a : QR ){
                if( Math.abs(a.x - e.x) < 1e-5 && Math.abs(a.y - e.y) < 1e-5 ) {
                    found = true;
                    break;
                }
            }
            assertTrue(String.format("Expected %s to match %s", QRExpected.toString(), QR.toString()), found);
        }

        // Test Other
        List<Point> OtherExpected = new ArrayList<>();
        OtherExpected.add( new Point(438, 721) );
        OtherExpected.add( new Point(430, 58) );

        for( Point e : OtherExpected ){
            boolean found = false;
            for( Point a : Other ){
                if( Math.abs(a.x - e.x) < 1e-5 && Math.abs(a.y - e.y) < 1e-5 ) {
                    found = true;
                    break;
                }
            }
            assertTrue(String.format("Expected %s to match %s", OtherExpected.toString(), Other.toString()), found);
        }
    }
}
