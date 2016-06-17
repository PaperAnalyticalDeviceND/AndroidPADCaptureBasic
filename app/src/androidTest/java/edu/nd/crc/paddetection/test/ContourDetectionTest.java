package edu.nd.crc.paddetection.test;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.support.test.filters.SmallTest;
import android.support.test.runner.AndroidJUnit4;
import android.test.AndroidTestCase;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.opencv.android.Utils;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Point3;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

import edu.nd.crc.paddetection.ContourDetection;
import edu.nd.crc.paddetection.WhiteBalance;

import static org.hamcrest.Matchers.closeTo;
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

    @Test
    public void test_qr_order(){
        // Input
        List<Point> qr = new ArrayList<>();
        qr.add( new Point(167, 59) );
        qr.add( new Point(72, 61) );
        qr.add( new Point(72, 156) );

        List<Point> other = new ArrayList<>();
        other.add( new Point(430, 58) );
        other.add( new Point(438, 721) );

        // Sort
        ContourDetection.SortQR(qr, other);

        // Test QR
        List<Point> qrexpected = new ArrayList<>();
        qrexpected.add( new Point(72, 61) );
        qrexpected.add( new Point(72, 156) );
        qrexpected.add( new Point(167, 59) );

        for( Point e : qrexpected ){
            boolean found = false;
            for( Point a : qr ){
                if( Math.abs(a.x - e.x) < 1e-5 && Math.abs(a.y - e.y) < 1e-5 ) {
                    found = true;
                    break;
                }
            }
            assertTrue(String.format("Expected %s to match %s", qrexpected.toString(), qr.toString()), found);
        }

        // Test Other
        List<Point> otherexpected = new ArrayList<>();
        other.add( new Point(438, 721) );
        other.add( new Point(430, 58) );

        for( Point e : otherexpected ){
            boolean found = false;
            for( Point a : other ){
                if( Math.abs(a.x - e.x) < 1e-5 && Math.abs(a.y - e.y) < 1e-5 ) {
                    found = true;
                    break;
                }
            }
            assertTrue(String.format("Expected %s to match %s", otherexpected.toString(), other.toString()), found);
        }
    }

    @Test
    public void test_match_points(){
        // Load Input
        List<Point3> input = new ArrayList<>();
        input.add(new Point3(721, 438, 16));
        input.add(new Point3(58, 430, 16));
        input.add(new Point3(59, 167, 22));
        input.add(new Point3(61, 72, 22));
        input.add(new Point3(156, 72, 22));

        // Run Function
        List<Point> Source = new ArrayList<>(), SourceTest = new ArrayList<>();
        List<Point> Destination = new ArrayList<>(), DestinationTest = new ArrayList<>();

        ContourDetection.MatchPoints(input, Source, SourceTest, Destination, DestinationTest);

        // Expected
        List<Point> eSource = new ArrayList<>();
        eSource.add(new Point(438, 721));
        eSource.add(new Point(430, 58));
        eSource.add(new Point(72, 61));
        eSource.add(new Point(72, 156));

        for( Point e : eSource ){
            boolean found = false;
            for( Point a : Source ){
                if( Math.abs(a.x - e.x) < 1e-5 && Math.abs(a.y - e.y) < 1e-5 ) {
                    found = true;
                    break;
                }
            }
            assertTrue(String.format("Expected %s to match %s", eSource.toString(), Source.toString()), found);
        }

        List<Point> eSourceTest = new ArrayList<>();
        eSourceTest.add(new Point(167, 59));

        for( Point e : eSourceTest ){
            boolean found = false;
            for( Point a : SourceTest ){
                if( Math.abs(a.x - e.x) < 1e-5 && Math.abs(a.y - e.y) < 1e-5 ) {
                    found = true;
                    break;
                }
            }
            assertTrue(String.format("Expected %s to match %s", eSourceTest.toString(), eSourceTest.toString()), found);
        }

        List<Point> eDestination = new ArrayList<>();
        eSource.add(new Point(686, 1163));
        eSource.add(new Point(686, 77));
        eSource.add(new Point(82, 64));
        eSource.add(new Point(82, 226));

        for( Point e : eDestination ){
            boolean found = false;
            for( Point a : Destination ){
                if( Math.abs(a.x - e.x) < 1e-5 && Math.abs(a.y - e.y) < 1e-5 ) {
                    found = true;
                    break;
                }
            }
            assertTrue(String.format("Expected %s to match %s", eDestination.toString(), Destination.toString()), found);
        }

        List<Point> eDestinationTest = new ArrayList<>();
        eDestinationTest.add(new Point(244, 64));

        for( Point e : eDestinationTest ){
            boolean found = false;
            for( Point a : DestinationTest ){
                if( Math.abs(a.x - e.x) < 1e-5 && Math.abs(a.y - e.y) < 1e-5 ) {
                    found = true;
                    break;
                }
            }
            assertTrue(String.format("Expected %s to match %s", eDestinationTest.toString(), DestinationTest.toString()), found);
        }
    }

    @Test
    public void test_transform_points() {
        // Input
        List<Point> source = new ArrayList<>();
        source.add(new Point(398.5, 225.5));
        source.add(new Point(388.5, 1173.5));

        List<Point> destination = new ArrayList<>();
        destination.add(new Point(387, 214));
        destination.add(new Point(387, 1164));

        // Calculate Actual
        Mat actual = ContourDetection.TransformPoints(source, destination);

        // Load Expected
        Mat expected = new Mat(2, 3, CvType.CV_32FC1);
        expected.put(0, 0, 1.002, 0.0106, -14.6797 );
        expected.put(1, 0, -0.0106, 1.002, -7.7386 );

        // Compare
        for( int i = 0; i < actual.rows(); i++ ) {
            for (int j = 0; j < actual.cols(); j++) {
                assertThat(String.format("Value[%d,%d] Matches", i, j), actual.get(i, j)[0], closeTo(expected.get(i, j)[0], 1e-4));
            }
        }
    }

    @Test
    public void test_rectify_image(){
        // Parse input image file
        Bitmap aBM = BitmapFactory.decodeStream(this.getClass().getResourceAsStream("/rectify-input.png"));
        assertThat(aBM, is(notNullValue()));

        // Convert to OpenCV matrix
        Mat aMat = new Mat();
        Utils.bitmapToMat(aBM, aMat);

        Mat input = new Mat();
        Imgproc.cvtColor(aMat, input, Imgproc.COLOR_BGRA2RGB);

        // Parse input template file
        Bitmap tBM = BitmapFactory.decodeStream(this.getClass().getResourceAsStream("/rectify-template.png"));
        assertThat(tBM, is(notNullValue()));

        // Convert to OpenCV matrix
        Mat tMat = new Mat();
        Utils.bitmapToMat(tBM, tMat);

        Mat template = new Mat();
        Imgproc.cvtColor(tMat, template, Imgproc.COLOR_BGRA2GRAY);

        // Run white balance
        Mat actual = ContourDetection.RectifyImage(input, template);

        // Parse expected image file
        Bitmap eBM = BitmapFactory.decodeStream(this.getClass().getResourceAsStream("/rectify-expected.png"));
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
