package edu.nd.crc.paddetection;

import android.content.Intent;
import android.net.Uri;
import android.os.Environment;
import android.util.Log;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Point3;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.highgui.Highgui;
import org.opencv.imgproc.Imgproc;
import org.opencv.imgproc.Moments;

import java.io.File;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.Collections;
import java.util.Date;
import java.util.List;
import java.util.Vector;

/**
 * Created by Omegaice on 6/16/16.
 */
public class ContourDetection {
    private static final double IMAGE_WIDTH = 600.0;

    public static class DataPoint implements Comparable<DataPoint>  {
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

    public static List<Point3> GetMarkers(Mat mat) {
        Mat input = mat.clone();
        WhiteBalance.InPlace(input);

        Mat gray = new Mat();
        Imgproc.cvtColor(input, gray, Imgproc.COLOR_RGB2GRAY);

        Log.i("Contours", String.format("Input size %f, %f", input.size().width, input.size().height));

        float ratio = (float)input.size().width / (float)IMAGE_WIDTH;

        Log.i("Contours", String.format("Ratio %f", ratio));

        Mat work = new Mat();
        Imgproc.resize(gray, work, new Size(IMAGE_WIDTH, (input.size().height * IMAGE_WIDTH) / input.size().width), 0, 0, Imgproc.INTER_LINEAR );
        //Mat output = work.clone();

        Log.i("Contours", String.format("Working size %f, %f", work.size().width, work.size().height));

        Mat work_blur = new Mat();
        Imgproc.blur(work, work_blur, new Size(2, 2));

        Mat edges = work.clone();
        Imgproc.Canny(edges, edges, 50, 150, 3, true);

        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(edges, contours, hierarchy, Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE);

        List<Point3> points = new ArrayList<>();
        if( contours.size() > 0 ) {
            int iBuff[] = new int[(int) (hierarchy.total() * hierarchy.channels())];
            hierarchy.get(0, 0, iBuff);

            Vector<Integer> Markers = new Vector<>();
            for (int i = 0; i < contours.size(); i++) {
                int k = i;
                int c = 0;

                while (iBuff[k * 4 + 2] != -1) {
                    k = iBuff[k * 4 + 2];
                    c = c + 1;
                }

                if (iBuff[k * 4 + 2] != -1) {
                    c = c + 1;
                }

                if (c >= 3) {
                    Markers.add(i);
                }
            }

            List<DataPoint> order = new Vector<>();
            for (int i = 0; i < Markers.size(); i++) {
                //Imgproc.drawContours(output, contours, Markers.get(i), new Scalar(255, 200, 0), 2, 8, hierarchy, 0, new Point(0, 0));

                Moments mum = Imgproc.moments(contours.get(Markers.get(i)), false);
                Point mc = new Point(mum.get_m10() / mum.get_m00(), mum.get_m01() / mum.get_m00());

                //calculate distance to nearest edge
                float dist = Math.min(Math.min(Math.min((float) mc.x, (float) (IMAGE_WIDTH - mc.x)), (float) mc.y), (float) (input.size().height - mc.y));

                Rect box = Imgproc.boundingRect(contours.get(Markers.get(i)));

                float dia = Math.max((float) box.width, (float) box.height) * 0.5f;


                Log.i("Contours", String.format("Marker %d Distance: %f Diameter: %f", i, dist, dia));

                //only add it if sensible
                if (dia < 30 && dia > 15) {
                    order.add(new DataPoint(i, dist, dia, mc));
                }
            }

            for (int i = 0; i < order.size(); i++) {
                if (order.get(i).valid) {
                    for (int j = i + 1; j < order.size(); j++) {
                        if (order.get(j).valid) {
                            double ix = order.get(i).Center.x;
                            double iy = order.get(i).Center.y;
                            double jx = order.get(j).Center.x;
                            double jy = order.get(j).Center.y;

                            if (Math.abs(ix - jx) < 5 && Math.abs(iy - jy) < 5) {
                                if (order.get(i).Diameter < order.get(j).Diameter) {
                                    order.get(i).valid = false;
                                    break;
                                } else {
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

            for (int i = 0; i < order.size(); i++) {
                if (order.get(i).valid) {
                    com.x += order.get(i).Center.x;
                    com.y += order.get(i).Center.y;
                    pcountf += 1.0;
                }
            }

            com.x /= pcountf;
            com.y /= pcountf;
            //Core.circle(output, com, 10, new Scalar(0, 255, 255), 1, 8, 0);

            //count points
            int pcount = 0;

            //loop
            for (int j = 0; j < order.size(); j++) {
                if (order.get(j).valid) {
                    float dia = 0.0f;
                    Point mcd = order.get(j).Center;

                    //if top LHS then QR code marker
                    if (mcd.x < (com.x + 30) && mcd.y < com.y) {
                        dia = 27;
                    } else {
                        dia = 20;
                    }

                    //Core.circle(output, mcd, (int) dia, new Scalar(0, 0, 255), 1, 8, 0);

                    points.add(new Point3((int) (mcd.y * ratio + 0.5), (int) (mcd.x * ratio + 0.5), (int) (dia * ratio + 0.5)));
                    Log.i("Contours", String.format("Point: %d, %d, %d", (int) (mcd.y * ratio + 0.5), (int) (mcd.x * ratio + 0.5), (int) (dia * ratio + 0.5)));
                    if (pcount++ >= 5) break;
                }
            }
        }
        return points;
    }

    public static Mat RectifyImage(Mat input, List<Point3> Markers){


        return new Mat();
    }

    public static void SeperateMarkers(List<Point3> Markers, List<Point> QR, List<Point> Other) {
        int meanSize = 0;
        for ( Point3 p : Markers ) {
            meanSize += p.z;
        }
        meanSize /= Markers.size();

        List<Double> QRSize = new ArrayList<>();
        List<Double> OtherSize = new ArrayList<>();

        // separate points and get averages
        double averageqr = 0;
        double averageouter = 0;
        for( Point3 point : Markers) {
            if( point.z > meanSize ) {
                QR.add(new Point(point.y, point.x));
                averageqr += point.z;
                QRSize.add(point.z);
            }else {
                Other.add(new Point(point.y, point.x));
                averageouter += point.z;
                OtherSize.add(point.z);
            }
        }
        // weed out additional points
        if( QR.size() > 3 ){
            averageqr /= QR.size();
            double maxdev = 0;
            int maxindex = -1;
            for( int i = 0; i < QR.size(); i++ ) {
                if (Math.abs(QRSize.get(i) - averageqr) > maxdev) {
                    maxdev = Math.abs(QRSize.get(i) - averageqr);
                    maxindex = i;
                }
            }
            if (maxindex > -1) {
                QR.remove(maxindex);
            }
        }
        // make sure no markers in QR if all QR defined
        if (QR.size() == 3 && QR.size() > 3) {
            // get average point and maximum x, y for QR markers
            double avx = 0, avy = 0;
            double maxx = 0, maxy = 0;
            for( int i = 0; i < 3; i++ ) {
                avx += QR.get(i).x;
                if( QR.get(i).x > maxx ){
                    maxx = QR.get(i).x;
                }

                avy += QR.get(i).y;
                if( QR.get(i).y > maxy ){
                    maxy = QR.get(i).y;
                }
            }
            avx /= 3;
            avy /= 3;
            maxx -= avx;
            maxy -= avy;
            
            // check if any outer points in QR domain
            for(int i = 0; i < Other.size(); i++ ){
                if( (Other.get(i).x - avx) < maxx && (Other.get(i).y - avy) < maxy ){
                    //Other.pop(i);
                    break;
                }
            }
        }

        // still too many outer points?
        if( Other.size() > 3 ){
            averageouter /= Other.size();
            double maxdev = 0;
            int maxindex = -1;
            for(int i = 0; i < Other.size(); i++ ) {
                if( Math.abs(OtherSize.get(i) - averageouter) > maxdev ) {
                    maxdev = Math.abs(OtherSize.get(i) - averageouter);
                    maxindex = i;
                }
            }
            if( maxindex > -1 ){
                Other.remove(maxindex);
            }
        }
    }
}
