package edu.nd.crc.paddetection;

import android.content.Intent;
import android.net.Uri;
import android.os.Environment;
import android.util.Log;

import org.opencv.core.Core;
import org.opencv.core.CvType;
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
import java.util.Arrays;
import java.util.Calendar;
import java.util.Collections;
import java.util.Date;
import java.util.List;
import java.util.Vector;

import static com.google.common.primitives.Doubles.min;
import static java.lang.Math.abs;
import static java.lang.Math.atan2;

/**
 * Created by Omegaice on 6/16/16.
 */
public class ContourDetection {
    private static final double IMAGE_WIDTH = 720.0;

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

    //get fudicial points, mark onto current image (mRgbaModified)
    public static boolean GetFudicialLocations(Mat mRgbaModified, Mat work, List<Point> src_points, boolean portrait){

        //get analasis/real ratio
        float ratio;

        if(portrait){
            ratio = (float)mRgbaModified.size().width / (float)IMAGE_WIDTH;
        }else{
            ratio = (float)mRgbaModified.size().height / (float)IMAGE_WIDTH;
        }

        //draw target fiducials
        //[85, 1163], [686, 1163], [686, 77], [82, 64], [82, 226], [244, 64]
        //portrait    comDisplay = new Point(mc.x * ratio, mc.y * ratio);
        //landscape comDisplay = new Point((mc.y) * ratio, (720-mc.x) * ratio);
        //center = 360
        double horiz_line = 730 / 2;
        double scale_ratio = min(work.size().height / 1220, 1.0) * .95;
        if(scale_ratio > .85){
            scale_ratio = 0.85;
        }
        double scale_offset = ((work.size().height - (1163 * scale_ratio)) / 2) - (64 * scale_ratio);
        //Log.i("ContoursOut", String.format("Offs %f", scale_offset));

        List<Integer> f_locs = Arrays.asList(85, 1163, 686, 1163, 686, 77, 82, 64, 82, 226, 244, 64);
        Scalar wt_color = new Scalar(255, 255, 255, 10);
        for(int i=0; i<6; i++) {
            int x = (int)((f_locs.get(i*2) - horiz_line) * scale_ratio + horiz_line - 10); //based on 730 width artwork
            int y = (int)(f_locs.get(i*2+1) * scale_ratio + scale_offset);

            Point pnt1 = new Point((y - 15) * ratio, (720 - x - 15) * ratio);
            Point pnt2 = new Point((y + 15) * ratio, (720 - x + 15) * ratio);
            Point pnt3 = new Point((y - 8) * ratio, (720 - x - 8) * ratio);
            Point pnt4 = new Point((y + 8) * ratio, (720 - x + 8) * ratio);
            Core.rectangle(mRgbaModified, pnt1, pnt2, wt_color, 2, 8, 0);
            Core.rectangle(mRgbaModified, pnt3, pnt4, wt_color, 2, 8, 0);
        }

        //ratio = (float)1.5;

        //Log.i("ContoursOut", "Sizes " + work.size().width + " height " + work.size().height + " ratio " + ratio);
        //Mat work_blur = new Mat();
        Mat edges = work.clone();
        Imgproc.blur(edges, edges, new Size(4, 4));

        //Mat edges = work.clone();
        Imgproc.Canny(edges, edges, 25 , 75, 3, true);

        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(edges, contours, hierarchy, Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE);

        if( contours.size() > 0 ) {
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

            List<DataPoint> order = new Vector<>();
            List<Point> outer = new Vector<>();
            List<Point> qr = new Vector<>();
            List<Float> diameter = new Vector<>();
            for( int i=0; i < Markers.size(); i++){
                //Imgproc.drawContours(mRgbaModified, contours, Markers.get(i), new Scalar(255, 200, 0), 2, 8, hierarchy, 0, new Point(0, 0));

                Moments mum = Imgproc.moments(contours.get(Markers.get(i)), false);
                Point mc = new Point( mum.get_m10()/mum.get_m00() , mum.get_m01()/mum.get_m00() );

                //calculate distance to nearest edge
                float dist = Math.min(Math.min(Math.min((float)mc.x, (float)(IMAGE_WIDTH - mc.x)), (float)mc.y), (float)(mRgbaModified.size().height - mc.y));

                Rect box = Imgproc.boundingRect(contours.get(Markers.get(i)));

                float dia = Math.max((float)box.width, (float)box.height) * 0.5f;
                float asprat = Math.max((float)box.width/(float)box.height, (float)box.height/(float)box.width);
                //only add it if sensible
                //Image is now 600 wide (x) and 337 high (y). Wax fiducials in y region 130-210 and z region 100-550.
                boolean nexcluded = !(mc.x > 300 && mc.x < 420);

                /*if(portrait) {
                    nexcluded = !(mc.x > 300 && mc.x < 420);
                }else{
                    nexcluded = !(mc.y > 300 && mc.y < 420);
                }*/

                //test valid point
                if(asprat < 1.5 && dia < 45 && dia > 5 && nexcluded){
                    //check of nearby points
                    boolean pnearby = false;

                    //loop over current points
                    for(int j=0; j<order.size(); j++){
                        float diff = (float)Math.sqrt((order.get(j).Center.x - mc.x) * (order.get(j).Center.x - mc.x) + (order.get(j).Center.y - mc.y) * (order.get(j).Center.y - mc.y));
                        if(diff < 10){
                            //Log.i("ContoursOut", "Dup Location " + mc.x + ", " + mc.y);
                            pnearby = true;
                            break;
                        }
                    }

                    //if unique point, add it.
                    if(!pnearby) {
                        //Log.i("ContoursOut", "Org Location " + mc.x + ", " + mc.y + " dia " + dia);
                        //save point
                        //DataPoint dat = [i, mc.x, mc.y, dia];
                        order.add(new DataPoint(i, dist, dia, mc));

                        boolean isQR = mc.y < 640 && mc.x < 320;

                        /*if(portrait){
                            isQR = mc.y < 640 && mc.x < 320;
                        }else{
                            isQR = mc.x < 640 && mc.y > 320;
                        }*/

                        //draw contour
                        //QR or outer?
                        Scalar color = new Scalar(0, 255, 0, 255);
                        //if (mc.y < 640 && mc.x < 320) {
                        if ( isQR ){ //mc.x < 640 && mc.y > 320) {
                            qr.add(new Point(mc.x, mc.y));
                            color = new Scalar(255, 0, 0, 255);
                        } else {
                            outer.add(new Point(mc.x, mc.y));
                        }

                        //draw COM circles if not used
                        //scale back to image size
                        Point comDisplay;
                        if(portrait) {
                            comDisplay = new Point(mc.x * ratio, mc.y * ratio);
                        }else{
                            comDisplay = new Point((mc.y) * ratio, (720-mc.x) * ratio);
                        }
                        Core.circle(mRgbaModified, comDisplay, 10, color, 2, 8, 0);

                    }
                    //Log.i("ContoursOut", "Location " + mc.x + ", " + mc.y);
                 }
            }

            //test if we have data
            if(outer.size() + qr.size() >= 5){
                //List<Point>
                order_points(src_points, outer, qr, ratio);

                return true;
            }

        }

        //return status
        return false;
    }

    //order outer and qr points
    private static boolean order_points(List<Point> src_points, List<Point> outer, List<Point> qr, float ratio) {
        //return data
        Log.i("ContoursOut", String.format("order points " + outer.size()));
        //sort outer~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if(outer.size() == 3) { //all points
            //transpoints = [[85, 1163], [686, 1163], [686, 77]];
            //max point, bottom RHS
            double odist_0_max = 0;
            int oindx = 0;
            for (int i = 0; i < 3; i++) {
                double otmp_dist_0 = outer.get(i).x * outer.get(i).x + outer.get(i).y * outer.get(i).y;
                if (otmp_dist_0 > odist_0_max) {
                    odist_0_max = otmp_dist_0;
                    oindx = i;
                }
            }

            //lowest x (LHS)
            double odist_x_min = 9999999;
            int oindxx = 0;
            for (int i = 0; i < 3; i++) {
                if (i == oindx) continue;

                if (outer.get(i).x < odist_x_min) {
                    odist_x_min = outer.get(i).x;
                    oindxx = i;
                }
            }

            //LHS outer fiducial
            //src_points.push(outer[oindxx]);
            src_points.add(new Point(outer.get(oindxx).y * ratio, (720 - outer.get(oindxx).x) * ratio));
            //src_points.add(new Point(outer.get(oindxx).x, outer.get(oindxx).y));

            //saved max fudicial
            //src_points.push(outer[oindx]);
            src_points.add(new Point(outer.get(oindx).y * ratio, (720 - outer.get(oindx).x) * ratio));
            //src_points.add(new Point(outer.get(oindx).x, outer.get(oindx).y));

            //remaining fiducial
            for (int i = 0; i < 3; i++) {
                if (i == oindx || i == oindxx) continue;
                //LHS QR fiducial
                //src_points.push(outer[i]);
                src_points.add(new Point(outer.get(i).y * ratio, (720 - outer.get(i).x) * ratio));
                //src_points.add(new Point(outer.get(i).x, outer.get(i).y));
            }
        }else{ //only 2 points
            //get angle
            double delta_x = outer.get(1).x - outer.get(0).x;
            double delta_y = outer.get(1).y - outer.get(0).y;
            double theta_radians = atan2(delta_y, delta_x);
            Log.i("ContoursOut", String.format("Angle %f", theta_radians));
            if(abs(theta_radians) < .26){ //<26'
                if(outer.get(1).x > outer.get(0).x){
                    src_points.add(new Point(outer.get(0).y * ratio, (720 - outer.get(0).x) * ratio));
                    src_points.add(new Point(outer.get(1).y * ratio, (720 - outer.get(1).x) * ratio));
                }else{
                    src_points.add(new Point(outer.get(1).y * ratio, (720 - outer.get(1).x) * ratio));
                    src_points.add(new Point(outer.get(0).y * ratio, (720 - outer.get(0).x) * ratio));
                }

                src_points.add(new Point(-1, -1));
            }else if(abs(theta_radians) > 1.3){ //>75'
                src_points.add(new Point(-1, -1));

                if(outer.get(1).y < outer.get(0).y){
                    src_points.add(new Point(outer.get(0).y * ratio, (720 - outer.get(0).x) * ratio));
                    src_points.add(new Point(outer.get(1).y * ratio, (720 - outer.get(1).x) * ratio));
                }else{
                    src_points.add(new Point(outer.get(1).y * ratio, (720 - outer.get(1).x) * ratio));
                    src_points.add(new Point(outer.get(0).y * ratio, (720 - outer.get(0).x) * ratio));
                }
            }else{ //else oblique
                if(outer.get(1).x > outer.get(0).x){
                    src_points.add(new Point(outer.get(0).y * ratio, (720 - outer.get(0).x) * ratio));
                    src_points.add(new Point(-1, -1));
                    src_points.add(new Point(outer.get(1).y * ratio, (720 - outer.get(1).x) * ratio));
                }else{
                    src_points.add(new Point(outer.get(1).y * ratio, (720 - outer.get(1).x) * ratio));
                    src_points.add(new Point(-1, -1));
                    src_points.add(new Point(outer.get(0).y * ratio, (720 - outer.get(0).x) * ratio));
                }
            }
        }

        //sort qr~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        //transqrpoints = [[82, 64], [82, 226], [244, 64]];
        if(qr.size() == 3) { //all points
            double dist_0_min = 9999999;
            int indx = 0;
            for (int i = 0; i < 3; i++) {
                double tmp_dist_0 = qr.get(i).x * qr.get(i).x + qr.get(i).y * qr.get(i).y;
                if (tmp_dist_0 < dist_0_min) {
                    dist_0_min = tmp_dist_0;
                    indx = i;
                }
            }


            //smallest x
            double dist_x_min = 9999999;
            int indxx = 0;
            for (int i = 0; i < 3; i++) {
                if (i == indx) continue;

                if (qr.get(i).x < dist_x_min) {
                    dist_x_min = qr.get(i).x;
                    indxx = i;
                }
            }


            //remaining fiducial
            for (int i = 0; i < 3; i++) {
                if (i == indx || i == indxx) continue;
                //LHS QR fiducial
                //src_points.push(qr[i]);
                src_points.add(new Point(qr.get(i).y * ratio, (720 - qr.get(i).x) * ratio));
                //src_points.add(new Point(qr.get(i).x, qr.get(i).y));
            }

            //min fiducial, top LHS
            //src_points.push(qr[indx]);
            src_points.add(new Point(qr.get(indx).y * ratio, (720 - qr.get(indx).x) * ratio));
            //src_points.add(new Point(qr.get(indx).x, qr.get(indx).y));

            //LHS QR fiducial
            //src_points.push(qr[indxx]);
            src_points.add(new Point(qr.get(indxx).y * ratio, (720 - qr.get(indxx).x) * ratio));
            //src_points.add(new Point(qr.get(indxx).x, qr.get(indxx).y));
        }else{ //only 2 ponts
            //get angle
            double delta_x = qr.get(1).x - qr.get(0).x;
            double delta_y = qr.get(1).y - qr.get(0).y;
            double theta_radians = atan2(delta_y, delta_x);
            Log.i("ContoursOut", String.format("Angle qr %f", theta_radians));
            if(abs(theta_radians) < .26){ //<26'
                if(qr.get(1).x > qr.get(0).x){
                    src_points.add(new Point(qr.get(1).y * ratio, (720 - qr.get(1).x) * ratio));
                    src_points.add(new Point(qr.get(0).y * ratio, (720 - qr.get(0).x) * ratio));
                }else{
                    src_points.add(new Point(qr.get(0).y * ratio, (720 - qr.get(0).x) * ratio));
                    src_points.add(new Point(qr.get(1).y * ratio, (720 - qr.get(1).x) * ratio));
                }

                src_points.add(new Point(-1, -1));
            }else if(abs(theta_radians) > 1.3){ //>75'
                src_points.add(new Point(-1, -1));

                if(qr.get(1).y < qr.get(0).y){
                    src_points.add(new Point(qr.get(1).y * ratio, (720 - qr.get(1).x) * ratio));
                    src_points.add(new Point(qr.get(0).y * ratio, (720 - qr.get(0).x) * ratio));
                }else{
                    src_points.add(new Point(qr.get(0).y * ratio, (720 - qr.get(0).x) * ratio));
                    src_points.add(new Point(qr.get(1).y * ratio, (720 - qr.get(1).x) * ratio));
                }
            }else{ //else oblique
                if(qr.get(1).x > qr.get(0).x){
                    src_points.add(new Point(qr.get(1).y * ratio, (720 - qr.get(1).x) * ratio));
                    src_points.add(new Point(-1, -1));
                    src_points.add(new Point(qr.get(0).y * ratio, (720 - qr.get(0).x) * ratio));
                }else{
                    src_points.add(new Point(qr.get(0).y * ratio, (720 - qr.get(0).x) * ratio));
                    src_points.add(new Point(-1, -1));
                    src_points.add(new Point(qr.get(1).y * ratio, (720 - qr.get(1).x) * ratio));
                }
            }
        }

        //return OK
        return true;
    }

    public static Mat TransformPoints(List<Point> Source, List<Point> Destination) {
        Point centroid_a = new Point(0, 0);
        Point centroid_b = new Point(0, 0);
        for (int i = 0; i < Source.size(); i++) {
            centroid_a.x += Source.get(i).x;
            centroid_a.y += Source.get(i).y;
            centroid_b.x += Destination.get(i).x;
            centroid_b.y += Destination.get(i).y;
        }
        centroid_a.x /= Source.size();
        centroid_a.y /= Source.size();
        centroid_b.x /= Source.size();
        centroid_b.y /= Source.size();

        // Remove Centroids
        List<Point> new_src = Source;
        List<Point> new_dst = Destination;
        for (int i = 0; i < Source.size(); i++) {
            new_src.set(i, new Point(new_src.get(i).x - centroid_a.x, new_src.get(i).y - centroid_a.y));
            new_dst.set(i, new Point(new_dst.get(i).x - centroid_b.y, new_dst.get(i).y - centroid_b.y));
        }

        // Get rotation
        Point v1 = new Point((new_src.get(0).x - new_src.get(1).x), (new_src.get(0).y - new_src.get(1).y));
        Point v2 = new Point((new_dst.get(0).x - new_dst.get(1).x), (new_dst.get(0).y - new_dst.get(1).y));
        double ang = atan2(v2.y, v2.x) - atan2(v1.y, v1.x);
        double cosang = Math.cos(ang);
        double sinang = Math.sin(ang);

        // Create rotation matrix
        Mat R = new Mat(2, 2, CvType.CV_32FC1);
        R.put(0, 0, cosang, -sinang);
        R.put(1, 0, sinang, cosang);

        // Calculate Scaling
        double sum_ss = 0;
        double sum_tt = 0;
        for (int i = 0; i < Source.size(); i++) {
            sum_ss += new_src.get(i).x * new_src.get(i).x;
            sum_ss += new_src.get(i).y * new_src.get(i).y;

            Mat pt = new Mat(1, 2, CvType.CV_32FC1);
            pt.put(0, 0, new_src.get(i).x, new_src.get(i).y);

            Mat res = new Mat();
            Core.gemm(R, pt.t(), 1, new Mat(), 0, res, 0);

            sum_tt += new_dst.get(i).x * res.get(0, 0)[0];
            sum_tt += new_dst.get(i).y * res.get(1, 0)[0];
        }

        // Scale Matrix
        Core.multiply(R, new Scalar(sum_tt / sum_ss), R);

        // Calculate Translation
        Mat C_A = new Mat(1, 2, CvType.CV_32FC1);
        C_A.put(0, 0, -centroid_a.x, -centroid_a.y);

        Mat C_B = new Mat(1, 2, CvType.CV_32FC1);
        C_B.put(0, 0, centroid_b.x, centroid_b.y);

        Mat CAR = new Mat();
        Core.gemm(R, C_A.t(), 1, new Mat(), 0, CAR, 0);

        Mat TL = new Mat();
        Core.add(C_B.t(), CAR, TL);

        // Combine Results
        Mat T = new Mat(2, 3, CvType.CV_32FC1);
        T.put(0, 0, R.get(0, 0)[0], R.get(0, 1)[0], TL.get(0, 0)[0]);
        T.put(1, 0, R.get(1, 0)[0], R.get(1, 1)[0], TL.get(1, 0)[0]);

        return T;
    }

    public static boolean RectifyImage(Mat input, Mat Template, Mat fringe_warped, List<Point> src_points, float dest_points[]){

        Log.i("ContoursOut", String.format("Pre get persp"));

        //create points
        float data[] = new float[8];
        float data2[] = new float[8];
        int j = 0;

        //copy data to structures
        for(int i=0; i<4; i++){
            if(j == 4) j++; //miss first fiducial
            while(src_points.get(j).x < 0){
                j++;
            }
            double datinx = src_points.get(j).x;
            double datiny = src_points.get(j).y;
            if(datinx > 0 && datiny > 0){
                data[2*i] = (float)datinx;
                data[2*i+1] = (float)datiny;
                data2[2*i] = dest_points[j*2];
                data2[2*i+1] = dest_points[j*2+1];
                j++;
            }
        }

        //set artwork points
        Mat destinationpoints = new Mat(4, 2, CvType.CV_32F);
        destinationpoints.put(0, 0, data2);

        Mat points = new Mat(4, 2,CvType.CV_32F);
        points.put(0, 0, data);

        //get check point, chose tp LHS QR if 3 QR, else one of QR
        double checkdata[] = new double[3];
        double checksdata[] = new double[2];

        int check_idx = 4;

        //check point 4 exists
        if(src_points.get(check_idx).x < 0){
            check_idx++;
        }
        //put in array
        checkdata[0] = src_points.get(check_idx).x;
        checkdata[1] = src_points.get(check_idx).y;
        checkdata[2] = 1.0;//,

        //set taerget point
        checksdata[0] = dest_points[check_idx * 2];
        checksdata[1] = dest_points[check_idx * 2 + 1];

        Mat checks = new Mat(3, 1,CvType.CV_64F);
        checks.put(0, 0, checkdata);

        //get transformation
        Mat TI = Imgproc.getPerspectiveTransform(points, destinationpoints);//TransformPoints(points, destinationpoints);
        Log.i("ContoursOut", String.format("TI %s, %s.",TI.toString(), checks.toString()));

        Mat work = new Mat();
        //Imgproc.resize(input, work, new Size(720, 1280), 0, 0, Imgproc.INTER_LINEAR );

        Imgproc.warpPerspective(input, work, TI, new Size(690 + 40, 1230 + 20));


        //checks
        Mat transformedChecks = new Mat(3, 1,CvType.CV_64F);
        Core.gemm(TI, checks, 1, new Mat(), 0, transformedChecks, 0);

        //get error norm
        double norm2 = 0;
        for(int i=0; i<2; i++){
            double xerror = transformedChecks.get(i,0)[0] / transformedChecks.get(2,0)[0] - checksdata[i];
            norm2 += xerror * xerror;
        }

        double norm = Math.sqrt(norm2);

        Log.i("ContoursOut", String.format("Points (%f, %f, %f) %f.",
                transformedChecks.get(0,0)[0], transformedChecks.get(1,0)[0],
                transformedChecks.get(2,0)[0], norm));

        //abort if transformation error
        if(norm > 15){
            return false;
        }

        //Mat work = new Mat();
        //Imgproc.resize(input, work, new Size(730, (input.size().height * 730) / input.size().width), 0, 0, Imgproc.INTER_LINEAR );
        //input.copyTo(work);

        Mat im_warped_nb = new Mat();
        Imgproc.cvtColor(work, im_warped_nb, Imgproc.COLOR_RGB2GRAY);

        File SDlocation = Environment.getExternalStorageDirectory();
        File padImageDirectory = new File(SDlocation + "/PAD/Test");
        padImageDirectory.mkdirs();

        Mat mTemp = new Mat();
        File outputFile = new File(padImageDirectory, "image.jpeg");
        Imgproc.cvtColor(im_warped_nb, mTemp, Imgproc.COLOR_GRAY2RGBA);
        Highgui.imwrite(outputFile.getPath(), mTemp);

        File outputFiles = new File(padImageDirectory, "template.jpeg");
        Imgproc.cvtColor(Template, mTemp, Imgproc.COLOR_GRAY2RGBA);
        Highgui.imwrite(outputFiles.getPath(), mTemp);



        Mat result = new Mat( );
        Imgproc.matchTemplate(im_warped_nb, Template, result, Imgproc.TM_CCOEFF_NORMED);

        List<Point> cellPoints = new ArrayList<>();

        Mat cellmask = Mat.ones(result.size(), CvType.CV_8UC1);
        double cellmaxVal = 1;
        double cellthr = 0.70;

        while( cellPoints.size() < 2 && cellmaxVal > cellthr) {
            Core.MinMaxLocResult mmResult = Core.minMaxLoc(result, cellmask);
            if( cellmaxVal <= cellthr ){
                break;
            }
            Log.d("Contour", String.format("Max cell point location %f, %f, %f", mmResult.maxLoc.x, mmResult.maxLoc.y, cellmaxVal));

            cellPoints.add(new Point(mmResult.maxLoc.x + Template.size().width / 2.0 - 0, mmResult.maxLoc.y + Template.size().height / 2.0 - 0));

            List<Point> rect = new ArrayList<>();
            rect.add( new Point(mmResult.maxLoc.x - Template.size().width / 2, mmResult.maxLoc.y - Template.size().height / 2));
            rect.add( new Point(mmResult.maxLoc.x + Template.size().width / 2, mmResult.maxLoc.y - Template.size().height / 2));
            rect.add( new Point(mmResult.maxLoc.x + Template.size().width / 2, mmResult.maxLoc.y + Template.size().height / 2));
            rect.add( new Point(mmResult.maxLoc.x - Template.size().width / 2, mmResult.maxLoc.y + Template.size().height / 2));

            List<MatOfPoint> poly = new ArrayList<>();
            MatOfPoint mp = new MatOfPoint();
            mp.fromList(rect);
            poly.add(mp);
            Core.fillPoly(cellmask, poly, new Scalar(0));
        }

        if( cellPoints.size() != 2) {
            Log.d("Contour", String.format("Error: Wax target not found with > 0.70 confidence."));
            // ERROR
            return false;
        }

        double dist1 = ((cellPoints.get(0).x - 387) * (cellPoints.get(0).x - 387)) + ((cellPoints.get(0).y - 214) * (cellPoints.get(0).y - 214));
        double dist2 = ((cellPoints.get(0).x - 387) * (cellPoints.get(0).x - 387)) + ((cellPoints.get(0).y - 1164) * (cellPoints.get(0).y - 1164));

        if( dist1 > dist2){
            Point temp = cellPoints.get(0);
            cellPoints.set(0, cellPoints.get(1));
            cellPoints.set(1, temp);
            //Log.d("Contour", String.format("Flipped %d, %d", cellPoints.get(0).x, cellPoints.get(0).y));
        }

        // do SVD for rotation/translation?
        List<Point> comparePoints = new ArrayList<>();
        comparePoints.add(new Point(387, 214));
        comparePoints.add(new Point(387, 1164));

        Log.d("Contour", String.format("Wax Points %s actual %s",cellPoints.toString(), comparePoints.toString()));

        Mat TICP = new Mat(3, 3, CvType.CV_32FC1);
        if( cellPoints.size() > 1 ) {
            Mat TCP = TransformPoints(cellPoints, comparePoints);

            // get full matrix
            Log.d("Contour", String.format("Mat %f %f",TCP.get(0,2)[0],TCP.get(1,2)[0]));

            TICP.put(0, 0, TCP.get(0,0)[0], TCP.get(0,1)[0], TCP.get(0,2)[0]);
            TICP.put(1, 0, TCP.get(1,0)[0], TCP.get(1,1)[0], TCP.get(1,2)[0]);
            TICP.put(2, 0, 0, 0, 1);
        }

        float[] tBuff = new float[9];
        TICP.get(0, 0, tBuff);
        Log.d("Contour", String.format("TICP %f %f %f \n %f %f %f \n %f %f %f", tBuff[0], tBuff[1], tBuff[2], tBuff[3], tBuff[4], tBuff[5], tBuff[6], tBuff[7], tBuff[8]));

        //Mat fringe_warped = new Mat();
        Imgproc.warpPerspective(work, fringe_warped, TICP, new Size(690 + 40, 1220), Imgproc.BORDER_REPLICATE);

        for( int i = 0; i < 13; i++ ) {
            double px = 706 - 53 * i;
            Core.line(fringe_warped, new Point(px, 339 + 20), new Point(px, 1095), new Scalar(0, 255, 0), 1);
        }

        Core.line(fringe_warped, new Point(comparePoints.get(0).x,comparePoints.get(0).y-5), new Point(comparePoints.get(0).x,comparePoints.get(0).y+5), new Scalar(0,255,0),1);
        Core.line(fringe_warped, new Point(comparePoints.get(0).x-5,comparePoints.get(0).y), new Point(comparePoints.get(0).x+5,comparePoints.get(0).y), new Scalar(0,255,0),1);
        Core.line(fringe_warped, new Point(comparePoints.get(1).x,comparePoints.get(1).y-5), new Point(comparePoints.get(1).x,comparePoints.get(1).y+5), new Scalar(0,255,0),1);
        Core.line(fringe_warped, new Point(comparePoints.get(1).x-5,comparePoints.get(1).y), new Point(comparePoints.get(1).x+5,comparePoints.get(1).y), new Scalar(0,255,0),1);

        //wax markers
        //Core.circle(fringe_warped, new Point(387,214), 10, new Scalar(0,0,255), 2, 6, 0);
        //Core.circle(fringe_warped, new Point(387,1164), 10, new Scalar(0,0,255), 2, 6, 0);

        return true;
    }

}
