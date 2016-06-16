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


        Log.d("Contour", String.format("%f %f\n%f %f", centroid_a.x, centroid_a.y, centroid_b.x, centroid_b.y));

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
        double ang = Math.atan2(v2.y, v2.x) - Math.atan2(v1.y, v1.x);
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

    public static Mat RectifyImage(Mat input, List<Point3> Markers){
        /*List<Point> QR = new ArrayList<>(), Other = new ArrayList<>();
        SeperateMarkers(Markers, QR, Other);

        List<Point> src_points = new ArrayList<>(), dst_points = new ArrayList<>();
        List<Point> src_tests = new ArrayList<>(), dst_tests = new ArrayList<>();

        if( QR.size() >= 2 && Other.size() >= 2 ){
            SortQR(QR, Other);

            int pcount = 0;

            List<Point> transpoints = new ArrayList<>();
            transpoints.add(new Point(85, 1163));
            transpoints.add(new Point(686, 1163));
            transpoints.add(new Point(686, 77));

            for( int i = 0; i < 3; i++ ) {
                if( Other.get(i).x >= 0 ){
                    src_points.add(Other.get(i));
                    dst_points.add(transpoints.get(i));
                    pcount += 1;
                }
            }

            List<Point> transqrpoints = new ArrayList<>();
            transqrpoints.add(new Point(82, 64));
            transqrpoints.add(new Point(82, 226));
            transqrpoints.add(new Point(244, 64));

            if( QR.size() == 3 ){
                if( Other.get(0).x >= 0) {
                    QR.get(1).x = -1;
                }
            }

            for( int i = 0; i < 3; i++ ) {
                if( QR.get(i).x >= 0 ){
                    if( pcount < 4){
                        src_points.add(QR.get(i));
                        dst_points.add(transqrpoints.get(i));
                        pcount += 1;
                    }else{
                        src_tests.add(QR.get(i));
                        dst_tests.add(transqrpoints.get(i));
                    }
                }
            }
        }

        if( src_points.size() < 4){
            // ERROR
        }


        Mat src_mat=new Mat(4,1, CvType.CV_32FC2);
        double buff[] = new double[(int)src_mat.total() * src_mat.channels()];
        src_mat.get(0, 0, buff);
        for( int i = 0; i < src_points.size(); i++ ){
            buff[i*2+0] = src_points.get(i).x;
            buff[i*2+1] = src_points.get(i).y;
        }
        src_mat.put(0, 0, buff);

        Mat dst_mat=new Mat(4,1, CvType.CV_32FC2);
        dst_mat.get(0, 0, buff);
        for( int i = 0; i < dst_points.size(); i++ ){
            buff[i*2+0] = dst_points.get(i).x;
            buff[i*2+1] = dst_points.get(i).y;
        }
        dst_mat.put(0, 0, buff);

        Mat TI = Imgproc.getPerspectiveTransform(src_mat, dst_mat);

        double maxerror = 0;
        for( int i = 0; i < src_tests.size(); i++ ) {
            Mat point = new Mat(1, 4, CvType.CV_32FC2);
            point.put(0, 0, src_tests.get(0).x, src_tests.get(0).y, 1.0);

            Mat result = TI.mul(point.t());
            result.put(0, 0, result.get(0,0)[0] / result.get(0,2)[0]);
            result.put(0, 1, result.get(0,1)[0] / result.get(0,2)[0]);
            result.put(0, 2, 1.0);

            double eX = (result.get(0,0)[0] - dst_tests.get(i).x) * (result.get(0,0)[0] - dst_tests.get(i).x);
            double eY = (result.get(0,1)[0] - dst_tests.get(i).y) * (result.get(0,1)[0] - dst_tests.get(i).y);

            double error = Math.sqrt(eX + eY);
            if( error > maxerror ) {
                maxerror = error;
            }
        }

        Log.d("Contour", String.format("Transformation maximum error: %d", maxerror));
        if( maxerror > 15.0 ){
            // ERROR
        }

        Mat im_warped = new Mat();
        Imgproc.warpPerspective(input, im_warped, TI, new Size(690 + 40, 1230 + 20), Imgproc.BORDER_REPLICATE);

        Mat tIn = Highgui.imread(templatefile, Imgproc.CV_LOAD_IMAGE_GRAYSCALE);

        Mat template = new Mat();
        tIn.convertTo(template, CvType.CV_32FC1, 255, 0);

        Mat m = new Mat( new Size(1,3), CvType.CV_32FC1 );
        double cBuff[] = new double[(int)m.total() * m.channels()];
        m.get(0, 0, cBuff);
        buff[0] = 0.163;
        buff[1] = 0.837;
        buff[2] = 1.0;
        m.put(0, 0, cBuff);

        Mat im_warped_nb = new Mat();
        Core.transform(im_warped, im_warped_nb, m);

        Mat fgim_warped_nb = new Mat();
        im_warped_nb.convertTo(fgim_warped_nb, CvType.CV_32FC1);

        Mat result = new Mat();
        Imgproc.matchTemplate(fgim_warped_nb, result, template, Imgproc.CV_TM_CCOEFF_NORMED);

        List<Point> cellPoints = new ArrayList<>();

        Mat cellmask = Mat.ones(result.size(), CvType.CV_8UC1);
        double cellmaxVal = 1;
        double cellthr = 0.70;

        while( cellPoints.size() < 2 && cellmaxVal > cellthr) {
            Core.MinMaxLocResult mmResult = Core.minMaxLoc(result, cellmask);
            if( cellmaxVal <= cellthr ){
                break;
            }
            Log.d("Contour", String.format("Max cell point location %d, %d, %d", mmResult.maxLoc.x, mmResult.maxLoc.y, cellmaxVal));

            cellPoints.add(new Point(mmResult.maxLoc.x + template.size().width / 2.0 - 0, mmResult.maxLoc.y + template.size().height / 2.0 - 0));

            List<Point> rect = new ArrayList<>();
            rect.add( new Point(mmResult.maxLoc.x - template.size().width / 2, mmResult.maxLoc.y - template.size().height / 2));
            rect.add( new Point(mmResult.maxLoc.x + template.size().width / 2, mmResult.maxLoc.y - template.size().height / 2));
            rect.add( new Point(mmResult.maxLoc.x + template.size().width / 2, mmResult.maxLoc.y + template.size().height / 2));
            rect.add( new Point(mmResult.maxLoc.x - template.size().width / 2, mmResult.maxLoc.y + template.size().height / 2));

            List<MatOfPoint> poly = new ArrayList<>();
            MatOfPoint mp = new MatOfPoint();
            mp.fromList(rect);
            poly.add(mp);
            Core.fillPoly(cellmask, poly, new Scalar(0));
        }

        if( cellPoints.size() != 2) {
            Log.d("Contour", String.format("Error: Wax target not found with > 0.70 confidence."));
            // ERROR
        }

        double dist1 = ((cellPoints.get(0).x - 387) * (cellPoints.get(0).x - 387)) + ((cellPoints.get(0).y - 214) * (cellPoints.get(0).y - 214));
        double dist2 = ((cellPoints.get(0).x - 387) * (cellPoints.get(0).x - 387)) + ((cellPoints.get(0).y - 1164) * (cellPoints.get(0).y - 1164));

        if( dist1 > dist2){
            Point temp = cellPoints.get(0);
            cellPoints.set(0, cellPoints.get(1);
            cellPoints.set(1, temp);
            Log.d("Contour", String.format("Flipped %d, %d", cellPoints.get(0).x, cellPoints.get(0).y));
        }

        int k = 0;
        for( int i = 0; i < cellPoints.size(); i++ ) {
            Core.circle(im_warped, new Point((int)cellPoints.get(i).x, (int)cellPoints.get(i).y), 17, new Scalar(255, 255, 255, 255), 2);
            Core.putText(im_warped, String.format("%d",k), new Point((int)cellPoints.get(i).x, (int)cellPoints.get(i).y), Core.FONT_HERSHEY_PLAIN, 2.0, new Scalar(255, 255, 255))
            k += 1;
        }

        // do SVD for rotation/translation?
        int artwork = -1;
        comparePoints = [[[387, 214], [387, 1164]], [[387-17, 214], [387, 1164]], [[387+5, 214], [387-11, 1164]]]
        if( cellPoints.size() > 1 ) {
            TCPA = [];
            for( int i = 0; i < comparePoints.size(); i++ ) {
                TCPA.append(RotTrans2Points(cellPoints, comparePoints[i]));
            }

            double minangle = Double.MAX_VALUE;
            for( int i = 0; i < comparePoints.size(); i++ ) {
                Log.d("Contour", String.format("data %f",TCPA[i].A[0][1]));
                double iangl = Math.abs(Math.asin(Math.min(TCPA[i].A[0][1],1.0));
                if( iangl < minangle ){
                    minangle = iangl;
                    artwork = i;
                }
            }

            Log.d("Contour", String.format("Minimum angle was at index %d", artwork );
            TCP = TCPA[artwork];

            // get full matrix
            Log.d("Contour", String.format("Mat",TCP.A[0][2],TCP.A[1][2]
            Mat TICP = np.matrix([
                [TCP.A[0][0], TCP.A[0][1], TCP.A[0][2]],
                [TCP.A[1][0], TCP.A[1][1], TCP.A[1][2]],
                [0, 0, 1]
            ]);
        }

        // put transformed markers on image
        Mat pnt1 = new Mat();
        pnt1.put(0, 0, cellPoints.get(0).x, cellPoints.get(0).y, 1.0);
        Mat trans1 = TICP.mul(pnt1.t());

        Mat pnt2 = new Mat();
        pnt1.put(0, 0, cellPoints.get(1).x, cellPoints.get(1).y, 1.0);
        Mat trans2 = TICP.mul(pnt2.t());

        Core.line(im_warped, new Point((int)(trans1.get(0,0)[0])+10, (int)(trans1.get(0,1)[0])), new Point((int)(trans1.get(0,0)[0])-10, (int)(trans1.get(0,1)[0])), new Scalar(255,0,0), 2);
        Core.line(im_warped, new Point((int)(trans1.get(0,0)[0]), (int)(trans1.get(0,1)[0])+10), new Point((int)(trans1.get(0,0)[0]), (int)(trans1.get(0,1)[0])-10), new Scalar(255,0,0), 2);
        Core.line(im_warped, new Point((int)(trans2.get(0,0)[0])+10, (int)(trans2.get(0,1)[0])), new Point((int)(trans2.get(0,0)[0])-10, (int)(trans2.get(0,1)[0])), new Scalar(255,0,0), 2);
        Core.line(im_warped, new Point((int)(trans2.get(0,0)[0]), (int)(trans2.get(0,1)[0])+10), new Point((int)(trans2.get(0,0)[0]), (int)(trans2.get(0,1)[0])-10), new Scalar(255,0,0), 2);

        for(int i = 0; i < 13; i++ ) {
            double px = 706 - 53 * i;

            pnt1 = new Mat();
            pnt1.put(0, 0, px, 339, 1.0);
            trans1 = TICP.mul(pnt1.t());

            pnt2 = new Mat();
            pnt2.put(0, 0, px, 1095, 1.0);
            trans2 = TICP.mul(pnt2.t());

            Core.line(im_warped, new Point((int)(trans1.get(0,0)[0]),(int)(trans1.get(0,1)[0])),new Point((int)(trans2.get(0,0)[0]),(int)(trans2.get(0,1)[0])),new Scalar(0, 0, 255), 2);
            Core.line(im_warped, new Point(px, 339), new Point(px, 1095), new Scalar(0, 255, 0), 2);
        }

        // actual transformed fringes
        Mat TALL = TICP.mul(TI);

        Mat fringe_warped = new Mat();
        Imgproc.warpPerspective(input, fringe_warped, TALL, new Size(690 + 40, 1220), Imgproc.BORDER_REPLICATE);

        for( int i = 0; i < 13; i++ ) {
            double px = 706 - 53 * i;
            Core.line(fringe_warped, new Point(px, 339 + 20), new Point(px, 1095), new Scalar(0, 255, 0), 1);
        }

        targetloc = comparePoints[artwork];
        Core.line(fringe_warped, new Point(targetloc[0][0],targetloc[0][1]-5), new Point(targetloc[0][0],targetloc[0][1]+5), new Scalar(0,255,0),1);
        Core.line(fringe_warped, new Point(targetloc[0][0]-5,targetloc[0][1]), new Point(targetloc[0][0]+5,targetloc[0][1]), new Scalar(0,255,0),1);
        Core.line(fringe_warped, new Point(targetloc[1][0],targetloc[1][1]-5), new Point(targetloc[1][0],targetloc[1][1]+5), new Scalar(0,255,0),1);
        Core.line(fringe_warped, new Point(targetloc[1][0]-5,targetloc[1][1]), new Point(targetloc[1][0]+5,targetloc[1][1]), new Scalar(0,255,0),1);

        return fringe_warped;*/
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

    public static void SortQR(List<Point> QR, List<Point> Other) {
        if( QR.size() == 3 ) {
            Point qr_top_left = new Point(9999, 9999);
            Point qr_top_right = new Point(0, 0);
            Point qr_bot_left = new Point(0, 0);

            for( Point point : QR ) {
                if( point.x > qr_top_right.x ){
                    qr_top_right = point;
                }

                if( point.y > qr_bot_left.y ){
                    qr_bot_left = point;
                }
            }

            for( Point point : QR ) {
                if( point != qr_top_right && point != qr_bot_left ) {
                    qr_top_left = point;
                }
            }
            QR.clear();
            QR.add(qr_top_left); QR.add(qr_bot_left); QR.add(qr_top_right);
        } else {
            double dist = Math.sqrt((QR.get(0).x - QR.get(1).x) * (QR.get(0).x - QR.get(1).x) + (QR.get(0).y - QR.get(1).y) * (QR.get(0).y - QR.get(1).y)) / 3.0;
            if( Math.abs(QR.get(0).x - QR.get(1).x) < dist ){
                if( QR.get(0).y < QR.get(1).y ){
                    List<Point> temp = new ArrayList<>();
                    temp.add(QR.get(0)); temp.add(QR.get(1)); temp.add(new Point( -1, -1 ));
                    QR.clear(); QR.addAll(temp);
                }else{
                    List<Point> temp = new ArrayList<>();
                    temp.add(QR.get(1)); temp.add(QR.get(0)); temp.add(new Point( -1, -1 ));
                    QR.clear(); QR.addAll(temp);
                }
            }else{
                if( Math.abs(QR.get(0).y - QR.get(1).y) < dist ){
                    if( QR.get(0).x < QR.get(1).x ){
                        List<Point> temp = new ArrayList<>();
                        temp.add(QR.get(0)); temp.add(new Point( -1, -1 )); temp.add(QR.get(1));
                        QR.clear(); QR.addAll(temp);
                    }else{
                        List<Point> temp = new ArrayList<>();
                        temp.add(QR.get(1)); temp.add(new Point( -1, -1 )); temp.add(QR.get(0));
                        QR.clear(); QR.addAll(temp);
                    }
                }else{
                    if( QR.get(0).x < QR.get(1).x ){
                        List<Point> temp = new ArrayList<>();
                        temp.add(new Point( -1, -1 )); temp.add(QR.get(0)); temp.add(QR.get(1));
                        QR.clear(); QR.addAll(temp);
                    }else{
                        List<Point> temp = new ArrayList<>();
                        temp.add(new Point( -1, -1 )); temp.add(QR.get(1)); temp.add(QR.get(0));
                        QR.clear(); QR.addAll(temp);
                    }
                }
            }
        }

        if( Other.size() == 3 ) {
            Point top_right = new Point(0, 9999);
            Point bottom_right = new Point(0, 0);
            Point bottom_left = new Point(9999, 0);

            for( Point point : Other ) {
                if( point.x > top_right.x && point.y < top_right.y ){
                    top_right = point;
                }

                if( point.x < bottom_left.x && point.y > bottom_left.y ){
                    bottom_left = point;
                }
            }

            for( Point point : Other ) {
                if( point != top_right && point != bottom_left ) {
                    bottom_right = point;
                }
            }
            Other = new ArrayList<>();
            Other.add(bottom_left); Other.add(bottom_right); Other.add(top_right);
        }else{
            double dist = Math.sqrt((Other.get(0).x - Other.get(1).x) * (Other.get(0).x - Other.get(1).x) + (Other.get(0).y - Other.get(1).y) * (Other.get(0).y - Other.get(1).y)) / 3.0;

            if( Math.abs(Other.get(0).x - Other.get(1).x) < dist ){
                if( Other.get(0).y < Other.get(1).y ){
                    List<Point> temp = new ArrayList<>();
                    temp.add(new Point( -1, -1 )); temp.add(Other.get(1)); temp.add(Other.get(0));
                    Other.clear(); Other.addAll(temp);
                }else{
                    List<Point> temp = new ArrayList<>();
                    temp.add(new Point( -1, -1 )); temp.add(Other.get(0)); temp.add(Other.get(1));
                    Other.clear(); Other.addAll(temp);
                }
            }else{
                if( Math.abs(Other.get(0).y - Other.get(1).y) < dist ){
                    if(Other.get(0).x < Other.get(1).x ){
                        List<Point> temp = new ArrayList<>();
                        temp.add(Other.get(0)); temp.add(Other.get(1)); temp.add(new Point( -1, -1 ));
                        Other.clear(); Other.addAll(temp);
                    }else{
                        List<Point> temp = new ArrayList<>();
                        temp.add(Other.get(1)); temp.add(Other.get(0)); temp.add(new Point( -1, -1 ));
                        Other.clear(); Other.addAll(temp);
                    }
                }else{
                    if(Other.get(0).x < Other.get(1).x ){
                        List<Point> temp = new ArrayList<>();
                        temp.add(Other.get(0)); temp.add(new Point( -1, -1 )); temp.add(Other.get(1));
                        Other.clear(); Other.addAll(temp);
                    }else{
                        List<Point> temp = new ArrayList<>();
                        temp.add(Other.get(1)); temp.add(new Point( -1, -1 )); temp.add(Other.get(0));
                        Other.clear(); Other.addAll(temp);
                    }
                }
            }
        }
    }
}
