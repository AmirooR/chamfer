#include "TemplateMatcher.hpp"
#include "precomp.hpp"
#include <opencv2/opencv.hpp>
#include <dirent.h>
#include <stdio.h>

using namespace std;
using namespace cv;


int main( int argc, const char** argv )
{
    string root("/home/amir/src/intrinsics/util/");
    string models_dir = root + string("model_contours/");
    string imgList = root + string("imgList.txt");
    string mpb_dir = root + string("mPb_contours_100/");
    /*Dir *dir;
    struct dirent *ent;
    if( (dir = opendir(img_dir.c_str())) != NULL )
    {

    }*/
    FILE *fd = fopen(imgList.c_str(), "r");
    for(size_t ff = 0; ff < 284; ff++)
    {
        
        char name[255];
        fscanf(fd,"%s",name);
        printf("[%d] - %s\n", ff, name);
        string templ = models_dir + string(name);
        string image = mpb_dir + string(name);
        Mat img = imread(image.c_str(), 0);
        Mat tpl = imread(templ.c_str(), 0);

        if (img.empty() || tpl.empty())
        {
            cout << "Could not read image file " << image << " or " << templ << "." << endl;
            return -1;
        }
        Mat cimg;
        cvtColor(img, cimg, CV_GRAY2BGR);

        // if the image and the template are not edge maps but normal grayscale images,
        // you might want to uncomment the lines below to produce the maps. You can also
        // run Sobel instead of Canny.

        // Canny(img, img, 5, 50, 3);
        // Canny(tpl, tpl, 5, 50, 3);

        vector<vector<Point> > results;
        vector<float> costs;
        float dt_truncate = 125.0f;
        int best = chamerMatching( img, tpl, results, costs, 1, 10, 5.0, 3,3, 15, 0.9, 2.0, 0.5, dt_truncate );
        if( best < 0 )
        {
            cout << "matching not found" << endl;
            continue;
            //return -1;
        }
        cout << "Results: "<<endl;
        Mat dt;
        Mat annotated_img;
        annotated_img.create(img.size(), CV_32SC2);
        dt.create(img.size(), CV_32FC1);
        dt.setTo(0);
        computeDistanceTransform2(img, dt, annotated_img, 30.0);
        //imshow("dt",dt);
        
        normalize( dt, dt, 0, 1., cv::NORM_MINMAX);
        //imshow("normalized",dt);
        //    imshow("annotated",annotated_img);

        configuration_t config;
        {
            config.do_fast_update = false;//true;
            config.normalize_scales = true;
            config.normalize_dt = true;
            config.lambda_dc = .02f;//0.02f;
            config.gamma = 2000.1f;//2000.0f;
            config.learning_rate = 0.0001f;//0.00012f;
            config.alpha = 0.0008f;//0.00005f;
            config.max_iter = 8423;
            config.dt_truncate = dt_truncate;
            config.dt_a = 1.0;
            config.dt_b = 1.5;
        }

        vector<Point> my_results;
        int sample_step = 10;
        if(true)
        {
            for(size_t i = 0; i < results[best].size(); i+=sample_step)
            {
                my_results.push_back( results[best][i] );
            }
        }
        else
        {
            my_results = results[best];
        }

        TemplateMatcher t_matcher(img, tpl, my_results, config);
        float m_loss = t_matcher.compute_loss();
        cout<< "loss: "<<m_loss<<endl;
        vector<Point2f> best_points = t_matcher.minimize();
        m_loss = t_matcher.compute_loss();
        cout<<"new loss: "<<m_loss<<endl;

        /*for(size_t j = 0; j < results[best].size(); j+= 30)
        {
            Mat dimg = cimg.clone();
            Point pt = results[best][j];
            //dimg.at<Vec3b>(pt) = Vec3b(0,0,255);
            circle(dimg, pt, 3,Scalar(0,0,255));
            //imshow("points",dimg);
           // waitKey(0);
        }*/


        //for(size_t j = 0; j < results.size(); j++)
        for(size_t j = 0; j < 1; j++)
        {
            Mat dimg = cimg.clone();
            size_t i, n = results[j].size();
            for( i = 0; i < n; i++ )
            {
                Point pt = results[j][i];
                if( pt.inside(Rect(0, 0, cimg.cols, cimg.rows)) )
                    if( j != best )
                        dimg.at<Vec3b>(pt) = Vec3b(0, 255, 0);
                    else
                        dimg.at<Vec3b>(pt) = Vec3b(0, 0, 255);

            }
            vector<Point2f> obj_result = best_points;
            n = obj_result.size();
            for( i = 0; i < n; i++)
            {
                Point pt = obj_result[i];
                Point pt2 = obj_result[(i+1)%n];
                if( pt.inside(Rect(0, 0, cimg.cols, cimg.rows)) && pt2.inside(Rect(0, 0, cimg.cols, cimg.rows)) )
                {
                    line(dimg, pt, pt2, Scalar(255,0,0), 2); 
                    line(dimg, pt, my_results[i], Scalar(0,255,255),1);
                }
                //dimg.at<Vec3b>(pt) = Vec3b(255,0,0);
            }
            //imshow("result", dimg);
            string w_name("result_");
            w_name = w_name + name;
            imwrite(w_name.c_str(), dimg);

            //waitKey(0);

        }

    }
    fclose(fd);
    return 0;
}
