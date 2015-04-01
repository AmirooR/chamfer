#pragma once

#include <opencv2/opencv.hpp>
#include <iostream>
#include <queue>

using namespace cv;
using namespace std;


typedef struct {
    float phi; // rotation angle
    float s; // scale
}param_t;

typedef struct{
    bool do_fast_update;
    bool normalize_scales;
}configuration_t;


void computeDistanceTransform(Mat& edges_img, Mat& dist_img, Mat& annotate_img, float truncate_dt, float a = 1.0, float b = 1.5)
{
    int d[][2] = { {-1,-1}, { 0,-1}, { 1,-1},
            {-1,0},          { 1,0},
            {-1,1}, { 0,1}, { 1,1} };


    Size s = edges_img.size();
    int w = s.width;
    int h = s.height;
    // set distance to the edge pixels to 0 and put them in the queue
    std::queue<std::pair<int,int> > q;

    for (int y=0;y<h;++y) {
        for (int x=0;x<w;++x) {
            // initialize
            if (&annotate_img!=NULL) {
                annotate_img.at<Vec2i>(y,x)[0]=x;
                annotate_img.at<Vec2i>(y,x)[1]=y;
            }

            uchar edge_val = edges_img.at<uchar>(y,x);
            if( (edge_val!=0) ) {
                q.push(std::make_pair(x,y));
                dist_img.at<float>(y,x)= 0;
            }
            else {
                dist_img.at<float>(y,x)=-1;
            }
        }
    }

    // breadth first computation of distance transform
    std::pair<int,int> crt;
    while (!q.empty()) {
        crt = q.front();
        q.pop();

        int x = crt.first;
        int y = crt.second;

        float dist_orig = dist_img.at<float>(y,x);
        float dist;

        for (size_t i=0;i<sizeof(d)/sizeof(d[0]);++i) {
            int nx = x + d[i][0];
            int ny = y + d[i][1];

            if (nx<0 || ny<0 || nx>=w || ny>=h) continue;

            if (std::abs(d[i][0]+d[i][1])==1) {
                dist = (dist_orig)+a;
            }
            else {
                dist = (dist_orig)+b;
            }

            float dt = dist_img.at<float>(ny,nx);

            if (dt==-1 || dt>dist) {
                dist_img.at<float>(ny,nx) = dist;
                q.push(std::make_pair(nx,ny));

                if (&annotate_img!=NULL) {
                    annotate_img.at<Vec2i>(ny,nx)[0]=annotate_img.at<Vec2i>(y,x)[0];
                    annotate_img.at<Vec2i>(ny,nx)[1]=annotate_img.at<Vec2i>(y,x)[1];
                }
            }
        }
    }
    // truncate dt

    if (truncate_dt>0) {
        Mat dist_img_thr = dist_img.clone();
        threshold(dist_img, dist_img_thr, truncate_dt,0.0 ,THRESH_TRUNC);
        dist_img_thr.copyTo(dist_img);
    }
}



class TemplateMatcher
{
    vector<Point> initialPoints;
    float gamma; // scale factor for scales in d_2
    Mat lambda; // scale factor for d_2 in loss function
    Mat DT; // distance_transform image
    Mat grad_DT_x; // gradient of DT in x direction
    Mat grad_DT_y; // gradient of DT in y direction
    vector<Point> currentPoints;
    Mat edgeImg; // target edge image
    Mat template_contour; // template contour image
    float learning_rate; // learning rate
    vector<param_t> params; // parameters
    vector<param_t> current_gradients; //gradient of params
    vector<float> scale_norms; // scaling of each params.s
    configuration_t config;
    int max_iter;

    public:
    TemplateMatcher(Mat& edgeImg, vector<Point>& initialPoints, float gamma, float learning_rate, int max_iter);
    float computeLoss();
    void minimize_single_step();
    void minimize();
    vector<param_t> getParams(){return params;}
    vector<Point> getCurrentPoints(){return currentPoints;}
    void initialize_params();
};


