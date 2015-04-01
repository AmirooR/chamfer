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

void computeDistanceTransformA(Mat& edges_img, Mat& dist_img, Mat& annotate_img, float truncate_dt, float a = 1.0, float b = 1.5);



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
