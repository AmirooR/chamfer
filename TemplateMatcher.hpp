#pragma once

#include <opencv2/opencv.hpp>
#include <iostream>
#include <queue>

using namespace cv;
using namespace std;


typedef struct param_t {
    float phi; // rotation angle
    float s; // scale
	param_t(const param_t& p)
	{
		phi = p.phi;
		s = p.s;
	}
	param_t()
	{
		phi = 0.0f;
		s = 1.0f;
	}
	
	param_t(float _phi, float _s):phi(_phi), s(_s){}
}param_t;

typedef struct configuration_t {
    bool do_fast_update;
    bool normalize_scales;
	float gamma; // scale factor for scales in d_2
	float learning_rate; // learning rate
	int max_iter; // max. number of iterations
	float dt_truncate; // DT truncation
	float dt_a;
	float dt_b;
}configuration_t;

void computeDistanceTransform2(Mat& edges_img, Mat& dist_img, Mat& annotate_img, float truncate_dt, float a = 1.0, float b = 1.5);
inline float euclidean_dist(Point2f& p1, Point2f& p2);


class TemplateMatcher
{
    vector<Point2f> initialPoints;
    vector<Point2f> currentPoints;
	Mat lambda; // scale factor for d_2 in loss function
    Mat DT; // distance_transform image
    Mat grad_x_DT; // gradient of DT in x direction
    Mat grad_y_DT; // gradient of DT in y direction    
    Mat edgeImg; // target edge image
    Mat template_contour; // template contour image
    vector<param_t> params; // parameters
    vector<param_t> current_gradients; //gradient of params
    vector<float> scale_norms; // scaling of each params.s
	Mat annotated_img; // correspondances for DT
    configuration_t config;
	Point2f origin;
    

    public:
    TemplateMatcher(Mat& edgeImg, Mat& template_contour, vector<Point>& initialPoints, configuration_t config);
    float computeLoss();
    void minimize_single_step();
    void minimize();
    vector<param_t> getParams(){return params;}
    vector<Point2f> getCurrentPoints(){return currentPoints;}
    void initialize_params(){/*TODO: implement me */};
};
