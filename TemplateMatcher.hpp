#pragma once

#include <opencv2/opencv.hpp>
#include <iostream>
#include <queue>

using namespace cv;
using namespace std;

#define PIPI4 M_PI * M_PI * 4
#define BIG_LOSS 9999999.9

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
	//TODO: assuming that contour points are ordered in a chain
	
    vector<Point2f> initialPoints;
    vector<Point2f> currentPoints;
	vector<float> lambda; // scale factor for d_2 in loss function
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
    float compute_loss();
    void minimize_single_step();
    void minimize();
	inline void apply_transform(Point2f& point, param_t& param, Point2f& new_point)
	{
		new_point = point * param.s;
		new_point.x += cos(param.phi)*point.x - sin(param.phi)*point.y;
		new_point.y += cos(param.phi)*point.y + sin(param.phi)*point.x;
	}
	
	inline Point2f imageToLocal(Point2f& image_point)
	{
		Point2f p;
		p.x = image_point.x - origin.x;
		p.y = origin.y - image_point.y;
		return p;
	}
	
	inline Point2f localToImage(Point2f& local_point)
	{
		Point2f p;
		p.x = local_point.x + origin.x;
		p.y = origin.y - local_point.y;
		return p;
	}
	
	virtual float param_distance(param_t& p1, param_t& p2, int id1, int id2)
	{
		float scale_id1 = 1.0f, scale_id2 = 1.0f;
		if(config.normalize_scales)
		{
			scale_id1 = 1.0f/scale_norms[id1];
			scale_id2 = 1.0f/scale_norms[id2];
		}	

		return (p1.phi - p2.phi)*(p1.phi - p2.phi)/PIPI4 + (p1.s/scale_id1 - p2.s/scale_id2)*(p1.s/scale_id1 - p2.s/scale_id2);
	}
    vector<param_t> getParams(){return params;}
    vector<Point2f> getCurrentPoints(){return currentPoints;}
    void initialize_params(){/*TODO: implement me */};
};
