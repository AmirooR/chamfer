#include "TemplateMatcher.hpp"

inline float euclidean_dist2f(Point2f& p1, Point2f& p2)
{
	Point2f c = p1 - p2;
	return sqrt( (p1.x - p2.x)*(p1.x - p2.x) + (p1.y - p2.y)*(p1.y - p2.y) );
}

void computeDistanceTransform2(Mat& edges_img, Mat& dist_img, Mat& annotate_img, float truncate_dt, float a, float b)
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


TemplateMatcher::TemplateMatcher(Mat& _edgeImg, Mat& _template_contour, vector<Point>& _initialPoints, configuration_t _config):
    config(_config), origin(0,0)
{
    edgeImg = _edgeImg.clone();
	template_contour = _template_contour.clone();
	
    initialPoints.resize( _initialPoints.size() );
	currentPoints.resize( _initialPoints.size() );
    for(int i = 0; i < _initialPoints.size(); ++i)
    {
        initialPoints[i] = _initialPoints[i];
		currentPoints[i] = _initialPoints[i];
    }
	
	annotated_img.create(edgeImg.size(), CV_32SC2);
	DT.create(edgeImg.size(), CV_32FC1);
	DT.setTo(0);	
	computeDistanceTransform2(edgeImg, DT, annotated_img, config.dt_truncate, config.dt_a, config.dt_b);
	
	lambda.create(edgeImg.size(), CV_32FC1);
	lambda.setTo(1.0f); //TODO: how to calc. lambda
	
	//computing gradients
	Scharr( DT, grad_x_DT, CV_32FC1, 1, 0, 1, 0, BORDER_DEFAULT );
	Scharr( DT, grad_y_DT, CV_32FC1, 0, 1, -1, 0, BORDER_DEFAULT ); //TODO: check for scale == -1
	
	params.resize( _initialPoints.size(), param_t());
	initialize_params();
	current_gradients.resize( _initialPoints.size(), param_t(0,0));
	
	for(size_t i = 0; i < _initialPoints.size(); ++i)
	{
		origin.x += _initialPoints[i].x;
		origin.y += _initialPoints[i].y;
	}
	origin.x = origin.x/_initialPoints.size();
	origin.y = origin.y/_initialPoints.size();
	
	if(config.normalize_scales)
	{
		scale_norms.resize( _initialPoints.size(),1.0f);
		for(size_t i = 0; i < initialPoints.size(); ++i)
		{
			scale_norms[i] = euclidean_dist2f(origin, initialPoints[i]);
		}
	}
	
}
