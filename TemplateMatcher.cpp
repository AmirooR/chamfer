#include "TemplateMatcher.hpp"

inline float euclidean_dist2f(Point2f& p1, Point2f& p2)
{
	Point2f c = p1 - p2;
	return sqrt( c.x*c.x + c.y*c.y );
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
    cout<< "Number of points: "<<_initialPoints.size() <<endl;
	
    initialPoints.resize( _initialPoints.size() );
	currentPoints.resize( _initialPoints.size() );
    for(size_t i = 0; i < _initialPoints.size(); ++i)
    {
        initialPoints[i] = _initialPoints[i];
		currentPoints[i] = _initialPoints[i];
    }
	
	annotated_img.create(edgeImg.size(), CV_32SC2);
	DT.create(edgeImg.size(), CV_32FC1);
	DT.setTo(0);	
	computeDistanceTransform2(edgeImg, DT, annotated_img, config.dt_truncate, config.dt_a, config.dt_b);
	
	//lambda.create(edgeImg.size(), CV_32FC1);
	//lambda.setTo(1.0f); //TODO: how to calc. lambda
	lambda.resize(_initialPoints.size(), 1.0);
	
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

    cout<< "Origin: "<<origin<<endl;
	
	if(config.normalize_scales)
	{
        
		scale_norms.resize( _initialPoints.size(),1.0f);
		for(size_t i = 0; i < initialPoints.size(); ++i)
		{
			scale_norms[i] = euclidean_dist2f(origin, initialPoints[i]);
		}
        cout<<endl;
	}
	
}

float TemplateMatcher::compute_loss()
{
	//TODO: asserts
	float loss = 0.0f;
	for(size_t i = 0; i < currentPoints.size(); ++i)
	{
		//TODO: can make it faster
		Point2f _newPoint;
		Point2f _newPoint_image;
		Point2f current_local = imageToLocal(currentPoints[i]);
		apply_transform( current_local, params[i], _newPoint);
		_newPoint_image = localToImage(_newPoint);
		
		if( (_newPoint_image.x >= 1) && (_newPoint_image.y >= 1) && (_newPoint_image.x < (edgeImg.cols-1)) && (_newPoint_image.y < (edgeImg.rows-1)))
		{
			/* data term */
			loss += bilinear_interpolate(DT, _newPoint_image);			
			/* pairwise term */
			loss += lambda[i] * param_distance(params[i], params[(i+1)%params.size()], i, (i+1)%params.size()); //TODO: assuming the contour is closed
		}
		else
		{
            cout<<"id: "<<i<<" x: "<<_newPoint_image.x<<" y: "<< _newPoint_image.y<<endl;
            cout<<"currentPoints[i]: "<<currentPoints[i]<<endl;
            cout<<"current_local: "<<current_local<<endl;
            cout<<"p.s: "<<params[i].s<<" p.phi: "<<params[i].phi<<endl;
            cout<<"_newPoint: "<<_newPoint<<endl;
            cout<<"_newPoint_image: "<< _newPoint_image <<endl;
			return BIG_LOSS;
		}
	}
	
	//loss += lambda[currentPoints.size()-1] * param_distance(params[currentPoints.size()-1], params[0], currentPoints.size()-1, 0);
	return loss;
}

void TemplateMatcher::minimize_single_step()
{
    //current_gradients.resize( currentPoints.size(), param_t(0,0) ); //TODO: check to be correct
    std::fill(current_gradients.begin(), current_gradients.end(), param_t(0,0));

    for(size_t i = 0; i < current_gradients.size(); ++i)
    {
        Point2f _newPoint;
        Point2f _newPoint_image;
        Point2f current_local = imageToLocal(currentPoints[i]);
        apply_transform( current_local, params[i], _newPoint);
        _newPoint_image = localToImage(_newPoint);

        if( (_newPoint_image.x >= 1) && (_newPoint_image.y >= 1) && (_newPoint_image.x < (edgeImg.cols-1)) && (_newPoint_image.y < (edgeImg.rows-1)))
        {
            float grad_x = bilinear_interpolate(grad_x_DT, _newPoint_image);
            float grad_y = bilinear_interpolate(grad_y_DT, _newPoint_image);

            /* \frac{\partial dist}{\partial s_n} */
            float d_s = grad_x * current_local.x + grad_y * current_local.y; 
            float normalize_1 = 1.0f, normalize_2 = 1.0f;
            size_t next_id = (i+1)%params.size();
            if( config.normalize_scales)
            {
                normalize_1 = 1.0f/scale_norms[i];
                normalize_2 = 1.0f/scale_norms[next_id];
            }

            /* \frac{\partial d2}{\partial s_n}*/
            d_s += lambda[i] * 2 * config.gamma * normalize_1 * ( params[i].s * normalize_1 - params[next_id].s * normalize_2); 
            current_gradients[i].s += d_s;

            /* \frac{\partial d2}{\partial s_{n+1}} */
            float ds_2 = lambda[i] * (-2) * config.gamma * normalize_2 * (params[i].s * normalize_1 - params[next_id].s * normalize_2);
            current_gradients[next_id].s += ds_2;

            /* \frac{\partial dist}{\partial \phi_n}*/
            float d_phi = grad_x * ( -sin( params[i].phi ) * current_local.x - cos( params[i].phi ) * current_local.y ) +
                grad_y * (cos( params[i].phi)*current_local.x - sin(params[i].phi)*current_local.y );

            /* \frac{\partial d2}{\partial \phi_n} */
            d_phi += lambda[i] *  phi_minus( params[i].phi, params[next_id].phi) / M_PI;
            current_gradients[i].phi += d_phi;

            /* \frac{\partial d2}{\partial \phi_{n+1} } */
            float d_phi_2 = -lambda[i] * phi_minus( params[i].phi, params[next_id].phi) / M_PI;
            current_gradients[next_id].phi += d_phi_2;

            if(config.do_fast_update)
            {
                /* update params */
                param_t par;
                par.s = params[i].s - config.learning_rate * current_gradients[i].s;
                par.phi = correct_phi( params[i].phi - config.learning_rate * current_gradients[i].phi);

                /* update current point */ 
                // TODO: what to do for the next_id? (at least the 'last to first' point contribution  will be lost)
                Point2f _new_local;
                apply_transform( current_local, par, _new_local);
                Point2f _new_p = localToImage( _new_local);
                
                if(_new_p.x > 1 && _new_p.y > 1 && _new_p.x < edgeImg.cols -1 && _new_p.y < edgeImg.rows - 1 )
                {
                    currentPoints[i] = _new_p;
                    params[i] = par;
                }
            }

        }

    }

    if(!config.do_fast_update)
    {
        for(size_t i = 0; i < params.size(); ++i)
        {
            param_t par;
            par.s = params[i].s - config.learning_rate * current_gradients[i].s;
            par.phi = correct_phi( params[i].phi - config.learning_rate * current_gradients[i].phi);
            Point2f current_local = imageToLocal(currentPoints[i]);
            Point2f _new_local;
            apply_transform( current_local, par, _new_local);
            Point2f _new_p = localToImage( _new_local);
			
            if(_new_p.x > 1 && _new_p.y > 1 && _new_p.x < edgeImg.cols -1 && _new_p.y < edgeImg.rows - 1 )
            {
                currentPoints[i] = _new_p;
                params[i] = par;
            }

        }
    } 
}

vector<Point2f> TemplateMatcher::minimize()
{
	float best_loss = BIG_LOSS;
	vector<Point2f> best_points;
	for(int i = 0; i < config.max_iter; i++)
	{		
		minimize_single_step();
		float n_loss = compute_loss();
		if(n_loss < best_loss)
		{
			best_loss = n_loss;
			best_points = currentPoints;
		}
		cout<<"\t ** LOSS: "<< n_loss << endl;
	}
	return best_points;
}
