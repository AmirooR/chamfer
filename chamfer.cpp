#include "TemplateMatcher.hpp"

static void help()
{

   cout << "\nThis program demonstrates Chamfer matching -- computing a distance between an \n"
            "edge template and a query edge image.\n"
            "Usage: \n"
            "./chamfer <image edge map> <template edge map>,"
            " By default the inputs are logo_in_clutter.png logo.png\n";
}

const char* keys =
{
    "{1| |logo_in_clutter.png|image edge map    }"
    "{2| |logo.png               |template edge map}"
};

int main( int argc, const char** argv )
{

    help();
    CommandLineParser parser(argc, argv, keys);

    string image = parser.get<string>("1");
    string templ = parser.get<string>("2");
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
    int best = chamerMatching( img, tpl, results, costs, 1, 10, 5.0, 3,3, 15, 0.8, 3.0, 0.4, 35 );
    if( best < 0 )
    {
        cout << "matching not found" << endl;
        return -1;
    }
    cout << "Results: "<<endl;
    Mat dt;
    Mat annotated_img;
    annotated_img.create(img.size(), CV_32SC2);
    dt.create(img.size(), CV_32FC1);
    dt.setTo(0);
    computeDistanceTransform(img, dt, annotated_img, 30.0);
    imshow("dt",dt);
//    normalize( dt, dt, 0, 1., cv::NORM_MINMAX);
//    imshow("normalized",dt);
//    imshow("annotated",annotated_img);


   

    for(int j = 0; j < results.size(); j++)
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
	imshow("result", dimg);

    	waitKey(0);

     }

    
    return 0;
}
