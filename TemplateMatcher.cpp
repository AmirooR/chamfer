#include <TemplateMatcher.hpp>

TemplateMatcher(Mat& _edgeImg, vector<Point>& _initialPoints, float _gamma = 1.0f, float _learning_rate = 1.0f, int _max_iter = 1000):
    gamma(_gamma), learning_rate(_learning_rate), max_iter(_max_iter)
{
    edgeImg = _edgeImg.clone();
    initialPoints.resize( _initialPoints.size() );
    for(int i = 0; i < _initialPoints.size(); ++i)
    {
        initialPoints[i] = _initialPoints[i];
    }

    
}
