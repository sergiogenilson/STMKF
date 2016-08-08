#ifndef STMKF_H
#define STMKF_H

#include <iostream>

using namespace std;
#include "opencv2/core.hpp"
#include <opencv2/core/utility.hpp>
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <omp.h>
using namespace cv;

class STMKF{
public:
    STMKF(Mat firstFrame, float q, float r, int maskSize, int d, double sigmaValue);
    Mat newFrame(Mat frame);
    int frameCount;
    double bilateralTime, blurTime, modifiedKalmanTime;

private:
    Mat xPredicted;
    Mat pPredicted;
    Mat xCorrection;
    Mat pCorrection;
    Mat k;
    Mat blured;
    Mat r;
    float q;
    double sigmas;
    int d;
    int maskSize;
    int height, width;
    //auxiliares
    Mat floatFrame;
    Mat delta;
    Mat aux;
    Mat aux2;
    Mat bfFrame;
    Mat auxU;
};
#endif