#ifndef STMKF_H
#define STMKF_H

#include <iostream>

using namespace std;
#include "opencv2/core.hpp"
#include <opencv2/core/utility.hpp>
#include "opencv2/highgui.hpp"
#include "opencv2/cudaoptflow.hpp"
#include "opencv2/cudaarithm.hpp"
#include "opencv2/cudaimgproc.hpp"
#include "opencv2/cudafilters.hpp"
#include "opencv2/gpu/gpu.hpp"
using namespace cv;
using namespace cv::cuda;

class STMKF{
public:
    STMKF(GpuMat firstFrame, float q, float r, int maskSize, int d, double sigmaValue);
    void newFrame(GpuMat frame, GpuMat dst);

private:
    GpuMat xPredicted;
    GpuMat pPredicted;
    GpuMat xCorrection;
    GpuMat pCorrection;
    GpuMat k;
    GpuMat blured;
    GpuMat r;
    float q;
    double sigmas;
    int d;
    int maskSize;
    Ptr<cuda::Filter> blurFilter;
};
#endif
