/*
STMKF - A Real-Time Spatio-Temporal Vídeo Denoising Method with Kalman-based and Bilateral Filters Fusion
Copyright (C) 2016  Sergio Genilson Pfleger sergiogenilson@gmail.com

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

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
    void newFrame(GpuMat frame, GpuMat dst, Stream stream);

private:
    GpuMat xPredicted;
    GpuMat pPredicted;
    GpuMat xCorrection;
    GpuMat pCorrection;
    GpuMat k;
    GpuMat blured;
    float r;
    float q;
    double sigmas;
    int d;
    int maskSize;
    Ptr<cuda::Filter> blurFilter;
    //auxiliares
    GpuMat floatFrame;
    GpuMat delta;
    GpuMat aux;
    GpuMat aux2;
    GpuMat bfFrame;
    GpuMat auxU;
};
#endif
