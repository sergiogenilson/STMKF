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
