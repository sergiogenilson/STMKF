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

#include "stmkf.h"
#include <sys/time.h>

STMKF::STMKF(Mat firstFrame, float q, float r, int maskSize, int d, double sigmaValue) {
    int height = firstFrame.rows;
    int width = firstFrame.cols;
    xPredicted = Mat(height, width, CV_32F);
    xPredicted.setTo(1);
    pPredicted = Mat(height, width, CV_32F);
    pPredicted.setTo(1);
    firstFrame.convertTo(xCorrection, CV_32F);    // xCorrection = firstFrame;
    pCorrection = Mat(height, width, CV_32F);
    pCorrection.setTo(1);
    k = Mat(height, width, CV_32F);
    k.setTo(0.5);
    blured = Mat(height, width, CV_32F);
    blured.setTo(0);
    this->r = Mat(height, width, CV_32F);
    this->r.setTo(r);
    this->q = q;
    this->maskSize = maskSize;
    this->d = d;
    this->sigmas = sigmaValue;

    // auxiliares...
    delta = Mat(height, width, CV_32F);
    aux = Mat(height, width, CV_32F);
    bfFrame = Mat(height, width, CV_32F);
    auxU = Mat(height, width, CV_8U);

    frameCount = 0;
    bilateralTime = 0;
    blurTime = 0;
    modifiedKalmanTime = 0;
}

Mat STMKF::newFrame(Mat frame) {
    frame.convertTo(floatFrame, CV_32F);
    blur(floatFrame, aux, Size(maskSize,maskSize));
    bilateralFilter(floatFrame, bfFrame, d, sigmas, sigmas);
    //--- STMKF-CORE ---//
    delta = blured - aux;
    blured = aux.clone();
    r = 1+(r)/(1+k);
    xPredicted = xCorrection.clone();
    pPredicted = pCorrection+q*delta.mul(delta);
    k = pPredicted/(pPredicted+r);
    xCorrection = (1-k).mul(xPredicted + k.mul(floatFrame - xPredicted)) + k.mul(bfFrame);
    pCorrection = pPredicted.mul(1-k);
    //--- END OF STMKF-CORE ---//
    xCorrection.convertTo(auxU, CV_8U);
    return auxU;
}
