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

STMKF::STMKF(GpuMat firstFrame, float q, float r, int maskSize, int d, double sigmaValue) {
    int height = firstFrame.rows;
    int width = firstFrame.cols;
    xPredicted = GpuMat(height, width, CV_32F);
    xPredicted.setTo(1);
    pPredicted = GpuMat(height, width, CV_32F);
    pPredicted.setTo(1);
    firstFrame.convertTo(xCorrection, CV_32F);    // xCorrection = firstFrame;
    pCorrection = GpuMat(height, width, CV_32F);
    pCorrection.setTo(1);
    k = GpuMat(height, width, CV_32F);
    k.setTo(0.5);
    blured = GpuMat(height, width, CV_32F);
    blured.setTo(0);
    this->q = q;
    this->r = r;
    this->maskSize = maskSize;
    this->d = d;
    this->sigmas = sigmaValue;
    blurFilter = cuda::createBoxFilter(firstFrame.type(), firstFrame.type(), Size(maskSize,maskSize));

    // auxiliares...
    delta = GpuMat(height, width, CV_32F);
    aux = GpuMat(height, width, CV_32F);
    aux2 = GpuMat(height, width, CV_32F);
    bfFrame = GpuMat(height, width, CV_32F);

    auxU = GpuMat(height, width, CV_8U);
}

void STMKF::newFrame(GpuMat frame, GpuMat dst, Stream stream) {

    frame.convertTo(floatFrame,CV_32F, stream);

    blurFilter->apply(frame,auxU,stream);
//        auxU.download(blur);
//        imshow("Blured", blur);
    auxU.convertTo(aux,CV_32F,stream);
    cuda::bilateralFilter(floatFrame, bfFrame, d, sigmas, sigmas, BORDER_DEFAULT, stream);
//        bfFrame.convertTo(auxU,CV_32F);
//        auxU.download(bil);
//        imshow("Bilateral", bil);
    cuda::subtract(blured, aux, delta,noArray(),-1,stream);
//        blured.convertTo(auxU, CV_32F);
//        auxU.download(del);
//        imshow("Delta", del);
    aux.copyTo(blured,stream);
    xCorrection.copyTo(xPredicted,stream);
    //pPredicted = pCorrection+q*delta*delta;
    cuda::multiply(delta, delta, delta, 1, -1, stream);
    cuda::multiply(delta, q, delta, 1, -1, stream);
    cuda::add(pCorrection,delta,pPredicted, noArray(), -1, stream);

    //k = pPredicted/(pPredicted+r);
    cuda::add(pPredicted,r,aux, noArray(), -1, stream);
    cuda::divide(pPredicted,aux,k, 1, -1, stream);
//        cuda::multiply(k,255,aux);
//        aux.convertTo(auxU,CV_32F);
//        auxU.download(k_);
//        imshow("k", k_);
    //xCorrection = (1-k)*(xPredicted+k*(frame-xPredicted))+k*bf;
    cuda::subtract(floatFrame, xPredicted, aux, noArray(), -1, stream);
    cuda::multiply(k, aux, aux,  1, -1, stream);
    cuda::add(xPredicted, aux, aux, noArray(), -1, stream);
    cuda::subtract(1, k, aux2, noArray(), -1, stream);
    cuda::multiply(aux2, aux, aux, 1, -1, stream);
    //pCorrection = pPredicted*(1-k);
    cuda::multiply(pPredicted, aux2, pCorrection, 1, -1, stream);
    //continuation of xCorrection calcutaltion
    cuda::multiply(k,bfFrame,aux2,1,-1,stream);
    cuda::add(aux,aux2,xCorrection, noArray(), -1, stream);
    xCorrection.convertTo(dst, CV_8U,stream);
//        dst.download(final);
//        imshow("Output", final);
}
