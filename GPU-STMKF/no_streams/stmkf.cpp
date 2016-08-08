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
    k = GpuMat(height, width, CV_64F);
    k.setTo(0.5);
    blured = GpuMat(height, width, CV_32F);
    blured.setTo(0);
    this->r.setTo(r);
    this->q = q;
    this->maskSize = maskSize;
    this->d = d;
    this->sigmas = sigmaValue;
    blurFilter = cuda::createBoxFilter(firstFrame.type(), firstFrame.type(), Size(maskSize,maskSize));
}

void STMKF::newFrame(GpuMat frame, GpuMat dst) {
    int height = frame.rows;
    int width = frame.cols;
    GpuMat floatFrame;
    GpuMat delta = GpuMat(height, width, CV_32F);
    GpuMat aux = GpuMat(height, width, CV_32F);
    GpuMat aux2 = GpuMat(height, width, CV_32F);
    GpuMat bfFrame = GpuMat(height, width, CV_32F);
    GpuMat auxU = GpuMat(height, width, CV_8U);
    frame.convertTo(floatFrame,CV_32F);
    blurFilter->apply(frame,auxU);
    auxU.convertTo(aux,CV_32F);
    cuda::bilateralFilter(floatFrame, bfFrame, d, sigmas, sigmas);
    cuda::subtract(blured, aux, delta);
    blured = aux.clone();
    cuda::add(1,k,aux2);
    cuda::divide(r,aux2,aux2);
    cuda::add(1,aux2,r);
    xPredicted = xCorrection.clone();
    cuda::multiply(delta,delta,delta);
    cuda::multiply(delta,q,delta);
    cuda::add(pCorrection,delta,pPredicted);
    cuda::add(pPredicted,r,aux);
    cuda::divide(pPredicted,aux,k);
    cuda::subtract(floatFrame, xPredicted, aux);
    cuda::multiply(k, aux, aux);
    cuda::add(xPredicted, aux, aux);
    cuda::subtract(1,k,aux2);
    cuda::multiply(aux2,aux,aux);
    cuda::multiply(pPredicted, aux2, pCorrection);
    cuda::multiply(k,bfFrame,aux2);
    cuda::add(aux,aux2,xCorrection);
    xCorrection.convertTo(dst, CV_8U);
}
