#include "stmkf.h"
#include <omp.h>
#include <sstream>
#include <string>

STMKF::STMKF(Mat firstFrame, float q, float r, int maskSize, int d, double sigmaValue) {
    height = firstFrame.rows;
    width = firstFrame.cols;
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
    bilateralTime = 0.0;
    blurTime = 0.0;
    modifiedKalmanTime = 0.0;
}

Mat STMKF::newFrame(Mat frame) {
    frameCount++;
    frame.convertTo(floatFrame, CV_32F);
    double start_time = omp_get_wtime();
    blur(floatFrame, aux, Size(maskSize,maskSize));
    blurTime += omp_get_wtime() - start_time;
    start_time = omp_get_wtime();
    bilateralFilter(floatFrame, bfFrame, d, sigmas, sigmas);
    bilateralTime += omp_get_wtime() - start_time;
    start_time = omp_get_wtime();
    xPredicted = xCorrection.clone();
    int i,j;
    Mat aux2 = Mat(height, width, CV_32F);
#pragma omp parallel for private(i,j)
    for (i = 0; i < height; i++){
        for (j = 0; j < width; j++) {
            delta.at<float>(i,j) = blured.at<float>(i,j) - aux.at<float>(i,j);
            r.at<float>(i,j) = 1 + r.at<float>(i,j) / (1 + k.at<float>(i,j));
            pPredicted.at<float>(i,j) = pCorrection.at<float>(i,j) + q * delta.at<float>(i,j) * delta.at<float>(i,j);
            k.at<float>(i,j) = pPredicted.at<float>(i,j)/(pPredicted.at<float>(i,j)+r.at<float>(i,j));
            xCorrection.at<float>(i,j) = (1-k.at<float>(i,j)) * (xPredicted.at<float>(i,j) + k.at<float>(i,j) * (floatFrame.at<float>(i,j) - xPredicted.at<float>(i,j))) + k.at<float>(i,j) * bfFrame.at<float>(i,j);
            pCorrection.at<float>(i,j) = pPredicted.at<float>(i,j) * (1 - k.at<float>(i,j));
        }
    }
    blured = aux.clone();
    xCorrection.convertTo(auxU, CV_8U);
    modifiedKalmanTime += omp_get_wtime() - start_time;
    return auxU;
}
