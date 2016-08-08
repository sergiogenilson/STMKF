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

#include <iostream>
using namespace std;
#include "opencv2/core.hpp"
#include <opencv2/core/utility.hpp>
#include "opencv2/highgui.hpp"
#include "opencv2/cudaoptflow.hpp"
#include "opencv2/cudaarithm.hpp"
#include "opencv2/cudaimgproc.hpp"
#include "stmkf.h"
#include <math.h>
#include <sys/time.h>

using namespace cv;
using namespace cv::cuda;

int main (int argc, char* argv[])
{
    if (getCudaEnabledDeviceCount() < 1) {
        cout << "No CUDA devices found" << endl;
        return 0;
    }
    cout << "Enabled CUDA devices: " << getCudaEnabledDeviceCount() << endl;
    setDevice(0);
    cout << "Chosen devise: --------------------> " << DeviceInfo(getDevice()).name() << endl;
    try {
        VideoCapture videoSource;
        if(!videoSource.open(argv[1])) {
            printf("Problemas ao carregar video...\n");
            return 0;
        }
        videoSource.set(CV_CAP_PROP_CONVERT_RGB, 0);

        Mat noise, noised, output;

        GpuMat src,gray;

        float noiseStd = 10;
        float q = 0.026;
        float r = 1;
        int bilateralKernelSize = 3;
        float bilateralSigmas = 50.0;

        int frameCount = 0;

        Mat frame, frameAux;
        videoSource >> frameAux;
       // cout << "Video Size: " << frameAux.cols << "x" << frameAux.rows << endl;
        cuda::createContinuous(frameAux.cols, frameAux.rows, CV_8UC3, frame);
        cuda::createContinuous(frameAux.cols, frameAux.rows, CV_8U, output);
        registerPageLocked(frame);
        registerPageLocked(output);
        frameAux.copyTo(frame);

        src.upload(frame);
        cuda::cvtColor(src, gray, CV_RGB2GRAY);
        GpuMat dst = GpuMat(gray.size(), gray.type());
        STMKF stmkf = STMKF(gray, q, r, 5, bilateralKernelSize, bilateralSigmas);

        int numberOfStreams = 3;
        cout << "Using " << numberOfStreams << " streams; ";
        cuda::Stream stream [numberOfStreams];

        bool go = true;

        gettimeofday(&start, NULL);
        while(go) {

            for (int i = 0; i < numberOfStreams; i++) {
                stream[i].waitForCompletion();
                imshow("New Frame", frame);
                cv::randn(noise, 0, noiseStd);
                noised = frame + noise;
                imshow("Noised", noised);
                src.upload(noised,stream[i]);
                cuda::cvtColor(src, gray, CV_RGB2GRAY, 0, stream[i]);

                stmkf.newFrame(gray, dst, stream[i]);
                dst.download(output, stream[i]);

                if (output.size().width>0)
                    imshow("Filtered", output);

                waitKey(1);
                videoSource >> frame;
                frameCount++;
                if (frame.empty()) {
                    cout << "The end!" << endl;
                    go = false;
                    break;
                }
            }
        }
        cout << endl;
    }
    catch(const Exception& ex) 	{
        cout << "Error: " << ex.what() << std::endl;
    }
    return 0;
}

