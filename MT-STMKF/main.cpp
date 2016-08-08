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
#include "opencv2/imgproc.hpp"
#include "stmkf.h"
#include <math.h>
#include <sys/time.h>
#include <omp.h>

using namespace cv;

int main(int argc, char *argv[]) {
    if (argc < 3) {
        cerr << "Usage: " << argv[0] << " YOUR_VIDEO.EXT NUMBER_OF_THREADS" << std::endl;
        return 1;
    }
    try {
        float noiseStd = 10;
        float q = 0.026;
        float r = 1;
        int bilateralKernelSize = 3;
        float bilateralSigmas = 50.0;

        int frameCount = 0;
        VideoCapture videoSource;
        if(!videoSource.open(argv[1])) {
            cout << "ERROR on load video..." << endl;
            return 0;
        }
        videoSource.set(CV_CAP_PROP_CONVERT_RGB, 0);
        Mat frame, gray, noised, filtered;
        videoSource >> frame;
        Mat noise = Mat(frame.size(), CV_16S);
       
        setNumThreads(atoi(argv[2]));
        omp_set_num_threads(atoi(argv[2]));

		cvtColor(frame, gray, CV_RGB2GRAY);
        STMKF stmkf = STMKF(gray, q, r, 5, bilateralKernelSize, bilateralSigmas);

        while (1) {
            imshow("Original Frame", frame);  //without show images are faster
            imshow("Gray Frame", gray);
            randn(noise, 0, noiseStd);
            add(gray, noise, noised, noArray(), CV_8U);
            imshow("Noised", noised);
            filtered = stmkf.newFrame(noised);
            imshow("Filtered", filtered);
            waitKey(1); // set to 0 for step by step
            videoSource >> frame;
            frameCount++;
            if (frame.empty() ) {
                cout << endl << "Video ended!" << endl;
                break;
            }
            cvtColor(frame, gray, CV_RGB2GRAY);
        }
    } catch(const Exception& ex) 	{
        cout << "Error: " << ex.what() << endl;
    }

    return 0;
}
