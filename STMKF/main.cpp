#include <iostream>
using namespace std;

#include "opencv2/core.hpp"
#include <opencv2/core/utility.hpp>
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "stmkf.h"
#include <math.h>
#include <sys/time.h>

using namespace cv;

int main(int argc, char *argv[]) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " YOUR_VIDEO.EXT" << std::endl;
        return 1;
    }
    try {
        float noiseStd = 10;
        float q = 0.026;
        float r = 1;
        int bilateralKernelSize = 3;
        float bilateralSigmas = 50.0;

        setNumThreads(1);

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

        cvtColor(frame, gray, CV_RGB2GRAY);
        STMKF stmkf = STMKF(gray, q, r, 5, bilateralKernelSize, bilateralSigmas);

        while (1) {
            imshow("Original Frame", frame);
            imshow("Gray Frame", gray);
            randn(noise, 0, noiseStd);
            add(gray, noise, noised, noArray(), CV_8U);
            imshow("Noised", noised);
            filtered = stmkf.newFrame(noised);
            imshow("Filtered", filtered);
            waitKey(1);  //change to 0 do see step by step
            videoSource >> frame;
            frameCount++;
            if (frame.empty()) {
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
