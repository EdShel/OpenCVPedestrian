#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/ml.hpp>
#include <iostream>
using namespace std;

int main(int argc, char *argv[])
{
    cv::Mat person = cv::imread("12.png");
    cv::Rect rect(301, 0, 80, 200);
    cv::rectangle(person, rect, cv::Scalar(0, 0, 255));

    cv::Mat cropped = person(rect);

    // cv::imshow("P", person);
    // cv::imshow("C", cropped);
    // cv::waitKey();

    // person.convertTo(person, CV_32F, 1 / 255.0);

    // cv::Mat gx, gy;
    // cv::Sobel(person, gx, CV_32F, 1, 0, 1);
    // cv::Sobel(person, gy, CV_32F, 0, 1, 1);

    // cv::Mat mag, angle;
    // cv::cartToPolar(gx, gy, mag, angle, true);

    // cv::imshow("Inout", mag);
    // cv::waitKey();

    // auto svm = cv::ml::SVM::create();
    // svm->setType(cv::ml::SVM::C_SVC);
    // svm->setKernel(cv::ml::SVM::LINEAR);

    cv::Mat imgGrayscale;
    cropped.convertTo(imgGrayscale, CV_8U);

    cv::HOGDescriptor hog(
        cv::Size(80, 200),                           // winSize
        cv::Size(20, 20),                            // blocksize
        cv::Size(10, 10),                            // blockStride,
        cv::Size(10, 10),                            // cellSize,
        9,                                           // nbins,
        1,                                           // derivAper,
        -1,                                          // winSigma,
        cv::HOGDescriptor::HistogramNormType::L2Hys, // histogramNormType,
        0.2,                                         // L2HysThresh,
        1,                                           // gammal correction,
        64,                                          // nlevels=64
        1);                                          // Use signed gradients
    std::vector<float> descriptors;
    hog.compute(imgGrayscale, descriptors);

    // std::cout << "descriptors: " << descriptors.size();
    for (float i : descriptors)
        std::cout << i << ' ';
    cv::waitKey();

    return 0;
}