#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/ml.hpp>
#include <iostream>
#include <fstream>

#define PATCH_W 80
#define PATCH_H 200

#define LABEL_PEDESTRIAN 1
#define LABEL_BACKGROUND -1

std::vector<float> get_svm_detector(const cv::Ptr<cv::ml::SVM> &svm);

int main(int argc, char *argv[])
{
    auto svm = cv::ml::SVM::load("pedestrian_model.yml");
    std::string testImageFile = "../train/1.png";
    cv::Mat testImage = cv::imread(testImageFile);

    cv::HOGDescriptor hog(
        cv::Size(PATCH_W, PATCH_H),                  // winSize
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
    std::vector<cv::Mat> testDataList;

    int stride = 4;
    for (int x = 0; x + PATCH_W < testImage.cols; x += stride)
    {
        cv::Rect slice(x, 0, PATCH_W, PATCH_H);
        hog.compute(testImage(slice), descriptors);
        testDataList.push_back(cv::Mat(descriptors).clone());
    }

    int testDataRows = testDataList.size();
    int testDataCols = testDataList[0].rows;
    cv::Mat testDataMatrix(testDataRows, testDataCols, CV_32FC1);
    cv::Mat transposeTmpMatrix(1, testDataCols, CV_32FC1);

    for (size_t i = 0; i < testDataList.size(); i++)
    {
        cv::transpose(testDataList[i], transposeTmpMatrix);
        transposeTmpMatrix.copyTo(testDataMatrix.row((int)i));
    }

    cv::Mat results;
    svm->predict(testDataMatrix, results, cv::ml::ROW_SAMPLE);
    // float r = svm->predict(testDataMatrix);

    for (int strideX = 0; strideX < results.rows; strideX++)
    {
        if (results.at<float>(1, strideX) == LABEL_BACKGROUND)
        {
            continue;
        }

        int x = strideX * stride;
        cv::Rect box(x, 0, PATCH_W, PATCH_H);
        cv::rectangle(testImage, box, cv::Scalar(0, 0, (int)(((float)x / testImage.cols) * 255)));
    }

    cv::imshow("Test image", testImage);
    cv::waitKey();

    // hog.setSVMDetector(get_svm_detector(svm));

    // svm->predict()

    // std::vector<cv::Rect> detections;
    // std::vector<double> foundWeights;
    // hog.detectMultiScale(testImage, detections, foundWeights);
    // for (size_t j = 0; j < detections.size(); j++)
    // {
    //     cv::Scalar color = cv::Scalar(0, 0, 255);
    //     rectangle(testImage, detections[j], color);
    // }

    // std::ifstream trainAnnotationsFile;
    // trainAnnotationsFile.open("../train/train-processed.idl");
    // if (!trainAnnotationsFile.is_open())
    // {
    //     std::cout << "Can't open training annotations file." << std::endl;
    //     return 1;
    // }

    // cv::HOGDescriptor hog(
    //     cv::Size(PATCH_W, PATCH_H),                           // winSize
    //     cv::Size(20, 20),                            // blocksize
    //     cv::Size(10, 10),                            // blockStride,
    //     cv::Size(10, 10),                            // cellSize,
    //     9,                                           // nbins,
    //     1,                                           // derivAper,
    //     -1,                                          // winSigma,
    //     cv::HOGDescriptor::HistogramNormType::L2Hys, // histogramNormType,
    //     0.2,                                         // L2HysThresh,
    //     1,                                           // gammal correction,
    //     64,                                          // nlevels=64
    //     1);                                          // Use signed gradients
    // std::vector<float> descriptors;
    // std::vector<cv::Mat> trainDataList;
    // std::vector<int> labelsList;
    // cv::RNG rng;

    // while (true)
    // {
    //     int imageNo, y1, x1, y2, x2;
    //     trainAnnotationsFile >> imageNo;
    //     trainAnnotationsFile >> y1;
    //     trainAnnotationsFile >> x1;
    //     trainAnnotationsFile >> y2;
    //     trainAnnotationsFile >> x2;

    //     if (trainAnnotationsFile.eof())
    //     {
    //         break;
    //     }

    //     std::cout << imageNo << std::endl;

    //     cv::Rect pedestrianBox(x1, y1, x2 - x1, y2 - y1);
    //     std::string trainImageFile = "../train/" + std::to_string(imageNo) + ".png";
    //     cv::Mat trainImage = cv::imread(trainImageFile, cv::ImreadModes::IMREAD_GRAYSCALE);
    //     if (trainImage.empty())
    //     {
    //         std::cout << "Can't open " << trainImageFile << std::endl;
    //         trainAnnotationsFile.close();
    //         return 1;
    //     }

    //     int backgroundX;
    //     do
    //     {
    //         backgroundX = rng.uniform(0, trainImage.cols - PATCH_W + 1);
    //     } while (backgroundX + PATCH_W >= x1 && backgroundX <= x2);
    //     cv::Rect backgroundBox(backgroundX, 0, PATCH_W, PATCH_H);

    //     hog.compute(trainImage(pedestrianBox), descriptors);
    //     trainDataList.push_back(cv::Mat(descriptors).clone());
    //     labelsList.push_back(LABEL_PEDESTRIAN);

    //     hog.compute(trainImage(backgroundBox), descriptors);
    //     trainDataList.push_back(cv::Mat(descriptors).clone());
    //     labelsList.push_back(LABEL_BACKGROUND);
    // }

    // trainAnnotationsFile.close();

    // int trainDataRows = trainDataList.size();
    // int trainDataCols = trainDataList[0].rows;
    // cv::Mat trainDataMatrix(trainDataRows, trainDataCols, CV_32FC1);
    // cv::Mat transposeTmpMatrix(1, trainDataCols, CV_32FC1);

    // for (size_t i = 0; i < trainDataList.size(); i++)
    // {
    //     cv::transpose(trainDataList[i], transposeTmpMatrix);
    //     transposeTmpMatrix.copyTo(trainDataMatrix.row((int)i));
    // }

    // auto svm = cv::ml::SVM::create();
    // svm->setType(cv::ml::SVM::C_SVC);
    // svm->setKernel(cv::ml::SVM::LINEAR);
    // svm->trainAuto(trainDataMatrix, cv::ml::ROW_SAMPLE, labelsList);

    // svm->save("pedestrian_model.yml");

    return 0;
}

std::vector<float> get_svm_detector(const cv::Ptr<cv::ml::SVM> &svm)
{
    // get the support vectors
    cv::Mat sv = svm->getSupportVectors();
    const int sv_total = sv.rows;
    // get the decision function
    cv::Mat alpha, svidx;
    double rho = svm->getDecisionFunction(0, alpha, svidx);
    std::vector<float> hog_detector(sv.cols + 1);
    memcpy(&hog_detector[0], sv.ptr(), sv.cols * sizeof(hog_detector[0]));
    hog_detector[sv.cols] = (float)-rho;
    return hog_detector;
}