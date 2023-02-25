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

void createHog(cv::FileStorage &params, cv::HOGDescriptor &hog);
cv::Mat stdVectorToSamplesCvMat(std::vector<cv::Mat> &vec);

int trainMain();
int testMain();
int detectPedestrians(cv::Ptr<cv::ml::SVM> &svm, cv::HOGDescriptor &hog, std::string imageFile, std::vector<cv::Rect> &pedestriansResult);
void groupSlices(std::vector<cv::Rect> &rectangles, std::vector<cv::Rect> &overlaps);
std::string fileNameWithoutExtension(std::string path);
std::vector<int> getImagesSorted(std::string imagesDirectory);

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        std::cout << "Not enough arguments." << std::endl;
        return 1;
    }

    std::string commandType = argv[1];
    if (commandType == "train")
    {
        return trainMain();
    }
    if (commandType == "test")
    {
        return testMain();
    }

    std::cout << "Unknown command type." << std::endl;
    return 1;
}

int trainMain()
{
    std::ifstream trainAnnotationsFile;
    trainAnnotationsFile.open("../train/train-processed.idl");
    if (!trainAnnotationsFile.is_open())
    {
        std::cout << "Can't open training annotations file." << std::endl;
        return 1;
    }

    cv::FileStorage params("../params.yml", cv::FileStorage::READ);
    int backgroundSamples = params["backgroundSamples"];

    cv::HOGDescriptor hog;
    createHog(params, hog);

    std::vector<float> descriptors;
    std::vector<cv::Mat> trainDataList;
    std::vector<int> labelsList;
    cv::RNG rng;

    while (true)
    {
        int imageNo, y1, x1, y2, x2;
        trainAnnotationsFile >> imageNo;
        trainAnnotationsFile >> y1;
        trainAnnotationsFile >> x1;
        trainAnnotationsFile >> y2;
        trainAnnotationsFile >> x2;

        if (trainAnnotationsFile.eof())
        {
            break;
        }

        bool isValidationSample = imageNo % 5 == 0;
        if (isValidationSample)
        {
            continue;
        }
        if (imageNo < 200)
        {
            continue;
        }

        std::cout << imageNo << std::endl;

        cv::Rect pedestrianBox(x1, y1, x2 - x1, y2 - y1);
        std::string trainImageFile = "../train/" + std::to_string(imageNo) + ".png";
        cv::Mat trainImage = cv::imread(trainImageFile, cv::ImreadModes::IMREAD_GRAYSCALE);
        if (trainImage.empty())
        {
            std::cout << "Can't open " << trainImageFile << std::endl;
            trainAnnotationsFile.close();
            return 1;
        }

        hog.compute(trainImage(pedestrianBox), descriptors);
        trainDataList.push_back(cv::Mat(descriptors).clone());
        labelsList.push_back(LABEL_PEDESTRIAN);

        for (int i = 0; i < backgroundSamples; i++)
        {
            int backgroundX;
            do
            {
                backgroundX = rng.uniform(0, trainImage.cols - PATCH_W + 1);
            } while (backgroundX + PATCH_W >= x1 && backgroundX <= x2);
            cv::Rect backgroundBox(backgroundX, 0, PATCH_W, PATCH_H);

            hog.compute(trainImage(backgroundBox), descriptors);
            trainDataList.push_back(cv::Mat(descriptors).clone());
            labelsList.push_back(LABEL_BACKGROUND);
        }
    }

    trainAnnotationsFile.close();

    int trainDataRows = trainDataList.size();
    int trainDataCols = trainDataList[0].rows;
    cv::Mat trainDataMatrix(trainDataRows, trainDataCols, CV_32FC1);
    cv::Mat transposeTmpMatrix(1, trainDataCols, CV_32FC1);

    for (size_t i = 0; i < trainDataList.size(); i++)
    {
        cv::transpose(trainDataList[i], transposeTmpMatrix);
        transposeTmpMatrix.copyTo(trainDataMatrix.row((int)i));
    }

    auto svm = cv::ml::SVM::create();
    svm->setType(cv::ml::SVM::C_SVC);
    svm->setKernel(cv::ml::SVM::LINEAR);
    svm->trainAuto(trainDataMatrix, cv::ml::ROW_SAMPLE, labelsList);

    svm->save("pedestrian_model.yml");

    return 0;
}

void createHog(cv::FileStorage &params, cv::HOGDescriptor &hog)
{
    hog.winSize = cv::Size(PATCH_W, PATCH_H);
    hog.histogramNormType = cv::HOGDescriptor::HistogramNormType::L2Hys;

    hog.blockSize.width = params["blockSizeX"];
    hog.blockSize.height = params["blockSizeY"];

    hog.blockStride.width = params["blockStrideX"];
    hog.blockStride.height = params["blockStrideY"];

    hog.cellSize.width = params["cellSizeX"];
    hog.cellSize.height = params["cellSizeY"];

    hog.nbins = 9;
    hog.derivAperture = params["derivAperture"];
    hog.winSigma = params["winSigma"];
    hog.L2HysThreshold = params["L2HysThreshold"];
    hog.gammaCorrection = static_cast<int>(params["gammaCorrection"]) != 0;
    hog.nlevels = params["nlevels"];
    hog.signedGradient = static_cast<int>(params["signedGradient"]) != 0;
}

cv::Mat stdVectorToSamplesCvMat(std::vector<cv::Mat> &vec)
{
    int testDataRows = vec.size();
    int testDataCols = vec[0].rows;
    cv::Mat testDataMatrix(testDataRows, testDataCols, CV_32FC1);
    cv::Mat transposeTmpMatrix(1, testDataCols, CV_32FC1);

    for (size_t i = 0; i < vec.size(); i++)
    {
        cv::transpose(vec[i], transposeTmpMatrix);
        transposeTmpMatrix.copyTo(testDataMatrix.row((int)i));
    }

    return testDataMatrix;
}

// std::vector<float> getSvmDetector(const cv::Ptr<cv::ml::SVM> &svm)
// {
//     cv::Mat sv = svm->getSupportVectors();
//     const int sv_total = sv.rows;
//     cv::Mat alpha, svidx;
//     double rho = svm->getDecisionFunction(0, alpha, svidx);
//     std::vector<float> hog_detector(sv.cols + 1);
//     memcpy(&hog_detector[0], sv.ptr(), sv.cols * sizeof(hog_detector[0]));
//     hog_detector[sv.cols] = (float)-rho;
//     return hog_detector;
// }

//

int testMain()
{
    bool showDetections = true;
    std::string imagesDirectory = "../test-public";
    std::vector<int> imagesNumbers = getImagesSorted(imagesDirectory);

    auto svm = cv::ml::SVM::load("pedestrian_model.yml");

    cv::FileStorage params("../params.yml", cv::FileStorage::READ);
    cv::HOGDescriptor hog;
    createHog(params, hog);

    std::string outputFileName = "test-processed.idl";
    std::ofstream output;
    output.open(outputFileName);

    std::vector<cv::Rect> results;
    for (auto b = imagesNumbers.begin(), e = imagesNumbers.end(); b != e; b++)
    {
        int imageNumber = *b;
        std::string image = imagesDirectory + "/" + std::to_string(*b) + ".png";

        int detectResultCode = detectPedestrians(svm, hog, image, results);
        if (detectResultCode != 0)
        {
            std::cout << "Error during detection" << std::endl;

            output.close();
            return detectResultCode;
        }

        for (auto rb = results.begin(), re = results.end(); rb != re; rb++)
        {
            cv::Rect rect = *rb;
            output << imageNumber
                   << '\t' << rect.y
                   << '\t' << rect.x
                   << '\t' << rect.y + rect.height
                   << '\t' << rect.x + rect.width
                   << '\n';
        }

        results.clear();
    }

    output.close();

    return 0;
}

std::vector<int> getImagesSorted(std::string imagesDirectory)
{
    std::vector<std::string> files;
    cv::glob(imagesDirectory + "/*.png", files, false);
    std::vector<int> result;
    for (auto b = files.begin(), e = files.end(); b != e; b++)
    {
        std::string filePath = *b;
        std::string fileName = fileNameWithoutExtension(filePath);
        int fileNumber = std::stoi(fileName);
        result.push_back(fileNumber);
    }
    std::sort(result.begin(), result.end());

    return result;
}

std::string fileNameWithoutExtension(std::string path)
{
    int dot = path.find_last_of(".");
    int slash = path.find_last_of("/\\");
    return path.substr(slash + 1, dot - slash - 1);
}

int detectPedestrians(cv::Ptr<cv::ml::SVM> &svm, cv::HOGDescriptor &hog, std::string imageFile, std::vector<cv::Rect> &pedestriansResult)
{
    cv::Mat testImage = cv::imread(imageFile);
    if (testImage.empty())
    {
        std::cout << "Cannot read image " << imageFile << std::endl;
        return 1;
    }

    std::vector<float> descriptors;
    std::vector<cv::Mat> testDataList;

    int stride = 4;
    for (int x = 0; x + PATCH_W < testImage.cols; x += stride)
    {
        cv::Rect slice(x, 0, PATCH_W, PATCH_H);
        hog.compute(testImage(slice), descriptors);
        testDataList.push_back(cv::Mat(descriptors).clone());
    }

    cv::Mat testDataMatrix = stdVectorToSamplesCvMat(testDataList);

    cv::Mat results;
    svm->predict(testDataMatrix, results, cv::ml::ROW_SAMPLE);
    std::vector<cv::Rect> pedestrianBoxes;

    for (int strideX = 0; strideX < results.rows; strideX++)
    {
        if (results.at<float>(1, strideX) == LABEL_BACKGROUND)
        {
            continue;
        }

        int x = strideX * stride;
        cv::Rect box(x, 0, PATCH_W, PATCH_H);
        pedestrianBoxes.push_back(box);

        cv::rectangle(testImage, box, cv::Scalar(0, 0, 255));
    }

    groupSlices(pedestrianBoxes, pedestriansResult);

    for (auto b = pedestriansResult.begin(), e = pedestriansResult.end(); b != e; b++)
    {
        cv::rectangle(testImage, *b, cv::Scalar(255, 0, 0));
    }

    cv::imshow("Test image", testImage);
    cv::waitKey();

    return 0;
}

void groupSlices(std::vector<cv::Rect> &rectangles, std::vector<cv::Rect> &overlaps)
{
    for (auto rb = rectangles.begin(), re = rectangles.end(); rb != re; rb++)
    {
        cv::Rect currentRect = *rb;
        bool merged = false;
        for (auto ob = overlaps.begin(), oe = overlaps.end(); ob != oe; ob++)
        {
            cv::Rect overlap = currentRect & (*ob);
            if (overlap.area() > 0)
            {
                *ob = currentRect & (*ob);
                merged = true;
                break;
            }
        }

        if (!merged)
        {
            overlaps.push_back(currentRect);
        }
    }
}

void evalMain()
{
    // trainAnnotationsFile.open("../train/train-processed.idl");
    // if (!trainAnnotationsFile.is_open())
    // {
    //     std::cout << "Can't open training annotations file." << std::endl;
    //     return 1;
    // }

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

    //     bool isValidationSample = imageNo % 5 == 0;
    //     if (isValidationSample)
    //     {
    //         continue;
    //     }
    // }

    // trainAnnotationsFile.close();
}