#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/ml.hpp>
#include <iostream>
#include <fstream>
#include <set>

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
int evalMain(std::string correct, std::string detected);
int readAnnotations(std::string annotationsFile, std::map<int, std::vector<cv::Rect>> &result);
void getKeys(std::map<int, std::vector<cv::Rect>> map, std::set<int> &result);

bool showDetectionImages = true;

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
    if (commandType == "eval")
    {
        return evalMain("../test-public/test-processed.idl", "test-processed.idl");
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

int testMain()
{
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
        // For some reason the last row might be garbage SOMETIMES, hmmmm
        if (results.at<float>(1, strideX) != LABEL_PEDESTRIAN)
        {
            continue;
        }

        int x = strideX * stride;
        cv::Rect box(x, 0, PATCH_W, PATCH_H);
        pedestrianBoxes.push_back(box);
    }

    groupSlices(pedestrianBoxes, pedestriansResult);

    for (auto b = pedestriansResult.begin(), e = pedestriansResult.end(); b != e; b++)
    {
        cv::rectangle(testImage, *b, cv::Scalar(255, 0, 0));
    }

    if (showDetectionImages)
    {
        cv::imshow("Test image", testImage);
        if (cv::waitKey(3000) == -1)
        {
            showDetectionImages = false;
        }
    }

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
                *ob = (*ob) | currentRect;
                merged = true;
                break;
            }
        }

        if (!merged)
        {
            overlaps.push_back(currentRect);
        }
    }

    for (auto ob = overlaps.begin(), oe = overlaps.end(); ob != oe; ob++)
    {
        ob->x -= (PATCH_W - ob->width) / 2;
        ob->width = PATCH_W;
    }
}

int evalMain(std::string correct, std::string detected)
{
    int truePositives = 0;  // Detected pedestrians who overlap at least 50% with the correct pedestrians
    int falsePositives = 0; // Detected pedestrians who overlap less than 50% with the correct pedestrians
    int correctCount = 0;

    std::map<int, std::vector<cv::Rect>> correctMap, detectedMap;
    if (readAnnotations(correct, correctMap) != 0)
    {
        return 1;
    }
    if (readAnnotations(detected, detectedMap) != 0)
    {
        return 1;
    }

    std::set<int> images;
    getKeys(correctMap, images);
    getKeys(detectedMap, images);

    for (auto b = images.begin(), e = images.end(); b != e; b++)
    {
        int imageId = *b;

        if (correctMap.count(imageId) == 0)
        {
            falsePositives += detectedMap[imageId].size();
            continue;
        }
        correctCount += correctMap[imageId].size();

        if (detectedMap.count(imageId) == 0)
        {
            continue;
        }

        std::vector<cv::Rect> correctPedestrians = correctMap[imageId];
        std::vector<cv::Rect> detectedPedestrians = detectedMap[imageId];

        for (auto db = detectedPedestrians.begin(), de = detectedPedestrians.end(); db != de; db++)
        {
            cv::Rect detectedBox = *db;
            bool isCorrectMatch = false;
            for (auto cb = correctPedestrians.begin(), ce = correctPedestrians.end(); cb != ce; cb++)
            {
                cv::Rect correctBox = *cb;
                int correctBoxAreaSize = correctBox.area();
                int overlapAreaSize = (correctBox & detectedBox).area();

                if (overlapAreaSize >= correctBoxAreaSize / 2)
                {
                    isCorrectMatch = true;
                    break;
                }
            }

            if (isCorrectMatch)
            {
                truePositives++;
            }
            else
            {
                falsePositives++;
            }
        }
    }

    double recall = ((double)truePositives) / correctCount;
    double precision = ((double)truePositives) / (truePositives + falsePositives);

    std::cout << "Recall   : " << recall << std::endl;
    std::cout << "Precision: " << precision << std::endl;

    return 0;
}

int readAnnotations(std::string annotationsFile, std::map<int, std::vector<cv::Rect>> &result)
{
    std::ifstream file;
    file.open(annotationsFile);
    if (!file.is_open())
    {
        std::cout << "Can't open annotations file " << annotationsFile << std::endl;
        return 1;
    }

    while (true)
    {
        int imageNo, y1, x1, y2, x2;
        file >> imageNo;
        file >> y1;
        file >> x1;
        file >> y2;
        file >> x2;

        if (file.eof())
        {
            break;
        }

        cv::Rect box(x1, y1, x2 - x1, y2 - y1);
        result[imageNo].push_back(box);
    }

    file.close();
    return 0;
}

void getKeys(std::map<int, std::vector<cv::Rect>> map, std::set<int> &result)
{
    for (auto b = map.begin(), e = map.end(); b != e; b++)
    {
        result.insert((*b).first);
    }
}