//
// Created by tiemajor on 11/20/21.
//

#include <filesystem>
#include "FramePreprocessor.hpp"
#include "FeatureDetector.hpp"
#include "FeatureMatcher.hpp"
#include "MatchFilterer.hpp"
#include "PoseEstimator.hpp"
#include "OpenCVVisualizer.hpp"


void processFrame(
        const std::shared_ptr<slam::Frame> &previousFrame,
        const std::shared_ptr<slam::Frame> &currentFrame
        ) {



}

cv::Mat getCameraMatrix() {

//    TUI
//    double fx = 190.9784771;
//    double fy = 190.9733070;
//    double cx = 254.9317060;
//    double cy = 256.8974428;

    double fx = 718.856;
    double fy = 718.856;
    double cx = 607.1928;
    double cy = 185.2157;

    cv::Mat K(3, 3, CV_64F);
    K.at<double>(0, 0) = fx;
    K.at<double>(0, 1) = 0;
    K.at<double>(0, 2) = cx;
    K.at<double>(1, 0) = 0;
    K.at<double>(1, 1) = fy;
    K.at<double>(1, 2) = cy;
    K.at<double>(2, 0) = 0;
    K.at<double>(2, 1) = 0;
    K.at<double>(2, 2) = 1;
    return K;
}

cv::Mat getDistCoeff() {
    //TUI
//	return (cv::Mat_<double>(1, 4) << 0.00348238, 0.000715034, -0.0020532361718, 0.000202936);
//kitti
    return (cv::Mat_<double>(1, 4) << 0, 0, 0, 0);

}

void loop(const std::vector<std::string> &files) {
    std::shared_ptr<slam::Frame> prevFrame(nullptr);

    std::shared_ptr<slam::IFramePreprocessor> framePreprocessor = std::make_shared<slam::FramePreprocessor>();
    std::shared_ptr<slam::IFeatureDetector> featureDetector = std::make_shared<slam::FeatureDetector>();
    std::shared_ptr<slam::IFeatureMatcher> featureMatcher = std::make_shared<slam::FeatureMatcher>();
    std::shared_ptr<slam::IMatchFilterer> matchFilterer = std::make_shared<slam::MatchFilterer>();
    std::shared_ptr<slam::IPoseEstimator> poseEstimator = std::make_shared<slam::PoseEstimator>();
    std::shared_ptr<slam::IVisualizer> visualizer = std::make_shared<slam::OpenCVVisualizer>();

    visualizer->start();

    for (const std::string &path: files) {
        cv::Mat opencvFrame = cv::imread(path);
        auto currentFrame = std::make_shared<slam::Frame>(opencvFrame);
        currentFrame->K = getCameraMatrix();
        currentFrame->distCoeffs = getDistCoeff();

        if (prevFrame == nullptr) {
            currentFrame->id = 1;
        } else {
            currentFrame->id = prevFrame->id + 1;
        }

        double t = (double)getTickCount();

        framePreprocessor->preprocessFrame(currentFrame);
        currentFrame->keypoints = featureDetector->extractKeypoints(currentFrame);
        currentFrame->descriptors = featureDetector->extractDescriptors(currentFrame);
        if (prevFrame != nullptr) {
            auto matches = featureMatcher->matchKeypoints(prevFrame, currentFrame);
            matches = matchFilterer->filterMatches(matches);
            auto pose = poseEstimator->estimatePose(prevFrame, currentFrame, matches);
            visualizer->addPoint(pose);
            cv::imshow("frame", currentFrame->getAnnotatedFrame(matches));
        }

        t = 1000 * ((double)getTickCount() - t) / getTickFrequency();
		std::cout << "Frame time (ms): " << t << std::endl;

        cv::waitKey(5);
        prevFrame = currentFrame;
    }
    visualizer->stop();
}

std::vector<std::string> loadImages(const std::string &dirPath) {
    std::vector<std::string> files;
    for (const auto &entry: std::filesystem::directory_iterator(dirPath)) {
        files.push_back(entry.path().string());
    }
    std::sort(files.begin(), files.end());
    return files;
}

int main(int ac, char **av) {
//    std::vector<std::string> files = loadImages(
//            "/home/tiemajor/projects/perso/opencv/ORB_SLAM3/Examples/Datasets/TUM_VI/dataset-room1_512_16/dso/cam0/images/");
    std::vector<std::string> files = loadImages(
            "/home/tiemajor/projects/perso/opencv/slam/image_0");
    loop(files);
    return 0;
}