//
// Created by tiemajor on 11/20/21.
//

#include <FramePreprocessor.hpp>

slam::FramePreprocessor::FramePreprocessor() {

}

slam::FramePreprocessor::~FramePreprocessor() {

}

void slam::FramePreprocessor::preprocessFrame(const std::shared_ptr<slam::Frame> &frame) {
    cv::Mat newCameraMatrix = cv::getOptimalNewCameraMatrix(
            frame->K,
            frame->distCoeffs,
            frame->mat.size(),
            0
    );

    cv::Mat map1, map2;

    cv::undistort(frame->mat.clone(), frame->mat, frame->K, frame->distCoeffs, newCameraMatrix);
}

