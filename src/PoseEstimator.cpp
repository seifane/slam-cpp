//
// Created by tiemajor on 11/20/21.
//

#include "PoseEstimator.hpp"

slam::PoseEstimator::PoseEstimator() {

}

slam::PoseEstimator::~PoseEstimator() {

}

cv::Point3d slam::PoseEstimator::estimatePose(const std::shared_ptr <Frame> &previousFrame,
                                       const std::shared_ptr <Frame> &currentFrame,
                                       std::vector<std::pair<cv::KeyPoint, cv::KeyPoint>> &matches) {
    std::vector<cv::Point2f> pointsPrevFrame;
    std::vector<cv::Point2f> udPointsPrevFrame;
    std::vector<cv::Point2f> pointsCurrentFrame;
    std::vector<cv::Point2f> udPointsCurrentFrame;

    for (const auto &match : matches) {
        pointsPrevFrame.push_back(match.first.pt);
        pointsCurrentFrame.push_back(match.second.pt);
    }

    if (pointsPrevFrame.empty() || pointsCurrentFrame.empty()) {
        throw std::runtime_error("No points");
    }

//	cv::undistortPoints(pointsPrevFrame, udPointsPrevFrame, getCameraMatrix(), getDistCoeff());
//	cv::undistortPoints(pointsCurrentFrame, udPointsCurrentFrame, getCameraMatrix(), getDistCoeff());


    cv::Mat mask;

    cv::Mat F = cv::findFundamentalMat(pointsPrevFrame, pointsCurrentFrame, mask);
    cv::Mat E = currentFrame->K.t() * F * currentFrame->K;
    cv::Mat R;
    cv::Mat T;

    std::vector<std::pair<cv::KeyPoint, cv::KeyPoint>> filteredMatches;
    for (int i = 0; i < mask.rows; i++) {
        auto res = mask.at<bool>(i, 0);
        if (res) {
            filteredMatches.push_back(matches[i]);
            udPointsPrevFrame.push_back(pointsPrevFrame[i]);
            udPointsCurrentFrame.push_back(pointsCurrentFrame[i]);
        }
    }

    matches.clear();
    for (const auto &i : filteredMatches) {
        matches.push_back(i);
    }


    cv::recoverPose(E, udPointsPrevFrame, udPointsCurrentFrame, currentFrame->K, R, T);
    cv::SVD svd(E, cv::SVD::FULL_UV);
//	std::cout << " U = " << svd.u << " w = " << svd.w << " Vt = " << svd.vt << std::endl;
    std::cout << "Err : " << (svd.w.at<double>(0, 0) - svd.w.at<double>(0, 1)) / svd.w.at<double>(0, 1) * 100 << std::endl;
//	std::cout << "R = " << R << std::endl << "T = " << T << std::endl;
//	std::cout << "isValid = " << CheckCoherentRotation(R) << std::endl;

    if (!previousFrame->R.empty()) {
        currentFrame->T = previousFrame->T + previousFrame->R * T;
        currentFrame->R = previousFrame->R * R;
    } else {
        currentFrame->R = R.clone();
        currentFrame->T = T.clone();
    }

    return cv::Point3d(currentFrame->T.at<double>(0, 0), currentFrame->T.at<double>(0, 1), currentFrame->T.at<double>(0, 2));
}
