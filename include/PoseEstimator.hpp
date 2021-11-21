//
// Created by tiemajor on 11/20/21.
//

#ifndef SLAM_POSEESTIMATOR_HPP
#define SLAM_POSEESTIMATOR_HPP

#include "IPoseEstimator.hpp"
#include "../Frame.hpp"

namespace slam {
    class PoseEstimator: public IPoseEstimator {
    public:
        PoseEstimator();
        ~PoseEstimator();

        cv::Point3d estimatePose(
                const std::shared_ptr<Frame> &prevFrame,
                const std::shared_ptr<Frame> &currentFrame,
                std::vector<std::pair<cv::KeyPoint, cv::KeyPoint>> &matches
                );
    };
}

#endif //SLAM_POSEESTIMATOR_HPP
