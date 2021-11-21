//
// Created by tiemajor on 11/20/21.
//

#ifndef SLAM_IPOSEESTIMATOR_HPP
#define SLAM_IPOSEESTIMATOR_HPP

#include <opencv4/opencv2/core.hpp>
#include "Frame.hpp"

namespace slam {
    class IPoseEstimator {
    public:
        virtual ~IPoseEstimator() = default;

        virtual cv::Point3d estimatePose(
                const std::shared_ptr<Frame> &prevFrame,
                const std::shared_ptr<Frame> &currentFrame,
                std::vector<std::pair<cv::KeyPoint, cv::KeyPoint>> &matches
                ) = 0;
    };
}

#endif //SLAM_IPOSEESTIMATOR_HPP
