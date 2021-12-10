//
// Created by tiemajor on 11/21/21.
//

#ifndef SLAM_HELPERS_HPP
#define SLAM_HELPERS_HPP

#include <opencv4/opencv2/core.hpp>

namespace slam {
    cv::Mat getMatFromPoint(const cv::Point2f &point) {
        cv::Mat out(3, 1, CV_64F);
        out.at<double>(0, 0) = point.x;
        out.at<double>(1, 0) = point.y;
        out.at<double>(2, 0) = 1;
        return out;
    }
}
#endif //SLAM_HELPERS_HPP
