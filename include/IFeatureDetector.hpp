//
// Created by tiemajor on 11/20/21.
//

#ifndef SLAM_IFEATUREDETECTOR_HPP
#define SLAM_IFEATUREDETECTOR_HPP
namespace slam {
    class IFeatureDetector {
    public:
        virtual ~IFeatureDetector() = default;

        virtual std::vector<cv::KeyPoint> extractKeypoints(const std::shared_ptr<slam::Frame>& frame) = 0;
        virtual cv::Mat extractDescriptors(const std::shared_ptr<slam::Frame> &frame) = 0;
    };
}
#endif //SLAM_IFEATUREDETECTOR_HPP
