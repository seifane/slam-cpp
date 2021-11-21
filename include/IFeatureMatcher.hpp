//
// Created by tiemajor on 11/19/21.
//

#ifndef SLAM_IFEATUREMATCHER_HPP
#define SLAM_IFEATUREMATCHER_HPP
namespace slam {
	class IFeatureMatcher {
	public:
		virtual ~IFeatureMatcher() = default;

		virtual std::vector<std::pair<cv::KeyPoint, cv::KeyPoint>> matchKeypoints(
                const std::shared_ptr<slam::Frame> &previousFrame,
                const std::shared_ptr<slam::Frame> &currentFrame
        ) = 0;
	};
}
#endif //SLAM_IFEATUREMATCHER_HPP
