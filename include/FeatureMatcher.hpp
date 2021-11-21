//
// Created by tiemajor on 11/19/21.
//

#ifndef SLAM_FEATUREMATCHER_HPP
#define SLAM_FEATUREMATCHER_HPP

#include <vector>
#include <opencv2/core/types.hpp>
#include "../Frame.hpp"
#include "IFeatureMatcher.hpp"

namespace slam {
	class FeatureMatcher: public IFeatureMatcher {
	public:
		FeatureMatcher();
		~FeatureMatcher();

		std::vector<std::pair<cv::KeyPoint, cv::KeyPoint>> matchKeypoints(
				const std::shared_ptr<slam::Frame> &previousFrame,
				const std::shared_ptr<slam::Frame> &currentFrame
				);

	private:
		cv::FlannBasedMatcher matcher;
	};
}

#endif //SLAM_FEATUREMATCHER_HPP
