//
// Created by tiemajor on 11/19/21.
//

#include "../include/FeatureMatcher.hpp"

slam::FeatureMatcher::FeatureMatcher() {
	this->matcher = cv::FlannBasedMatcher(cv::makePtr<cv::flann::LshIndexParams>(12, 20, 2));
}

slam::FeatureMatcher::~FeatureMatcher() = default;

std::vector<std::pair<cv::KeyPoint, cv::KeyPoint>> slam::FeatureMatcher::matchKeypoints(
		const std::shared_ptr<slam::Frame> &previousFrame,
		const std::shared_ptr<slam::Frame> &currentFrame
) {
	std::vector<std::pair<cv::KeyPoint, cv::KeyPoint>> finalMatches;

	std::vector<std::vector<cv::DMatch>> knnMatches;
	if (previousFrame->descriptors.rows == 0 || currentFrame->descriptors.rows == 0) {
		return finalMatches;
	}
	this->matcher.knnMatch(previousFrame->descriptors, currentFrame->descriptors, knnMatches, 2);

	for (auto match: knnMatches) {
		if (match.empty()) {
			continue;
		}
		if (match[0].distance < 0.85f * match[1].distance) {
			auto lastPoint = previousFrame->keypoints[match[0].queryIdx];
			auto currentPoint = currentFrame->keypoints[match[0].trainIdx];

			auto pair = std::pair(lastPoint, currentPoint);
			finalMatches.push_back(pair);
		}
	}
	return finalMatches;
}

