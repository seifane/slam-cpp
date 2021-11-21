//
// Created by tiemajor on 11/19/21.
//

#include <vector>
#include <opencv2/core/types.hpp>
#include "../Frame.hpp"
#include "IFeatureDetector.hpp"

#ifndef SLAM_FEATUREDETECTOR_HPP
#define SLAM_FEATUREDETECTOR_HPP

namespace slam {
	class FeatureDetector: public IFeatureDetector {
	public:
		FeatureDetector();
		~FeatureDetector();

		std::vector<cv::KeyPoint> extractKeypoints(const std::shared_ptr<slam::Frame>& frame);
        cv::Mat extractDescriptors(const std::shared_ptr<slam::Frame> &frame);
	private:
		cv::Ptr<cv::FastFeatureDetector> detector;
        cv::Ptr<cv::ORB> _orbDetector;
	};
}

#endif //SLAM_FEATUREDETECTOR_HPP
