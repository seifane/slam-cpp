//
// Created by tiemajor on 11/19/21.
//

#include "../include/FeatureDetector.hpp"

slam::FeatureDetector::FeatureDetector() {
	this->detector = cv::FastFeatureDetector::create();
    this->_orbDetector = cv::ORB::create(1000, 1.2, 8, 15);
}

slam::FeatureDetector::~FeatureDetector() = default;

std::vector<cv::KeyPoint> slam::FeatureDetector::extractKeypoints(const std::shared_ptr<slam::Frame> &frame) {
	std::vector<cv::KeyPoint> keypoints;

	int divFactor = 10;

	int blockSizeWidth = frame->mat.cols / divFactor;
	int blockSizeHeight = frame->mat.rows / divFactor;

	for (int i = 0; i < divFactor; i++) {
		for (int j = 0; j < divFactor; j++) {
			std::vector<cv::KeyPoint> blockKeypoints;

			int iCorrect = i * blockSizeWidth;
			if (iCorrect > frame->mat.cols - blockSizeWidth) {
				iCorrect = frame->mat.cols - blockSizeWidth;
			}
			int jCorrect = j * blockSizeHeight;
			if (jCorrect > frame->mat.rows - blockSizeHeight) {
				jCorrect = frame->mat.rows - blockSizeHeight;
			}

			cv::Rect chunk = cv::Rect(iCorrect, jCorrect, blockSizeWidth, blockSizeHeight);
//			cv::rectangle(frame, chunk, cv::Scalar(255, 255, 255));
			cv::Mat frameBlock = frame->getMat()(chunk).clone();

			this->detector->setThreshold(20);
			this->detector->detect(frameBlock, blockKeypoints);
			if (blockKeypoints.size() < 5) {
				this->detector->setThreshold(6);
				blockKeypoints.clear();
				this->detector->detect(frameBlock, blockKeypoints);
			}

			cv::KeyPointsFilter::retainBest(blockKeypoints, 4);

			for (cv::KeyPoint item : blockKeypoints) {
				item.pt.x += iCorrect;
				item.pt.y += jCorrect;
				keypoints.push_back(item);
			}
		}
	}
	return keypoints;
}

cv::Mat slam::FeatureDetector::extractDescriptors(const std::shared_ptr<slam::Frame> &frame) {
    cv::Mat out;
    this->_orbDetector->compute(frame->getMat(), frame->getKeypoints(), out);
    return out;
}



