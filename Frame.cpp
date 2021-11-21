//
// Created by tiemajor on 11/16/21.
//

#include "Frame.hpp"

#include <utility>

slam::Frame::Frame() = default;

slam::Frame::Frame(cv::Mat mat): mat(std::move(mat))  {

}

slam::Frame::Frame(cv::Mat mat, std::vector<cv::KeyPoint> keypoints, cv::Mat descriptors):
		mat(std::move(mat)), keypoints(std::move(keypoints)), descriptors(std::move(descriptors)){

}

slam::Frame::~Frame() = default;

void slam::Frame::setMat(cv::Mat mat) {
	this->mat = std::move(mat);
}

void slam::Frame::setKeypoints(std::vector<cv::KeyPoint> keypoints) {
	this->keypoints = std::move(keypoints);
}

void slam::Frame::setDescriptors(cv::Mat descriptors) {
	this->descriptors = std::move(descriptors);
}

const cv::Mat &slam::Frame::getMat() {
	return mat;
}

std::vector<cv::KeyPoint> &slam::Frame::getKeypoints() {
	return keypoints;
}

const cv::Mat &slam::Frame::getDescriptors() {
	return descriptors;
}

cv::Mat slam::Frame::getAnnotatedFrame(const std::vector<std::pair<cv::KeyPoint, cv::KeyPoint>> &matches) {
	cv::Mat annotatedFrame = this->mat.clone();

	for (auto match : matches) {
		cv::circle(annotatedFrame, match.second.pt, 2, cv::Scalar(255, 255, 255));
//			cv::circle(displayFrame, match.second.pt, 4, cv::Scalar(0, 0, 255));
		cv::line(annotatedFrame, match.first.pt, match.second.pt, cv::Scalar(0, 0, 255), 1, cv::LINE_4);
	}
	return annotatedFrame;
}

