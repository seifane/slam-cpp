#include <iostream>
#include <filesystem>
#include <vector>
#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/features2d.hpp>
#include <opencv4/opencv2/core/types.hpp>

#include "vfc.h"

std::vector<std::string> loadImages(const std::string &dirPath) {
	std::vector<std::string> files;
	for (const auto &entry: std::filesystem::directory_iterator(dirPath)) {
		files.push_back(entry.path().string());
	}
	std::sort(files.begin(), files.end());
	return files;
}

auto orbDetector = cv::ORB::create(1000, 1.2, 8, 15);
auto fastDetector = cv::FastFeatureDetector::create();
auto flannedBaseMatcher = cv::FlannBasedMatcher(cv::makePtr<cv::flann::LshIndexParams>(12, 20, 2));
cv::Ptr<cv::DescriptorMatcher> bfMatcher = cv::BFMatcher::create();


std::pair<int, int> getAdjustedCoordsForBlock(const cv::Mat &frame, int splitFactor, int x, int y, int blockX, int blockY) {
	int blockSizeWidth = frame.cols / splitFactor;
	int blockSizeHeight = frame.rows / splitFactor;

	int iCorrect = blockX * blockSizeWidth;
	if (iCorrect > frame.cols - blockSizeWidth) {
		iCorrect = frame.cols - blockSizeWidth;
	}
	int jCorrect = blockY * blockSizeHeight;
	if (jCorrect > frame.rows - blockSizeHeight) {
		jCorrect = frame.rows - blockSizeHeight;
	}

	return std::make_pair(x + iCorrect, y + jCorrect);
}

std::vector<cv::KeyPoint> prevKeypoints;
cv::Mat prevDescriptors;

std::vector<std::pair<cv::KeyPoint, cv::KeyPoint>> matchKeypoints(const cv::Mat &frame,
					const std::vector<cv::KeyPoint>& keypoints1,
					const cv::Mat& descriptors1,
					const std::vector<cv::KeyPoint>& keypoints2,
					const cv::Mat& descriptors2) {
	std::vector<std::pair<cv::KeyPoint, cv::KeyPoint>> finalMatches;
	if (prevKeypoints.empty()) {
		return finalMatches;
	}

	std::vector<std::vector<cv::DMatch>> knnMatches;
	flannedBaseMatcher.knnMatch(descriptors1, descriptors2, knnMatches, 2);

	for (auto match: knnMatches) {
		if (match.size() == 0) {
			continue;
		}
		if (match[0].distance < 0.8f * match[1].distance) {
			auto lastPoint = keypoints1[match[0].queryIdx];
			auto currentPoint = keypoints2[match[0].trainIdx];

			auto pair = std::pair(lastPoint, currentPoint);
			finalMatches.push_back(pair);
		}
	}
	return finalMatches;
}

std::vector<std::pair<cv::KeyPoint, cv::KeyPoint>> matchKeypointsBf(const cv::Mat &frame,
																  const std::vector<cv::KeyPoint>& keypoints1,
																  const cv::Mat& descriptors1,
																  const std::vector<cv::KeyPoint>& keypoints2,
																  const cv::Mat& descriptors2) {
	std::vector<std::pair<cv::KeyPoint, cv::KeyPoint>> finalMatches;
	if (prevKeypoints.empty()) {
		return finalMatches;
	}

	std::vector<cv::DMatch> knnMatches;
	bfMatcher->match(descriptors1, descriptors2, knnMatches);

	for (auto match: knnMatches) {
		auto lastPoint = keypoints1[match.queryIdx];
		auto currentPoint = keypoints2[match.trainIdx];

		auto pair = std::pair(lastPoint, currentPoint);
		finalMatches.push_back(pair);
	}
	return finalMatches;
}

cv::Mat extractDescriptors(const cv::Mat &frame, std::vector<cv::KeyPoint> &points) {
	cv::Mat out;
	orbDetector->compute(frame, points, out);
	return out;
}

std::vector<std::pair<cv::KeyPoint, cv::KeyPoint>> filterVFC(
		const std::vector<std::pair<cv::KeyPoint, cv::KeyPoint>> &matches
		) {
	std::vector<cv::Point2f> X;
	std::vector<cv::Point2f> Y;

	for (const auto &match: matches) {
		X.push_back(match.first.pt);
		Y.push_back(match.second.pt);
	}

	VFC vfc;

	vfc.setData(X, Y);
	vfc.optimize();
	std::vector<int> matchIdxs = vfc.obtainCorrectMatch();

	std::vector<std::pair<cv::KeyPoint, cv::KeyPoint>> out;

	for (int matchIdx: matchIdxs) {
		out.push_back(matches[matchIdx]);
	}
	return out;
}

std::vector<cv::KeyPoint> extractKeypoints(const cv::Mat &frame) {
	std::vector<cv::KeyPoint> out;
	int divFactor = 10;

	int blockSizeWidth = frame.cols / divFactor;
	int blockSizeHeight = frame.rows / divFactor;


	for (int i = 0; i < divFactor; i++) {
		for (int j = 0; j < divFactor; j++) {
			std::vector<cv::KeyPoint> blockKeypoints;

			int iCorrect = i * blockSizeWidth;
			if (iCorrect > frame.cols - blockSizeWidth) {
				iCorrect = frame.cols - blockSizeWidth;
			}
			int jCorrect = j * blockSizeHeight;
			if (jCorrect > frame.rows - blockSizeHeight) {
				jCorrect = frame.rows - blockSizeHeight;
			}

			cv::Rect chunk = cv::Rect(iCorrect, jCorrect, blockSizeWidth, blockSizeHeight);
//			cv::rectangle(frame, chunk, cv::Scalar(255, 255, 255));
			cv::Mat frameBlock = frame(chunk).clone();

			fastDetector->setThreshold(20);
			fastDetector->detect(frameBlock, blockKeypoints);
			if (blockKeypoints.size() < 5) {
				fastDetector->setThreshold(6);
				blockKeypoints.clear();
				fastDetector->detect(frameBlock, blockKeypoints);
			}

			cv::KeyPointsFilter::retainBest(blockKeypoints, 4);

			for (cv::KeyPoint item : blockKeypoints) {
				item.pt.x += iCorrect;
				item.pt.y += jCorrect;
				out.push_back(item);
			}
		}
	}

	return out;
}

void processFrame(const cv::Mat &frame) {
	auto keypoints = extractKeypoints(frame);
	auto descriptors = extractDescriptors(frame, keypoints);

	auto matches = matchKeypoints(frame, keypoints, descriptors, prevKeypoints, prevDescriptors);
//	auto matches = matchKeypointsBf(frame, keypoints, descriptors, prevKeypoints, prevDescriptors);

	auto filteredMatches = filterVFC(matches);

	for (const auto& kp : keypoints) {
//			cv::circle(frame, kp.pt, 4, cv::Scalar(255, 255, 255));
	}

	for (auto match : filteredMatches) {
		cv::circle(frame, match.first.pt, 4, cv::Scalar(255, 255, 255));
		cv::line(frame, match.second.pt, match.first.pt, cv::Scalar(0, 0, 255), 1, cv::LINE_4);
	}
	prevKeypoints = keypoints;
	prevDescriptors = descriptors;
}

int main (int ac, char **av) {
	std::vector<std::string> files = loadImages(
			"/home/tiemajor/projects/perso/opencv/ORB_SLAM3/Examples/Datasets/TUM_VI/dataset-room1_512_16/dso/cam0/images/");

	int i = 0;
	for (const auto &path: files) {
		auto frame = cv::imread(path);

		double t = (double)getTickCount();

		processFrame(frame);

		t = 1000 * ((double)getTickCount() - t) / getTickFrequency();
		std::cout << "Frame time (ms): " << t << std::endl;
		if (t > 33.33) {
			std::cout << "§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§" << std::endl;
		}

		cv::imshow("win", frame);
		cv::waitKey(25);
		i++;
		if (i > 50) {
		}
	}

	return 0;
}