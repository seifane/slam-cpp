#include <iostream>
#include <filesystem>
#include <vector>
#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/features2d.hpp>
#include <opencv4/opencv2/core/types.hpp>

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
cv::Ptr<cv::DescriptorMatcher> matcher = cv::BFMatcher::create();


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

std::map<std::pair<int, int>, std::vector<cv::KeyPoint>> prevKeypoints;
std::map<std::pair<int, int>, cv::Mat> prevDescriptors;

std::vector<std::pair<cv::KeyPoint, cv::KeyPoint>> matchKeypoints(const cv::Mat &frame,
					const std::map<std::pair<int, int>, std::vector<cv::KeyPoint>>& keypoints1,
					const std::map<std::pair<int, int>, cv::Mat>& descriptors1,
					const std::map<std::pair<int, int>, std::vector<cv::KeyPoint>>& keypoints2,
					const std::map<std::pair<int, int>, cv::Mat>& descriptors2) {
	std::vector<std::pair<cv::KeyPoint, cv::KeyPoint>> finalMatches;
	if (prevKeypoints.empty()) {
		return finalMatches;
	}

	int blockSizeWidth = frame.cols / 10;
	int blockSizeHeight = frame.rows / 10;

	for (int i = 0; i < 10; i++) {
		for (int j = 0; j < 10; j++) {
			auto currentBlock = std::make_pair(i, j);

			if (descriptors1.at(currentBlock).rows < 2) {
				continue;
			}

			std::vector<cv::KeyPoint> nextFrameKeypoints;
			cv::Mat nextFrameDescriptors;


			for (int ii = i - 1; ii < i + 1; ii++) {
				if (ii < 0 || ii >= 10) {
					continue;
				}
				for (int jj = j - 1; jj < j + 1; jj++) {
					if (jj < 0 || jj >= 10) {
						continue;
					}
					auto nextFrameBlock = std::make_pair(ii, jj);
					nextFrameKeypoints.insert(nextFrameKeypoints.end(),
											  keypoints2.at(nextFrameBlock).begin(),
											  keypoints2.at(nextFrameBlock).end());
					nextFrameDescriptors.push_back(descriptors2.at(nextFrameBlock));
				}
			}
			if (nextFrameDescriptors.rows < 2) {
				continue;
			}

			std::vector<std::vector<cv::DMatch>> knnMatches;
			flannedBaseMatcher.knnMatch(descriptors1.at(currentBlock), nextFrameDescriptors, knnMatches, 2);

			/*std::vector<cv::DMatch> matches;
			std::cout << descriptors1.at(currentBlock).type() << "  " << nextFrameDescriptors.type() << std::endl;
			auto desc1 = descriptors1.at(currentBlock);
			matcher->match( desc1, nextFrameDescriptors, matches);

			for (const auto &match: matches) {
				auto lastPoint = keypoints1.at(currentBlock)[match.queryIdx];
				auto currentPoint = nextFrameKeypoints[match.trainIdx];

				auto pair = std::pair(lastPoint, currentPoint);
				finalMatches.push_back(pair);
			}*/

			for (auto match: knnMatches) {
				if (match.size() == 0) {
					continue;
				}

				if (match.size() == 1 || match[0].distance < 0.6f * match[1].distance) {
					auto lastPoint = keypoints1.at(currentBlock)[match[0].queryIdx];
					auto currentPoint = nextFrameKeypoints[match[0].trainIdx];

					auto pair = std::pair(lastPoint, currentPoint);
					finalMatches.push_back(pair);
				}
			}
		}
	}
	return finalMatches;
}

std::map<std::pair<int, int>, cv::Mat> extractDescriptors(const cv::Mat &frame,
																	   std::map<std::pair<int, int>, std::vector<cv::KeyPoint>> &points) {
	std::map<std::pair<int, int>, cv::Mat> out;

	int blockSizeWidth = frame.cols / 10;
	int blockSizeHeight = frame.rows / 10;

	for (int i = 0; i < 10; i++) {
		for (int j = 0; j < 10; j++) {
			std::pair<int, int> blockCoords = std::make_pair(i, j);
			cv::Mat descriptors;
			orbDetector->compute(frame, points.at(blockCoords), descriptors);
			out[blockCoords] = descriptors;
		}
	}
	return out;
}

std::map<std::pair<int, int>, std::vector<cv::KeyPoint>> extractKeypoints(const cv::Mat &frame) {
	std::map<std::pair<int, int>, std::vector<cv::KeyPoint>> out;

	int blockSizeWidth = frame.cols / 10;
	int blockSizeHeight = frame.rows / 10;


	for (int i = 0; i < 10; i++) {
		for (int j = 0; j < 10; j++) {
			std::pair<int, int> blockCoords = std::make_pair(i, j);

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
				fastDetector->setThreshold(7);
				blockKeypoints.clear();
				fastDetector->detect(frameBlock, blockKeypoints);
			}

			cv::KeyPointsFilter::retainBest(blockKeypoints, 5);

			out[blockCoords].reserve(5);
			for (cv::KeyPoint item : blockKeypoints) {
				item.pt.x += iCorrect;
				item.pt.y += jCorrect;
				out[blockCoords].push_back(item);
			}
		}
	}



	return out;
}

void processFrame(const cv::Mat &frame) {
	auto keypoints = extractKeypoints(frame);
	auto descriptors = extractDescriptors(frame, keypoints);

	auto matches = matchKeypoints(frame, keypoints, descriptors, prevKeypoints, prevDescriptors);

	for (const auto& block : keypoints) {
		for (const auto &kp :  block.second) {
			cv::circle(frame, kp.pt, 4, cv::Scalar(255, 255, 255));
		}
	}

	for (auto match : matches) {
		cv::line(frame, match.second.pt, match.first.pt, cv::Scalar(0, 255, 255), 1, cv::LINE_4);
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
		processFrame(frame);
		cv::imshow("win", frame);
		cv::waitKey(25);
		i++;
		if (i > 50) {
		}
	}

	return 0;
}