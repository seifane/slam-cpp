#include <iostream>
#include <filesystem>
#include <vector>
#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/features2d.hpp>
#include <opencv4/opencv2/core/types.hpp>

#include "Ransac.hpp"


auto orbDectector = cv::ORB::create(1000, 1.2, 8, 15);
auto fastDetector = cv::FastFeatureDetector::create();
cv::Ptr<cv::DescriptorMatcher> matcher = cv::BFMatcher::create();

//cv::Ptr<cv::DescriptorMatcher> flannMatcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);

std::vector<cv::KeyPoint> prevKeypoints;
cv::Mat prevDescriptors;


std::vector<std::string> loadImages(const std::string &dirPath) {
	std::vector<std::string> files;
	for (const auto &entry: std::filesystem::directory_iterator(dirPath)) {
		files.push_back(entry.path().string());
	}
	std::sort(files.begin(), files.end());
	return files;
}

void extractFeatures(const cv::Mat &frame) {
	std::vector<cv::KeyPoint> keypoints;
	cv::Mat descriptors;
	auto frameCopy = frame.clone();

	auto flannedBaseMatcher = cv::FlannBasedMatcher(cv::makePtr<cv::flann::LshIndexParams>(12, 20, 2));


	int blockSizeWidth = frame.cols / 1;
	int blockSizeHeight = frame.rows / 1;

	for (int i = 0; i < blockSizeWidth - 1; i++) {
		for (int j = 0; j < blockSizeHeight - 1; j++) {
			std::vector<cv::KeyPoint> localKeypoints;

			int iCorrect = i * blockSizeWidth;
			if (iCorrect > frame.cols - blockSizeWidth) {
				iCorrect = frame.cols - blockSizeWidth;
			}
			int jCorrect = j * blockSizeHeight;
			if (jCorrect > frame.rows - blockSizeHeight) {
				jCorrect = frame.rows - blockSizeHeight;
			}

			auto chunk = cv::Rect(iCorrect, jCorrect, blockSizeWidth, blockSizeHeight);

            cv::rectangle(frame, chunk, cv::Scalar(255, 255, 255));
			cv::Mat frameBlock = frameCopy(chunk).clone();

			fastDetector->setThreshold(20);
			fastDetector->detect(frameBlock, localKeypoints);
			if (localKeypoints.size() < 4) {
				fastDetector->setThreshold(7);
				localKeypoints.clear();
				fastDetector->detect(frameBlock, localKeypoints);
			}

			int addedCount = 0;

			/*cv::KeyPointsFilter::retainBest(localKeypoints, 2);

			for (auto keypoint: localKeypoints) {
				keypoint.pt.x = keypoint.pt.x + iCorrect;
				keypoint.pt.y = keypoint.pt.y + jCorrect;
				keypoints.push_back(keypoint);

				addedCount++;
				if (addedCount > 5) {
					break;
				}
			}*/
		}
	}
	orbDectector->compute(frameCopy, keypoints, descriptors);
//    cv::drawKeypoints(frame, keypoints, frame);

	if (!prevKeypoints.empty() && false) {
		std::vector<std::vector<cv::DMatch>> knnMatches;
		flannedBaseMatcher.knnMatch(prevDescriptors, descriptors, knnMatches, 2);

		float totalX = 0;
		float totalY = 0;
		int addedCount = 0;

		std::vector<std::pair<cv::KeyPoint, cv::KeyPoint>> filteredMatches;
		std::vector<cv::Point> matchVectors;

		for (auto match: knnMatches) {
			if (match.size() != 2) {
				continue;
			}
			if (match[0].distance < 0.8f * match[1].distance) {
				auto lastPoint = prevKeypoints[match[0].queryIdx];
				auto currentPoint = keypoints[match[0].trainIdx];

				float vx = lastPoint.pt.x - currentPoint.pt.x;
				float vy = lastPoint.pt.y - currentPoint.pt.y;

				totalX += vx;
				totalY += vy;

				matchVectors.emplace_back(vx, vy);

				addedCount++;

				auto pair = std::pair(currentPoint, lastPoint);
				filteredMatches.emplace_back(pair);
				auto color = cv::Scalar(255, 255, 255);
				cv::circle(frame, pair.first.pt, 4, color);
				cv::line(frame, pair.second.pt, pair.first.pt, color, 1, cv::LINE_4);
			}
		}

		/*auto bestInliners = executeRANSAC(matchVectors, .8f, 1000);

		for (int i = 0; i < filteredMatches.size(); i++) {
			auto m = filteredMatches[i];

			cv::Scalar color;
			if (bestInliners.count(i)) {
				color = cv::Scalar(255, 255, 255);
				cv::circle(frame, m.first.pt, 4, color);
				cv::line(frame, m.second.pt, m.first.pt, color, 1, cv::LINE_4);
			} else {
				color = cv::Scalar(0, 0, 0);
			}


		}

		for (auto index: bestInliners) {

		}*/





		/* float meanX, meanY;
		 meanX = totalX / addedCount;
		 meanY = totalY / addedCount;

		 std::cout << "meanX = " << meanX << " Mean Y " << meanY << std::endl;


		 float marg = .1f;

		 for (auto kpsPair : filteredMatches) {
			 float vx = kpsPair.first.pt.x - kpsPair.second.pt.x;
			 float vy = kpsPair.first.pt.y - kpsPair.second.pt.y;
			 if (vx > meanX - (meanX * marg) && vx < meanX + (meanX * marg) && vy > meanY - (meanY * marg) && vy < meanY + (meanY * marg)) {

			 }
		 }*/


		/*std::vector<cv::DMatch> matches;
		matcher->match(prevDescriptors, descriptors, matches);
		for (auto match: matches) {
			if (match.distance > 400) {
				continue;
			}
			auto lastPoint = prevKeypoints[match.queryIdx];
			auto currentPoint = keypoints[match.trainIdx];

			cv::circle(frame, currentPoint.pt, 4, cv::Scalar(255, 0, 0));
			cv::line(frame, lastPoint.pt, currentPoint.pt, cv::Scalar(255, 255, 255), 1, cv::LINE_4);
		}*/
	}

	prevKeypoints = keypoints;
	prevDescriptors = descriptors;
}

cv::Mat getCameraMatrix() {
	float fx = 517.306408;
	float fy = 516.469215;
	float cx = 318.643040;
	float cy = 255.313989;

	return (cv::Mat_<float>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
}

cv::Mat getDistCoeff() {
	float k1 = 0.262383;
	float k2 = -0.953104;
	float p1 = -0.005358;
	float p2 = 0.002628;
	float k3 = 1.163314;

	return (cv::Mat_<float>(1, 5) << k1, k2, p1, p2, k3);
}

void computeImageBounds(const cv::Mat &frame) {
	cv::Mat cameraMatrix = getCameraMatrix();
	cv::Mat distCoeffs = getDistCoeff();

	cv::Mat mat(4,2,CV_32F);
	mat.at<float>(0,0)=0.0; mat.at<float>(0,1)=0.0;
	mat.at<float>(1,0)=frame.cols; mat.at<float>(1,1)=0.0;
	mat.at<float>(2,0)=0.0; mat.at<float>(2,1)=frame.rows;
	mat.at<float>(3,0)=frame.cols; mat.at<float>(3,1)=frame.rows;

	// Undistort corners
	mat=mat.reshape(2);
	cv::undistortPoints(mat,mat,cameraMatrix,distCoeffs,cv::Mat(),cameraMatrix);
	mat=mat.reshape(1);

	float mnMinX = std::min(floor(mat.at<float>(0,0)),floor(mat.at<float>(2,0)));
	float mnMaxX = std::max(ceil(mat.at<float>(1,0)),ceil(mat.at<float>(3,0)));
	float mnMinY = std::min(floor(mat.at<float>(0,1)),floor(mat.at<float>(1,1)));
	float mnMaxY = std::max(ceil(mat.at<float>(2,1)),ceil(mat.at<float>(3,1)));

	cv::line(frame, cv::Point(mnMinX,  mnMinY), cv::Point(mnMaxX, mnMaxY), cv::Scalar(255, 255, 0));
//	cv::rectangle(frame, cv::Rect(mnMinX, mnMinY, 10, 10), cv::Scalar(255, 255, 255));
}

void undistorneKeypoints(std::vector<cv::KeyPoint> keypoints) {

}

void processFrame(const cv::Mat &frame) {
	std::vector<cv::KeyPoint> kps;
	cv::Mat descriptors;

	float fx = 517.306408;
	float fy = 516.469215;
	float cx = 318.643040;
	float cy = 255.313989;

	float k1 = 0.262383;
	float k2 = -0.953104;
	float p1 = -0.005358;
	float p2 = 0.002628;
	float k3 = 1.163314;


	cv::Mat cameraMatrix = (cv::Mat_<float>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
	cv::Mat distCoeffs = (cv::Mat_<float>(1, 5) << k1, k2, p1, p2, k3);
	cv::Mat newCameraMatrix = cv::getOptimalNewCameraMatrix(
			cameraMatrix,
			distCoeffs,
			frame.size(),
			0
	);

	cv::Mat map1, map2;


	cv::Mat undistortedImage;


//	cv::undistort(frame, undistortedImage, cameraMatrix, distCoeffs, newCameraMatrix);
	cv::Size imageSize(cv::Size(frame.cols,frame.rows));

	cv::initUndistortRectifyMap(cameraMatrix,
								distCoeffs,
								cv::Mat(),
								cv::getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, imageSize, 1, imageSize, 0),
								imageSize,
								CV_16SC2,
								map1,
								map2);
	cv::remap(frame, undistortedImage, map1, map2, cv::INTER_LINEAR);

//	computeImageBounds(undistortedImage);
	extractFeatures(undistortedImage);

//    siftDectector->detectAndCompute(frame, cv::noArray(), kps, descriptors);


	/* if (!prevKeypoints.empty()) {
		 std::vector<std::vector<cv::DMatch>> matches;

		 flannMatcher.knnMatch(prevDescriptors, descriptors, matches, 2);
		 for (auto match : matches) {
			 if (match[0].distance < 0.3 * match[1].distance) {
				 auto lastPoint = prevKeypoints[match[0].queryIdx];
				 auto currentPoint = kps[match[0].trainIdx];

				 cv::line(frame, lastPoint.pt, currentPoint.pt, cv::Scalar(255, 255, 255), 1, cv::LINE_4);
			 }
		 }
	 }*/

	/*orbDectector->detectAndCompute(undistortedImage, cv::noArray(), kps, descriptors);

	if (!prevKeypoints.empty()) {
		std::vector<cv::DMatch> matches;
		matcher->match(prevDescriptors, descriptors, matches);
		for (auto match: matches) {
			if (match.distance > 64) {
				continue;
			}
			auto lastPoint = prevKeypoints[match.queryIdx];
			auto currentPoint = kps[match.trainIdx];

			cv::circle(undistortedImage, currentPoint.pt, 4, cv::Scalar(255, 0, 0));
			cv::line(undistortedImage, lastPoint.pt, currentPoint.pt, cv::Scalar(255, 255, 255), 1, cv::LINE_4);
		}
	}

	//cv::drawKeypoints(frame, kps, frame);
	prevKeypoints = kps;
	prevDescriptors = descriptors;*/
	cv::imshow("undistorted", undistortedImage);
}

int main() {
	std::cout << "Hello, World!" << std::endl;

	std::vector<std::string> files = loadImages(
			"/home/tiemajor/projects/perso/opencv/ORB_SLAM3/Examples/Datasets/TUM_VI/dataset-room1_512_16/dso/cam0/images/");

	int i = 0;
	for (const auto &path: files) {
		auto frame = cv::imread(path);
		processFrame(frame);
		//cv::imshow("win", frame);
		cv::waitKey(25);
		i++;
		if (i > 50) {
		}
	}

	return 0;
}
