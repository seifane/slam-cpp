#include <iostream>
#include <filesystem>
#include <vector>
#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/features2d.hpp>
#include <opencv4/opencv2/core/types.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/viz.hpp>

#include "Frame.hpp"
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
std::vector<cv::Affine3d> points;

std::shared_ptr<slam::Frame> prevFrame;

cv::Vec3f cumPos;

auto vizz = cv::makePtr<cv::viz::Viz3d>("win");


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

std::vector<std::pair<cv::KeyPoint, cv::KeyPoint>> matchKeypoints(const std::shared_ptr<slam::Frame> &previousFrame,
																  const std::shared_ptr<slam::Frame> &currentFrame) {
	std::vector<std::pair<cv::KeyPoint, cv::KeyPoint>> finalMatches;

	std::vector<std::vector<cv::DMatch>> knnMatches;
	if (previousFrame->getDescriptors().rows == 0 || currentFrame->getDescriptors().rows == 0) {
		return finalMatches;
	}
	flannedBaseMatcher.knnMatch(previousFrame->getDescriptors(), currentFrame->getDescriptors(), knnMatches, 2);

	for (auto match: knnMatches) {
		if (match.empty()) {
			continue;
		}
		if (match[0].distance < 0.85f * match[1].distance) {
			auto lastPoint = previousFrame->getKeypoints()[match[0].queryIdx];
			auto currentPoint = currentFrame->getKeypoints()[match[0].trainIdx];

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

cv::Mat extractDescriptors(const std::shared_ptr<slam::Frame> &frame) {
	cv::Mat out;
	orbDetector->compute(frame->getMat(), frame->getKeypoints(), out);
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

std::vector<cv::KeyPoint> extractKeypoints(const std::shared_ptr<slam::Frame>& frame) {
	std::vector<cv::KeyPoint> out;
	int divFactor = 10;

	int blockSizeWidth = frame->getMat().cols / divFactor;
	int blockSizeHeight = frame->getMat().rows / divFactor;


	for (int i = 0; i < divFactor; i++) {
		for (int j = 0; j < divFactor; j++) {
			std::vector<cv::KeyPoint> blockKeypoints;

			int iCorrect = i * blockSizeWidth;
			if (iCorrect > frame->getMat().cols - blockSizeWidth) {
				iCorrect = frame->getMat().cols - blockSizeWidth;
			}
			int jCorrect = j * blockSizeHeight;
			if (jCorrect > frame->getMat().rows - blockSizeHeight) {
				jCorrect = frame->getMat().rows - blockSizeHeight;
			}

			cv::Rect chunk = cv::Rect(iCorrect, jCorrect, blockSizeWidth, blockSizeHeight);
//			cv::rectangle(frame, chunk, cv::Scalar(255, 255, 255));
			cv::Mat frameBlock = frame->getMat()(chunk).clone();

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

cv::Mat getCameraMatrix() {

	/*
	 * TUIdouble fx = 190.9784771;
	double fy = 190.9733070;
	double cx = 254.9317060;
	double cy = 256.8974428;*/

	double fx = 718.856;
	double fy = 718.856;
	double cx = 607.1928;
	double cy = 185.2157;

	cv::Mat K(3, 3, CV_64F);
	K.at<double>(0, 0) = fx;
	K.at<double>(0, 1) = 0;
	K.at<double>(0, 2) = cx;
	K.at<double>(1, 0) = 0;
	K.at<double>(1, 1) = fy;
	K.at<double>(1, 2) = cy;
	K.at<double>(2, 0) = 0;
	K.at<double>(2, 1) = 0;
	K.at<double>(2, 2) = 1;
	return K;

//	return (cv::Mat_<float>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
}

cv::Mat getDistCoeff() {
	float k1 = 0.262383;
	float k2 = -0.953104;
	float p1 = -0.005358;
	float p2 = 0.002628;
	float k3 = 1.163314;

//	return (cv::Mat_<double>(1, 4) << 0.00348238, 0.000715034, -0.0020532361718, 0.000202936);
	return (cv::Mat_<double>(1, 4) << 0, 0, 0, 0);

}

bool CheckCoherentRotation(const cv::Mat_<double>& R) {

	if (fabsf(determinant(R)) - 1.0 > 1e-07) {

		cerr << "rotation matrix is invalid" << endl;

		return false;

	}

	return true;

}

bool degeneracyCheck(
		const std::vector<std::pair<cv::KeyPoint, cv::KeyPoint>> &matches,
		const cv::Mat &R,
		const cv::Mat &T
		) {
	for (const auto &match: matches) {
		cv::Mat secondPtMat = (cv::Mat_<double>(3, 1) << match.second.pt.x, match.second.pt.y,  1);
		cv::Mat firstz = ((R.row(0) - match.second.pt.x * R.row(2)) * (T) /
				(R.row(0) - match.second.pt.x * R.row(2))) * (secondPtMat);
		cv::Mat first3dPoint = (cv::Mat_<double>(3, 1) << match.first.pt.x * firstz.at<double>(0, 0), match.second.pt.x * firstz.at<double>(0, 0), firstz.at<double>(0, 0));
		cv::Mat second3dPoint = R.t() * first3dPoint - R.t() * T;

		std::cout << firstz << first3dPoint << second3dPoint << std::endl;

		if (first3dPoint.at<double>(0, 2) < 0 || second3dPoint.at<double>(0, 2) < 0) {
			return false;
		}
	}
	return true;
}

void getRTfromE(const cv::Mat &E, cv::Mat &R, cv::Mat &T, const std::vector<std::pair<cv::KeyPoint, cv::KeyPoint>> &matches) {
	cv::Mat U, S, Vt;
	cv::SVD svd(E, SVD::FULL_UV);
	std::cout << " U = " << svd.u << " w = " << svd.w << " Vt = " << svd.vt << std::endl;
	std::cout << "Err : " << (svd.w.at<double>(0, 0) - svd.w.at<double>(0, 1)) / svd.w.at<double>(0, 1) * 100 << std::endl;


	cv::Mat W = Mat::zeros(3, 3, CV_64F);
	W.at<double>(0,1) = -1;
	W.at<double>(1,0) =  1;
	W.at<double>(2,2) =  1;

	cv::Mat R1= svd.u * W * svd.vt;
	cv::Mat R2= svd.u * W.t() * svd.vt;
	if(determinant(R1) < 0)
		R1 = -1 * R1;
	if(determinant(R2) < 0)
		R2 = -1 * R2;

	R = svd.u * W.t() * svd.vt;
	T = svd.u.col(2);

	std::cout << "R1" << R1 << std::endl << "R2" << R2 << std::endl << "T" << T << std::endl;

	R = R;
	T = T;

	/*if (!degeneracyCheck(matches, R, T)) {
		R = svd.u * W * svd.vt;
		if (!degeneracyCheck(matches, R, T)) {
			T = -svd.u.col(2);
			if (!degeneracyCheck(matches, R, T)) {
				R = svd.u * W * svd.vt;
				if (!degeneracyCheck(matches, R, T)) {
					std::cout << "We didn't find anything" << std::endl;
				}
			}
		}
	}*/

	return;
}

void computePosition(const std::shared_ptr<slam::Frame> &previousFrame,
					 const std::shared_ptr<slam::Frame> &frame,
					 std::vector<std::pair<cv::KeyPoint, cv::KeyPoint>> &matches) {
	std::vector<cv::Point2f> pointsPrevFrame;
	std::vector<cv::Point2f> udPointsPrevFrame;
	std::vector<cv::Point2f> pointsCurrentFrame;
	std::vector<cv::Point2f> udPointsCurrentFrame;

	for (const auto &match : matches) {
		pointsPrevFrame.push_back(match.first.pt);
		pointsCurrentFrame.push_back(match.second.pt);
	}

	if (pointsPrevFrame.empty() || pointsCurrentFrame.empty()) {
		return;
	}

//	cv::undistortPoints(pointsPrevFrame, udPointsPrevFrame, getCameraMatrix(), getDistCoeff());
//	cv::undistortPoints(pointsCurrentFrame, udPointsCurrentFrame, getCameraMatrix(), getDistCoeff());


	cv::Mat mask;

	cv::Mat F = cv::findFundamentalMat(pointsPrevFrame, pointsCurrentFrame, mask);
	cv::Mat E = getCameraMatrix().t() * F * getCameraMatrix();
	cv::Mat R;
	cv::Mat T;

	std::vector<std::pair<cv::KeyPoint, cv::KeyPoint>> filteredMatches;
	for (int i = 0; i < mask.rows; i++) {
		auto res = mask.at<bool>(i, 0);
		if (res) {
			filteredMatches.push_back(matches[i]);
			udPointsPrevFrame.push_back(pointsPrevFrame[i]);
			udPointsCurrentFrame.push_back(pointsCurrentFrame[i]);
		}
	}

	matches.clear();
	for (const auto &i : filteredMatches) {
		matches.push_back(i);
	}


	cv::recoverPose(E, udPointsPrevFrame, udPointsCurrentFrame, getCameraMatrix(), R, T);
	cv::SVD svd(E, SVD::FULL_UV);
//	std::cout << " U = " << svd.u << " w = " << svd.w << " Vt = " << svd.vt << std::endl;
	std::cout << "Err : " << (svd.w.at<double>(0, 0) - svd.w.at<double>(0, 1)) / svd.w.at<double>(0, 1) * 100 << std::endl;
//	std::cout << "R = " << R << std::endl << "T = " << T << std::endl;
//	std::cout << "isValid = " << CheckCoherentRotation(R) << std::endl;

	if (!previousFrame->R.empty()) {
		frame->T = previousFrame->T + previousFrame->R * T;
		frame->R = previousFrame->R * R;
	} else {
		frame->R = R.clone();
		frame->T = T.clone();
	}


//	getRTfromE(E, R, T, matches);
//	cv::Affine3d pose(frame->R, frame->T);
//
//	cumPos += pose.translation();
//	std::cout << cumPos << std::endl;
//	points.push_back(pose);
//	vizz->showWidget("cameras_frames_and_lines", cv::viz::WTrajectory(points, viz::WTrajectory::BOTH, 0.1, viz::Color::green()));
	vizz->showWidget(format("frame%d", frame->id),
					 cv::viz::WSphere(cv::Point3d(frame->T.at<double>(0, 0), frame->T.at<double>(0, 1), frame->T.at<double>(0, 2)), 1));
	if (points.size() == 1) {
		vizz->setViewerPose(points[0]);
	}
}

void processFrame(const std::shared_ptr<slam::Frame>& frame) {
	if (frame->id == 165) {
		std::cout << "";
	}

	auto keypoints = extractKeypoints(frame);
	frame->setKeypoints(keypoints);
	auto descriptors = extractDescriptors(frame);
	frame->setDescriptors(descriptors);

	if (descriptors.rows == 0) {
		std::cout << std::endl;
	}

	if (prevFrame != nullptr) {
		auto matches = matchKeypoints(prevFrame, frame);
		std::vector<std::pair<cv::KeyPoint, cv::KeyPoint>> filteredMatches = filterVFC(matches);

		computePosition(prevFrame, frame, filteredMatches);

		auto displayFrame = frame->getMat().clone();
		for (auto match : filteredMatches) {
			cv::circle(displayFrame, match.second.pt, 2, cv::Scalar(255, 255, 255));
//			cv::circle(displayFrame, match.second.pt, 4, cv::Scalar(0, 0, 255));
			cv::line(displayFrame, match.first.pt, match.second.pt, cv::Scalar(0, 0, 255), 1, cv::LINE_4);
			cv::imshow("win", displayFrame);
		}
	} else {
		std::cout << "no previous frame" << std::endl;
	}

	prevFrame = frame;
}

void preprocessFrame(const cv::Mat &frame) {
	cv::Mat cameraMatrix = getCameraMatrix();
	cv::Mat distCoeffs = getDistCoeff();
	cv::Mat newCameraMatrix = cv::getOptimalNewCameraMatrix(
			cameraMatrix,
			distCoeffs,
			frame.size(),
			0
	);

	cv::Mat map1, map2;


	cv::Mat undistortedImage;

	cv::undistort(frame.clone(), frame, cameraMatrix, distCoeffs, newCameraMatrix);
//	cv::Size imageSize(cv::Size(frame.cols,frame.rows));
//
//	cv::initUndistortRectifyMap(cameraMatrix,
//								distCoeffs,
//								cv::Mat(),
//								cv::getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, imageSize, 1, imageSize, 0),
//								imageSize,
//								CV_16SC2,
//								map1,
//								map2);
//	cv::remap(frame.clone(), frame, map1, map2, cv::INTER_LINEAR);
}

int main (int ac, char **av) {
//	std::vector<std::string> files = loadImages(
//			"/home/tiemajor/projects/perso/opencv/ORB_SLAM3/Examples/Datasets/TUM_VI/dataset-room1_512_16/dso/cam0/images/");
	std::vector<std::string> files = loadImages(
			"/home/tiemajor/projects/perso/opencv/slam/image_0");

	std::vector<slam::Frame> frames;

	int i = 0;
	for (const auto &path: files) {
		auto frame = cv::imread(path);

		double t = (double)getTickCount();
		preprocessFrame(frame);
		auto currentFrame = std::make_shared<slam::Frame>(std::move(frame));

		if (prevFrame != nullptr) {
			currentFrame->id = prevFrame->id + 1;
		} else {
			currentFrame->id = 1;
		}

		processFrame(currentFrame);

		t = 1000 * ((double)getTickCount() - t) / getTickFrequency();
//		std::cout << "Frame time (ms): " << t << std::endl;
		if (t > 33.33) {
//			std::cout << "§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§§" << std::endl;
		}

		cv::waitKey(10);
		vizz->spinOnce(10);
		i++;
		if (i > 50) {
		}
	}

	return 0;
}