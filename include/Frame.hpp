//IFramePreprocessor
// Created by tiemajor on 11/16/21.
//
#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/features2d.hpp>
#include <opencv4/opencv2/core/types.hpp>
#include <vector>

#ifndef SLAM_FRAME_HPP
#define SLAM_FRAME_HPP

namespace slam {
	class Frame {
	public:
		Frame();
		Frame(cv::Mat);
		Frame(cv::Mat, std::vector<cv::KeyPoint>, cv::Mat);
		~Frame();

		void setMat(cv::Mat);
		const cv::Mat &getMat();

		void setKeypoints(std::vector<cv::KeyPoint>);
		std::vector<cv::KeyPoint> &getKeypoints();

		void setDescriptors(cv::Mat);
		const cv::Mat &getDescriptors();

		cv::Mat getAnnotatedFrame(const std::vector<std::pair<cv::KeyPoint, cv::KeyPoint>> &);

        //Image data
        int id;
        cv::Mat mat;
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;

        //Camera intrinsics
        cv::Mat K;
        cv::Mat distCoeffs;

        //Computed
        cv::Mat R;
        cv::Mat T;
        cv::Mat cameraPose;
	};
}

#endif //SLAM_FRAME_HPP
