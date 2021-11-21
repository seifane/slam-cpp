//
// Created by tiemajor on 11/19/21.
//

#ifndef SLAM_IMATCHFILTERER_HPP
#define SLAM_IMATCHFILTERER_HPP

namespace slam {
	class IMatchFilterer {
	public:
		virtual ~IMatchFilterer() = default;

		virtual std::vector<std::pair<cv::KeyPoint, cv::KeyPoint>> filterMatches(
				const std::vector<std::pair<cv::KeyPoint, cv::KeyPoint>> &matches
				) = 0;
	};
}

#endif //SLAM_IMATCHFILTERER_HPP
