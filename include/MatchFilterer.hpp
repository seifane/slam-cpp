//
// Created by tiemajor on 11/19/21.
//

#ifndef SLAM_MATCHFILTERER_HPP
#define SLAM_MATCHFILTERER_HPP

#include <vector>
#include <utility>
#include <opencv2/core/types.hpp>
#include "IMatchFilterer.hpp"
#include "../vfc.h"

namespace slam {
	class MatchFilterer: public IMatchFilterer {
	public:
		MatchFilterer();
		~MatchFilterer();

		std::vector<std::pair<cv::KeyPoint, cv::KeyPoint>> filterMatches(
				const std::vector<std::pair<cv::KeyPoint, cv::KeyPoint>> &matches
				);
	};
}

#endif //SLAM_MATCHFILTERER_HPP
