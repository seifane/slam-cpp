//
// Created by tiemajor on 11/19/21.
//

#include "../include/MatchFilterer.hpp"

slam::MatchFilterer::MatchFilterer() {

}

slam::MatchFilterer::~MatchFilterer() {

}

std::vector<std::pair<cv::KeyPoint, cv::KeyPoint>> slam::MatchFilterer::filterMatches(const std::vector<std::pair<cv::KeyPoint, cv::KeyPoint>> &matches) {
	std::vector<std::pair<cv::KeyPoint, cv::KeyPoint>> filteredMatches;

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


	for (int matchIdx: matchIdxs) {
		filteredMatches.push_back(matches[matchIdx]);
	}
	return filteredMatches;
}

