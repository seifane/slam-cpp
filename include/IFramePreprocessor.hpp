//
// Created by tiemajor on 11/19/21.
//

#ifndef SLAM_IFRAMEPREPROCESSOR_HPP
#define SLAM_IFRAMEPREPROCESSOR_HPP

#include <memory>
#include "Frame.hpp"

namespace slam {
	class IFramePreprocessor {
	public:
		virtual ~IFramePreprocessor() = default;

		virtual void preprocessFrame(const std::shared_ptr<slam::Frame> &) = 0;
	};
}

#endif //SLAM_IFRAMEPREPROCESSOR_HPP
