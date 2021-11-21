//
// Created by tiemajor on 11/20/21.
//

#ifndef SLAM_FRAMEPREPROCESSOR_HPP
#define SLAM_FRAMEPREPROCESSOR_HPP

#include "IFramePreprocessor.hpp"

namespace slam {
    class FramePreprocessor: public IFramePreprocessor {
    public:
        FramePreprocessor();
        ~FramePreprocessor();

        void preprocessFrame(const std::shared_ptr<slam::Frame> &);
    };
}

#endif //SLAM_FRAMEPREPROCESSOR_HPP
