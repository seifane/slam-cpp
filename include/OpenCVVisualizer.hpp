//
// Created by tiemajor on 11/20/21.
//

#ifndef SLAM_OPENCVVISUALIZER_HPP
#define SLAM_OPENCVVISUALIZER_HPP

#include <opencv2/core/types.hpp>
#include <opencv2/viz/viz3d.hpp>
#include <thread>
#include "IVisualizer.hpp"

namespace slam {
    class OpenCVVisualizer: public IVisualizer {
    public:
        OpenCVVisualizer();
        ~OpenCVVisualizer();

        void loop();
        void start();
        void stop();
        void addPoint(cv::Point3d);

    private:
        cv::Ptr<cv::viz::Viz3d> _viz;
        std::vector<cv::Point3d> _points;
        std::thread _thread;
        bool _isRunning;
    };
}

#endif //SLAM_OPENCVVISUALIZER_HPP
