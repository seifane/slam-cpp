//
// Created by tiemajor on 11/20/21.
//

#include "../include/OpenCVVisualizer.hpp"

slam::OpenCVVisualizer::OpenCVVisualizer(): _isRunning(false) {
    this->_viz = cv::makePtr<cv::viz::Viz3d>("Visualizer");
}

slam::OpenCVVisualizer::~OpenCVVisualizer() {
}

void slam::OpenCVVisualizer::loop() {
    this->_viz->spin();
}

void slam::OpenCVVisualizer::start() {
    if (!this->_isRunning) {
        this->_thread = std::thread(&slam::OpenCVVisualizer::loop, this);
        this->_isRunning = true;
    }
}

void slam::OpenCVVisualizer::stop() {
    this->_viz->close();
    this->_thread.join();
    this->_isRunning = false;
}

void slam::OpenCVVisualizer::addPoint(cv::Point3d point) {
    this->_points.push_back(point);
    this->_viz->showWidget(
            cv::format("point%lu", this->_points.size()),
            cv::viz::WSphere(point, 1)
            );
//    this->_viz->spinOnce(1, true);
    /*if (this->_points.size() == 1) {
        this->_viz->setViewerPose(point);
    }*/
}

