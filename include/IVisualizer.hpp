//
// Created by tiemajor on 11/20/21.
//

#ifndef SLAM_IVISUALIZER_HPP
#define SLAM_IVISUALIZER_HPP

namespace slam {
    class IVisualizer {
    public:
        virtual ~IVisualizer() = default;

        virtual void loop() = 0;
        virtual void start() = 0;
        virtual void stop() = 0;
        virtual void addPoint(cv::Point3d) = 0;
    };
}

#endif //SLAM_IVISUALIZER_HPP
