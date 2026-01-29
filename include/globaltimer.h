// MPILS: Multi-Phase Iterated Local Search Tuner
//
// Author: Youri Rigaud
// License: GNU GPLv3

#ifndef GLOBAL_TIMER_H
#define GLOBAL_TIMER_H

#include <chrono>

class GlobalTimer {
    private:
        inline static std::chrono::_V2::system_clock::time_point start_time_;

    public:
        static void start() {
            start_time_ = std::chrono::high_resolution_clock::now();
        }

        static int elapsedSeconds() {
            auto now_time = std::chrono::high_resolution_clock::now();
            return std::chrono::duration_cast<std::chrono::seconds>(now_time - start_time_).count();
        }
};

#endif