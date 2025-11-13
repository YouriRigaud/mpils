// MPILS: Multi-Phase Iterated Local Search Tuner
//
// Author: Youri Rigaud
// License: GNU GPLv3

#ifndef LOGGER_H
#define LOGGER_H

#include <iostream>
#include <sstream>
#include <string>
#include <chrono>
#include <iomanip>

enum class Verbosity {
    Normal,
    Debug,
};

class Logger {
    private:
        Verbosity level_;
        std::ostream* out_;

        template<typename... Args>
        void print(const std::string& prefix, Args&&... args) const {
            auto now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
            (*out_) << prefix;
            (*out_) << std::put_time(std::localtime(&now), "[%H:%M:%S] ");
            ((*out_) << ... << args) << std::endl;
        }

    public:
        Logger(Verbosity level, std::ostream& out): level_(level), out_(&out) {}

        void setLevel(Verbosity lvl) { level_ = lvl; }
        void setOutput(std::ostream& output) { out_ = &output; }

        template<typename... Args>
        void info(Args&&... args) const {
            if (level_ >= Verbosity::Normal)
                print("[INFO]:", std::forward<Args>(args)...);
        }

        template<typename... Args>
        void debug(Args&&... args) const {
            if (level_ == Verbosity::Debug)
                print("[DEBUG]:", std::forward<Args>(args)...);
        }
};

#endif // LOGGER_H