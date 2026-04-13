// MPILS: Multi-Phase Iterated Local Search Tuner
//
// Author: Youri Rigaud
// License: GNU GPLv3

#ifndef SOLVER_TIME_MODE_H
#define SOLVER_TIME_MODE_H

#include <stdexcept>
#include <string>

enum class SolverTimeMode {
    Seconds,
    Ticks
};

inline std::string solverTimeModeToString(SolverTimeMode mode) {
    switch (mode) {
        case SolverTimeMode::Seconds:
            return "seconds";
        case SolverTimeMode::Ticks:
            return "ticks";
    }

    throw std::runtime_error("Unknown solver time mode.");
}

inline SolverTimeMode parseSolverTimeMode(const std::string& value) {
    if (value == "seconds") {
        return SolverTimeMode::Seconds;
    }
    if (value == "ticks") {
        return SolverTimeMode::Ticks;
    }

    throw std::runtime_error("Unknown solver time mode: " + value);
}

#endif // SOLVER_TIME_MODE_H
