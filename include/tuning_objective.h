// MPILS: Multi-Phase Iterated Local Search Tuner
//
// Author: Youri Rigaud
// License: GNU GPLv3

#ifndef TUNING_OBJECTIVE_H
#define TUNING_OBJECTIVE_H

#include <stdexcept>
#include <string>

enum class TuningObjective {
    Gap,
    UpperBound
};

inline std::string tuningObjectiveToString(TuningObjective objective) {
    switch (objective) {
        case TuningObjective::Gap:
            return "gap";
        case TuningObjective::UpperBound:
            return "upper_bound";
    }

    return "unknown";
}

inline TuningObjective parseTuningObjective(const std::string& value) {
    if (value == "gap") {
        return TuningObjective::Gap;
    }
    if (value == "upper_bound") {
        return TuningObjective::UpperBound;
    }

    throw std::runtime_error("Unknown tuning objective: " + value + ". Expected 'gap' or 'upper_bound'.");
}

#endif // TUNING_OBJECTIVE_H
