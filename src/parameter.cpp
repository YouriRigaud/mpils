// MPILS: Multi-Phase Iterated Local Search Tuner
//
// Author: Youri Rigaud
// License: GNU GPLv3

#include "../include/parameter.h"

std::string Value::getString() const {
    if (std::holds_alternative<int>(value_)) {
        return std::to_string(std::get<int>(value_));
    } else if (std::holds_alternative<double>(value_)) {
        return std::to_string(std::get<double>(value_));
    } else if (std::holds_alternative<std::string>(value_)) {
        return std::get<std::string>(value_);
    }
    return "";
}
