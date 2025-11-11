// MPILS: Multi-Phase Iterated Local Search Tuner
//
// Author: Youri Rigaud
// License: GNU GPLv3

#include "../include/parameter.h"

Parameter::Parameter(
    int i, const std::string& n, const std::string& t,
    const Value& dv, const std::vector<Value>& vals
): index(i), name(n), type(t), default_value(dv), values(vals) {}

