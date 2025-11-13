// MPILS: Multi-Phase Iterated Local Search Tuner
//
// Author: Youri Rigaud
// License: GNU GPLv3

#ifndef CONFIGURATION_H
#define CONFIGURATION_H

#include "parameter.h"

#include <vector>
#include <map>
#include <cmath>

class Configuration {
    private:
        std::vector<Parameter> parameters_; // Liste of the parameters in the configuration
        std::map<std::string, Value> parameter_values_; // Map of parameter names to their values
        double objective_value_ = std::nan(""); // Objective value if computed
};

// Classe abstraite pour représenter une configuration de paramètres
// Il doit y avoir le nombre de paramètres, et pour chaque paramètre, sa valeur et le cout si calcule

// On va avoir CplexConfiguration et GurobiConfiguration qui hérite de Configuration

#endif // CONFIGURATION_H