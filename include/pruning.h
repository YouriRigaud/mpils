// MPILS: Multi-Phase Iterated Local Search Tuner
//
// Author: Youri Rigaud
// License: GNU GPLv3

#ifndef PRUNING_H
#define PRUNING_H

#include "logger.h"
#include "tuner_memory.h"
#include "parameter_space.h"

class Pruning {
    private:
        Logger& logger_;
        TunerMemory& memory_;
        ParameterSpace& parameter_space_;
        int& iteration_;

        void writeLearnerFile(const std::string& learner_file);

        std::vector<std::vector<std::pair<std::string, Value>>> extractForbiddenTuples();
        
        void applyPruning(std::vector<std::vector<std::pair<std::string, Value>>>& forbidden_tuples);

    public:
        Pruning(
            Logger& logger,
            TunerMemory& memory,
            ParameterSpace& parameter_space,
            int& iteration
        ): logger_(logger),
           memory_(memory),
           parameter_space_(parameter_space),
           iteration_(iteration)
        {}

        void run();
};

#endif // PRUNING_H