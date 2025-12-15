// MPILS: Multi-Phase Iterated Local Search Tuner
//
// Author: Youri Rigaud
// License: GNU GPLv3

#ifndef EXPANSION_H
#define EXPANSION_H

#include "logger.h"
#include "tuner_memory.h"
#include "parameter_space.h"

#include <string>

struct EvaluateParameterOutput {
    Parameter& parameter;
    std::vector<Configuration> configurations;
};

struct ClassifyParameterOutput {
    Parameter& parameter;
    bool toSelect;
    bool toDiscard;
};

class Expansion {
    private:
        Logger& logger_;
        TunerMemory& memory_;
        ParameterSpace& parameter_space_;
        const std::string instance_file_;
        const std::string solver_log_file_;
        int& iteration_;
        int evaluation_budget_;
        int nb_threads_solver_;
        double cutoff_solver_time_;

        const std::string expansion_working_dir_ = "tuner_working_dir/expansion/";

        const std::vector<std::reference_wrapper<Parameter>> selectParameters();

        const std::vector<EvaluateParameterOutput> evaluateParameters(const std::vector<std::reference_wrapper<Parameter>>& parameters);

        const std::vector<ClassifyParameterOutput> classifyParameters(const std::vector<EvaluateParameterOutput>& evaluation_results);

        void updateParameterFlags(const std::vector<ClassifyParameterOutput>& classified_parameters);

    public:
        Expansion(
            Logger& logger,
            TunerMemory& memory,
            ParameterSpace& parameter_space,
            const std::string& instance_file,
            const std::string& solver_log_file,
            int& iteration,
            int evaluation_budget,
            int nb_threads_solver,
            double cutoff_solver_time
        ): logger_(logger),
           memory_(memory),
           parameter_space_(parameter_space),
           instance_file_(instance_file),
           solver_log_file_(solver_log_file),
           iteration_(iteration),
           evaluation_budget_(evaluation_budget),
              nb_threads_solver_(nb_threads_solver),
              cutoff_solver_time_(cutoff_solver_time)
        {}

        void run();
};

#endif // EXPANSION_H