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

struct CreateConfigurationsOutput {
    Parameter& parameter;
    Configuration configuration;
    std::string config_file_path;
};

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

        const std::vector<CreateConfigurationsOutput> createConfigurationsFiles(const std::vector<std::reference_wrapper<Parameter>>& parameters);

        const std::vector<EvaluateParameterOutput> evaluateParameters(const std::vector<CreateConfigurationsOutput>& configuration_files);

        const std::vector<ClassifyParameterOutput> classifyParameters(const std::vector<EvaluateParameterOutput>& evaluation_results);

        void updateParameterFlags(const std::vector<ClassifyParameterOutput>& classified_parameters);

#ifdef USE_MPI
        void launchExpansionWorkers();
        void waitExpansionWorkers();
#endif

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

#ifdef USE_MPI
class ExpansionWorker {
    private:
        int worker_id_;
        int iteration_;
        std::string instance_file_;
        std::string solver_log_file_;
        int nb_threads_solver_;
        double cutoff_solver_time_;

        std::vector<std::pair<int, std::string>> configs_to_evaluate_; // Pair of (config_id, config_file_path)
        std::vector<std::pair<int, double>> evaluation_results_; // Pair of (config_id, objective_value)

        void receiveConfigsToEvaluateFromMaster();
        void evaluateConfigurations();
        void sendConfigsResultToMaster();

    public:
        ExpansionWorker(int worker_id, int iteration, const std::string& instance_file, const std::string& solver_log_file, int nb_threads_solver, double cutoff_solver_time)
            : worker_id_(worker_id), iteration_(iteration), instance_file_(instance_file), solver_log_file_(solver_log_file), nb_threads_solver_(nb_threads_solver), cutoff_solver_time_(cutoff_solver_time) {}

        void run();
};
#endif

#endif // EXPANSION_H