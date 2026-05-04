// MPILS: Multi-Phase Iterated Local Search Tuner
//
// Author: Youri Rigaud
// License: GNU GPLv3

#ifndef EXPANSION_H
#define EXPANSION_H

#include "tuning_objective.h"
#include "solver_time_mode.h"
#include "logger.h"
#include "tuner_memory.h"
#include "parameter_space.h"

#include <string>
#include <stdexcept>

enum class ExpansionSelectRule {
    Strict,
    Inclusive
};

enum class ExpansionValueStrategy {
    All,
    FirstLast
};

inline std::string expansionSelectRuleToString(ExpansionSelectRule rule) {
    switch (rule) {
        case ExpansionSelectRule::Strict:
            return "strict";
        case ExpansionSelectRule::Inclusive:
            return "inclusive";
    }
    throw std::runtime_error("Unknown expansion select rule.");
}

inline ExpansionSelectRule parseExpansionSelectRule(const std::string& value) {
    if (value == "strict") {
        return ExpansionSelectRule::Strict;
    }
    if (value == "inclusive") {
        return ExpansionSelectRule::Inclusive;
    }
    throw std::runtime_error("Unknown expansion select rule: " + value);
}

inline std::string expansionValueStrategyToString(ExpansionValueStrategy strategy) {
    switch (strategy) {
        case ExpansionValueStrategy::All:
            return "all";
        case ExpansionValueStrategy::FirstLast:
            return "first_last";
    }
    throw std::runtime_error("Unknown expansion value strategy.");
}

inline ExpansionValueStrategy parseExpansionValueStrategy(const std::string& value) {
    if (value == "all") {
        return ExpansionValueStrategy::All;
    }
    if (value == "first_last") {
        return ExpansionValueStrategy::FirstLast;
    }
    throw std::runtime_error("Unknown expansion value strategy: " + value);
}

struct CreateConfigurationsOutput {
    Parameter& parameter;
    Configuration configuration;
    std::string config_file_path;
};

struct PrepareExpansionOutput {
    std::vector<CreateConfigurationsOutput> configuration_files;
    std::vector<std::reference_wrapper<Parameter>> skipped_parameters;
};

struct EvaluateParameterOutput {
    Parameter& parameter;
    std::vector<EvaluationRecord> evaluations;
};

struct ExpansionEvaluationResult {
    int config_id;
    double objective_value;
    int evaluated_time;
    std::optional<double> gap;
    std::optional<double> upper_bound;
    std::optional<double> lower_bound;
    std::optional<double> solver_runtime_seconds;
    SolverTerminationStatus solver_termination_status = SolverTerminationStatus::Normal;
};

struct ClassifyParameterOutput {
    Parameter& parameter;
    bool toSelect;
    bool toDiscard;
};

struct ExpansionRunStats {
    int evaluations_added = 0;
    int parameters_selected = 0;
    int parameters_discarded = 0;
    int parameters_skipped = 0;
};

class Expansion {
    private:
        struct ParameterClassificationMetrics {
            Parameter& parameter;
            double c_p;
            double s_p;
            bool selected_stage_1;
            bool discarded_by_threshold = false;
            bool dominated = false;
        };

        Logger& logger_;
        TunerMemory& memory_;
        ParameterSpace& parameter_space_;
        const std::string expansion_working_dir_;
        const std::string instance_file_;
        const std::string solver_log_file_;
        int& iteration_;
        int nb_parameter_to_evaluate_;
        int nb_threads_solver_;
        double cutoff_solver_time_;
        SolverTimeMode solver_time_mode_;
        SolverWatchdogOptions solver_watchdog_options_;
        TuningObjective tuning_objective_;
        ExpansionSelectRule select_rule_;
        ExpansionValueStrategy value_strategy_;
        double max_deviation_;
        bool enable_early_stop_;

        double best_objective_value_;

        const std::vector<std::reference_wrapper<Parameter>> selectParameters();

        std::vector<Value> selectValuesToEvaluate(const Parameter& param, const Configuration& base_config) const;

        const PrepareExpansionOutput createConfigurationsFiles(const std::vector<std::reference_wrapper<Parameter>>& parameters);

        const std::vector<EvaluateParameterOutput> evaluateParameters(const std::vector<CreateConfigurationsOutput>& configuration_files);

        const std::vector<ClassifyParameterOutput> classifyParameters(const std::vector<EvaluateParameterOutput>& evaluation_results);

        bool isInvalidExpansionObjective(double objective_value) const;

        bool shouldUseExpansionEarlyStop() const;

        bool isExpansionImprovement(double objective_value) const;

        std::vector<double> extractValidObjectives(const EvaluateParameterOutput& eval_output, int& invalid_count) const;

        ParameterClassificationMetrics computeParameterMetrics(Parameter& param, const std::vector<double>& valid_objectives) const;

        bool isSelectedByDirectImprovement(double c_p) const;

        bool shouldDiscardByDeviation(double s_p) const;

        bool doesParetoDominate(const ParameterClassificationMetrics& lhs, const ParameterClassificationMetrics& rhs) const;

        void markDominatedParameters(std::vector<ParameterClassificationMetrics>& metrics) const;

        ClassifyParameterOutput buildClassificationOutput(const ParameterClassificationMetrics& metric) const;

        void updateParameterFlags(const std::vector<ClassifyParameterOutput>& classified_parameters);

        void addToEvaluateParameters(
            Parameter& param,
            const Configuration& config,
            double objective_value,
            std::optional<double> gap,
            std::optional<double> upper_bound,
            std::optional<double> lower_bound,
            std::optional<double> solver_runtime_seconds,
            SolverTerminationStatus solver_termination_status,
            int evaluated_time,
            int worker_id,
            std::vector<EvaluateParameterOutput>& evaluation_outputs
        );

#ifdef USE_MPI
        void launchExpansionWorkers();
        void waitExpansionWorkers();
#endif

    public:
        Expansion(
            Logger& logger,
            TunerMemory& memory,
            ParameterSpace& parameter_space,
            const std::string& tuner_dir,
            const std::string& instance_file,
            const std::string& solver_log_file,
            int& iteration,
            int nb_parameter_to_evaluate,
            int nb_threads_solver,
            double cutoff_solver_time,
            SolverTimeMode solver_time_mode,
            SolverWatchdogOptions solver_watchdog_options,
            TuningObjective tuning_objective,
            ExpansionSelectRule select_rule,
            ExpansionValueStrategy value_strategy,
            double max_deviation,
            bool enable_early_stop
        ): logger_(logger),
           memory_(memory),
           parameter_space_(parameter_space),
           expansion_working_dir_(tuner_dir + "expansion/"),
           instance_file_(instance_file),
           solver_log_file_(solver_log_file),
           iteration_(iteration),
           nb_parameter_to_evaluate_(nb_parameter_to_evaluate),
           nb_threads_solver_(nb_threads_solver),
           cutoff_solver_time_(cutoff_solver_time),
           solver_time_mode_(solver_time_mode),
           solver_watchdog_options_(solver_watchdog_options),
           tuning_objective_(tuning_objective),
           select_rule_(select_rule),
           value_strategy_(value_strategy),
           max_deviation_(max_deviation),
           enable_early_stop_(enable_early_stop)
        {}

        ExpansionRunStats run();
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
        SolverTimeMode solver_time_mode_;
        SolverWatchdogOptions solver_watchdog_options_;
        TuningObjective tuning_objective_;
        double best_objective_value_;
        bool enable_early_stop_;
        Logger& logger_;

        std::vector<std::pair<int, std::string>> configs_to_evaluate_; // Pair of (config_id, config_file_path)
        std::vector<ExpansionEvaluationResult> evaluation_results_;

        void receiveConfigsToEvaluateFromMaster();
        void evaluateConfigurations();
        void sendConfigsResultToMaster();
        bool isExpansionImprovement(double objective_value) const;

    public:
        ExpansionWorker(int worker_id, int iteration, const std::string& instance_file, const std::string& solver_log_file, int nb_threads_solver, double cutoff_solver_time, SolverTimeMode solver_time_mode, SolverWatchdogOptions solver_watchdog_options, TuningObjective tuning_objective, double best_objective_value, bool enable_early_stop, Logger& logger)
            : worker_id_(worker_id), iteration_(iteration), instance_file_(instance_file), solver_log_file_(solver_log_file), nb_threads_solver_(nb_threads_solver), cutoff_solver_time_(cutoff_solver_time), solver_time_mode_(solver_time_mode), solver_watchdog_options_(solver_watchdog_options), tuning_objective_(tuning_objective), best_objective_value_(best_objective_value), enable_early_stop_(enable_early_stop), logger_(logger) {}

        void run();
};
#endif

#endif // EXPANSION_H
