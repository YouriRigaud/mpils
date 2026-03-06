// MPILS: Multi-Phase Iterated Local Search Tuner
//
// Author: Youri Rigaud
// License: GNU GPLv3

#ifndef ITERATED_LOCAL_SEARCH_H
#define ITERATED_LOCAL_SEARCH_H

#include "logger.h"
#include "tuner_memory.h"
#include "configuration.h"

class LocalSearchMemory {
    // Implementation of the memory component for the local search phase
    // This will contains an unordered_set of configurations already evaluated and their objective value
    private:
        Logger logger_;
        std::vector<EvaluationRecord> evaluations_; ///< List of all evaluations performed during the local search phase
        std::unordered_map<ConfigurationId, double> cache_; ///< Map of configuration ID to its objective value found during the local search phase, stored to avoid re-evaluating the same configuration multiple times

    public:
        /** @brief Constructor for the local search memory component */
        LocalSearchMemory(Logger logger): logger_(logger) {}

        /** @brief Constructor for the local search memory component with an initial configuration and objective value */
        LocalSearchMemory(Logger logger, Configuration initial_configuration, double initial_objective_value): logger_(logger) {
            // Initialize the local search cache memory with the initial configuration and its objective value
            cache_[initial_configuration.getConfigurationId()] = initial_objective_value;
        }

        void addEvaluation(const EvaluationRecord& record) {
            evaluations_.push_back(record);
            cache_[record.configuration_id] = record.objective_value;
        }

        std::optional<double> getCachedObjectiveValue(const Configuration& config) const {
            auto it = cache_.find(config.getConfigurationId());
            if (it != cache_.end()) {
                return it->second;
            } else {
                return std::nullopt;
            }
        }

        const std::vector<EvaluationRecord>& getEvaluations() const {
            return evaluations_;
        }
};

class ParameterSpace {

}

class IteratedLocalSearch {
    private:
        Logger logger_;
        LocalSearchMemory memory_;
        Options options_;
        bool stop_condition_met_ = false;

        Configuration acumbent_solution_;   ///< The best solution found so far during the local search phase
        Configuration current_configuration_;   ///< The current solution being evaluated during the local search phase

    public:
        struct Options {
            unsigned int random_seed = 0; // Random seed for reproducibility
            std::size_t evaluation_budget = 20; // Maximum number of solver evaluations to perform
            std::size_t perturbation_strength = 3; // Number of parameters to change during the perturbation step
            std::size_t random_initial_samples = 0; // Number of random configurations to sample for the initial solution
            double restart_probability = 0.1; // Probability of restarting from a random solution instead of perturbing the current solution
            bool accept_ties = false; // Whether to accept solutions with the same objective value as the current solution
            double acceptance_threshold = 0.01; // Threshold for accepting a new solution (relative improvement)
            bool use_mip_starts = false; // Whether to use MIP starts during the local search phase
            int nb_threads_solver = 2; // Number of threads for the solver
            double cutoff_solver_time = 15.0; // Cutoff time for each solver run
        };

        /** @brief Constructor for the iterated local search component */
        IteratedLocalSearch(
            Logger logger,
            const Options& options,
            const Configuration& initial_configuration
        ): logger_(logger),
           options_(options),
           acumbent_solution_(initial_configuration),
           current_configuration_(initial_configuration)
        {}

        /** @brief Constructor for the iterated local search component with an initial configuration already evaluated */
        IteratedLocalSearch(
            Logger logger,
            const Options& options,
            const Configuration& initial_configuration,
            double initial_objective_value
        ): logger_(logger, initial_configuration, initial_objective_value),
           options_(options),
           acumbent_solution_(initial_configuration),
           current_configuration_(initial_configuration)
        {}

        void run();

        std::vector<EvaluationRecord> getEvaluations() const {
            return memory_.getEvaluations();
        }
};

#endif // ITERATED_LOCAL_SEARCH_H