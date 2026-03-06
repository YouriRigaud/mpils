// MPILS: Multi-Phase Iterated Local Search Tuner
//
// Author: Youri Rigaud
// License: GNU GPLv3

#include "../include/iterated_local_search.h"
#include "../include/solver.h"

void IteratedLocalSearch::run() {
    logger_.info("Starting local search phase...");

    if (options_.initial_configuration == nullptr) {
        logger_.warn("No initial configuration provided for local search phase. This should not happen, check the tuner memory initialization.");
        return;
    }

    if (memory_.getCachedObjectiveValue(current_configuration_.getConfigurationId()).has_value()) {
        logger_.info("Initial configuration for local search phase already evaluated with objective value: ", memory_.getCachedObjectiveValue(current_configuration_.getConfigurationId()).value());
    } else {
        logger_.info("Evaluating initial configuration for local search phase...");
        // Evaluate the initial configuration and store the result in memory
        double initial_objective_value = evaluateConfiguration(*options_.initial_configuration);
        EvaluationRecord initial_record = createEvaluationRecord(options_, *options_.initial_configuration, initial_objective_value);
        memory_.addEvaluation(initial_record);
        logger_.info("Initial configuration evaluated with objective value: ", initial_objective_value);
    }
    
    //todo

    logger_.info("Local search phase completed.");
}