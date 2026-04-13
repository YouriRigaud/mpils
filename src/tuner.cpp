// MPILS: Multi-Phase Iterated Local Search Tuner
//
// Author: Youri Rigaud
// License: GNU GPLv3

#include "../include/tuner.h"
#include "../include/filesystem_utils.h"

#ifdef USE_MPI
#include <mpi.h>
#endif

#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <iomanip>

Parameter getParameterFromLine(const std::string& line) {
    // The line is formatted as: name value1 value2 ... [default_value]
    std::istringstream iss(line);
    std::string name;
    iss >> name;
    std::vector<Value> values;
    std::string token;
    Value default_value = Value(0); // Placeholder default value
    bool default_set = false;
    while (iss >> token) {
        if (token.front() == '[' && token.back() == ']') {
            // This is the default value
            std::string def_val_str = token.substr(1, token.size() - 2);
            default_value = Value(def_val_str);
            default_set = true;
        } else {
            values.push_back(Value(token));
        }
    }
    if (!default_set && !values.empty()) {
        default_value = values.front(); // Set first value as default if not specified
    }
    return Parameter(name, values, default_value);
}

void Tuner::printParameters(const std::vector<Parameter>& params) {
    std::ostringstream oss;
    oss << "Printing " << params.size() << " parameters:\n";
    for (const auto& p : params) {
        oss << "Parameter: " << p.getName() << ", Values: ";
        for (const auto& v : p.getValues()) {
            oss << v.getString() << " ";
        }
        oss << ", Default: " << p.getDefaultValue().getString() << "\n";
    }
    logger_.debug(oss.str());
}

std::vector<Parameter> readParametersFromFile(const std::string &parameters_file, Logger &logger) {
    // open the parameters file
    std::ifstream file(parameters_file);
    if (!file.is_open()) {
        logger.info("Could not open parameter file: ", parameters_file);
        throw std::runtime_error("Could not open parameter file: " + parameters_file);
    } else {
        logger.info("Reading parameters from file: ", parameters_file);
    }

    // read each line and parse parameters
    std::vector<Parameter> params;
    std::string line;
    while (std::getline(file, line)) {
        Parameter param = getParameterFromLine(line);
        params.push_back(param);
    }
    file.close();
    logger.info("Finished reading parameters.");
    return params;
}

std::vector<Parameter> Tuner::getParameters() {
    std::vector<Parameter> initial_params = readParametersFromFile(parameters_file_, logger_);
    printParameters(initial_params);
    return initial_params;
}

void Tuner::createWorkingDirectories() {
    ensureDirectoryExists(tuner_dir_);
    ensureDirectoryExists(tuner_dir_ + "solver/");
    ensureDirectoryExists(tuner_dir_ + "solver/outfiles/");
    ensureDirectoryExists(tuner_dir_ + "solver/mipstarts/");
    ensureDirectoryExists(tuner_dir_ + "expansion/");
    ensureDirectoryExists(tuner_dir_ + "config_for_mip_start/");
    ensureDirectoryExists(tuner_dir_ + "mip_start/");
    ensureDirectoryExists(tuner_dir_ + "iterated_local_search/");
    ensureDirectoryExists(tuner_dir_ + "iterated_local_search/search_space/");
    ensureDirectoryExists(tuner_dir_ + "iterated_local_search/local_results/");
    ensureDirectoryExists(tuner_dir_ + "param_ils/");
    ensureDirectoryExists(tuner_dir_ + "param_ils/parameter/");
    ensureDirectoryExists(tuner_dir_ + "param_ils/scenario/");
    ensureDirectoryExists(tuner_dir_ + "pruning/");
    ensureDirectoryExists(tuner_dir_ + "pruning/input/");
    ensureDirectoryExists(tuner_dir_ + "pruning/output/");
}

void Tuner::setAllParametersFlags() {
    int count_discarded_fixed = 0;
    int count_selected_tunable = 0;
    int count_residual_tunable = 0;

    for (auto& param : parameter_space_.getParameters()) {
        const bool is_tunable = param.getValues().size() > 1;

        if (!is_tunable) {
            param.setIsSelected(false);
            param.setIsTuned(false);
            param.setIsDiscarded(true);
            param.setIsResidual(false);
            count_discarded_fixed++;
        } else if (count_selected_tunable < nb_initial_selected_parameters_) {
            param.setIsSelected(true);
            param.setIsTuned(false);
            param.setIsDiscarded(false);
            param.setIsResidual(false);
            count_selected_tunable++;
        } else {
            param.setIsSelected(false);
            param.setIsTuned(false);
            param.setIsDiscarded(false);
            param.setIsResidual(true);
            count_residual_tunable++;
        }
    }

    logger_.info(
        "Set all parameter flags finished. Fixed discarded at setup: ", count_discarded_fixed,
        ", Tunable initially selected: ", count_selected_tunable,
        ", Tunable residual: ", count_residual_tunable
    );
}

void Tuner::setDefaultConfiguration() {
    // Create default configuration with default values
    std::map<std::string, Value> default_config_map;
    for (const auto& param : parameter_space_.getParameters()) {
        default_config_map.insert({param.getName(), param.getDefaultValue()});
    }
    Configuration default_config(default_config_map);
    // Note: Default configuration is not evaluated yet, so we do not add it to memory
    memory_.setDefaultConfiguration(default_config);
    logger_.info("Default configuration created with default parameter values.");
}

void Tuner::writeParametersIdToFile(const Configuration& config, const std::string& filename) {
    logger_.info("Writing parameter IDs of configuration to file: ", filename);

    ensureParentDirectoryForFile(filename);
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file to write parameters ID: " + filename);
    }
    file << "Name ID" << std::endl;
    int index = 1;
    for (const auto& pair : config.getConfigurationMap()) {
        file << pair.first << "\tP" << index << std::endl;
        index++;
    }
    file.close();

    logger_.info("Finished writing parameter IDs to file: ", filename);
}

void Tuner::setup() {
    logger_.info("Seting up the MPILS tuner");
    createWorkingDirectories();
    setAllParametersFlags();

    logger_.debug("Tuned Parameters:");
    printParameters(parameter_space_.getTunedParameters());
    logger_.debug("Selected Parameters:");
    printParameters(parameter_space_.getSelectedParameters());

    // Set default configuration in memory
    setDefaultConfiguration();

    logger_.debug("After adding selected to tuned via running, Tuned Parameters:");
    printParameters(parameter_space_.getTunedParameters());
    logger_.debug("After adding selected to tuned via running, Selected Parameters:");
    printParameters(parameter_space_.getSelectedParameters());


    // logger_.debug("Initial configuration check in memory:");
    // const Configuration& initial_config_check = getInitialConfiguration();
    // std::ostringstream oss;
    // oss << "Initial Configuration (not evaluated yet): ";
    // for (const auto& pair : initial_config_check.getConfiguration()) {
    //     oss << pair.first << "=" << pair.second.getString() << " ";
    // }
    // logger_.debug(oss.str());

    writeParametersIdToFile(memory_.getDefaultConfiguration(), tuner_dir_ + "parameter_ids.txt");    

    logger_.info("Tuner setup complete.");
}

bool Tuner::stopConditionMet() {
    if (iteration_ >= max_iterations_) {
        logger_.info("Stopping condition met: reached maximum iterations (", iteration_, ").");
        return true;
    }
    
    if (parameter_space_.getResidualParameters().empty()) {
        logger_.info("Stopping condition met: no more residual parameters to evaluate.");
        return true;
    }

    if (memory_.hasEvaluationAtOrBelowGap(0.0)) {
        logger_.info("Stopping condition met: satisfactory gap achieved.");
        return true;
    }
    
    return false;
}


void Tuner::run() {
    logger_.info("Running the MPILS tuner");
    while (true) {
        logger_.info("Starting iteration ", iteration_);

        // Exploration phase
        exploration_.run();

        if (exploration_only_) {
            logger_.info("Stopping after exploration phase because exploration-only mode is enabled.");
#ifdef USE_MPI
            sendStopOrderToWorkers();
#endif
            break;
        }
        
        // Check stopping condition
        if (stopConditionMet()) {
#ifdef USE_MPI
            sendStopOrderToWorkers();
#endif
            break;
        }

        // Expansion phase
        expansion_.run();

        // Check stopping condition
        if (memory_.hasEvaluationAtOrBelowGap(0.0)) {
            logger_.info("Stopping condition met: satisfactory gap achieved.");
#ifdef USE_MPI
            sendStopOrderToWorkers();
#endif
            break;
        }

        // Pruning phase
        pruning_.run();
        
        
        logger_.info("Completed iteration ", iteration_);
        iteration_++;
    }
    logger_.info("Tuner run complete.");
}

#ifdef USE_MPI
void Tuner::sendStopOrderToWorkers() {
    logger_.info("Sending stop order to all worker processes.");
    WorkerOrder stop_order;
    stop_order.step = 3; // 3 indicates stop
    stop_order.iteration = iteration_;
    MPI_Bcast(&stop_order, sizeof(WorkerOrder), MPI_BYTE, 0, MPI_COMM_WORLD);
    logger_.info("Stop order sent to all worker processes.");
}

void Worker::run() {
    std::cout << "Worker " << worker_id_ << " starting." << std::endl;
    while (true) {
        receiveOrderFromMaster();
        if (stopConditionMet()) {
            break;
        }
        if (worker_step_ == 1) {
            runExplorationPhase();
        } else if (worker_step_ == 2) {
            runExpansionPhase();
        }
    }
    std::cout << "Worker " << worker_id_ << " finished." << std::endl;
}

void Worker::receiveOrderFromMaster() {
    // Implementation to receive order from master process, this means updating worker_step_ and iteration_
    // Worker waits two ints from master: worker_step_ and iteration_
    WorkerOrder order;
    MPI_Bcast(&order, sizeof(WorkerOrder), MPI_BYTE, 0, MPI_COMM_WORLD);
    worker_step_ = order.step;
    iteration_ = order.iteration;
    if (worker_step_ == 1) {
        nb_evaluations_ = order.nb_evaluations;
    }
    std::cout << "Worker " << worker_id_ << " received order for step " << worker_step_ << "." << std::endl;
}

void Worker::runExplorationPhase() {
    std::cout << "Worker " << worker_id_ << " running exploration phase for iteration " << iteration_ << "." << std::endl;
    const bool use_mip_start = enable_mip_starts_ && worker_id_ == 1;
    switch (local_search_backend_) {
        case LocalSearchBackend::IteratedLocalSearch:
            setLocalSearchWorker(std::make_unique<IteratedLocalSearchWorker>(
                worker_id_,
                iteration_,
                nb_evaluations_,
                nb_threads_solver_,
                cutoff_solver_time_,
                solver_time_mode_,
                instance_file_,
                solver_log_file_,
                tuning_objective_,
                base_seed_,
                use_shared_cache_,
                use_mip_start,
                random_worker_initial_configs_
            ));
            break;
        case LocalSearchBackend::ParamILS:
            setLocalSearchWorker(std::make_unique<ParamILSWorker>(
                worker_id_,
                iteration_,
                tuning_objective_,
                base_seed_,
                solver_time_mode_,
                use_mip_start,
                use_shared_cache_,
                random_worker_initial_configs_
            ));
            break;
    }
    local_search_worker_->run();
    MPI_Barrier(MPI_COMM_WORLD); // Ensure all workers finish before proceeding
    worker_step_ = 0; // Set to waiting state
    std::cout << "Worker " << worker_id_ << " completed exploration phase." << std::endl;
}

void Worker::runExpansionPhase() {
    std::cout << "Worker " << worker_id_ << " running expansion phase for iteration " << iteration_ << "." << std::endl;
    std::string solver_log_file_worker = solver_log_file_ + "_iteration_expansion_" + std::to_string(iteration_) + "_worker_" + std::to_string(worker_id_);
    std::cout << "Worker " << worker_id_ << " will use solver log file: " << solver_log_file_worker << std::endl;
    // Implementation of expansion phase logic
    setExpansionWorker(std::make_unique<ExpansionWorker>(worker_id_, iteration_, instance_file_, solver_log_file_worker, nb_threads_solver_, cutoff_solver_time_, solver_time_mode_, tuning_objective_));
    expansion_worker_->run();
    MPI_Barrier(MPI_COMM_WORLD); // Ensure all workers finish before proceeding
    worker_step_ = 0; // Set to waiting state
    std::cout << "Worker " << worker_id_ << " completed expansion phase." << std::endl;
}
#endif
