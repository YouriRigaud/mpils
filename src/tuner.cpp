// MPILS: Multi-Phase Iterated Local Search Tuner
//
// Author: Youri Rigaud
// License: GNU GPLv3

#include "../include/tuner.h"
#include "../include/parameter.h"

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

void Tuner::setAllParametersFlags() {
    int index = 1;
    for (auto& param : parameter_space_.getParameters()) {
        if (index <= nb_initial_selected_parameters_) {
            param.setIsSelected(true);
            param.setIsTuned(false);
            param.setIsDiscarded(false);
            param.setIsResidual(false);
        } else {
            param.setIsSelected(false);
            param.setIsTuned(false);
            param.setIsDiscarded(false);
            param.setIsResidual(true);
        }
        index++;
    }
    logger_.info("Set all parameter flags finished.");
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

    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file to write parameters ID: " + filename);
    }
    file << "Name ID" << std::endl;
    int index = 1;
    for (const auto& pair : config.getConfiguration()) {
        file << pair.first << "\tP" << index << std::endl;
        index++;
    }
    file.close();

    logger_.info("Finished writing parameter IDs to file: ", filename);
}

void Tuner::setup() {
    logger_.info("Seting up the MPILS tuner");
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
    if (iteration_ >= 15) {
        logger_.info("Stopping condition met: reached maximum iterations (", iteration_, ").");
        return true;
    }
    
    if (parameter_space_.getResidualParameters().empty()) {
        logger_.info("Stopping condition met: no more residual parameters to evaluate.");
        return true;
    }

    if (memory_.getBestConfiguration() != nullptr) {
        double best_objective = memory_.getBestConfiguration()->getObjective();
        if (best_objective <= 0.01) {
            logger_.info("Stopping condition met: satisfactory objective value achieved (", best_objective, ").");
            return true;
        }
    }
    
    return false;
}


void Tuner::run() {
    logger_.info("Running the MPILS tuner");
    while (true) {
        logger_.info("Starting iteration ", iteration_);

        // Exploration phase
        exploration_.run();
        
        // Check stopping condition
        if (stopConditionMet()) {
#ifdef USE_MPI
            sendStopOrderToWorkers();
#endif
            break;
        }

        // Expansion phase
        expansion_.run();

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
    std::cout << "Worker " << worker_id_ << " received order for step " << worker_step_ << "." << std::endl;
}

void Worker::runExplorationPhase() {
    std::cout << "Worker " << worker_id_ << " running exploration phase for iteration " << iteration_ << "." << std::endl;
    setLocalSearchWorker(std::make_unique<ParamILSWorker>(worker_id_, iteration_));
    local_search_worker_->run();
    MPI_Barrier(MPI_COMM_WORLD); // Ensure all workers finish before proceeding
    worker_step_ = 0; // Set to waiting state
    std::cout << "Worker " << worker_id_ << " completed exploration phase." << std::endl;
}

void Worker::runExpansionPhase() {
    std::cout << "Worker " << worker_id_ << " running expansion phase for iteration " << iteration_ << "." << std::endl;
    std::string solver_log_file_worker = solver_log_file_ + "_iteration_expansion_" + std::to_string(iteration_) + "_worker_" + std::to_string(worker_id_);
    // Implementation of expansion phase logic
    setExpansionWorker(std::make_unique<ExpansionWorker>(worker_id_, iteration_, instance_file_, solver_log_file_worker, nb_threads_solver_, cutoff_solver_time_));
    expansion_worker_->run();
    MPI_Barrier(MPI_COMM_WORLD); // Ensure all workers finish before proceeding
    worker_step_ = 0; // Set to waiting state
    std::cout << "Worker " << worker_id_ << " completed expansion phase." << std::endl;
}
#endif