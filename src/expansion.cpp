// MPILS: Multi-Phase Iterated Local Search Tuner
//
// Author: Youri Rigaud
// License: GNU GPLv3

#include "../include/expansion.h"
#include "../include/solver.h"
#include "../include/tuner.h"
#include "../include/globaltimer.h"

#ifdef USE_MPI
#include <mpi.h>
#endif

#include <algorithm>

void Expansion::run() {
    logger_.info("Starting expansion phase...");

    best_actual_config_ = memory_.getBestConfiguration();
    if (best_actual_config_ == nullptr) {
            logger_.info("No best configuration in memory, cannot start expansion phase at iteration ", iteration_, ".");
            return;
    }

    const std::vector<std::reference_wrapper<Parameter>> expansion_parameters = selectParameters();
    if (expansion_parameters.empty()) {
        logger_.info("No expansion parameters selected, skipping expansion.");
        return;
    }
    logger_.info("Selected expansion parameters: ", expansion_parameters.size());

    const std::vector<CreateConfigurationsOutput> configuration_files = createConfigurationsFiles(expansion_parameters);

#ifdef USE_MPI
    launchExpansionWorkers();
#endif

    const std::vector<EvaluateParameterOutput> evaluation_results = evaluateParameters(configuration_files);

#ifdef USE_MPI
    waitExpansionWorkers();
#endif

    logger_.info("Evaluated expansion parameters: ", evaluation_results.size());

    const std::vector<ClassifyParameterOutput> classified_parameters = classifyParameters(evaluation_results);

    updateParameterFlags(classified_parameters);


    int count_selected = 0;
    int count_discarded = 0;
    for (const auto& cp : classified_parameters) {
        if (cp.toSelect) count_selected++;
        if (cp.toDiscard) count_discarded++;
    }
    logger_.debug("Classified parameters at iteration ", iteration_, " - To Select: ", count_selected, ", To Discard: ", count_discarded);
    for (const auto& cp : classified_parameters) {
        logger_.debug("Parameter: ", cp.parameter.getName(), ", To Select: ", cp.toSelect ? "Yes" : "No", ", To Discard: ", cp.toDiscard ? "Yes" : "No");
    }

    int total_configs = 0;
    for (const auto& evaluation_result : evaluation_results) {
        total_configs += evaluation_result.configurations.size();
    }

    logger_.info("Added ", total_configs, " configurations to memory from expansion phase at iteration ", iteration_);


    logger_.info("Expansion phase completed.");
}

const std::vector<std::reference_wrapper<Parameter>> Expansion::selectParameters() {
    logger_.info("Selecting parameters for expansion phase...");
    std::vector<std::reference_wrapper<Parameter>> residual_parameters = parameter_space_.getResidualParameters();
    if (residual_parameters.empty()) {
        logger_.info("No residual parameters available for expansion.");
        return {};
    }

    std::vector<std::reference_wrapper<Parameter>> selected_parameters;
    int count = 0;
    for (auto& param : residual_parameters) {
        if (count >= nb_parameter_to_evaluate_) {
            break;
        }
        selected_parameters.push_back(param);
        count++;
    }
    
    return selected_parameters;
}

const std::vector<CreateConfigurationsOutput> Expansion::createConfigurationsFiles(const std::vector<std::reference_wrapper<Parameter>>& parameters) {
    logger_.info("Creating configuration files for expansion parameters...");
    std::vector<CreateConfigurationsOutput> configuration_files_outputs;

    for (auto& param_ref : parameters) {
        Parameter& param = param_ref.get();
        logger_.info("Creating configurations for parameter: ", param.getName());

        // Generate configurations for the parameter
        const Configuration* best_config = memory_.getBestConfiguration();
        if (best_config == nullptr) {
            logger_.info("No best configuration in memory, using default configuration for expansion.");
            best_config = &memory_.getDefaultConfiguration();
        }

        // This method evaluate only 2 values
	    std::vector<Value> valueToEvaluate;
	    const auto& values = param.getValues();

	    if (values.size() == 1) {
	        valueToEvaluate.push_back(values.front());
	    } else {
            valueToEvaluate.push_back(values.front());
    	    valueToEvaluate.push_back(values.back());
        }

        for (const auto& value : valueToEvaluate) {
            std::string config_file_path = expansion_working_dir_ + "config_param_" + param.getName() + "_" + value.getString() + "_iter_" + std::to_string(iteration_) + ".prm";
            std::map<std::string, Value> config_map = best_config->getConfiguration();
            config_map.insert_or_assign(param.getName(), value);

            Configuration config(config_map);
            config.generateConfigFile(config_file_path);
            
            configuration_files_outputs.push_back({param, config, config_file_path});
            logger_.debug("Created configuration file for parameter ", param.getName(), " with value ", value.getString(), " at ", config_file_path);
        }
    }
    logger_.info("Created ", configuration_files_outputs.size(), " configuration files for expansion phase at iteration ", iteration_, ".");
    return configuration_files_outputs;
}

void Expansion::addToEvaluateParameters(Parameter& param, const Configuration& config, double objective_value, int evaluated_time, int worker_id, std::vector<EvaluateParameterOutput>& evaluation_outputs) {
    // Find or create EvaluateParameterOutput for the parameter
    auto it = std::find_if(evaluation_outputs.begin(), evaluation_outputs.end(),
                           [&param](const EvaluateParameterOutput& epo) { return &epo.parameter == &param; });
    if (it == evaluation_outputs.end()) {
        evaluation_outputs.push_back({param, {}});
        it = std::prev(evaluation_outputs.end());
    }

    // Add the evaluated configuration
    Configuration evaluated_config = config;
    evaluated_config.setObjective(objective_value, evaluated_time);
    it->configurations.push_back(evaluated_config);
    memory_.addConfiguration(evaluated_config, worker_id, iteration_, 1); // Add to memory with worker_id, iteration and phase for logging
}

const std::vector<EvaluateParameterOutput> Expansion::evaluateParameters(const std::vector<CreateConfigurationsOutput>& configuration_files_outputs) {
    logger_.info("Evaluating expansion parameters...");
    std::vector<EvaluateParameterOutput> evaluation_outputs;
    std::vector<std::pair<int, std::string>> configs_to_evaluate; // Pair of (config_id, config_file_path)
    configs_to_evaluate.reserve(configuration_files_outputs.size());
    for (size_t i = 0; i < configuration_files_outputs.size(); ++i) {
        configs_to_evaluate.emplace_back(i, configuration_files_outputs[i].config_file_path);
    }
#ifdef USE_MPI
    // Split configs_to_evaluate between master and workers
    int nb_workers = 1; // Default to 1 for non-MPI
    MPI_Comm_size(MPI_COMM_WORLD, &nb_workers);
    int configs_per_worker = configs_to_evaluate.size() / nb_workers;
    int remainder = configs_to_evaluate.size() % nb_workers;
    // Master evaluates its share
    std::vector<std::pair<int, std::string>> master_configs_to_evaluate(configs_to_evaluate.begin(), configs_to_evaluate.begin() + configs_per_worker + (remainder > 0 ? 1 : 0));
    // Send configs to workers
    for (int worker_id = 1; worker_id < nb_workers; ++worker_id) {
        int start_index = worker_id * configs_per_worker + std::min(worker_id, remainder);
        int end_index = start_index + configs_per_worker + (worker_id < remainder ? 1 : 0); // Send a remainder config if still at least one left
        int num_configs = end_index - start_index;
        MPI_Send(&num_configs, 1, MPI_INT, worker_id, 0, MPI_COMM_WORLD);
        for (int i = start_index; i < end_index; ++i) {
            int config_id = configs_to_evaluate[i].first;
            const std::string& config_file_path = configs_to_evaluate[i].second;
            MPI_Send(&config_id, 1, MPI_INT, worker_id, 0, MPI_COMM_WORLD);
            int path_length = config_file_path.size();
            MPI_Send(&path_length, 1, MPI_INT, worker_id, 0, MPI_COMM_WORLD);
            MPI_Send(config_file_path.c_str(), path_length, MPI_CHAR, worker_id, 0, MPI_COMM_WORLD);
        }
    }
#else
    std::vector<std::pair<int, std::string>> master_configs_to_evaluate = configs_to_evaluate;
#endif
    // Master evaluates its configurations
    std::string solver_log_file_master = solver_log_file_ + "_iteration_expansion_" + std::to_string(iteration_) + "_worker_0";
    for (const auto& config_pair : master_configs_to_evaluate) {
        int config_id = config_pair.first;
        const std::string& config_file_path = config_pair.second;

        CPLEXSolver solver(
            logger_,
            instance_file_,
            config_file_path,
            solver_log_file_master,
            nb_threads_solver_,
            cutoff_solver_time_
        );

        solver.solve();
        double objective_value = solver.getObjectiveValue();
        int evaluated_time = GlobalTimer::elapsedSeconds();

        const auto& create_output = configuration_files_outputs[config_id];
        addToEvaluateParameters(create_output.parameter, create_output.configuration, objective_value, evaluated_time, 0, evaluation_outputs);
    }
#ifdef USE_MPI
    // Receive results from workers
    for (int worker_id = 1; worker_id < nb_workers; ++worker_id) {
        MPI_Status status;
        int num_results;
        MPI_Recv(&num_results, 1, MPI_INT, worker_id, 0, MPI_COMM_WORLD, &status);
        for (int i = 0; i < num_results; ++i) {
            int config_id;
            double objective_value;
            int evaluated_time;
            MPI_Recv(&config_id, 1, MPI_INT, worker_id, 0, MPI_COMM_WORLD, &status);
            MPI_Recv(&objective_value, 1, MPI_DOUBLE, worker_id, 0, MPI_COMM_WORLD, &status);
            MPI_Recv(&evaluated_time, 1, MPI_INT, worker_id, 0, MPI_COMM_WORLD, &status);

            const auto& create_output = configuration_files_outputs[config_id];
            addToEvaluateParameters(create_output.parameter, create_output.configuration, objective_value, evaluated_time, worker_id, evaluation_outputs);
        }
    }
#endif

    return evaluation_outputs;
}

const std::vector<ClassifyParameterOutput> Expansion::classifyParameters(const std::vector<EvaluateParameterOutput>& evaluation_results) {
    logger_.info("Classifying expansion parameters...");
    std::vector<ClassifyParameterOutput> classified_parameters;

    for (const auto& eval_output : evaluation_results) {
        Parameter& param = eval_output.parameter;
        const auto& configurations = eval_output.configurations;

        // Simple classification logic: if the best configuration with this parameter is better than the best overall, select it
        //TODO: More sophisticated classification logic can be implemented here

        double best_actual_objective = best_actual_config_->getObjective();
        double param_best_objective = std::numeric_limits<double>::max();
        for (const auto& config : configurations) {
            if (config.getObjective() < param_best_objective) {
                param_best_objective = config.getObjective();
            }
        }

        bool toSelect = param_best_objective < best_actual_objective;
        bool toDiscard = !toSelect && (param_best_objective >= best_actual_objective * 1.0); // Discard if significantly worse

        classified_parameters.push_back({param, toSelect, toDiscard});
    }

    return classified_parameters;
}

void Expansion::updateParameterFlags(const std::vector<ClassifyParameterOutput>& classified_parameters) {
    logger_.info("Updating parameter flags based on classification...");
    for (const auto& cp : classified_parameters) {
        Parameter& param = cp.parameter;
        if (cp.toSelect) {
            param.setIsSelected(true);
            param.setIsResidual(false);
            logger_.info("Parameter ", param.getName(), " selected for tuning.");
        } else if (cp.toDiscard) {
            param.setIsDiscarded(true);
            param.setIsResidual(false);
            logger_.info("Parameter ", param.getName(), " discarded from tuning.");
        } else {
            logger_.info("Parameter ", param.getName(), " remains residual.");
        }
    }
}

#ifdef USE_MPI
void ExpansionWorker::run() {
    std::cout << "Expansion Worker " << worker_id_ << " starting expansion for iteration " << iteration_ << "." << std::endl;
    receiveConfigsToEvaluateFromMaster();
    evaluateConfigurations();
    sendConfigsResultToMaster();
    std::cout << "Expansion Worker " << worker_id_ << " completed expansion." << std::endl;
}

void ExpansionWorker::receiveConfigsToEvaluateFromMaster() {
    // Implementation to receive configurations to evaluate from master
    MPI_Status status;
    int num_configs;
    MPI_Recv(&num_configs, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);

    for (int i = 0; i < num_configs; ++i) {
        int config_id;
        int path_length;
        MPI_Recv(&config_id, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
        MPI_Recv(&path_length, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
        std::vector<char> path_buffer(path_length);
        MPI_Recv(path_buffer.data(), path_length, MPI_CHAR, 0, 0, MPI_COMM_WORLD, &status);
        std::string config_file_path(path_buffer.begin(), path_buffer.end());
        configs_to_evaluate_.emplace_back(config_id, config_file_path);
    }
}

void ExpansionWorker::evaluateConfigurations() {
    Logger logger(Verbosity::Normal, std::cout);
    for (const auto& config_pair : configs_to_evaluate_) {
        int config_id = config_pair.first;
        const std::string& config_file_path = config_pair.second;

        CPLEXSolver solver(
            logger,
            instance_file_,
            config_file_path,
            solver_log_file_,
            nb_threads_solver_,
            cutoff_solver_time_
        );

        solver.solve();
        double objective_value = solver.getObjectiveValue();
        int elapsed_time = GlobalTimer::elapsedSeconds();

        evaluation_results_.emplace_back(config_id, std::make_pair(objective_value, elapsed_time));
    }
}

void ExpansionWorker::sendConfigsResultToMaster() {
    // Implementation to send evaluated configurations results back to master
    int num_results = evaluation_results_.size();
    MPI_Send(&num_results, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);

    for (const auto& result_pair : evaluation_results_) {
        int config_id = result_pair.first;
        double objective_value = result_pair.second.first;
        int evaluated_time = result_pair.second.second;
        MPI_Send(&config_id, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        MPI_Send(&objective_value, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        MPI_Send(&evaluated_time, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }
}
#endif

#ifdef USE_MPI
void Expansion::launchExpansionWorkers() {
    logger_.info("Launching Expansion Workers via MPI...");
    // Implementation to launch expansion workers using MPI
    // Broadcast to give all workers worker_step = 2 and iteration_
    WorkerOrder order;
    order.step = 2; // expansion step
    order.iteration = iteration_;
    MPI_Bcast(&order, sizeof(WorkerOrder), MPI_BYTE, 0, MPI_COMM_WORLD);
    logger_.info("Expansion Workers launched.");
}

void Expansion::waitExpansionWorkers() {
    logger_.info("Waiting for Expansion Workers via MPI...");
    // Implementation to wait for expansion workers using MPI
    MPI_Barrier(MPI_COMM_WORLD);
    logger_.info("Expansion Workers have completed.");
}
#endif