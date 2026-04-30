// MPILS: Multi-Phase Iterated Local Search Tuner
//
// Author: Youri Rigaud
// License: GNU GPLv3

#include "../include/expansion.h"
#include "../include/solver.h"
#include "../include/tuner.h"
#include "../include/globaltimer.h"
#include "../include/configuration.h"

#ifdef USE_MPI
#include <mpi.h>
#endif

#include <algorithm>
#include <cmath>
#include <limits>

ExpansionRunStats Expansion::run() {
    logger_.info("Starting expansion phase...");

    best_objective_value_ = memory_.getBestObjectiveWithoutMipStart();

    const std::vector<std::reference_wrapper<Parameter>> expansion_parameters = selectParameters();
    if (expansion_parameters.empty()) {
        logger_.info("No expansion parameters selected, skipping expansion.");
        return {};
    }
    logger_.info("Selected expansion parameters: ", expansion_parameters.size());

    const PrepareExpansionOutput preparation_output = createConfigurationsFiles(expansion_parameters);
    const std::vector<CreateConfigurationsOutput>& configuration_files = preparation_output.configuration_files;

#ifdef USE_MPI
    launchExpansionWorkers();
#endif

    const std::vector<EvaluateParameterOutput> evaluation_results = evaluateParameters(configuration_files);

#ifdef USE_MPI
    waitExpansionWorkers();
#endif

    logger_.info("Evaluated expansion parameters: ", evaluation_results.size());

    std::vector<ClassifyParameterOutput> classified_parameters;
    classified_parameters.reserve(preparation_output.skipped_parameters.size() + evaluation_results.size());

    for (Parameter& param : preparation_output.skipped_parameters) {
        logger_.info(
            "Parameter ", param.getName(),
            " skipped during expansion and marked for discard because no alternative value remains to evaluate."
        );
        classified_parameters.push_back({param, false, true});
    }

    const std::vector<ClassifyParameterOutput> evaluated_classified_parameters = classifyParameters(evaluation_results);
    for (const auto& classified_parameter : evaluated_classified_parameters) {
        classified_parameters.push_back(classified_parameter);
    }

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
        total_configs += evaluation_result.evaluations.size();
    }

    logger_.info("Added ", total_configs, " evaluations to memory from expansion phase at iteration ", iteration_);


    logger_.info("Expansion phase completed.");
    return ExpansionRunStats{
        total_configs,
        count_selected,
        count_discarded,
        static_cast<int>(preparation_output.skipped_parameters.size())
    };
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

std::vector<Value> Expansion::selectValuesToEvaluate(const Parameter& param, const Configuration& base_config) const {
    std::vector<Value> candidate_values;
    const std::vector<Value> values = param.getValues();

    switch (value_strategy_) {
        case ExpansionValueStrategy::All:
            candidate_values = values;
            break;
        case ExpansionValueStrategy::FirstLast:
            if (values.size() <= 1) {
                candidate_values = values;
            } else {
                candidate_values.push_back(values.front());
                if (!(values.back() == values.front())) {
                    candidate_values.push_back(values.back());
                }
            }
            break;
    }

    auto current_value_it = base_config.getConfigurationMap().find(param.getName());
    if (current_value_it == base_config.getConfigurationMap().end()) {
        return candidate_values;
    }

    std::vector<Value> filtered_values;
    filtered_values.reserve(candidate_values.size());
    for (const Value& value : candidate_values) {
        if (!(value == current_value_it->second)) {
            filtered_values.push_back(value);
        }
    }

    return filtered_values;
}

const PrepareExpansionOutput Expansion::createConfigurationsFiles(const std::vector<std::reference_wrapper<Parameter>>& parameters) {
    logger_.info("Creating configuration files for expansion parameters...");
    PrepareExpansionOutput output;

    for (auto& param_ref : parameters) {
        Parameter& param = param_ref.get();
        logger_.info("Creating configurations for parameter: ", param.getName());

        // Generate configurations for the parameter, use the best configuration without MIP start as base to avoid bias from MIP starts
        const Configuration* best_config = memory_.getBestConfigurationWithoutMipStart();
        if (best_config == nullptr) {
            logger_.info("No best configuration without MIP start in memory, using default configuration for expansion.");
            best_config = &memory_.getDefaultConfiguration();
        }

        std::vector<Value> values_to_evaluate = selectValuesToEvaluate(param, *best_config);
        if (values_to_evaluate.empty()) {
            logger_.info(
                "Skipping parameter ", param.getName(),
                " during expansion because no alternative value remains after excluding the current base configuration value."
            );
            output.skipped_parameters.push_back(param);
            continue;
        }

        logger_.debug(
            "Parameter ", param.getName(),
            " will be evaluated on ", values_to_evaluate.size(),
            " alternative value(s) after excluding the current base configuration value."
        );

        for (const auto& value : values_to_evaluate) {
            std::string config_file_path = expansion_working_dir_ + "config_param_" + param.getName() + "_" + value.getString() + "_iter_" + std::to_string(iteration_) + ".prm";
            std::map<std::string, Value> config_map = best_config->getConfigurationMap();
            config_map.insert_or_assign(param.getName(), value);

            Configuration config(config_map);
            config.generateConfigFile(config_file_path);
            
            output.configuration_files.push_back({param, config, config_file_path});
            logger_.debug("Created configuration file for parameter ", param.getName(), " with value ", value.getString(), " at ", config_file_path);
        }
    }
    logger_.info("Created ", output.configuration_files.size(), " configuration files for expansion phase at iteration ", iteration_, ".");
    logger_.info("Skipped ", output.skipped_parameters.size(), " expansion parameter(s) with no alternative value to evaluate.");
    return output;
}

void Expansion::addToEvaluateParameters(
    Parameter& param,
    const Configuration& config,
    double objective_value,
    std::optional<double> gap,
    std::optional<double> upper_bound,
    std::optional<double> lower_bound,
    std::optional<double> solver_runtime_seconds,
    int evaluated_time,
    int worker_id,
    std::vector<EvaluateParameterOutput>& evaluation_outputs
) {
    // Find or create EvaluateParameterOutput for the parameter
    auto it = std::find_if(evaluation_outputs.begin(), evaluation_outputs.end(),
                           [&param](const EvaluateParameterOutput& epo) { return &epo.parameter == &param; });
    if (it == evaluation_outputs.end()) {
        evaluation_outputs.push_back({param, {}});
        it = std::prev(evaluation_outputs.end());
    }

    // Add the evaluation
    RecordEvaluationOptions options;
    options.mip_start_used = false; // Expansion phase does not use MIP starts
    options.produced_mip_start = false; // Expansion phase does not produce MIP starts
    
    options.objective_value = objective_value;
    options.gap = gap;
    options.upper_bound = upper_bound;
    options.lower_bound = lower_bound;
    options.solver_runtime_seconds = solver_runtime_seconds;
    options.time_evaluated = evaluated_time;
    options.worker_id = worker_id;
    options.iteration = iteration_;
    options.phase = 1; // Phase 1 for expansion

    EvaluationId eval_id = memory_.recordEvaluation(config, options);
    EvaluationRecord eval_record = *memory_.getEvaluationById(eval_id);

    it->evaluations.push_back(eval_record);
}

const std::vector<EvaluateParameterOutput> Expansion::evaluateParameters(const std::vector<CreateConfigurationsOutput>& configuration_files_outputs) {
    logger_.info("Evaluating expansion parameters...");
    std::vector<EvaluateParameterOutput> evaluation_outputs;
    std::vector<std::pair<int, std::string>> configs_to_evaluate; // Pair of (config_id, config_file_path)
    configs_to_evaluate.reserve(configuration_files_outputs.size());
    for (size_t i = 0; i < configuration_files_outputs.size(); ++i) {
        configs_to_evaluate.emplace_back(i, configuration_files_outputs[i].config_file_path);
    }
    const bool use_expansion_early_stop = shouldUseExpansionEarlyStop();
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
    auto evaluate_single_config = [&](int config_id, const std::string& config_file_path) -> double {
        CPLEXSolver solver(
            logger_,
            instance_file_,
            config_file_path,
            solver_log_file_master,
            nb_threads_solver_,
            cutoff_solver_time_,
            solver_time_mode_,
            tuning_objective_
        );

        solver.solve();
        const double objective_value = solver.getObjectiveValue();
        std::optional<double> gap = solver.getGap();
        std::optional<double> upper_bound = solver.getUpperBound();
        std::optional<double> lower_bound = solver.getLowerBound();
        std::optional<double> solver_runtime_seconds = solver.getSolveTimeSeconds();
        int evaluated_time = GlobalTimer::elapsedSeconds();

        const auto& create_output = configuration_files_outputs[config_id];
        addToEvaluateParameters(create_output.parameter, create_output.configuration, objective_value, gap, upper_bound, lower_bound, solver_runtime_seconds, evaluated_time, 0, evaluation_outputs);

        return objective_value;
    };

#ifdef USE_MPI
    if (use_expansion_early_stop && nb_workers > 1) {
        int local_count = static_cast<int>(master_configs_to_evaluate.size());
        int max_configs_per_rank = 0;
        MPI_Allreduce(&local_count, &max_configs_per_rank, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

        for (int round = 0; round < max_configs_per_rank; ++round) {
            int local_found_improvement = 0;

            if (round < local_count) {
                const auto& config_pair = master_configs_to_evaluate[round];
                const double objective_value = evaluate_single_config(config_pair.first, config_pair.second);

                if (isExpansionImprovement(objective_value)) {
                    local_found_improvement = 1;
                    const auto& create_output = configuration_files_outputs[config_pair.first];
                    logger_.info(
                        "Expansion early stop improvement found on master by parameter ", create_output.parameter.getName(),
                        " with objective value ", objective_value, "."
                    );
                }
            }

            int global_early_stop = 0;
            MPI_Allreduce(&local_found_improvement, &global_early_stop, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
            if (global_early_stop != 0) {
                logger_.info(
                    "Expansion early stop activated in MPI after ", evaluation_outputs.size(),
                    " master-side evaluation(s). Remaining not-yet-evaluated parameters stay residual."
                );
                break;
            }
        }
    } else
#endif
    {
        for (const auto& config_pair : master_configs_to_evaluate) {
            const double objective_value = evaluate_single_config(config_pair.first, config_pair.second);

            if (use_expansion_early_stop && isExpansionImprovement(objective_value)) {
                const auto& create_output = configuration_files_outputs[config_pair.first];
                logger_.info(
                    "Expansion early stop triggered by parameter ", create_output.parameter.getName(),
                    " with objective value ", objective_value,
                    ". Remaining not-yet-evaluated parameters stay residual."
                );
                break;
            }
        }
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
            int has_gap;
            double gap_value = 0.0;
            int has_upper_bound;
            double upper_bound_value = 0.0;
            int has_lower_bound;
            double lower_bound_value = 0.0;
            int has_solver_runtime;
            double solver_runtime_value = 0.0;
            int evaluated_time;
            MPI_Recv(&config_id, 1, MPI_INT, worker_id, 0, MPI_COMM_WORLD, &status);
            MPI_Recv(&objective_value, 1, MPI_DOUBLE, worker_id, 0, MPI_COMM_WORLD, &status);
            MPI_Recv(&has_gap, 1, MPI_INT, worker_id, 0, MPI_COMM_WORLD, &status);
            if (has_gap != 0) {
                MPI_Recv(&gap_value, 1, MPI_DOUBLE, worker_id, 0, MPI_COMM_WORLD, &status);
            }
            MPI_Recv(&has_upper_bound, 1, MPI_INT, worker_id, 0, MPI_COMM_WORLD, &status);
            if (has_upper_bound != 0) {
                MPI_Recv(&upper_bound_value, 1, MPI_DOUBLE, worker_id, 0, MPI_COMM_WORLD, &status);
            }
            MPI_Recv(&has_lower_bound, 1, MPI_INT, worker_id, 0, MPI_COMM_WORLD, &status);
            if (has_lower_bound != 0) {
                MPI_Recv(&lower_bound_value, 1, MPI_DOUBLE, worker_id, 0, MPI_COMM_WORLD, &status);
            }
            MPI_Recv(&has_solver_runtime, 1, MPI_INT, worker_id, 0, MPI_COMM_WORLD, &status);
            if (has_solver_runtime != 0) {
                MPI_Recv(&solver_runtime_value, 1, MPI_DOUBLE, worker_id, 0, MPI_COMM_WORLD, &status);
            }
            MPI_Recv(&evaluated_time, 1, MPI_INT, worker_id, 0, MPI_COMM_WORLD, &status);

            const auto& create_output = configuration_files_outputs[config_id];
            addToEvaluateParameters(
                create_output.parameter,
                create_output.configuration,
                objective_value,
                has_gap != 0 ? std::optional<double>(gap_value) : std::nullopt,
                has_upper_bound != 0 ? std::optional<double>(upper_bound_value) : std::nullopt,
                has_lower_bound != 0 ? std::optional<double>(lower_bound_value) : std::nullopt,
                has_solver_runtime != 0 ? std::optional<double>(solver_runtime_value) : std::nullopt,
                evaluated_time,
                worker_id,
                evaluation_outputs
            );
        }
    }
#endif

    return evaluation_outputs;
}

bool Expansion::isInvalidExpansionObjective(double objective_value) const {
    return !std::isfinite(objective_value) || objective_value == std::numeric_limits<double>::max();
}

bool Expansion::shouldUseExpansionEarlyStop() const {
    if (!enable_early_stop_) {
        return false;
    }
    return true;
}

bool Expansion::isExpansionImprovement(double objective_value) const {
    return objective_value < best_objective_value_;
}

std::vector<double> Expansion::extractValidObjectives(const EvaluateParameterOutput& eval_output, int& invalid_count) const {
    std::vector<double> valid_objectives;
    valid_objectives.reserve(eval_output.evaluations.size());
    invalid_count = 0;

    for (const auto& evaluation : eval_output.evaluations) {
        const double objective_value = evaluation.objective_value;
        if (isInvalidExpansionObjective(objective_value)) {
            invalid_count++;
            continue;
        }
        valid_objectives.push_back(objective_value);
    }

    return valid_objectives;
}

bool Expansion::isSelectedByDirectImprovement(double c_p) const {
    switch (select_rule_) {
        case ExpansionSelectRule::Strict:
            return c_p < best_objective_value_;
        case ExpansionSelectRule::Inclusive:
            return c_p <= best_objective_value_;
    }

    return false;
}

bool Expansion::shouldDiscardByDeviation(double s_p) const {
    return s_p > max_deviation_;
}

Expansion::ParameterClassificationMetrics Expansion::computeParameterMetrics(Parameter& param, const std::vector<double>& valid_objectives) const {
    double c_p = std::numeric_limits<double>::max();
    double squared_deviation_sum = 0.0;

    for (double objective_value : valid_objectives) {
        c_p = std::min(c_p, objective_value);
        const double deviation = objective_value - best_objective_value_;
        squared_deviation_sum += deviation * deviation;
    }

    double s_p = 0.0;
    if (valid_objectives.size() >= 2) {
        s_p = std::sqrt(squared_deviation_sum / static_cast<double>(valid_objectives.size() - 1));
    }

    return {param, c_p, s_p, isSelectedByDirectImprovement(c_p), shouldDiscardByDeviation(s_p), false};
}

bool Expansion::doesParetoDominate(const ParameterClassificationMetrics& lhs, const ParameterClassificationMetrics& rhs) const {
    return
        (lhs.c_p < rhs.c_p && lhs.s_p >= rhs.s_p) ||
        (lhs.c_p <= rhs.c_p && lhs.s_p > rhs.s_p);
}

void Expansion::markDominatedParameters(std::vector<ParameterClassificationMetrics>& metrics) const {
    for (size_t i = 0; i < metrics.size(); ++i) {
        if (metrics[i].selected_stage_1) {
            continue;
        }

        for (size_t j = 0; j < metrics.size(); ++j) {
            if (i == j || metrics[j].selected_stage_1) {
                continue;
            }

            if (doesParetoDominate(metrics[j], metrics[i])) {
                metrics[i].dominated = true;
                logger_.debug(
                    "Parameter ", metrics[i].parameter.getName(),
                    " is dominated by parameter ", metrics[j].parameter.getName(), "."
                );
                break;
            }
        }
    }
}

ClassifyParameterOutput Expansion::buildClassificationOutput(const ParameterClassificationMetrics& metric) const {
    const bool toSelect = metric.selected_stage_1 || !metric.dominated;
    const bool toDiscard = !metric.selected_stage_1 && metric.dominated;

    logger_.debug(
        "Parameter ", metric.parameter.getName(),
        " metrics - c_p: ", metric.c_p,
        ", s_p: ", metric.s_p,
        ", selected_stage_1: ", metric.selected_stage_1 ? "Yes" : "No",
        ", discarded_by_threshold: ", metric.discarded_by_threshold ? "Yes" : "No",
        ", dominated: ", metric.dominated ? "Yes" : "No"
    );

    return {metric.parameter, toSelect, toDiscard};
}

const std::vector<ClassifyParameterOutput> Expansion::classifyParameters(const std::vector<EvaluateParameterOutput>& evaluation_results) {
    logger_.info("Classifying expansion parameters...");
    std::vector<ParameterClassificationMetrics> metrics;
    metrics.reserve(evaluation_results.size());
    std::vector<ClassifyParameterOutput> classified_parameters;
    classified_parameters.reserve(evaluation_results.size());

    for (const auto& eval_output : evaluation_results) {
        Parameter& param = eval_output.parameter;
        int invalid_evaluation_count = 0;
        std::vector<double> valid_objectives = extractValidObjectives(eval_output, invalid_evaluation_count);

        if (valid_objectives.empty()) {
            logger_.debug(
                "Parameter ", param.getName(),
                " discarded before classification because all evaluations were invalid."
            );
            classified_parameters.push_back({param, false, true});
            continue;
        }

        ParameterClassificationMetrics metric = computeParameterMetrics(param, valid_objectives);

        logger_.debug(
            "Parameter ", param.getName(),
            " valid evaluations: ", valid_objectives.size(),
            ", invalid evaluations: ", invalid_evaluation_count,
            ", c_p: ", metric.c_p,
            ", s_p: ", metric.s_p,
            ", selected_stage_1: ", metric.selected_stage_1 ? "Yes" : "No",
            ", discarded_by_threshold: ", metric.discarded_by_threshold ? "Yes" : "No"
        );

        if (metric.discarded_by_threshold) {
            classified_parameters.push_back({param, false, true});
            continue;
        }

        metrics.push_back(metric);
    }

    markDominatedParameters(metrics);

    for (const auto& metric : metrics) {
        classified_parameters.push_back(buildClassificationOutput(metric));
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
    logger_.info("Expansion Worker ", worker_id_, " starting expansion for iteration ", iteration_, ".");
    receiveConfigsToEvaluateFromMaster();
    evaluateConfigurations();
    sendConfigsResultToMaster();
    logger_.info("Expansion Worker ", worker_id_, " completed expansion.");
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

bool ExpansionWorker::isExpansionImprovement(double objective_value) const {
    return objective_value < best_objective_value_;
}

void ExpansionWorker::evaluateConfigurations() {
    auto evaluate_single_config = [&](int config_id, const std::string& config_file_path) -> double {
        CPLEXSolver solver(
            logger_,
            instance_file_,
            config_file_path,
            solver_log_file_,
            nb_threads_solver_,
            cutoff_solver_time_,
            solver_time_mode_,
            tuning_objective_
        );

        solver.solve();
        double objective_value = solver.getObjectiveValue();
        std::optional<double> gap = solver.getGap();
        std::optional<double> upper_bound = solver.getUpperBound();
        std::optional<double> lower_bound = solver.getLowerBound();
        std::optional<double> solver_runtime_seconds = solver.getSolveTimeSeconds();
        int elapsed_time = GlobalTimer::elapsedSeconds();

        evaluation_results_.push_back({config_id, objective_value, elapsed_time, gap, upper_bound, lower_bound, solver_runtime_seconds});
        return objective_value;
    };

    int world_size = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (enable_early_stop_ && world_size > 1) {
        int local_count = static_cast<int>(configs_to_evaluate_.size());
        int max_configs_per_rank = 0;
        MPI_Allreduce(&local_count, &max_configs_per_rank, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

        for (int round = 0; round < max_configs_per_rank; ++round) {
            int local_found_improvement = 0;

            if (round < local_count) {
                const auto& config_pair = configs_to_evaluate_[round];
                const double objective_value = evaluate_single_config(config_pair.first, config_pair.second);

                if (isExpansionImprovement(objective_value)) {
                    local_found_improvement = 1;
                    logger_.info(
                        "Expansion Worker ", worker_id_,
                        " found an improving configuration with objective value ",
                        objective_value, "."
                    );
                }
            }

            int global_early_stop = 0;
            MPI_Allreduce(&local_found_improvement, &global_early_stop, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
            if (global_early_stop != 0) {
                logger_.info("Expansion Worker ", worker_id_, " stopping early after global MPI expansion stop was activated.");
                break;
            }
        }
        return;
    }

    for (const auto& config_pair : configs_to_evaluate_) {
        evaluate_single_config(config_pair.first, config_pair.second);
    }
}

void ExpansionWorker::sendConfigsResultToMaster() {
    // Implementation to send evaluated configurations results back to master
    int num_results = evaluation_results_.size();
    MPI_Send(&num_results, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);

    for (const auto& result : evaluation_results_) {
        int config_id = result.config_id;
        double objective_value = result.objective_value;
        int has_gap = result.gap.has_value() ? 1 : 0;
        double gap_value = result.gap.value_or(0.0);
        int has_upper_bound = result.upper_bound.has_value() ? 1 : 0;
        double upper_bound_value = result.upper_bound.value_or(0.0);
        int has_lower_bound = result.lower_bound.has_value() ? 1 : 0;
        double lower_bound_value = result.lower_bound.value_or(0.0);
        int has_solver_runtime = result.solver_runtime_seconds.has_value() ? 1 : 0;
        double solver_runtime_value = result.solver_runtime_seconds.value_or(0.0);
        int evaluated_time = result.evaluated_time;
        MPI_Send(&config_id, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        MPI_Send(&objective_value, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        MPI_Send(&has_gap, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        if (has_gap != 0) {
            MPI_Send(&gap_value, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        }
        MPI_Send(&has_upper_bound, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        if (has_upper_bound != 0) {
            MPI_Send(&upper_bound_value, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        }
        MPI_Send(&has_lower_bound, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        if (has_lower_bound != 0) {
            MPI_Send(&lower_bound_value, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        }
        MPI_Send(&has_solver_runtime, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        if (has_solver_runtime != 0) {
            MPI_Send(&solver_runtime_value, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        }
        MPI_Send(&evaluated_time, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }
}
#endif

#ifdef USE_MPI
void Expansion::launchExpansionWorkers() {
    logger_.info("Launching Expansion Workers via MPI...");
    // Implementation to launch expansion workers using MPI
    // Broadcast to give all workers worker_step = 2 and iteration_
    WorkerOrder order{};
    order.step = 2; // expansion step
    order.iteration = iteration_;
    order.nb_evaluations = 0;
    order.expansion_best_objective_value = best_objective_value_;
    order.expansion_enable_early_stop = enable_early_stop_ ? 1 : 0;
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
