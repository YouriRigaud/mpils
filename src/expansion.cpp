// MPILS: Multi-Phase Iterated Local Search Tuner
//
// Author: Youri Rigaud
// License: GNU GPLv3

#include "../include/expansion.h"
#include "../include/solver.h"

void Expansion::run() {
    logger_.info("Starting expansion phase...");

    const std::vector<std::reference_wrapper<Parameter>> expansion_parameters = selectParameters();
    if (expansion_parameters.empty()) {
        logger_.info("No expansion parameters selected, skipping expansion.");
        return;
    }
    logger_.info("Selected expansion parameters: ", expansion_parameters.size());

    const std::vector<EvaluateParameterOutput> evaluation_results = evaluateParameters(expansion_parameters);
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
    for (const auto& evaluation_results : evaluation_results) {
        memory_.addConfigurations(evaluation_results.configurations);
        total_configs += evaluation_results.configurations.size();
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
        if (count >= evaluation_budget_) {
            break;
        }
        selected_parameters.push_back(param);
        count++;
    }
    
    return selected_parameters;
}

const std::vector<EvaluateParameterOutput> Expansion::evaluateParameters(const std::vector<std::reference_wrapper<Parameter>>& parameters) {
    logger_.info("Evaluating expansion parameters...");
    std::vector<EvaluateParameterOutput> evaluation_outputs;
    std::string config_file_path;

    for (auto& param_ref : parameters) {
        Parameter& param = param_ref.get();
        logger_.info("Evaluating parameter: ", param.getName());

        // Placeholder: Generate configurations for the parameter
        std::vector<Configuration> generated_configurations;
        // For each value of the parameter, create a configuration from the best known configuration
        const Configuration* best_config = memory_.getBestConfiguration();
        if (best_config == nullptr) {
            logger_.info("No best configuration in memory, using default configuration for evaluation.");
            best_config = &memory_.getDefaultConfiguration();
        }
        for (const auto& value : param.getValues()) {
            config_file_path = expansion_working_dir_ + "config_param_" + param.getName() + "_" + value.getString() + "_iter_" + std::to_string(iteration_) + ".prm";
            std::map<std::string, Value> config_map = best_config->getConfiguration();
            config_map.insert_or_assign(param.getName(), value);

            Configuration config(config_map);
            config.generateConfigFile(config_file_path);

            CPLEXSolver solver(logger_, instance_file_, config_file_path, solver_log_file_, nb_threads_solver_, cutoff_solver_time_);
            solver.solve();
            double objective = solver.getObjectiveValue();
            config.setObjective(objective);

            generated_configurations.push_back(config);
            logger_.debug("Generated configuration for parameter ", param.getName(), " with value ", value.getString(), " - Objective: ", objective);
        }
        

        evaluation_outputs.push_back({param, generated_configurations});
    }

    return evaluation_outputs;
}

const std::vector<ClassifyParameterOutput> Expansion::classifyParameters(const std::vector<EvaluateParameterOutput>& evaluation_results) {
    logger_.info("Classifying expansion parameters...");
    std::vector<ClassifyParameterOutput> classified_parameters;

    for (const auto& eval_output : evaluation_results) {
        Parameter& param = eval_output.parameter;
        const auto& configurations = eval_output.configurations;

        // Simple classification logic: if the best configuration with this parameter is better than the best overall, select it
        const Configuration* best_config = memory_.getBestConfiguration();
        if (best_config == nullptr) {
            logger_.info("No best configuration in memory, cannot classify parameter ", param.getName());
            classified_parameters.push_back({param, false, false});
            continue;
        }

        //TODO: More sophisticated classification logic can be implemented here

        double best_objective = best_config->getObjective();
        double param_best_objective = std::numeric_limits<double>::max();
        for (const auto& config : configurations) {
            if (config.getObjective() < param_best_objective) {
                param_best_objective = config.getObjective();
            }
        }

        bool toSelect = param_best_objective < best_objective;
        bool toDiscard = !toSelect && (param_best_objective > best_objective * 1.0); // Discard if significantly worse

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
            param.setIsTuned(true);
            param.setIsResidual(false);
            logger_.info("Parameter ", param.getName(), " selected for tuning.");
        } else if (cp.toDiscard) {
            param.setIsDiscarded(true);
            param.setIsTuned(false);
            param.setIsResidual(false);
            logger_.info("Parameter ", param.getName(), " discarded from tuning.");
        } else {
            logger_.info("Parameter ", param.getName(), " remains residual.");
        }
    }
}