// MPILS: Multi-Phase Iterated Local Search Tuner
//
// Author: Youri Rigaud
// License: GNU GPLv3

#include "../include/expansion.h"
#include "../include/solver.h"

#include <algorithm>

void Expansion::run() {
    logger_.info("Starting expansion phase...");

    const std::vector<std::reference_wrapper<Parameter>> expansion_parameters = selectParameters();
    if (expansion_parameters.empty()) {
        logger_.info("No expansion parameters selected, skipping expansion.");
        return;
    }
    logger_.info("Selected expansion parameters: ", expansion_parameters.size());

    const std::vector<CreateConfigurationsOutput> configuration_files = createConfigurationsFiles(expansion_parameters);

    const std::vector<EvaluateParameterOutput> evaluation_results = evaluateParameters(configuration_files);
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

void addToEvaluateParameters(Parameter& param, const Configuration& config, double objective_value, std::vector<EvaluateParameterOutput>& evaluation_outputs) {
    // Find or create EvaluateParameterOutput for the parameter
    auto it = std::find_if(evaluation_outputs.begin(), evaluation_outputs.end(),
                           [&param](const EvaluateParameterOutput& epo) { return &epo.parameter == &param; });
    if (it == evaluation_outputs.end()) {
        evaluation_outputs.push_back({param, {}});
        it = std::prev(evaluation_outputs.end());
    }

    // Add the evaluated configuration
    Configuration evaluated_config = config;
    evaluated_config.setObjective(objective_value);
    it->configurations.push_back(evaluated_config);
}

const std::vector<EvaluateParameterOutput> Expansion::evaluateParameters(const std::vector<CreateConfigurationsOutput>& configuration_files_outputs) {
    logger_.info("Evaluating expansion parameters...");
    std::vector<EvaluateParameterOutput> evaluation_outputs;
    bool stop_evaluation {false};
    const Configuration* best_config = memory_.getBestConfiguration();
    if (best_config == nullptr) {
        logger_.info("No best configuration in memory, using default configuration for evaluation.");
        best_config = &memory_.getDefaultConfiguration();
    }

    for (auto& create_output : configuration_files_outputs) {
        Parameter& param = create_output.parameter;
        const std::string& config_file_path = create_output.config_file_path;

        logger_.info("Evaluating parameter: ", param.getName());

        std::vector<Configuration> evaluated_configurations;

        CPLEXSolver solver(
            logger_,
            instance_file_,
            config_file_path,
            solver_log_file_,
            nb_threads_solver_,
            cutoff_solver_time_
        );

        solver.solve();
        double objective_value = solver.getObjectiveValue();

        addToEvaluateParameters(param, create_output.configuration, objective_value, evaluation_outputs);
        std::string evaluated_value = create_output.configuration.getConfiguration().at(param.getName()).getString();
        logger_.info("Evaluated configuration for parameter ", param.getName(), " with value: ", evaluated_value, " with objective: ", objective_value);
    
        if (objective_value < best_config->getObjective()) {
            logger_.info("New best configuration found with objective: ", objective_value);
            stop_evaluation = true;
        }

        if (stop_evaluation) {
            break;
        }
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
        bool toDiscard = !toSelect && (param_best_objective >= best_objective * 1.0); // Discard if significantly worse

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
