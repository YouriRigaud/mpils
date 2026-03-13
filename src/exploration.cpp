// MPILS: Multi-Phase Iterated Local Search Tuner
//
// Author: Youri Rigaud
// License: GNU GPLv3

#include "../include/exploration.h"
#include "../include/tuner.h"
#include "../include/globaltimer.h"
#include "../include/solver.h"

#ifdef USE_MPI
#include <mpi.h>
#endif

#include <vector>
#include <string>
#include <fstream>
#include <filesystem>
#include <optional>

void Exploration::updateTunedParameters() {
    for (auto& param : parameter_space_.getParameters()) {
        if (param.isSelected()) {
            param.setIsTuned(true);
            param.setIsSelected(false);
        }
    }
}

std::vector<Configuration> Exploration::selectInitialConfigurations() {
    // Warning: The initial configurations vector must contain at least nb_workers_ configurations (these could be duplicates)
    // Implementation to select initial configurations for tuning phase
    logger_.info("Selecting initial configurations for tuning phase...");
    std::vector<Configuration> initial_configurations;
    if (memory_.getBestConfiguration() != nullptr) {
        logger_.info("Using best configuration from memory as initial configuration.");
        initial_configurations.push_back(*(memory_.getBestConfiguration()));
    } else {
        logger_.info("No best configuration in memory, using default initial configuration.");
        initial_configurations.push_back(memory_.getDefaultConfiguration()); // Return default configuration if no best found
    }
    //Todo: implementation is for test only, after use random configurations or other strategies
    while (initial_configurations.size() < static_cast<size_t>(nb_workers_)) {
        // add a random configuration, random only on parameter that are tuned, default value for others
        std::map<std::string, Value> config_map;
        for (const auto& param : parameter_space_.getParameters()) {
            if (param.isTuned()) {
                // select a random value from the parameter's possible values
                const auto& values = param.getValues();
                size_t random_index = rand() % values.size();
                config_map.insert_or_assign(param.getName(), values[random_index]);
            } else {
                config_map.insert_or_assign(param.getName(), param.getDefaultValue());
            }
        }
        initial_configurations.push_back(Configuration(config_map));
    }
    return initial_configurations;
}

int Exploration::selectNumberOfEvaluations() {
    // Implementation to select number of evaluations for tuning phase
    logger_.info("Selecting number of evaluations for tuning phase...");
    int factor;
    switch (iteration_)
    {
    case 1:
        factor = 2;
        return factor * (parameter_space_.getSelectedParameters().size());
    case 2:
        factor = 3;
        break;
    case 3:
        factor = 5;
        break;
    case 4:
        factor = 7;
        break;
    default:
        factor = 9;
        break;
    }
    return factor * parameter_space_.getSelectedParameters().size();
}

void Exploration::run() {
    logger_.info("Starting exploration phase...");
    int nb_evaluations = selectNumberOfEvaluations();
    if (nb_evaluations > 30) {
        nb_evaluations = 30;
    }
    updateTunedParameters();
    std::vector<Configuration> initial_configurations = selectInitialConfigurations();
    logger_.info("Number of evaluations for this tuning phase: ", nb_evaluations);

    setEngine(std::make_unique<IteratedLocalSearchEngine>(
        memory_,
        logger_,
        initial_configurations,
        parameter_space_,
        instance_file_,
        param_ils_instance_file_,
        solver_log_file_,
        nb_evaluations,
        iteration_,
        nb_threads_solver_,
        cutoff_solver_time_,
        nb_workers_
    ));

    if (!engine_) {
        logger_.info("No local search engine set for exploration.");
        setEngine(std::make_unique<IteratedLocalSearchEngine>(
            memory_,
            logger_,
            initial_configurations,
            parameter_space_,
            instance_file_,
            param_ils_instance_file_,
            solver_log_file_,
            nb_evaluations,
            iteration_,
            nb_threads_solver_,
            cutoff_solver_time_,
            nb_workers_
        ));
        logger_.info("Default IteratedLocalSearchEngine has been set.");
    }
    
    const std::vector<std::pair<int, std::vector<EvaluationRecord>>> evaluations = engine_->run();

    nb_evaluations = 0;
    for (const auto& evaluation_pair : evaluations) {
        int worker_id = evaluation_pair.first;
        const std::vector<EvaluationRecord>& worker_evaluations = evaluation_pair.second;
      //  memory_.addConfigurations(worker_configs, worker_id, iteration_, 0); // Phase 0 for exploration
        nb_evaluations += worker_evaluations.size();
    }
    
    logger_.info("Added", nb_evaluations, " evaluations to memory from exploration phase at iteration ", iteration_);

    logger_.info("Exploration phase completed.");
}

std::vector<std::pair<int, std::vector<EvaluationRecord>>> ParamILSEngine::run() {
    // Implementation of the ParamILS algorithm
    logger_.info("Running ParamILS Engine...");
    if (iteration_ > 1) {
        mip_start_ = true;
    }
    logger_.info("Mip start is ", mip_start_ ? "enabled" : "disabled", " for this iteration.");
    
    if (mip_start_) {
        setMipStartFile();
    }

    writeParamILSParameterFiles();
    writeParamILSScenarioFiles();

    // All workers call ParamILS at the same time and wait for each other
#ifdef USE_MPI
    launchLocalSearchWorkers();
#endif
    callParamILS();
#ifdef USE_MPI
    waitLocalSearchWorkers();
#endif

    std::vector<std::pair<int, std::vector<EvaluationRecord>>> results = getParamILSResults();

    logger_.info("ParamILS Engine completed.");
    return results;
}

void ParamILSEngine::writeParameterOptionsToFile(std::ofstream& myfile, int worker_id) {
    // Implementation to write the parameter space to file
    std::vector<std::pair<std::string, Value>>& forbidden_values = parameter_space_.getForbiddenValues();

    for (auto& param : parameter_space_.getParameters()) {
        Value initial_value = initial_configurations_[worker_id].getConfigurationMap().at(param.getName());
        if (worker_id == 1 && mip_start_ && !mip_start_file_.empty()) { // We use the same initial configuration for worker 1 as worker 0 but with a mip start 
            initial_value = initial_configurations_[0].getConfigurationMap().at(param.getName());
        }
        myfile << param.getName() << " {";
        if (param.isTuned()) {
            const auto& values = param.getValues();
            for (size_t i = 0; i < values.size(); ++i) {
                // Check if this value is forbidden
                bool is_forbidden = false;
                for (const auto& forbidden_pair : forbidden_values) {
                    if (forbidden_pair.first == param.getName() && forbidden_pair.second.getString() == values[i].getString()) {
                        is_forbidden = true;
                        // if it is the initial value, log a warning
                        if (values[i] == initial_value) {
                            logger_.info("Warning: Initial value for parameter ", param.getName(), " is forbidden. So unforbidden it for this worker ", i, " at iteration ", iteration_, ".");
                            is_forbidden = false;
                        }
                        break;
                    }
                }
                if (is_forbidden) {
                    continue; // Skip forbidden values
                }
                myfile << values[i].getString();
                if (i < values.size() - 1) {
                    myfile << ",";
                }
            }
        } else {
            myfile << initial_value.getString();
        }
        myfile << "} [" << initial_value.getString() << "]" << std::endl;
    }
    myfile << "process_mpi { " << worker_id << " } [ " << worker_id << " ]" << std::endl;
    myfile << "iteration { " << iteration_ << " } [ " << iteration_ << " ]" << std::endl;
    
    if (worker_id == 1 && mip_start_ && !mip_start_file_.empty()) {
        myfile << "mip_start { " << mip_start_file_ << " } [ " << mip_start_file_ << " ]" << std::endl;
    }
}

void ParamILSEngine::writeForbiddenOptionsToFile(std::ofstream& myfile, int worker_id) {
    // Implementation to write forbidden options to file
    std::vector<std::vector<std::pair<std::string, Value>>>& forbidden_tuples = parameter_space_.getForbiddenTuples();
    // For each forbidden tuple we have to look if the initial configuration contains it, so we do not forbid it
    Configuration initial_config = initial_configurations_[worker_id];

    for (const auto& tuple : forbidden_tuples) {
        bool is_forbidden = true;
        for (const auto& pair : tuple) {
            const std::string& param_name = pair.first;
            const Value& forbidden_value = pair.second;
            if (initial_config.getConfigurationMap().at(param_name).getString() == forbidden_value.getString()) {
                continue;
            } else {
                is_forbidden = false;
                break;
            }
        }
        if (is_forbidden) {
            logger_.info("Warning: Initial configuration contains a forbidden tuple. So unforbidden it for this worker at iteration ", iteration_, ".");
            continue; // Skip this forbidden tuple
        }

        myfile << "{";
        for (size_t i = 0; i < tuple.size(); ++i) {
            myfile << tuple[i].first << "=" << tuple[i].second.getString();
            if (i < tuple.size() - 1) {
                myfile << ", ";
            }
        }
        myfile << "}" << std::endl;
    }
}

void ParamILSEngine::writeConditionalCplexOptionsToFile(std::ofstream& myfile) {
    myfile << std::endl;
    myfile << std::endl;
    myfile << "Conditionals: " << std::endl;
    myfile << "mip_limits_gomorycand | mip_cuts_gomory in {0,1,2} # mip_cuts_gomory just can't be -1  " << std::endl;
    myfile << "mip_limits_strongcand | mip_strategy_variableselect in {3} " << std::endl;
    myfile << "mip_limits_strongit | mip_strategy_variableselect in {3} " << std::endl;
    myfile << "mip_limits_submipnodelim | mip_strategy_rinsheur in {0,5,10,20,40,80} # RINSHEUR not -1  " << std::endl;
    myfile << "mip_strategy_bbinterval | mip_strategy_nodeselect in {2} " << std::endl;
    myfile << "preprocessing_numpass | preprocessing_presolve in {yes} " << std::endl;
    myfile << "mip_strategy_order | mip_ordertype in {1,2,3} " << std::endl;
}

void ParamILSEngine::writeParamILSParameterFiles() {
    // Implementation to write the ParamILS parameter files
    logger_.info("Writing ParamILS parameter files...");

    for (int i = 0; i < nb_workers_; ++i) {
        std::string parameter_file_path = param_ils_working_dir_ + "parameter/parameter_file_" + std::to_string(iteration_) + "_worker_" + std::to_string(i) + ".txt";

        // open parameter file
        std::ofstream myfile;
        myfile.open(parameter_file_path);
        if (!myfile.is_open()) {
            logger_.info("Error opening ParamILS parameter file for writing.");
            return;
        }

        writeParameterOptionsToFile(myfile, i);

        writeForbiddenOptionsToFile(myfile, i);

        writeConditionalCplexOptionsToFile(myfile);

        myfile.close();
    }

    logger_.info("ParamILS parameter files written.");
}

void ParamILSEngine::writeParamILSScenarioFiles() {
    // Implementation to write the ParamILS scenario file
    logger_.info("Writing ParamILS scenario files...");

    for (int i = 0; i < nb_workers_; ++i) {
        std::string scenario_file_path = param_ils_working_dir_ + "scenario/scenario_file_" + std::to_string(iteration_) + "_worker_" + std::to_string(i) + ".txt";
        std::string parameter_file_path = param_ils_working_dir_ + "parameter/parameter_file_" + std::to_string(iteration_) + "_worker_" + std::to_string(i) + ".txt";
       
        std::string tuning_obj = "qual";

        std::ofstream myfile;
        myfile.open(scenario_file_path);
        myfile << "algo = ruby " + param_ils_dir_ + "cplex_wrapper.rb" << std::endl;
        myfile << "execdir = ." << std::endl;
        myfile << "deterministic = 1" << std::endl;
        myfile << "run_obj = " << tuning_obj << std::endl;
        myfile << "overall_obj = mean" << std::endl;
        myfile << "cutoff_time = " << cutoff_solver_time_ << std::endl;
        myfile << "maxEvals = " << max_evaluations_ << std::endl;
        myfile << "wallclock-limit = " << cutoff_solver_time_*max_evaluations_ << std::endl;
        myfile << "logfile = " << solver_log_file_ + "_iteration_paramils_" + std::to_string(iteration_) + "_worker_" + std::to_string(i) << std::endl;
        myfile << "paramfile = " << parameter_file_path << std::endl;
        myfile << "outdir = " + param_ils_working_dir_ + "paramils-out_" + std::to_string(iteration_) + "_worker_" + std::to_string(i) << std::endl;
        myfile << "instance_file = " << param_ils_instance_file_ << std::endl;
        myfile << "test_instance_file = " << param_ils_instance_file_ << std::endl;
        myfile.close();
    }

    logger_.info("ParamILS scenario files written.");
}

void ParamILSEngine::callParamILS() {
    // Implementation to call the ParamILS executable
    logger_.info("Calling ParamILS executable...");
    local_search_start_time_ = GlobalTimer::elapsedSeconds();

    std::string scenario_file_path = param_ils_working_dir_ + "scenario/scenario_file_" + std::to_string(iteration_) + "_worker_" + std::to_string(0) + ".txt";
    std::string command = "ruby " + param_ils_dir_ + param_ils_executable_ + " -numRun 0 -scenariofile " + scenario_file_path;
    int ret = system(command.c_str());
    if (ret != 0) {
        logger_.info("Error calling ParamILS executable.");
    }

    logger_.info("ParamILS executable call completed.");
}

const std::vector<std::pair<int, std::vector<EvaluationRecord>>> ParamILSEngine::getParamILSResults() {
    // Implementation to get results from ParamILS
    logger_.info("Retrieving ParamILS results...");
    std::vector<std::pair<int, std::vector<EvaluationRecord>>> results;
    // For each worker, parse its CPLEX log file
    for (int i = 0; i < nb_workers_; ++i) {
        std::vector<EvaluationRecord> worker_results = parseCplexResultsFromLogFile(0, i);
        results.insert(results.end(), std::make_pair(i, worker_results));
    }
    logger_.info("ParamILS results retrieved: ", results.size(), " configurations found.");
    return results;
}

const std::vector<EvaluationRecord> LocalSearchEngine::parseCplexResultsFromLogFile(int run_obj, int worker_id) {
    std::vector<EvaluationRecord> results;
    double local_search_elapsed_time = 0;

    std::string solver_log_worker = solver_log_file_ + "_iteration_paramils_" + std::to_string(iteration_) + "_worker_" + std::to_string(worker_id);

    std::ifstream file(solver_log_worker);
    if (!file) {
        logger_.info("Cannot open CPLEX log file: " + solver_log_worker);
        return results;
    }

    std::string line;
    bool inside_block = false;
    std::map<std::string, Value> params;
    double gap = -1.0;
    double time_sec = -1.0;

    auto flush_block = [&]() {
        if (!inside_block) return;
        if (params.empty()) return;
        if (gap < 0 && time_sec < 0) return;

        double obj = (run_obj == 0 ? gap : time_sec);

        // Before adding, ensure all parameters from parameter space are included
        for (const auto& param : parameter_space_.getParameters()) {
            if (params.find(param.getName()) == params.end()) {
                const Value default_value = param.getDefaultValue();
                params.emplace(param.getName(), default_value);
            }
        }

        // Get the elapsed time of the config evaluation
        if (time_sec >= 0) {
            local_search_elapsed_time += time_sec;
        } else {
            local_search_elapsed_time += cutoff_solver_time_;
        }
        int config_elapsed_time = local_search_start_time_ + local_search_elapsed_time;

        // Record the evaluation result for this configuration
        RecordEvaluationOptions options;
        if (worker_id == 1 && mip_start_) {
            options.mip_start_used = true;
            options.used_mip_start_id = memory_.getMipStartToUse();
             logger_.info("Using MIP start with id ", options.used_mip_start_id.value(), " for this evaluation of worker ", worker_id, " at iteration ", iteration_, ".");
        } else {
            options.mip_start_used = false;
        }
        options.produced_mip_start = false; // Exploration phase does not produce MIP starts
        options.objective_value = obj;
        options.time_evaluated = config_elapsed_time;
        options.worker_id = worker_id;
        options.phase = 0; // Phase 0 for exploration
        options.iteration = iteration_;

        EvaluationId eval_id = memory_.recordEvaluation(Configuration(params), options);

        EvaluationRecord eval = *memory_.getEvaluationById(eval_id);
        results.push_back(eval);

        params.clear();
        gap = -1.0;
        time_sec = -1.0;
        inside_block = false;
    };

    while (std::getline(file, line)) {

        // Start of new configuration block
        if (line.rfind("Version identifier:", 0) == 0) {
            flush_block();     // finish previous block
            inside_block = true;
            continue;
        }

        // Parse CPXPARAM lines
        if (inside_block && line.rfind("CPXPARAM_", 0) == 0) {
            // Format:
            // CPXPARAM_Name   value
            std::istringstream iss(line);
            std::string name;
            std::string val;
            iss >> name >> val;
            params.emplace(name, Value(val));
            continue;
        }

        // GAP PARSING (robust for all CPLEX formats)
        if (inside_block && line.find("gap") != std::string::npos) {
        
            // Case 1: "(gap is infinite)"
            if (line.find("gap is infinite") != std::string::npos) {
                gap = 1e100;
                continue;
            }
        
            // Case 2: "(gap is X)"
            if (line.find("gap is") != std::string::npos) {
                size_t pos = line.find("gap is") + 6;
                try {
                    gap = std::stod(line.substr(pos));
                } catch (...) {
                    gap = 1e100;
                }
                continue;
            }
        
            // Case 3: "(gap = 602, 79.95%)"
            if (line.find("gap =") != std::string::npos) {
            
                size_t pos = line.find("gap =") + 5;
                std::string rest = line.substr(pos);
            
                // Expected: "602, 79.95%)"
                size_t comma = rest.find(',');
                size_t percent = rest.find('%');
            
                if (comma != std::string::npos && percent != std::string::npos) {
                    std::string rel = rest.substr(comma + 1, percent - comma - 1);
                    try {
                        gap = std::stod(rel);  // use relative gap
                    } catch (...) {
                        gap = 1e100;
                    }
                }
                continue;
            }
        }

        // Case "MIP - Integer optimal solution: ..."
        if (inside_block && line.find("MIP - Integer optimal solution:") != std::string::npos) {
            gap = 0.0;
            continue;
        }

        // Parse solution time
        if (inside_block && line.rfind("Solution time =", 0) == 0) {
            std::istringstream iss(line);
            std::string tmp;
            iss >> tmp >> tmp >> tmp >> time_sec; // "Solution time = 0.32"
            continue;
        }
    }

    // flush last block
    flush_block();

    return results;
}

void LocalSearchEngine::setMipStartFile() {
    Configuration best_config = memory_.getBestConfiguration() != nullptr ? *memory_.getBestConfiguration() : memory_.getDefaultConfiguration();
    mip_start_file_ = "tuner_working_dir/mip_start/mip_start_iteration_" + std::to_string(iteration_) + ".mst";
    std::string config_path = "tuner_working_dir/config_for_mip_start/config_mip_start_iteration_" + std::to_string(iteration_) + ".prm";
    best_config.generateConfigFile(config_path);
    // Call cplex on the best configuration found so far to generate the mip start file
    CPLEXSolver solver(logger_, instance_file_, config_path, solver_log_file_ + "_mip_start_" + std::to_string(iteration_), nb_threads_solver_, cutoff_solver_time_, mip_start_file_);
    solver.solve();

    if (!std::filesystem::exists(mip_start_file_)) {
        logger_.warn("Expected mip start file not found: ", mip_start_file_);
    }

    double objective_value = solver.getObjectiveValue();
    int evaluated_time = GlobalTimer::elapsedSeconds();
    logger_.info("Generated MIP start file for iteration ", iteration_, " with objective value ", objective_value, " and evaluation time ", evaluated_time, " seconds.");
    RecordEvaluationOptions options;
    options.mip_start_used = false; // This evaluation is not using a mip start, it is producing one
    options.objective_value = objective_value;
    options.time_evaluated = evaluated_time;
    options.worker_id = 0;
    options.phase = 0; // Phase 0 for exploration
    options.iteration = iteration_;
    options.produced_mip_start = true;
    options.mip_start_file = mip_start_file_;
    logger_.info("Producing mip start file path = '", mip_start_file_, "'");
    memory_.recordEvaluation(best_config, options);
}

std::optional<double> IteratedLocalSearchEngine::getKnownInitialObjective_() const {
    if (initial_configurations_.empty()) {
        return std::nullopt;
    }

    const Configuration& initial_config = initial_configurations_[0];
    try {
        const ConfigurationStats& stats = memory_.getConfigurationStatsById(initial_config.getConfigurationId());
        if (stats.nb_evaluations > 0) {
            return stats.best_objective;
        }
    } catch (...) {
        // configuration not yet known in memory
    }

    return std::nullopt;
}

void IteratedLocalSearchEngine::writeILSParameterOptionsToFile(std::ofstream& myfile) {
    std::vector<std::pair<std::string, Value>>& forbidden_values = parameter_space_.getForbiddenValues();

    const Configuration& initial_config = initial_configurations_[0];

    for (auto& param : parameter_space_.getParameters()) {
        Value initial_value = initial_config.getConfigurationMap().at(param.getName());

        myfile << param.getName() << " {";
        if (param.isTuned()) {
            const auto& values = param.getValues();
            bool first_written = true;
            for (const auto& value : values) {
                bool is_forbidden = false;
                for (const auto& forbidden_pair : forbidden_values) {
                    if (forbidden_pair.first == param.getName() &&
                        forbidden_pair.second.getString() == value.getString()) {
                        is_forbidden = true;
                        if (value == initial_value) {
                            logger_.info("Warning: Initial value for parameter ", param.getName(),
                                         " is forbidden. So unforbidden it for the ILS initial configuration at iteration ",
                                         iteration_, ".");
                            is_forbidden = false;
                        }
                        break;
                    }
                }

                if (is_forbidden) {
                    continue;
                }

                if (!first_written) {
                    myfile << ",";
                }
                myfile << value.getString();
                first_written = false;
            }
        } else {
            myfile << initial_value.getString();
        }
        myfile << "} [" << initial_value.getString() << "]" << std::endl;
    }
}

void IteratedLocalSearchEngine::writeILSForbiddenOptionsToFile(std::ofstream& myfile) {
    std::vector<std::vector<std::pair<std::string, Value>>>& forbidden_tuples = parameter_space_.getForbiddenTuples();
    const Configuration& initial_config = initial_configurations_[0];

    for (const auto& tuple : forbidden_tuples) {
        bool initial_contains_tuple = true;
        for (const auto& pair : tuple) {
            const std::string& param_name = pair.first;
            const Value& forbidden_value = pair.second;

            if (initial_config.getConfigurationMap().at(param_name).getString() == forbidden_value.getString()) {
                continue;
            } else {
                initial_contains_tuple = false;
                break;
            }
        }

        if (initial_contains_tuple) {
            logger_.info("Warning: Initial configuration contains a forbidden tuple. So unforbidden it for ILS at iteration ",
                         iteration_, ".");
            continue;
        }

        myfile << "{";
        for (size_t i = 0; i < tuple.size(); ++i) {
            myfile << tuple[i].first << "=" << tuple[i].second.getString();
            if (i < tuple.size() - 1) {
                myfile << ", ";
            }
        }
        myfile << "}" << std::endl;
    }
}

void IteratedLocalSearchEngine::writeILSConditionalCplexOptionsToFile(std::ofstream& myfile) {
    myfile << std::endl;
    myfile << "Conditionals:" << std::endl;

    myfile << "CPXPARAM_MIP_Limits_GomoryCand | CPXPARAM_MIP_Cuts_Gomory in {0,1,2} "
           << "# CPXPARAM_MIP_Cuts_Gomory just can't be -1" << std::endl;

    myfile << "CPXPARAM_MIP_Limits_StrongCand | CPXPARAM_MIP_Strategy_VariableSelect in {3}" << std::endl;
    myfile << "CPXPARAM_MIP_Limits_StrongIt | CPXPARAM_MIP_Strategy_VariableSelect in {3}" << std::endl;

    myfile << "CPXPARAM_MIP_Strategy_BBInterval | CPXPARAM_MIP_Strategy_NodeSelect in {2}" << std::endl;

    myfile << "CPXPARAM_Preprocessing_NumPass | CPXPARAM_Preprocessing_Presolve in {1}" << std::endl;

    myfile << "CPXPARAM_MIP_Strategy_Order | CPXPARAM_MIP_OrderType in {1,2,3}" << std::endl;
}

void IteratedLocalSearchEngine::writeILSInfoToFile(std::ofstream& myfile) {
    myfile << std::endl;
    myfile << "Info:" << std::endl;

    const std::optional<double> known_objective = getKnownInitialObjective_();
    if (known_objective.has_value()) {
        myfile << "Initial configuration evaluated: 1" << std::endl;
        myfile << "Objective value of the initial configuration: " << known_objective.value() << std::endl;
    } else {
        myfile << "Initial configuration evaluated: 0" << std::endl;
    }
}

void IteratedLocalSearchEngine::writeILSSearchSpaceFile() {
    logger_.info("Writing ILS search space file...");

    std::filesystem::create_directories(ils_working_dir_ + "search_space/");
    search_space_file_ = ils_working_dir_ + "search_space/search_space_file_" + std::to_string(iteration_) + ".txt";

    std::ofstream myfile(search_space_file_);
    if (!myfile.is_open()) {
        throw std::runtime_error("Error opening ILS search space file for writing: " + search_space_file_);
    }

    writeILSParameterOptionsToFile(myfile);
    writeILSForbiddenOptionsToFile(myfile);
    writeILSConditionalCplexOptionsToFile(myfile);
    writeILSInfoToFile(myfile);

    myfile.close();

    logger_.info("ILS search space file written: ", search_space_file_);
}

std::vector<EvaluationRecord> IteratedLocalSearchEngine::syncILSResultsToGlobalMemory_(
    const std::vector<std::pair<Configuration, EvaluationRecord>>& local_results
) {
    std::vector<EvaluationRecord> synced_results;

    const std::optional<double> known_initial_objective = getKnownInitialObjective_();
    const Configuration& initial_config = initial_configurations_[0];

    for (const auto& pair : local_results) {
        const Configuration& config = pair.first;
        const EvaluationRecord& local_record = pair.second;

        // If the initial configuration was already known before launching ILS,
        // then the injected local-cache record should not be duplicated in global memory.
        if (known_initial_objective.has_value() &&
            config == initial_config &&
            local_record.objective_value == known_initial_objective.value()) {
            continue;
        }

        RecordEvaluationOptions options;
        options.objective_value = local_record.objective_value;
        options.time_evaluated = local_record.time_evaluated;
        options.worker_id = 0;
        options.iteration = iteration_;
        options.phase = 0;

        options.mip_start_used = config.useMipStart();
        if (config.useMipStart() && mip_start_) {
            options.used_mip_start_id = memory_.getMipStartToUse();
        }

        options.produced_mip_start = false;

        EvaluationId eval_id = memory_.recordEvaluation(config, options);
        const EvaluationRecord* global_eval = memory_.getEvaluationById(eval_id);
        if (global_eval != nullptr) {
            synced_results.push_back(*global_eval);
        }
    }

    return synced_results;
}

std::vector<std::pair<int, std::vector<EvaluationRecord>>> IteratedLocalSearchEngine::run() {
    logger_.info("Running IteratedLocalSearch Engine...");

    if (iteration_ > 1) {
        mip_start_ = true;
    }
    logger_.info("Mip start is ", mip_start_ ? "enabled" : "disabled", " for this iteration.");

    if (mip_start_) {
        setMipStartFile();
    }

    writeILSSearchSpaceFile();

    IteratedLocalSearch::Options ils_options;
    ils_options.search_space_file = search_space_file_;
    ils_options.instance_file = instance_file_;
    ils_options.working_directory = ils_working_dir_ + "run_" + std::to_string(iteration_);
    ils_options.random_seed = static_cast<unsigned int>(iteration_);
    ils_options.evaluation_budget = max_evaluations_;
    ils_options.perturbation_strength = 3;
    ils_options.random_initial_samples = 0;
    ils_options.restart_probability = 0.10;
    ils_options.accept_ties = false;
    ils_options.acceptance_threshold = 0.01;
    ils_options.use_mip_starts = mip_start_ && !mip_start_file_.empty();
    if (ils_options.use_mip_starts) {
        ils_options.mip_start_file = mip_start_file_;
    }
    ils_options.nb_threads_solver = nb_threads_solver_;
    ils_options.cutoff_solver_time = cutoff_solver_time_;

    ils_ = std::make_unique<IteratedLocalSearch>(logger_, ils_options);
    ils_->run();

    // This accessor must exist in IteratedLocalSearch.
    const auto local_results = ils_->getEvaluationsWithConfigurations();
    std::vector<EvaluationRecord> synced_results = syncILSResultsToGlobalMemory_(local_results);

    logger_.info("IteratedLocalSearch Engine completed.");
    return { {0, synced_results} };
}

#ifdef USE_MPI
void ParamILSWorker::callParamILS() {
    std::string scenario_file_path = param_ils_working_dir_ + "scenario/scenario_file_" + std::to_string(iteration_) + "_worker_" + std::to_string(worker_id_) + ".txt";
    std::string command = "ruby " + param_ils_dir_ + param_ils_executable_ + " -numRun 0 -scenariofile " + scenario_file_path;
    int ret = system(command.c_str());
    if (ret != 0) {
        std::cout << "Error calling ParamILS executable for worker " << worker_id_ << " at iteration " << iteration_ << std::endl;
    }
}

void LocalSearchEngine::launchLocalSearchWorkers() {
    logger_.info("Launching Local Search Workers via MPI...");
    // Implementation to launch local search workers using MPI
    // Broadcast to give all workers worker_step = 1 and iteration_
    WorkerOrder order;
    order.step = 1; // exploration step
    order.iteration = iteration_;
    MPI_Bcast(&order, sizeof(WorkerOrder), MPI_BYTE, 0, MPI_COMM_WORLD);
    logger_.info("Local Search Workers launched.");
}

void LocalSearchEngine::waitLocalSearchWorkers() {
    logger_.info("Waiting for Local Search Workers via MPI...");
    // Implementation to wait for local search workers using MPI
    MPI_Barrier(MPI_COMM_WORLD);
    logger_.info("Local Search Workers have completed.");
}
#endif