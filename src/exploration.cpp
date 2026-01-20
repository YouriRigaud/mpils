// MPILS: Multi-Phase Iterated Local Search Tuner
//
// Author: Youri Rigaud
// License: GNU GPLv3

#include "../include/exploration.h"

#include <vector>
#include <string>
#include <fstream>

void Exploration::updateTunedParameters() {
    for (auto& param : parameter_space_.getParameters()) {
        if (param.isSelected()) {
            param.setIsTuned(true);
            param.setIsSelected(false);
        }
    }
}

std::vector<Configuration> Exploration::selectInitialConfigurations() {
    // Implementation to select initial configurations for tuning phase
    logger_.info("Selecting initial configurations for tuning phase...");
    if (memory_.getBestConfiguration() != nullptr) {
        logger_.info("Using best configuration from memory as initial configuration.");
        return {*(memory_.getBestConfiguration())};
    } else {
        logger_.info("No best configuration in memory, using default initial configuration.");
        return {memory_.getDefaultConfiguration()}; // Return default configuration if no best found
    }
    return {};
}

int Exploration::selectNumberOfEvaluations() {
    // Implementation to select number of evaluations for tuning phase
    logger_.info("Selecting number of evaluations for tuning phase...");
    switch (iteration_)
    {
    case 1:
        return 2 * parameter_space_.getSelectedParameters().size();
    case 2:
        return 3 * parameter_space_.getSelectedParameters().size();
    case 3:
        return 5 * parameter_space_.getSelectedParameters().size();
    case 4:
        return 7 * parameter_space_.getSelectedParameters().size();
    default:
        return 9 * parameter_space_.getSelectedParameters().size();
    }
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

    setEngine(std::make_unique<ParamILSEngine>(logger_, initial_configurations[0], parameter_space_, instance_file_, solver_log_file_, nb_evaluations, iteration_, nb_threads_solver_, cutoff_solver_time_));


    if (!engine_) {
        logger_.info("No local search engine set for exploration.");
        setEngine(std::make_unique<ParamILSEngine>(logger_, initial_configurations[0], parameter_space_, instance_file_, solver_log_file_, nb_evaluations, iteration_, nb_threads_solver_, cutoff_solver_time_));
        logger_.info("Default ParamILSEngine has been set.");
    }
    
    const std::vector<Configuration> configs = engine_->run();
    logger_.info("Adding configurations : ", configs.size(), " to memory from exploration phase at iteration ", iteration_);
    memory_.addConfigurations(configs);

    logger_.info("Exploration phase completed.");
}

std::vector<Configuration> ParamILSEngine::run() {
    // Implementation of the ParamILS algorithm
    logger_.info("Running ParamILS Engine...");
    
    writeParamILSParameterFile();
    writeParamILSScenarioFile();
    callParamILS();
    std::vector<Configuration> results = getParamILSResults();

    logger_.info("ParamILS Engine completed.");
    return results;
}

void ParamILSEngine::writeParameterOptionsToFile(std::ofstream& myfile) {
    // Implementation to write the parameter space to file
    std::vector<std::pair<std::string, Value>>& forbidden_values = parameter_space_.getForbiddenValues();

    for (auto& param : parameter_space_.getParameters()) {
        const Value initial_value = initial_configuration_.getConfiguration().at(param.getName());
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
                            logger_.info("Warning: Initial value for parameter ", param.getName(), " is forbidden. So unforbidden it.");
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
}

void ParamILSEngine::writeForbiddenOptionsToFile(std::ofstream& myfile) {
    // Implementation to write forbidden options to file
//    std::vector<std::pair<std::string, Value>>& forbidden_values = parameter_space_.getForbiddenValues();
//    for (const auto& pair : forbidden_values) {
//        myfile << "{" << pair.first << "=" << pair.second.getString() << "}" << std::endl;
//    }
    std::vector<std::vector<std::pair<std::string, Value>>>& forbidden_tuples = parameter_space_.getForbiddenTuples();
    // For each forbidden tuple we have to look if the initial configuration contains it, so we do not forbid it
    Configuration initial_config = initial_configuration_;

    for (const auto& tuple : forbidden_tuples) {
        bool is_forbidden = true;
        for (const auto& pair : tuple) {
            const std::string& param_name = pair.first;
            const Value& forbidden_value = pair.second;
            if (initial_config.getConfiguration().at(param_name).getString() == forbidden_value.getString()) {
                continue;
            } else {
                is_forbidden = false;
                break;
            }
        }
        if (is_forbidden) {
            logger_.info("Warning: Initial configuration contains a forbidden tuple. So unforbidden it.");
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

void ParamILSEngine::writeParamILSParameterFile() {
    // Implementation to write the ParamILS parameter file
    logger_.info("Writing ParamILS parameter file...");
    std::string parameter_file_path = param_ils_working_dir_ + "parameter/parameter_file_" + std::to_string(iteration_) + ".txt";

    // open parameter file
    std::ofstream myfile;
    myfile.open(parameter_file_path);
    if (!myfile.is_open()) {
        logger_.info("Error opening ParamILS parameter file for writing.");
        return;
    }

    writeParameterOptionsToFile(myfile);

    writeForbiddenOptionsToFile(myfile);

    writeConditionalCplexOptionsToFile(myfile);

    myfile.close();

    logger_.info("ParamILS parameter file written.");
}

void ParamILSEngine::writeParamILSScenarioFile() {
    // Implementation to write the ParamILS scenario file
    logger_.info("Writing ParamILS scenario file...");
    std::string scenario_file_path = param_ils_working_dir_ + "scenario/scenario_file_" + std::to_string(iteration_) + ".txt";
    std::string parameter_file_path = param_ils_working_dir_ + "parameter/parameter_file_" + std::to_string(iteration_) + ".txt";
   
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
    myfile << "logfile = " << solver_log_file_ << std::endl;
    myfile << "paramfile = " << parameter_file_path << std::endl;
    myfile << "outdir = " + param_ils_working_dir_ + "paramils-out" << std::endl;
    myfile << "instance_file = " << instance_file_ << std::endl;
    myfile << "test_instance_file = " << instance_file_ << std::endl;
    myfile.close();

    logger_.info("ParamILS scenario file written.");
}

void ParamILSEngine::callParamILS() {
    // Implementation to call the ParamILS executable
    logger_.info("Calling ParamILS executable...");

    std::string scenario_file_path = param_ils_working_dir_ + "scenario/scenario_file_" + std::to_string(iteration_) + ".txt";
    std::string command = "ruby " + param_ils_dir_ + param_ils_executable_ + " -numRun 0 -scenariofile " + scenario_file_path;
    int ret = system(command.c_str());
    if (ret != 0) {
        logger_.info("Error calling ParamILS executable.");
    }

    logger_.info("ParamILS executable call completed.");
}

const std::vector<Configuration> ParamILSEngine::getParamILSResults() {
    // Implementation to get results from ParamILS
    logger_.info("Retrieving ParamILS results...");
    std::vector<Configuration> results = parseCplexResultsFromLogFile(0);
    // Placeholder implementation
    return results;
}

const std::vector<Configuration> LocalSearchEngine::parseCplexResultsFromLogFile(int run_obj) {
    std::vector<Configuration> results;

    std::ifstream file(solver_log_file_);
    if (!file) {
        logger_.info("Cannot open CPLEX log file: " + solver_log_file_);
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

        results.emplace_back(params, obj);

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
            iss >> tmp >> tmp >> time_sec; // "Solution time = 0.32"
            continue;
        }
    }

    // flush last block
    flush_block();

    return results;
}
