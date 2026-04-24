// MPILS: Multi-Phase Iterated Local Search Tuner
//
// Author: Youri Rigaud
// License: GNU GPLv3

#include "../include/exploration.h"
#include "../include/tuner.h"
#include "../include/globaltimer.h"
#include "../include/solver.h"
#include "../include/filesystem_utils.h"

#ifdef USE_MPI
#include <mpi.h>
#endif

#include <vector>
#include <string>
#include <fstream>
#include <filesystem>
#include <optional>
#include <random>

namespace {

std::uint32_t computeInitialConfigurationSeed(std::uint32_t base_seed, int iteration) {
    return base_seed + 100000u + static_cast<std::uint32_t>(iteration - 1);
}

std::string buildParamILSCommand(const std::string& executable, std::uint32_t num_run, const std::string& scenario_file_path) {
    return "ruby " + executable + " -numRun " + std::to_string(num_run) + " -scenariofile " + scenario_file_path;
}

std::string buildHistoricalCacheSeedFilePath(const std::string& ils_working_dir, int iteration) {
    return ils_working_dir + "search_space/shared_cache_seed_" + std::to_string(iteration) + ".txt";
}

void writeHistoricalCacheSeedEntriesToFile(
    const std::string& filename,
    const std::vector<CompactSharedCacheSeedEntry>& seeds
) {
    ensureParentDirectoryForFile(filename);
    std::ofstream myfile(filename);
    if (!myfile.is_open()) {
        throw std::runtime_error("Error opening historical cache seed file for writing: " + filename);
    }

    for (const auto& seed : seeds) {
        myfile << "ConfigurationId=" << seed.configuration_id << std::endl;
        myfile << "ObjectiveValue=" << seed.objective_value << std::endl;
    }
}

#ifdef USE_MPI
std::vector<CompactSharedCacheSeedEntry> readHistoricalCacheSeedEntriesFromFile(
    const std::string& filename,
    Logger& logger
) {
    std::vector<CompactSharedCacheSeedEntry> seeds;

    std::ifstream file(filename);
    if (!file.is_open()) {
        logger.info("Historical cache seed file not found: ", filename);
        return seeds;
    }

    std::string line;
    double objective_value = kMaxObjective;
    ConfigurationId configuration_id = 0;
    bool has_configuration_id = false;
    bool has_objective_value = false;

    while (std::getline(file, line)) {
        if (line.empty()) {
            continue;
        }

        const std::size_t eq_pos = line.find('=');
        if (eq_pos == std::string::npos) {
            continue;
        }

        const std::string key = line.substr(0, eq_pos);
        const std::string value = line.substr(eq_pos + 1);

        if (key == "ConfigurationId") {
            configuration_id = static_cast<ConfigurationId>(std::stoull(value));
            has_configuration_id = true;
        } else if (key == "ObjectiveValue") {
            objective_value = std::stod(value);
            has_objective_value = true;
        }

        if (has_configuration_id && has_objective_value) {
            seeds.push_back(CompactSharedCacheSeedEntry{configuration_id, objective_value});
            configuration_id = 0;
            objective_value = kMaxObjective;
            has_configuration_id = false;
            has_objective_value = false;
        }
    }

    logger.info("Loaded ", static_cast<int>(seeds.size()), " historical cache seed entries from ", filename);
    return seeds;
}
#endif

} // namespace

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
    std::mt19937 rng(computeInitialConfigurationSeed(base_seed_, iteration_));
    while (initial_configurations.size() < static_cast<size_t>(nb_workers_)) {
        // add a random configuration, random only on parameter that are tuned, default value for others
        std::map<std::string, Value> config_map;
        for (const auto& param : parameter_space_.getParameters()) {
            if (param.isTuned()) {
                // select a random value from the parameter's possible values
                const auto& values = param.getValues();
                std::uniform_int_distribution<std::size_t> dist(0, values.size() - 1);
                std::size_t random_index = dist(rng);
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
    int nb_evaluations = 0;
    if (number_of_evaluations_override_.has_value()) {
        nb_evaluations = number_of_evaluations_override_.value();
        logger_.info("Using explicit evaluation budget override: ", nb_evaluations);
    } else {
        nb_evaluations = selectNumberOfEvaluations();
        if (nb_evaluations > 30) {
            nb_evaluations = 30;
        }
        if (nb_evaluations < 5) {
            nb_evaluations = 5;
        }
    }
    updateTunedParameters();
    std::vector<Configuration> initial_configurations = selectInitialConfigurations();
    logger_.info("Number of evaluations for this tuning phase: ", nb_evaluations);
    logger_.info("ILS shared cache is ", use_shared_cache_ ? "enabled" : "disabled", " for this exploration phase.");
    logger_.info("Local search backend for this exploration phase: ", localSearchBackendToString(local_search_backend_));

    switch (local_search_backend_) {
        case LocalSearchBackend::IteratedLocalSearch:
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
                solver_time_mode_,
                nb_workers_,
                use_shared_cache_,
                base_seed_,
                tuning_objective_,
                enable_mip_starts_,
                random_worker_initial_configs_
            ));
            break;
        case LocalSearchBackend::ParamILS:
            setEngine(std::make_unique<ParamILSEngine>(
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
                solver_time_mode_,
                nb_workers_,
                base_seed_,
                tuning_objective_,
                enable_mip_starts_
            ));
            break;
    }

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
            solver_time_mode_,
            nb_workers_,
            use_shared_cache_,
            base_seed_,
            tuning_objective_,
            enable_mip_starts_,
            random_worker_initial_configs_
        ));
        logger_.info("Default IteratedLocalSearchEngine has been set.");
    }
    
    const std::vector<std::pair<int, std::vector<EvaluationRecord>>> evaluations = engine_->run();

    nb_evaluations = 0;
    for (const auto& evaluation_pair : evaluations) {
        const std::vector<EvaluationRecord>& worker_evaluations = evaluation_pair.second;
        nb_evaluations += worker_evaluations.size();
    }
    
    logger_.info("Added", nb_evaluations, " evaluations to memory from exploration phase at iteration ", iteration_);

    logger_.info("Exploration phase completed.");
}

std::vector<std::pair<int, std::vector<EvaluationRecord>>> ParamILSEngine::run() {
    // Implementation of the ParamILS algorithm
    logger_.info("Running ParamILS Engine...");
    if (mip_start_ && iteration_ > 1 && nb_workers_ > 1) {
        mip_start_ = true;
    } else {
        mip_start_ = false;
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
        ensureParentDirectoryForFile(parameter_file_path);
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
        const std::string solver_working_dir =
            std::filesystem::path(solver_log_file_).parent_path().parent_path().string();
        const std::string paramils_outdir =
            param_ils_working_dir_ + "paramils-out_" + std::to_string(iteration_) +
            "_worker_" + std::to_string(i);

        ensureParentDirectoryForFile(scenario_file_path);
        ensureDirectoryExists(paramils_outdir);
        std::ofstream myfile;
        myfile.open(scenario_file_path);
        myfile << "algo = ruby " + param_ils_dir_ + "cplex_wrapper.rb --threads " + std::to_string(nb_threads_solver_) + " --work-dir " + solver_working_dir << std::endl;
        myfile << "execdir = ." << std::endl;
        myfile << "deterministic = 1" << std::endl;
        myfile << "run_obj = " << tuning_obj << std::endl;
        myfile << "overall_obj = mean" << std::endl;
        myfile << "cutoff_time = " << cutoff_solver_time_ << std::endl;
        myfile << "maxEvals = " << max_evaluations_ << std::endl;
        myfile << "wallclock-limit = " << cutoff_solver_time_*max_evaluations_ << std::endl;
        myfile << "logfile = " << solver_log_file_ + "_iteration_paramils_" + std::to_string(iteration_) + "_worker_" + std::to_string(i) << std::endl;
        myfile << "paramfile = " << parameter_file_path << std::endl;
        myfile << "outdir = " << paramils_outdir << std::endl;
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
    const std::uint32_t num_run = computeLocalSearchRunSeed(base_seed_, iteration_, nb_workers_, 0);
    std::string command = buildParamILSCommand(param_ils_dir_ + param_ils_executable_, num_run, scenario_file_path);
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
        if (gap >= 0) {
            options.gap = gap;
        }
        options.upper_bound = std::nullopt;
        options.lower_bound = std::nullopt;
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
    mip_start_file_ = buildTunerPath("mip_start/mip_start_iteration_" + std::to_string(iteration_) + ".mst");
    std::string config_path = buildTunerPath("config_for_mip_start/config_mip_start_iteration_" + std::to_string(iteration_) + ".prm");
    best_config.generateConfigFile(config_path);
    std::string mip_void = ""; // We do not want to use a mip start file for the solver, we just want to generate it with cplex, so we give it an empty file that does not exist, so it does not use it but it generates it
    // Call cplex on the best configuration found so far to generate the mip start file
    CPLEXSolver solver(logger_, instance_file_, config_path, solver_log_file_ + "_mip_start_" + std::to_string(iteration_), nb_threads_solver_, cutoff_solver_time_, solver_time_mode_, mip_void, mip_start_file_, tuning_objective_);
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
    options.gap = solver.getGap();
    options.upper_bound = solver.getUpperBound();
    options.lower_bound = solver.getLowerBound();
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

std::vector<CompactSharedCacheSeedEntry> IteratedLocalSearchEngine::getSharedCacheSeedEntries_() const {
    std::vector<CompactSharedCacheSeedEntry> seeds;
    const auto historical_entries = memory_.getHistoricalCacheSeedEntries();
    seeds.reserve(historical_entries.size());
    for (const auto& entry : historical_entries) {
        seeds.push_back(CompactSharedCacheSeedEntry{
            entry.configuration.getConfigurationId(),
            entry.objective_value
        });
    }
    return seeds;
}

std::string IteratedLocalSearchEngine::getHistoricalCacheSeedFilePath_() const {
    return buildHistoricalCacheSeedFilePath(ils_working_dir_, iteration_);
}

void IteratedLocalSearchEngine::writeHistoricalCacheSeedFile_(
    const std::vector<CompactSharedCacheSeedEntry>& seeds
) const {
    const std::string seed_file_path = getHistoricalCacheSeedFilePath_();
    writeHistoricalCacheSeedEntriesToFile(seed_file_path, seeds);
    logger_.info("Historical cache seed file written: ", seed_file_path, " with ", static_cast<int>(seeds.size()), " entries.");
}

const Configuration& IteratedLocalSearchEngine::getInitialConfigurationForWorker_(int worker_id) const {
    if (!random_worker_initial_configs_ || worker_id <= 0) {
        return initial_configurations_[0];
    }
    if (worker_id == 1 && mip_start_) {
        return initial_configurations_[0];
    }
    if (worker_id < static_cast<int>(initial_configurations_.size())) {
        return initial_configurations_[worker_id];
    }
    return initial_configurations_[0];
}

std::string IteratedLocalSearchEngine::getILSSearchSpaceFilePath_(int worker_id) const {
    if (!random_worker_initial_configs_ || worker_id == 0) {
        return ils_working_dir_ + "search_space/search_space_file_" + std::to_string(iteration_) + ".txt";
    }
    return ils_working_dir_ + "search_space/search_space_file_" + std::to_string(iteration_) + "_worker_" + std::to_string(worker_id) + ".txt";
}

void IteratedLocalSearchEngine::writeILSParameterOptionsToFile(std::ofstream& myfile, int worker_id) {
    std::vector<std::pair<std::string, Value>>& forbidden_values = parameter_space_.getForbiddenValues();

    const Configuration& initial_config = getInitialConfigurationForWorker_(worker_id);

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

void IteratedLocalSearchEngine::writeILSForbiddenOptionsToFile(std::ofstream& myfile, int worker_id) {
    std::vector<std::vector<std::pair<std::string, Value>>>& forbidden_tuples = parameter_space_.getForbiddenTuples();
    const Configuration& initial_config = getInitialConfigurationForWorker_(worker_id);

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

void IteratedLocalSearchEngine::writeILSInfoToFile(std::ofstream& myfile, int worker_id) {
    myfile << std::endl;
    myfile << "Info:" << std::endl;

    const std::optional<double> known_objective =
        worker_id == 0 ? getKnownInitialObjective_() : std::nullopt;
    if (known_objective.has_value()) {
        myfile << "Initial configuration evaluated: 1" << std::endl;
        myfile << "Objective value of the initial configuration: " << known_objective.value() << std::endl;
    } else {
        myfile << "Initial configuration evaluated: 0" << std::endl;
    }
}

void IteratedLocalSearchEngine::writeILSSearchSpaceFile(int worker_id) {
    logger_.info("Writing ILS search space file for worker ", worker_id, "...");

    ensureDirectoryExists(ils_working_dir_ + "search_space/");
    ensureDirectoryExists(ils_working_dir_ + "local_results/");
    const std::string search_space_file_path = getILSSearchSpaceFilePath_(worker_id);
    if (worker_id == 0) {
        search_space_file_ = search_space_file_path;
    }

    ensureParentDirectoryForFile(search_space_file_path);
    std::ofstream myfile(search_space_file_path);
    if (!myfile.is_open()) {
        throw std::runtime_error("Error opening ILS search space file for writing: " + search_space_file_path);
    }

    writeILSParameterOptionsToFile(myfile, worker_id);
    writeILSForbiddenOptionsToFile(myfile, worker_id);
    writeILSConditionalCplexOptionsToFile(myfile);
    writeILSInfoToFile(myfile, worker_id);

    myfile.close();

    logger_.info("ILS search space file written: ", search_space_file_path);
}

std::vector<EvaluationRecord> IteratedLocalSearchEngine::syncILSResultsToGlobalMemory_(
    int worker_id,
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
        options.gap = local_record.gap;
        options.upper_bound = local_record.upper_bound;
        options.lower_bound = local_record.lower_bound;
        options.time_evaluated = local_record.time_evaluated;
        options.worker_id = worker_id;
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

    std::vector<std::pair<int, std::vector<EvaluationRecord>>> exploration_results;
    std::vector<CompactSharedCacheSeedEntry> historical_cache_seeds;

    if (mip_start_ && iteration_ > 1 && nb_workers_ > 1) {
        mip_start_ = true;
    } else {
        mip_start_ = false;
    }
    logger_.info("Mip start is ", mip_start_ ? "enabled" : "disabled", " for this iteration.");

    if (mip_start_) {
        setMipStartFile();
    }

    if (use_shared_cache_) {
        historical_cache_seeds = getSharedCacheSeedEntries_();
        writeHistoricalCacheSeedFile_(historical_cache_seeds);
    }

    writeILSSearchSpaceFile(0);
    if (random_worker_initial_configs_) {
        for (int worker_id = 1; worker_id < nb_workers_; ++worker_id) {
            writeILSSearchSpaceFile(worker_id);
        }
    }

#ifdef USE_MPI
    launchLocalSearchWorkers();
#endif

    IteratedLocalSearch::Options ils_options;
    ils_options.search_space_file = search_space_file_;
    ils_options.instance_file = instance_file_;
    ils_options.log_file_solver = solver_log_file_ + "_iteration_ils_" + std::to_string(iteration_) + "_worker_0";
    ils_options.working_directory = ils_working_dir_ + "run_" + std::to_string(iteration_) + "_worker_0";
    ils_options.random_seed = computeLocalSearchRunSeed(base_seed_, iteration_, nb_workers_, 0);
    ils_options.evaluation_budget = max_evaluations_;
    ils_options.perturbation_strength = 3;
    ils_options.random_initial_samples = 0;
    ils_options.restart_probability = 0.10;
    ils_options.accept_ties = false;
    ils_options.acceptance_threshold = 0.0;
    ils_options.use_shared_cache = use_shared_cache_;
    ils_options.shared_cache_seed_entries = historical_cache_seeds;
    // Worker 0 never uses MIP starts. Only MPI worker 1 may consume the
    // produced MIP start in later iterations.
    ils_options.use_mip_starts = false;
    ils_options.mip_start_file = std::nullopt;
    ils_options.nb_threads_solver = nb_threads_solver_;
    ils_options.cutoff_solver_time = cutoff_solver_time_;
    ils_options.solver_time_mode = solver_time_mode_;
    ils_options.tuning_objective = tuning_objective_;

    ils_ = std::make_unique<IteratedLocalSearch>(logger_, ils_options);
    ils_->run();

    // This accessor must exist in IteratedLocalSearch.
    const auto local_results = ils_->getEvaluationsWithConfigurations();
    std::vector<EvaluationRecord> synced_results = syncILSResultsToGlobalMemory_(0, local_results);
    exploration_results.push_back(std::make_pair(0, synced_results));

#ifdef USE_MPI
    waitLocalSearchWorkers();
#endif

    // Read the results of the other workers and sync them to global memory
    for (int worker_id = 1; worker_id < nb_workers_; ++worker_id) {
        std::vector<std::pair<Configuration, EvaluationRecord>> worker_local_results = readLocalResultsFromFile_(worker_id);
        std::vector<EvaluationRecord> worker_synced_results = syncILSResultsToGlobalMemory_(worker_id, worker_local_results);
        exploration_results.push_back(std::make_pair(worker_id, worker_synced_results));
    }    

    logger_.info("IteratedLocalSearch Engine completed.");
    return exploration_results;
}

std::vector<std::pair<Configuration, EvaluationRecord>> IteratedLocalSearchEngine::readLocalResultsFromFile_(int worker_id) {
    std::vector<std::pair<Configuration, EvaluationRecord>> local_results;

    const std::string local_results_file =
        ils_working_dir_ + "local_results/local_results_" + std::to_string(iteration_) +
        "_worker_" + std::to_string(worker_id) + ".txt";

    std::ifstream file(local_results_file);
    if (!file.is_open()) {
        logger_.info("Error opening local results file for reading: ", local_results_file);
        return local_results;
    }

    auto parseBool = [](const std::string& value) {
        return value == "1" || value == "true" || value == "True";
    };

    std::string line;
    bool in_configuration = false;
    std::map<std::string, Value> config_map;
    double objective_value = -1.0;
    std::optional<double> gap = std::nullopt;
    std::optional<double> upper_bound = std::nullopt;
    std::optional<double> lower_bound = std::nullopt;
    int time_evaluated = -1;
    bool mip_start_used = false;
    std::optional<MipStartId> used_mip_start_id = std::nullopt;
    bool produced_mip_start = false;

    while (std::getline(file, line)) {
        if (line.empty()) {
            continue;
        }

        if (line == "Configuration:") {
            in_configuration = true;
            config_map.clear();
            objective_value = -1.0;
            gap = std::nullopt;
            upper_bound = std::nullopt;
            lower_bound = std::nullopt;
            time_evaluated = -1;
            mip_start_used = false;
            used_mip_start_id = std::nullopt;
            produced_mip_start = false;
            continue;
        }

        if (!in_configuration) {
            continue;
        }

        if (line == "EndConfiguration") {
            EvaluationRecord record{};
            record.evaluation_id = 0;
            record.objective_value = objective_value;
            record.gap = gap;
            record.upper_bound = upper_bound;
            record.lower_bound = lower_bound;
            record.time_evaluated = time_evaluated;
            record.configuration_id = 0;
            record.mip_start_used = mip_start_used;
            record.used_mip_start_id = used_mip_start_id;
            record.mip_start_source_evaluation_id = std::nullopt;
            record.produced_mip_start = produced_mip_start;
            record.produced_mip_start_id = std::nullopt;
            record.worker_id = worker_id;
            record.iteration = iteration_;
            record.phase = 0;

            local_results.emplace_back(Configuration(config_map, mip_start_used), record);
            in_configuration = false;
            continue;
        }

        const std::size_t eq_pos = line.find('=');
        if (eq_pos == std::string::npos) {
            continue;
        }

        const std::string key = line.substr(0, eq_pos);
        const std::string value = line.substr(eq_pos + 1);

        if (key == "ObjectiveValue") {
            objective_value = std::stod(value);
        } else if (key == "Gap") {
            gap = std::stod(value);
        } else if (key == "UpperBound") {
            upper_bound = std::stod(value);
        } else if (key == "LowerBound") {
            lower_bound = std::stod(value);
        } else if (key == "TimeEvaluated") {
            time_evaluated = std::stoi(value);
        } else if (key == "MipStartUsed") {
            mip_start_used = parseBool(value);
        } else if (key == "UsedMipStartId") {
            if (value != "-1" && value != "18446744073709551615") {
                used_mip_start_id = static_cast<MipStartId>(std::stoull(value));
            }
        } else if (key == "ProducedMipStart") {
            produced_mip_start = parseBool(value);
        } else {
            config_map.emplace(key, Value(value));
        }
    }

    return local_results;
}


#ifdef USE_MPI
void ParamILSWorker::callParamILS() {
#ifdef USE_MPI
    int nb_workers = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &nb_workers);
#else
    const int nb_workers = 1;
#endif
    std::string scenario_file_path = param_ils_working_dir_ + "scenario/scenario_file_" + std::to_string(iteration_) + "_worker_" + std::to_string(worker_id_) + ".txt";
    const std::uint32_t num_run = computeLocalSearchRunSeed(base_seed_, iteration_, nb_workers, worker_id_);
    std::string command = buildParamILSCommand(param_ils_dir_ + param_ils_executable_, num_run, scenario_file_path);
    int ret = system(command.c_str());
    if (ret != 0) {
        std::cout << "Error calling ParamILS executable for worker " << worker_id_ << " at iteration " << iteration_ << std::endl;
    }
}

void LocalSearchEngine::launchLocalSearchWorkers() {
    logger_.info("Launching Local Search Workers via MPI...");
    // Implementation to launch local search workers using MPI
    // Broadcast to give all workers worker_step = 1 and iteration_
    WorkerOrder order{};
    order.step = 1; // exploration step
    order.iteration = iteration_;
    order.nb_evaluations = max_evaluations_;
    order.expansion_best_objective_value = 0.0;
    order.expansion_enable_early_stop = 0;
    MPI_Bcast(&order, sizeof(WorkerOrder), MPI_BYTE, 0, MPI_COMM_WORLD);
    logger_.info("Local Search Workers launched.");
}

void LocalSearchEngine::waitLocalSearchWorkers() {
    logger_.info("Waiting for Local Search Workers via MPI...");
    // Implementation to wait for local search workers using MPI
    MPI_Barrier(MPI_COMM_WORLD);
    logger_.info("Local Search Workers have completed.");
}

void IteratedLocalSearchWorker::callIteratedLocalSearch() {
    int nb_workers = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &nb_workers);
    Logger worker_logger(Verbosity::Debug, std::cout);
    const std::string search_space_file =
        random_worker_initial_configs_ && worker_id_ > 0
            ? ils_working_dir_ + "search_space/search_space_file_" + std::to_string(iteration_) + "_worker_" + std::to_string(worker_id_) + ".txt"
            : ils_working_dir_ + "search_space/search_space_file_" + std::to_string(iteration_) + ".txt";
    if (mip_start_ && worker_id_ == 1 && iteration_ > 1) {
        mip_start_file_ = buildTunerPath("mip_start/mip_start_iteration_" + std::to_string(iteration_) + ".mst");
        worker_logger.info("Worker ", worker_id_, " at iteration ", iteration_, " will use MIP start file: ", mip_start_file_);
    } else {
        worker_logger.info("Worker ", worker_id_, " at iteration ", iteration_, " will not use a MIP start file.");
    }

    IteratedLocalSearch::Options ils_options;
    ils_options.search_space_file = search_space_file;
    ils_options.instance_file = instance_file_;
    ils_options.log_file_solver = solver_log_file_ + "_iteration_ils_" + std::to_string(iteration_) + "_worker_" + std::to_string(worker_id_);
    ils_options.working_directory = ils_working_dir_ + "run_" + std::to_string(iteration_) + "_worker_" + std::to_string(worker_id_);
    ils_options.random_seed = computeLocalSearchRunSeed(base_seed_, iteration_, nb_workers, worker_id_);
    ils_options.evaluation_budget = max_evaluations_;
    ils_options.perturbation_strength = 3;
    ils_options.random_initial_samples = 0;
    ils_options.restart_probability = 0.10;
    ils_options.accept_ties = false;
    ils_options.acceptance_threshold = 0.0;
    ils_options.use_shared_cache = use_shared_cache_;
    if (use_shared_cache_) {
        const std::string historical_cache_seed_file =
            buildHistoricalCacheSeedFilePath(ils_working_dir_, iteration_);
        ils_options.shared_cache_seed_entries =
            readHistoricalCacheSeedEntriesFromFile(historical_cache_seed_file, worker_logger);
    }
    ils_options.use_mip_starts = mip_start_ && !mip_start_file_.empty();
    if (ils_options.use_mip_starts) {
        ils_options.mip_start_file = mip_start_file_;
    }
    ils_options.nb_threads_solver = nb_threads_solver_;
    ils_options.cutoff_solver_time = cutoff_solver_time_;
    ils_options.solver_time_mode = solver_time_mode_;
    ils_options.tuning_objective = tuning_objective_;

    IteratedLocalSearch ils(worker_logger, ils_options);
    ils.run();

    const auto local_results = ils.getEvaluationsWithConfigurations();
    // Write all the local results to a file that will be read by the master process to sync with global memory
    std::string local_results_file = ils_working_dir_ + "local_results/local_results_" + std::to_string(iteration_) + "_worker_" + std::to_string(worker_id_) + ".txt";
    ensureDirectoryExists(ils_working_dir_ + "local_results/");
    ensureParentDirectoryForFile(local_results_file);
    std::ofstream myfile(local_results_file);
    if (!myfile.is_open()) {
        worker_logger.info("Error opening local results file for writing: ", local_results_file);
        return;
    }
    for (const auto& pair : local_results) {
        const Configuration& config = pair.first;
        const EvaluationRecord& record = pair.second;

        myfile << "Configuration:" << std::endl;
        for (const auto& param_pair : config.getConfigurationMap()) {
            myfile << param_pair.first << "=" << param_pair.second.getString() << std::endl;
        }
        myfile << "ObjectiveValue=" << record.objective_value << std::endl;
        if (record.gap.has_value()) {
            myfile << "Gap=" << record.gap.value() << std::endl;
        }
        if (record.upper_bound.has_value()) {
            myfile << "UpperBound=" << record.upper_bound.value() << std::endl;
        }
        if (record.lower_bound.has_value()) {
            myfile << "LowerBound=" << record.lower_bound.value() << std::endl;
        }
        myfile << "TimeEvaluated=" << record.time_evaluated << std::endl;
        myfile << "MipStartUsed=" << record.mip_start_used << std::endl;
        if (record.mip_start_used) {
            myfile << "UsedMipStartId=" << record.used_mip_start_id.value_or(-1) << std::endl;
        }
        myfile << "ProducedMipStart=" << record.produced_mip_start << std::endl;
        //Todo: we could also write the produced mip start file path if produced_mip_start is true but change memory for that to store the path
        myfile << "EndConfiguration" << std::endl;
    }
    myfile.close();
}
#endif
