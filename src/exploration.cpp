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
#include <algorithm>
#include <optional>
#include <random>

namespace {

std::uint32_t computeInitialConfigurationSeed(std::uint32_t base_seed, int iteration) {
    return base_seed + 100000u + static_cast<std::uint32_t>(iteration - 1);
}

std::string buildParamILSCommand(const std::string& executable, std::uint32_t num_run, const std::string& scenario_file_path, const std::string& stderr_log_path) {
    return "ruby " + executable + " -numRun " + std::to_string(num_run) + " -scenariofile " + scenario_file_path + " > /dev/null 2> " + stderr_log_path;
}

std::string buildHistoricalCacheSeedFilePath(const std::string& ils_working_dir, int iteration) {
    return ils_working_dir + "search_space/shared_cache_seed_" + std::to_string(iteration) + ".txt";
}

std::string buildCurrentMipStartInfoFilePath(const std::string& ils_working_dir, int iteration) {
    return ils_working_dir + "mip_start/current_mip_start_" + std::to_string(iteration) + ".txt";
}

struct CurrentMipStartInfo {
    std::string mip_start_file;
    std::optional<MipStartId> mip_start_id = std::nullopt;
    double best_upper_bound = kMaxObjective;
};

void writeCurrentMipStartInfoFile(
    const std::string& filename,
    const std::string& mip_start_file,
    std::optional<MipStartId> mip_start_id,
    double best_upper_bound
) {
    ensureParentDirectoryForFile(filename);
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Error opening current MIP start info file for writing: " + filename);
    }
    file << "MipStartFile=" << mip_start_file << std::endl;
    if (mip_start_id.has_value()) {
        file << "MipStartId=" << mip_start_id.value() << std::endl;
    }
    file << "BestUpperBound=" << best_upper_bound << std::endl;
}

CurrentMipStartInfo readCurrentMipStartInfoFile(
    const std::string& filename,
    Logger& logger
) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        logger.info("Current MIP start info file not found: ", filename);
        return {};
    }

    CurrentMipStartInfo info;
    std::string line;
    while (std::getline(file, line)) {
        const std::size_t eq_pos = line.find('=');
        if (eq_pos == std::string::npos) {
            continue;
        }

        const std::string key = line.substr(0, eq_pos);
        const std::string value = line.substr(eq_pos + 1);
        if (key == "MipStartFile") {
            info.mip_start_file = value;
        } else if (key == "MipStartId" && !value.empty()) {
            info.mip_start_id = static_cast<MipStartId>(std::stoull(value));
        } else if (key == "BestUpperBound" && !value.empty()) {
            info.best_upper_bound = std::stod(value);
        }
    }

    return info;
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

namespace {

Configuration withoutMipStart(const Configuration& config) {
    return Configuration(config.getConfigurationMap(), false);
}

Configuration sampleInitialConfiguration(
    ParameterSpace& parameter_space,
    std::mt19937& rng
) {
    std::map<std::string, Value> config_map;
    for (const auto& param : parameter_space.getParameters()) {
        if (param.isTuned()) {
            const auto& values = param.getValues();
            std::uniform_int_distribution<std::size_t> dist(0, values.size() - 1);
            config_map.insert_or_assign(param.getName(), values[dist(rng)]);
        } else {
            config_map.insert_or_assign(param.getName(), param.getDefaultValue());
        }
    }
    return Configuration(config_map, false);
}


} // namespace

std::vector<Configuration> Exploration::selectInitialConfigurations() {
    // Warning: The initial configurations vector must contain at least nb_workers_ configurations (these could be duplicates)
    // Implementation to select initial configurations for tuning phase
    logger_.info("Selecting initial configurations for tuning phase...");
    std::vector<Configuration> initial_configurations;
    std::mt19937 rng(computeInitialConfigurationSeed(base_seed_, iteration_));

    if (local_search_backend_ == LocalSearchBackend::IteratedLocalSearch) {
        const Configuration* global_best_exploration_without_mip =
            memory_.getBestExplorationConfigurationWithoutMipStart();
        if (global_best_exploration_without_mip != nullptr) {
            logger_.info("Using best exploration configuration without MIP start as ILS initial configuration.");
            initial_configurations.push_back(withoutMipStart(*global_best_exploration_without_mip));
        } else {
            logger_.info("No best exploration configuration without MIP start in memory, using default ILS initial configuration.");
            initial_configurations.push_back(withoutMipStart(memory_.getDefaultConfiguration()));
        }
    } else if (memory_.getBestConfiguration() != nullptr) {
        logger_.info("Using best configuration from memory as initial configuration.");
        initial_configurations.push_back(*(memory_.getBestConfiguration()));
    } else {
        logger_.info("No best configuration in memory, using default initial configuration.");
        initial_configurations.push_back(memory_.getDefaultConfiguration()); // Return default configuration if no best found
    }
    //Todo: implementation is for test only, after use random configurations or other strategies
    while (initial_configurations.size() < static_cast<size_t>(nb_workers_)) {
        initial_configurations.push_back(sampleInitialConfiguration(parameter_space_, rng));
    }
    return initial_configurations;
}

bool LocalSearchEngine::usesWorkerSpecificInitialConfigurations() const {
    return random_worker_initial_configs_ || force_worker_initial_configs_;
}

const Configuration& LocalSearchEngine::getInitialConfigurationForWorker(int worker_id) const {
    if (worker_id <= 0) {
        return initial_configurations_[0];
    }

#ifdef USE_MPI
    const int procs_per_ils = getParallelILSInfo().procs_per_ils;
#else
    const int procs_per_ils = 1;
#endif
    const int ils_group_id  = worker_id / procs_per_ils;

    if (ils_group_id == 1 && use_mip_start_) {
        if (mip_start_initial_config_policy_ == MipStartInitialConfigPolicy::BestConfig) {
            logger_.info(
                "Worker group 1 (M2) will start from worker 0 initial configuration for MIP start id ",
                memory_.getBestMipStartId().value_or(0),
                " because MIP-start initial configuration policy is best_config."
            );
            return initial_configurations_[0];
        }

        const Configuration* producer_config = memory_.getBestMipStartProducerConfiguration();
        if (producer_config != nullptr &&
            producer_config->getConfigurationId() != initial_configurations_[0].getConfigurationId()) {
            logger_.info(
                "Worker group 1 (M2) will start from MIP-start producer configuration ",
                producer_config->getConfigurationId(),
                " for MIP start id ",
                memory_.getBestMipStartId().value_or(0),
                "."
            );
            return *producer_config;
        }

        if (producer_config != nullptr) {
            logger_.info(
                "Worker group 1 (M2) producer config matches best cold config — falling back to random config for diversity."
            );
        } else {
            logger_.info(
                "Worker group 1 (M2) no producer config available — falling back to random config for diversity."
            );
        }
        if (worker_id < static_cast<int>(initial_configurations_.size())) {
            return initial_configurations_[worker_id];
        }
        return initial_configurations_[0];
    }

    if (mip_worker_strategy_ && ils_group_id == 2 && use_mip_start_) {
        logger_.info("Worker group 2 (M3) will start from best cold config (same as M1).");
        return initial_configurations_[0];
    }

    if (!usesWorkerSpecificInitialConfigurations()) {
        return initial_configurations_[0];
    }

    if (worker_id < static_cast<int>(initial_configurations_.size())) {
        return initial_configurations_[worker_id];
    }

    return initial_configurations_[0];
}

int Exploration::selectNumberOfEvaluations() {
    int idx = std::min(iteration_ - 1, static_cast<int>(exploration_budget_factors_.size()) - 1);
    int factor = exploration_budget_factors_[idx];
    return factor * static_cast<int>(parameter_space_.getSelectedParameters().size());
}

ExplorationRunStats Exploration::run() {
    logger_.info("Starting exploration phase...");
    int nb_evaluations = 0;
    if (number_of_evaluations_override_.has_value()) {
        nb_evaluations = number_of_evaluations_override_.value();
        logger_.info("Using explicit evaluation budget override: ", nb_evaluations);
    } else {
        nb_evaluations = selectNumberOfEvaluations();
    }
    if (paramils_wall_time_.has_value() &&
        local_search_backend_ == LocalSearchBackend::ParamILS &&
        !number_of_evaluations_override_.has_value()) {
        nb_evaluations = std::numeric_limits<int>::max();
        logger_.info("ParamILS wall-time budget set: using unlimited maxEvals (wall-time is the stopping criterion).");
    }
    if (exploration_budget_divisor_.has_value()) {
        const int divisor = exploration_budget_divisor_.value();
        const int original_nb_evaluations = nb_evaluations;
        nb_evaluations /= divisor;
        if (nb_evaluations < 1) {
            nb_evaluations = 1;
        }
        logger_.info("Divided exploration evaluation budget by ", divisor, ": ", original_nb_evaluations, " -> ", nb_evaluations);
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
                solver_watchdog_options_,
                nb_workers_,
                use_shared_cache_,
                base_seed_,
                tuning_objective_,
                enable_mip_starts_,
                random_worker_initial_configs_,
                mip_start_initial_config_policy_,
                local_search_backend_ == LocalSearchBackend::IteratedLocalSearch && nb_workers_ > 1 && (!enable_mip_starts_ || mip_worker_strategy_),
                mip_worker_strategy_
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
                solver_watchdog_options_,
                nb_workers_,
                base_seed_,
                tuning_objective_,
                enable_mip_starts_,
                mip_start_initial_config_policy_,
                paramils_wall_time_
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
            solver_watchdog_options_,
            nb_workers_,
            use_shared_cache_,
            base_seed_,
            tuning_objective_,
            enable_mip_starts_,
            random_worker_initial_configs_,
            mip_start_initial_config_policy_,
            local_search_backend_ == LocalSearchBackend::IteratedLocalSearch && nb_workers_ > 1 && !enable_mip_starts_
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
    return ExplorationRunStats{nb_evaluations};
}

std::vector<std::pair<int, std::vector<EvaluationRecord>>> ParamILSEngine::run() {
    // Implementation of the ParamILS algorithm
    logger_.info("Running ParamILS Engine...");
    if (use_mip_start_ && iteration_ > 1 && nb_workers_ > 1) {
        use_mip_start_ = true;
    } else {
        use_mip_start_ = false;
    }
    logger_.info("Mip start is ", use_mip_start_ ? "enabled" : "disabled", " for this iteration.");
    
    if (use_mip_start_) {
        setMipStartFile();
        if (mip_start_file_.empty()) {
            use_mip_start_ = false;
            logger_.info("MIP start use disabled for this iteration because no incumbent MIP start file is available.");
        }
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
    const Configuration& initial_config = getInitialConfigurationForWorker(worker_id);

    for (auto& param : parameter_space_.getParameters()) {
        Value initial_value = initial_config.getConfigurationMap().at(param.getName());
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
    
    if (worker_id == 1 && use_mip_start_ && !mip_start_file_.empty()) {
        myfile << "mip_start { " << mip_start_file_ << " } [ " << mip_start_file_ << " ]" << std::endl;
    }
}

void ParamILSEngine::writeForbiddenOptionsToFile(std::ofstream& myfile, int worker_id) {
    // Implementation to write forbidden options to file
    std::vector<std::vector<std::pair<std::string, Value>>>& forbidden_tuples = parameter_space_.getForbiddenTuples();
    // For each forbidden tuple we have to look if the initial configuration contains it, so we do not forbid it
    const Configuration& initial_config = getInitialConfigurationForWorker(worker_id);

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
        std::string algo_cmd = "algo = ruby " + param_ils_dir_ + "cplex_wrapper.rb"
            + " --threads " + std::to_string(nb_threads_solver_)
            + " --work-dir " + solver_working_dir;
        if (solver_time_mode_ == SolverTimeMode::Ticks) {
            algo_cmd += " --ticks";
        }
        myfile << algo_cmd << std::endl;
        myfile << "execdir = ." << std::endl;
        myfile << "deterministic = 1" << std::endl;
        myfile << "run_obj = " << tuning_obj << std::endl;
        myfile << "overall_obj = mean" << std::endl;
        myfile << "cutoff_time = " << cutoff_solver_time_ << std::endl;
        myfile << "maxEvals = " << max_evaluations_ << std::endl;
        const double wall_limit = paramils_wall_time_.has_value()
            ? paramils_wall_time_.value()
            : cutoff_solver_time_ * max_evaluations_;
        myfile << "wallclock-limit = " << wall_limit << std::endl;
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
    std::string stderr_log = param_ils_working_dir_ + "paramils_stderr_iter_" + std::to_string(iteration_) + "_worker_0.log";
    std::string command = buildParamILSCommand(param_ils_dir_ + param_ils_executable_, num_run, scenario_file_path, stderr_log);
    logger_.info("ParamILS command: ", command);
    int ret = system(command.c_str());
    if (ret != 0) {
        logger_.info("Error calling ParamILS executable (stderr: ", stderr_log, ").");
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
        if (worker_id == 1 && use_mip_start_) {
            options.mip_start_used = true;
            options.used_mip_start_id = used_mip_start_id_;
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
        if (time_sec >= 0) {
            options.solver_runtime_seconds = time_sec;
        }
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
    const auto best_mip_start_file = memory_.getBestMipStartFile();
    const auto best_mip_start_id = memory_.getBestMipStartId();
    if (!best_mip_start_file.has_value() || !best_mip_start_id.has_value()) {
        mip_start_file_.clear();
        used_mip_start_id_ = 0;
        logger_.info("No produced MIP start is available for iteration ", iteration_, ".");
        return;
    }

    mip_start_file_ = best_mip_start_file.value();
    used_mip_start_id_ = best_mip_start_id.value();
    logger_.info("Selected MIP start id ", used_mip_start_id_, " for iteration ", iteration_, ": ", mip_start_file_);
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

std::string IteratedLocalSearchEngine::getILSSearchSpaceFilePath_(int worker_id) const {
    if (!usesWorkerSpecificInitialConfigurations() || worker_id == 0) {
        return ils_working_dir_ + "search_space/search_space_file_" + std::to_string(iteration_) + ".txt";
    }
    return ils_working_dir_ + "search_space/search_space_file_" + std::to_string(iteration_) + "_worker_" + std::to_string(worker_id) + ".txt";
}

void IteratedLocalSearchEngine::writeILSParameterOptionsToFile(std::ofstream& myfile, int worker_id) {
    std::vector<std::pair<std::string, Value>>& forbidden_values = parameter_space_.getForbiddenValues();

    const Configuration& initial_config = getInitialConfigurationForWorker(worker_id);

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
    const Configuration& initial_config = getInitialConfigurationForWorker(worker_id);

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
        options.solver_runtime_seconds = local_record.solver_runtime_seconds;
        options.solver_termination_status = local_record.solver_termination_status;
        options.time_evaluated = local_record.time_evaluated;
        options.worker_id = worker_id;
        options.iteration = iteration_;
        options.phase = 0;

        options.mip_start_used = config.useMipStart();
        if (config.useMipStart() && local_record.used_mip_start_id.has_value()) {
            options.used_mip_start_id = local_record.used_mip_start_id;
        } else if (config.useMipStart() && use_mip_start_) {
            options.used_mip_start_id = used_mip_start_id_;
        }

        options.produced_mip_start = local_record.produced_mip_start;
        options.mip_start_file = local_record.produced_mip_start_file;

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
    const bool mip_start_production_enabled = use_mip_start_;

    if (use_mip_start_ && iteration_ > 1 && nb_workers_ > 1) {
        use_mip_start_ = true;
    } else {
        use_mip_start_ = false;
    }
    logger_.info("Mip start is ", use_mip_start_ ? "enabled" : "disabled", " for this iteration.");

    if (use_mip_start_) {
        setMipStartFile();
        if (mip_start_file_.empty()) {
            use_mip_start_ = false;
            logger_.info("MIP start use disabled for this iteration because no incumbent MIP start file is available.");
        } else {
            writeCurrentMipStartInfoFile(
                buildCurrentMipStartInfoFilePath(ils_working_dir_, iteration_),
                mip_start_file_,
                used_mip_start_id_,
                memory_.getBestMipStartUpperBound()
            );
        }
    }

    if (use_shared_cache_) {
        historical_cache_seeds = getSharedCacheSeedEntries_();
        writeHistoricalCacheSeedFile_(historical_cache_seeds);
    }

    writeILSSearchSpaceFile(0);
    if (usesWorkerSpecificInitialConfigurations()) {
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
    ils_options.used_mip_start_id = std::nullopt;
    ils_options.produce_mip_starts = mip_start_production_enabled;
    ils_options.best_mip_start_upper_bound = memory_.getBestMipStartUpperBound();
    ils_options.nb_threads_solver = nb_threads_solver_;
    ils_options.cutoff_solver_time = cutoff_solver_time_;
    ils_options.solver_time_mode = solver_time_mode_;
    ils_options.solver_watchdog_options = solver_watchdog_options_;
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
    std::optional<double> solver_runtime_seconds = std::nullopt;
    SolverTerminationStatus solver_termination_status = SolverTerminationStatus::Normal;
    int time_evaluated = -1;
    bool mip_start_used = false;
    std::optional<MipStartId> used_mip_start_id = std::nullopt;
    bool produced_mip_start = false;
    std::string produced_mip_start_file;

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
            solver_runtime_seconds = std::nullopt;
            solver_termination_status = SolverTerminationStatus::Normal;
            time_evaluated = -1;
            mip_start_used = false;
            used_mip_start_id = std::nullopt;
            produced_mip_start = false;
            produced_mip_start_file.clear();
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
            record.solver_runtime_seconds = solver_runtime_seconds;
            record.solver_termination_status = solver_termination_status;
            record.time_evaluated = time_evaluated;
            record.configuration_id = 0;
            record.mip_start_used = mip_start_used;
            record.used_mip_start_id = used_mip_start_id;
            record.mip_start_source_evaluation_id = std::nullopt;
            record.produced_mip_start = produced_mip_start;
            record.produced_mip_start_id = std::nullopt;
            record.produced_mip_start_file = produced_mip_start_file;
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
        } else if (key == "SolverRuntimeSeconds") {
            solver_runtime_seconds = std::stod(value);
        } else if (key == "SolverTerminationStatus") {
            solver_termination_status = parseSolverTerminationStatus(value);
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
        } else if (key == "ProducedMipStartFile") {
            produced_mip_start_file = value;
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
    std::string stderr_log = param_ils_working_dir_ + "paramils_stderr_iter_" + std::to_string(iteration_) + "_worker_" + std::to_string(worker_id_) + ".log";
    std::string command = buildParamILSCommand(param_ils_dir_ + param_ils_executable_, num_run, scenario_file_path, stderr_log);
    logger_.info("ParamILS command: ", command);
    int ret = system(command.c_str());
    if (ret != 0) {
        logger_.info("Error calling ParamILS executable for worker ", worker_id_, " at iteration ", iteration_, " (stderr: ", stderr_log, ")");
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
    const std::string search_space_file =
        (random_worker_initial_configs_ || (mip_worker_strategy_ && use_mip_start_) ||
         (!use_mip_start_ && !produce_mip_starts_ && nb_workers > 1)) && worker_id_ > 0
            ? ils_working_dir_ + "search_space/search_space_file_" + std::to_string(iteration_) + "_worker_" + std::to_string(worker_id_) + ".txt"
            : ils_working_dir_ + "search_space/search_space_file_" + std::to_string(iteration_) + ".txt";
    std::optional<MipStartId> used_mip_start_id = std::nullopt;
    CurrentMipStartInfo mip_start_info;
    if (produce_mip_starts_) {
        mip_start_info = readCurrentMipStartInfoFile(
            buildCurrentMipStartInfoFilePath(ils_working_dir_, iteration_),
            logger_
        );
    }
    const int ils_group_id_local = worker_id_ / getParallelILSInfo().procs_per_ils;
    if (use_mip_start_ && (ils_group_id_local == 1 || (mip_worker_strategy_ && ils_group_id_local == 2)) && iteration_ > 1) {
        mip_start_file_ = mip_start_info.mip_start_file;
        used_mip_start_id = mip_start_info.mip_start_id;
        logger_.info("Worker ", worker_id_, " at iteration ", iteration_, " will use MIP start file: ", mip_start_file_);
    } else {
        logger_.info("Worker ", worker_id_, " at iteration ", iteration_, " will not use a MIP start file. Current best MIP-start upper bound threshold is ", mip_start_info.best_upper_bound, ".");
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
            readHistoricalCacheSeedEntriesFromFile(historical_cache_seed_file, logger_);
    }
    ils_options.use_mip_starts = use_mip_start_ && !mip_start_file_.empty();
    if (ils_options.use_mip_starts) {
        ils_options.mip_start_file = mip_start_file_;
        ils_options.used_mip_start_id = used_mip_start_id;
    }
    ils_options.produce_mip_starts = produce_mip_starts_;
    ils_options.best_mip_start_upper_bound = mip_start_info.best_upper_bound;
    ils_options.nb_threads_solver = nb_threads_solver_;
    ils_options.cutoff_solver_time = cutoff_solver_time_;
    ils_options.solver_time_mode = solver_time_mode_;
    ils_options.solver_watchdog_options = solver_watchdog_options_;
    ils_options.tuning_objective = tuning_objective_;

    IteratedLocalSearch ils(logger_, ils_options);
    ils.run();

    const auto local_results = ils.getEvaluationsWithConfigurations();
    // Write all the local results to a file that will be read by the master process to sync with global memory
    std::string local_results_file = ils_working_dir_ + "local_results/local_results_" + std::to_string(iteration_) + "_worker_" + std::to_string(worker_id_) + ".txt";
    ensureDirectoryExists(ils_working_dir_ + "local_results/");
    ensureParentDirectoryForFile(local_results_file);
    std::ofstream myfile(local_results_file);
    if (!myfile.is_open()) {
        logger_.info("Error opening local results file for writing: ", local_results_file);
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
        if (record.solver_runtime_seconds.has_value()) {
            myfile << "SolverRuntimeSeconds=" << record.solver_runtime_seconds.value() << std::endl;
        }
        myfile << "SolverTerminationStatus=" << solverTerminationStatusToString(record.solver_termination_status) << std::endl;
        myfile << "TimeEvaluated=" << record.time_evaluated << std::endl;
        myfile << "MipStartUsed=" << record.mip_start_used << std::endl;
        if (record.mip_start_used) {
            myfile << "UsedMipStartId=" << record.used_mip_start_id.value_or(-1) << std::endl;
        }
        myfile << "ProducedMipStart=" << record.produced_mip_start << std::endl;
        if (record.produced_mip_start) {
            myfile << "ProducedMipStartFile=" << record.produced_mip_start_file << std::endl;
        }
        myfile << "EndConfiguration" << std::endl;
    }
    myfile.close();
}
#endif
