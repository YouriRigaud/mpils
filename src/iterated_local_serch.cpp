// MPILS: Multi-Phase Iterated Local Search Tuner
//
// Author: Youri Rigaud
// License: GNU GPLv3

#include "../include/iterated_local_search.h"
#include "../include/solver.h"
#include "../include/globaltimer.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <unordered_set>
#include <filesystem>
#include <cstdlib>

// ============================================================
// LocalSearchSpace helpers
// ============================================================

std::string LocalSearchSpace::trim_(const std::string& s) {
    std::size_t first = 0;
    while (first < s.size() && std::isspace(static_cast<unsigned char>(s[first]))) {
        ++first;
    }
    std::size_t last = s.size();
    while (last > first && std::isspace(static_cast<unsigned char>(s[last - 1]))) {
        --last;
    }
    return s.substr(first, last - first);
}

bool LocalSearchSpace::startsWith_(const std::string& s, const std::string& prefix) {
    return s.size() >= prefix.size() && s.compare(0, prefix.size(), prefix) == 0;
}

std::vector<std::string> LocalSearchSpace::split_(const std::string& s, char delimiter) {
    std::vector<std::string> tokens;
    std::stringstream ss(s);
    std::string token;
    while (std::getline(ss, token, delimiter)) {
        tokens.push_back(trim_(token));
    }
    return tokens;
}

std::string LocalSearchSpace::stripComment_(const std::string& line) {
    std::size_t pos = line.find('#');
    if (pos == std::string::npos) {
        return trim_(line);
    }
    return trim_(line.substr(0, pos));
}

void LocalSearchSpace::clear() {
    parameters_.clear();
    parameter_index_.clear();
    forbidden_tuples_.clear();
    conditionals_.clear();
    initial_configuration_ = Configuration();
    initial_configuration_evaluated_ = false;
    initial_configuration_objective_ = std::nullopt;
}

Value LocalSearchSpace::parseValue_(const std::string& token) const {
    const std::string t = trim_(token);
    if (t.empty()) {
        throw std::runtime_error("Cannot parse empty value token.");
    }

    // Important choice:
    // We store ALL search-space values as strings.
    // This matches the ParamILS / CPLEX text-file workflow and avoids stoi/stod issues.
    return Value(t);
}

std::vector<Value> LocalSearchSpace::parseValueList_(const std::string& values_str) const {
    std::vector<Value> values;
    const auto tokens = split_(values_str, ',');
    for (const auto& token : tokens) {
        if (!token.empty()) {
            values.push_back(parseValue_(token));
        }
    }
    return values;
}

void LocalSearchSpace::registerParameter_(const Parameter& parameter) {
    if (parameter_index_.find(parameter.getName()) != parameter_index_.end()) {
        throw std::runtime_error("Duplicate parameter in search space: " + parameter.getName());
    }
    parameter_index_[parameter.getName()] = parameters_.size();
    parameters_.push_back(parameter);
}

const Parameter& LocalSearchSpace::getParameter_(const std::string& parameter_name) const {
    auto it = parameter_index_.find(parameter_name);
    if (it == parameter_index_.end()) {
        throw std::runtime_error("Unknown parameter: " + parameter_name);
    }
    return parameters_.at(it->second);
}

bool LocalSearchSpace::valueInDomain_(const std::string& parameter_name, const Value& value) const {
    const auto& parameter = getParameter_(parameter_name);
    const auto values = parameter.getValues();
    return std::any_of(values.begin(), values.end(),
                       [&](const Value& v) { return v == value; });
}

void LocalSearchSpace::parseParameterLine_(const std::string& line) {
    const std::size_t brace_open = line.find('{');
    const std::size_t brace_close = line.find('}');
    const std::size_t bracket_open = line.find('[');
    const std::size_t bracket_close = line.find(']');

    if (brace_open == std::string::npos || brace_close == std::string::npos ||
        bracket_open == std::string::npos || bracket_close == std::string::npos) {
        throw std::runtime_error("Invalid parameter line format: " + line);
    }

    const std::string name = trim_(line.substr(0, brace_open));
    const std::string domain_str = trim_(line.substr(brace_open + 1, brace_close - brace_open - 1));
    const std::string default_str = trim_(line.substr(bracket_open + 1, bracket_close - bracket_open - 1));

    const std::vector<Value> domain = parseValueList_(domain_str);
    const Value default_value = parseValue_(default_str);

    if (domain.empty()) {
        throw std::runtime_error("Empty domain for parameter: " + name);
    }

    bool default_found = false;
    for (const auto& v : domain) {
        if (v == default_value) {
            default_found = true;
            break;
        }
    }
    if (!default_found) {
        throw std::runtime_error("Default value not in domain for parameter: " + name);
    }

    registerParameter_(Parameter(name, domain, default_value));
}

void LocalSearchSpace::parseForbiddenTupleLine_(const std::string& line) {
    if (line.size() < 2 || line.front() != '{' || line.back() != '}') {
        throw std::runtime_error("Invalid forbidden tuple line: " + line);
    }

    const std::string inside = trim_(line.substr(1, line.size() - 2));
    const auto assignments = split_(inside, ',');

    std::vector<std::pair<std::string, Value>> tuple;
    for (const auto& assignment : assignments) {
        const std::size_t eq_pos = assignment.find('=');
        if (eq_pos == std::string::npos) {
            throw std::runtime_error("Invalid forbidden tuple assignment: " + assignment);
        }

        const std::string param_name = trim_(assignment.substr(0, eq_pos));
        const std::string value_str = trim_(assignment.substr(eq_pos + 1));

        if (!isKnownParameter(param_name)) {
            throw std::runtime_error("Unknown parameter in forbidden tuple: " + param_name);
        }

        const Value value = parseValue_(value_str);
        tuple.emplace_back(param_name, value);
    }

    forbidden_tuples_.push_back(tuple);
}

void LocalSearchSpace::parseConditionalLine_(const std::string& line) {
    const std::size_t pipe_pos = line.find('|');
    const std::size_t in_pos = line.find(" in ");

    if (pipe_pos == std::string::npos || in_pos == std::string::npos || pipe_pos > in_pos) {
        throw std::runtime_error("Invalid conditional line: " + line);
    }

    const std::string child_name = trim_(line.substr(0, pipe_pos));
    const std::string parent_name = trim_(line.substr(pipe_pos + 1, in_pos - pipe_pos - 1));

    const std::size_t brace_open = line.find('{', in_pos);
    const std::size_t brace_close = line.find('}', in_pos);

    if (brace_open == std::string::npos || brace_close == std::string::npos || brace_open > brace_close) {
        throw std::runtime_error("Invalid conditional activation values in line: " + line);
    }

    if (!isKnownParameter(child_name)) {
        throw std::runtime_error("Unknown child parameter in conditional: " + child_name);
    }
    if (!isKnownParameter(parent_name)) {
        throw std::runtime_error("Unknown parent parameter in conditional: " + parent_name);
    }

    const std::string values_str = trim_(line.substr(brace_open + 1, brace_close - brace_open - 1));

    ConditionalActivation conditional;
    conditional.child_parameter = child_name;
    conditional.parent_parameter = parent_name;
    conditional.activating_values = parseValueList_(values_str);

    if (conditional.activating_values.empty()) {
        throw std::runtime_error("Conditional with empty activating set: " + line);
    }

    // Important:
    // We do NOT validate these activating values against the current phase domain
    // written in the search-space file, because the file may contain a restricted
    // sub-domain for this phase while the conditional still expresses the global
    // semantic dependency of the parameter.

    conditionals_.push_back(conditional);
}

void LocalSearchSpace::parseInfoLine_(const std::string& line) {
    if (startsWith_(line, "Initial configuration evaluated:")) {
        const std::string rhs = trim_(line.substr(std::string("Initial configuration evaluated:").size()));
        if (rhs == "1") {
            initial_configuration_evaluated_ = true;
        } else if (rhs == "0") {
            initial_configuration_evaluated_ = false;
        } else {
            throw std::runtime_error("Invalid value for 'Initial configuration evaluated': " + rhs);
        }
        return;
    }

    if (startsWith_(line, "Objective value of the initial configuration:")) {
        const std::string rhs = trim_(line.substr(std::string("Objective value of the initial configuration:").size()));
        initial_configuration_objective_ = std::stod(rhs);
        return;
    }
}

void LocalSearchSpace::loadFromFile(const std::string& filename) {
    clear();

    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open search space file: " + filename);
    }

    enum class ParseSection {
        PARAMETERS_AND_TUPLES,
        CONDITIONALS,
        INFO
    };

    ParseSection section = ParseSection::PARAMETERS_AND_TUPLES;
    std::string line;
    std::map<std::string, Value> initial_map;

    while (std::getline(file, line)) {
        line = stripComment_(line);
        if (line.empty()) {
            continue;
        }

        if (line == "Conditionals:" || line == "Conditionals") {
            section = ParseSection::CONDITIONALS;
            continue;
        }

        if (line == "Info:" || line == "Info") {
            section = ParseSection::INFO;
            continue;
        }

        if (section == ParseSection::PARAMETERS_AND_TUPLES) {
            if (!line.empty() && line.front() == '{') {
                parseForbiddenTupleLine_(line);
            } else {
                parseParameterLine_(line);
            }
            continue;
        }

        if (section == ParseSection::CONDITIONALS) {
            parseConditionalLine_(line);
            continue;
        }

        if (section == ParseSection::INFO) {
            parseInfoLine_(line);
            continue;
        }
    }

    for (const auto& parameter : parameters_) {
        initial_map.insert_or_assign(parameter.getName(), parameter.getDefaultValue());
    }

    initial_configuration_ = Configuration(initial_map, false);
    initial_configuration_ = normalizeConfiguration(initial_configuration_);

    if (initial_configuration_evaluated_ && !initial_configuration_objective_.has_value()) {
        throw std::runtime_error("Initial configuration is marked as evaluated but no objective value is provided.");
    }

    logger_.info("Local search space loaded from file: ", filename);
    logger_.info("Number of parameters in local search space: ", static_cast<int>(parameters_.size()));
    logger_.info("Number of forbidden tuples in local search space: ", static_cast<int>(forbidden_tuples_.size()));
    logger_.info("Number of conditionals in local search space: ", static_cast<int>(conditionals_.size()));
}

bool LocalSearchSpace::matchesForbiddenTuple_(const Configuration& config, const std::vector<std::pair<std::string, Value>>& tuple) const {
    const auto& conf = config.getConfigurationMap();
    for (const auto& [param_name, value] : tuple) {
        auto it = conf.find(param_name);
        if (it == conf.end()) {
            return false;
        }
        if (!(it->second == value)) {
            return false;
        }
    }
    return true;
}

bool LocalSearchSpace::conditionalSatisfied_(const Configuration& config, const ConditionalActivation& conditional) const {
    const auto& conf = config.getConfigurationMap();
    auto it = conf.find(conditional.parent_parameter);
    if (it == conf.end()) {
        throw std::runtime_error("Missing parent parameter in configuration: " + conditional.parent_parameter);
    }

    return std::any_of(conditional.activating_values.begin(), conditional.activating_values.end(),
                       [&](const Value& value) { return it->second == value; });
}

bool LocalSearchSpace::isParameterActive(const Configuration& config, const std::string& parameter_name) const {
    if (!isKnownParameter(parameter_name)) {
        throw std::runtime_error("Unknown parameter when checking activity: " + parameter_name);
    }

    bool has_conditional = false;
    for (const auto& conditional : conditionals_) {
        if (conditional.child_parameter == parameter_name) {
            has_conditional = true;
            if (!conditionalSatisfied_(config, conditional)) {
                return false;
            }
        }
    }

    if (!has_conditional) {
        return true;
    }

    return true;
}

std::vector<std::string> LocalSearchSpace::getActiveParameters(const Configuration& config) const {
    std::vector<std::string> active_parameters;
    for (const auto& parameter : parameters_) {
        if (isParameterActive(config, parameter.getName())) {
            active_parameters.push_back(parameter.getName());
        }
    }
    return active_parameters;
}

Configuration LocalSearchSpace::normalizeConfiguration(const Configuration& config) const {
    std::map<std::string, Value> normalized_map = config.getConfigurationMap();

    for (const auto& parameter : parameters_) {
        if (normalized_map.find(parameter.getName()) == normalized_map.end()) {
            normalized_map.insert_or_assign(parameter.getName(), parameter.getDefaultValue());
        }
    }

    Configuration normalized_config(normalized_map, config.useMipStart());

    for (const auto& parameter : parameters_) {
        const std::string param_name = parameter.getName();

        if (!isParameterActive(normalized_config, param_name)) {
            normalized_map.insert_or_assign(param_name, parameter.getDefaultValue());
        } else {
            if (!valueInDomain_(param_name, normalized_map.at(param_name))) {
                throw std::runtime_error("Active parameter value not in domain for parameter: " + param_name);
            }
        }
    }

    return Configuration(normalized_map, config.useMipStart());
}

bool LocalSearchSpace::isValidConfiguration(const Configuration& config) const {
    const Configuration normalized = normalizeConfiguration(config);
    const auto& conf = normalized.getConfigurationMap();

    for (const auto& parameter : parameters_) {
        auto it = conf.find(parameter.getName());
        if (it == conf.end()) {
            return false;
        }
        if (!valueInDomain_(parameter.getName(), it->second)) {
            return false;
        }
    }

    for (const auto& tuple : forbidden_tuples_) {
        if (matchesForbiddenTuple_(normalized, tuple)) {
            return false;
        }
    }

    return true;
}

std::vector<Configuration> LocalSearchSpace::generateNeighbors(const Configuration& config) const {
    const Configuration normalized = normalizeConfiguration(config);
    const auto active_parameters = getActiveParameters(normalized);

    std::vector<Configuration> neighbors;
    std::unordered_set<ConfigurationId> seen;

    for (const auto& active_param_name : active_parameters) {
        const auto& parameter = getParameter_(active_param_name);
        const Value current_value = normalized.getConfigurationMap().at(active_param_name);
        const auto domain = parameter.getValues();

        for (const auto& candidate_value : domain) {
            if (candidate_value == current_value) {
                continue;
            }

            std::map<std::string, Value> candidate_map = normalized.getConfigurationMap();
            candidate_map.insert_or_assign(active_param_name, candidate_value);

            Configuration candidate(candidate_map, normalized.useMipStart());
            candidate = normalizeConfiguration(candidate);

            if (!isValidConfiguration(candidate)) {
                continue;
            }

            if (candidate == normalized) {
                continue;
            }

            if (seen.insert(candidate.getConfigurationId()).second) {
                neighbors.push_back(candidate);
            }
        }
    }

    return neighbors;
}

Configuration LocalSearchSpace::sampleRandomConfiguration(std::mt19937& rng, bool use_mip_start) const {
    if (parameters_.empty()) {
        throw std::runtime_error("Cannot sample random configuration: local search space is empty.");
    }

    for (int attempt = 0; attempt < 10000; ++attempt) {
        std::map<std::string, Value> conf;
        for (const auto& parameter : parameters_) {
            const auto values = parameter.getValues();
            std::uniform_int_distribution<std::size_t> dist(0, values.size() - 1);
            conf.insert_or_assign(parameter.getName(), values.at(dist(rng)));
        }

        Configuration sampled(conf, use_mip_start);
        sampled = normalizeConfiguration(sampled);

        if (isValidConfiguration(sampled)) {
            return sampled;
        }
    }

    throw std::runtime_error("Failed to sample a valid random configuration after many attempts.");
}

// ============================================================
// IteratedLocalSearch implementation
// ============================================================

void IteratedLocalSearch::createSearchSpace_() {
    logger_.info("Creating local search space from parameter space file: ", options_.search_space_file);
    search_space_.loadFromFile(options_.search_space_file);
}

void IteratedLocalSearch::checkMipStartFile_() {
    if (options_.use_mip_starts) {
        if (options_.mip_start_file.has_value()) {
            logger_.info("Using MIP start file for local search: ", options_.mip_start_file.value());
        } else {
            options_.use_mip_starts = false;
            logger_.warn("MIP start option enabled but no MIP start file provided. Local search will proceed without MIP start.");
        }
    } else {
        logger_.info("MIP start option not enabled. Local search will proceed without MIP start.");
    }
}

void IteratedLocalSearch::initializeFromSearchSpace_() {
    current_configuration_ = search_space_.getInitialConfiguration();
    current_configuration_ = Configuration(
        current_configuration_.getConfigurationMap(),
        options_.use_mip_starts
    );
    incumbent_solution_ = current_configuration_;
}

void IteratedLocalSearch::injectInitialConfigurationIfAlreadyEvaluated_() {
    if (!search_space_.initialConfigurationAlreadyEvaluated()) {
        return;
    }

    if (!search_space_.getInitialConfigurationObjective().has_value()) {
        throw std::runtime_error("Initial configuration marked as evaluated but no objective value available.");
    }

    const double objective_value = search_space_.getInitialConfigurationObjective().value();
    EvaluationRecord initial_record = createEvaluationRecord_(current_configuration_, objective_value);
    memory_.addEvaluation(current_configuration_, initial_record);

    logger_.info("Injected initial configuration into local search cache with objective value: ", objective_value);
    updateStopConditionFromObjective_(objective_value);
}

void IteratedLocalSearch::computeInitialConfiguration_() {
    if (current_configuration_ == Configuration()) {
        throw std::runtime_error("Initial configuration for local search was not initialized.");
    }

    if (memory_.getCachedObjectiveValue(current_configuration_).has_value()) {
        logger_.info("Initial configuration already available in local search memory with objective value: ",
                     memory_.getCachedObjectiveValue(current_configuration_).value());
        return;
    }

    logger_.info("Evaluating initial configuration for local search phase...");
    const double initial_objective_value = evaluateConfiguration_(current_configuration_);
    logger_.info("Initial configuration evaluated with objective value: ", initial_objective_value);
}

void IteratedLocalSearch::computeRandomSampling_() {
    if (options_.random_initial_samples == 0 || terminationCriterionMet_()) {
        return;
    }

    logger_.info("Starting random initial sampling with ", static_cast<int>(options_.random_initial_samples), " random configurations...");

    for (std::size_t i = 0; i < options_.random_initial_samples; ++i) {
        if (terminationCriterionMet_()) {
            break;
        }

        Configuration sampled = search_space_.sampleRandomConfiguration(rng_, options_.use_mip_starts);
        logger_.info("Evaluating random sampled initial configuration ", static_cast<int>(i + 1), "/",
                     static_cast<int>(options_.random_initial_samples), "...");

        if (better_(sampled, current_configuration_)) {
            current_configuration_ = sampled;
            updateIncumbentIfNeeded_(current_configuration_);
        }
    }

    logger_.info("Random initial sampling completed.");
}

bool IteratedLocalSearch::better_(const Configuration& new_config, const Configuration& current_config) {
    const double new_obj = evaluateConfiguration_(new_config);
    const double current_obj = evaluateConfiguration_(current_config);

    if (options_.accept_ties) {
        return new_obj <= current_obj;
    }
    return new_obj < current_obj;
}

double IteratedLocalSearch::evaluateConfiguration_(const Configuration& config) {
    EvaluationRecord record = getOrEvaluate_(config);
    updateStopConditionFromObjective_(record.objective_value);
    return record.objective_value;
}

EvaluationRecord IteratedLocalSearch::getOrEvaluate_(const Configuration& config) {
    if (auto cached = memory_.getCachedEvaluation(config); cached.has_value()) {
        return cached.value();
    }

    if (nb_evaluations_ >= options_.evaluation_budget) {
        throw std::runtime_error("Evaluation budget exhausted while trying to evaluate a new configuration.");
    }

    EvaluationRecord record = runSolverAndCreateRecord_(config);
    memory_.addEvaluation(config, record);
    ++nb_evaluations_;
    return record;
}

int IteratedLocalSearch::elapsedSeconds_() const {
    const auto now = std::chrono::steady_clock::now();
    return static_cast<int>(std::chrono::duration_cast<std::chrono::seconds>(now - run_start_time_).count());
}

std::string IteratedLocalSearch::buildConfigFilePath_(const Configuration& config) const {
    return options_.working_directory + "/ils_config_" + std::to_string(config.getConfigurationId()) + ".prm";
}

std::string IteratedLocalSearch::buildLogFilePath_(const Configuration& config) const {
    return options_.working_directory + "/ils_log_" + std::to_string(config.getConfigurationId()) + ".log";
}

EvaluationRecord IteratedLocalSearch::runSolverAndCreateRecord_(const Configuration& config) {
    const double objective_value = runSolverAndGetObjective_(config);
    return createEvaluationRecord_(config, objective_value);
}

double IteratedLocalSearch::runSolverAndGetObjective_(const Configuration& config) {
    std::filesystem::create_directories(options_.working_directory);

    const std::string config_file_path = buildConfigFilePath_(config);
    const std::string log_file_path = buildLogFilePath_(config);

    config.generateConfigFile(config_file_path);

    logger_.info("Starting CPLEX solver on instance: ", options_.instance_file,
                 " with config: ", config_file_path);

    if (config.useMipStart() && options_.mip_start_file.has_value()) {
        std::string mip_start_file = options_.mip_start_file.value();
        CPLEXSolver solver(
            logger_,
            options_.instance_file,
            config_file_path,
            log_file_path,
            options_.nb_threads_solver,
            options_.cutoff_solver_time,
            mip_start_file
        );
        solver.solve();
        return solver.getObjectiveValue();
    } else {
        CPLEXSolver solver(
            logger_,
            options_.instance_file,
            config_file_path,
            log_file_path,
            options_.nb_threads_solver,
            options_.cutoff_solver_time
        );
        solver.solve();
        return solver.getObjectiveValue();
    }
}

EvaluationRecord IteratedLocalSearch::createEvaluationRecord_(const Configuration& config, double objective_value) {
    EvaluationRecord record;
    record.evaluation_id = static_cast<EvaluationId>(next_evaluation_id_++);
    record.objective_value = objective_value;
    record.time_evaluated = GlobalTimer::elapsedSeconds();
    record.configuration_id = config.getConfigurationId();

    record.mip_start_used = config.useMipStart();
    record.used_mip_start_id = std::nullopt;
    record.mip_start_source_evaluation_id = std::nullopt;

    record.produced_mip_start = false;
    record.produced_mip_start_id = std::nullopt;

    record.worker_id = -1;
    record.iteration = static_cast<int>(current_iteration_);
    record.phase = -1;

    return record;
}

std::vector<Configuration> IteratedLocalSearch::generateNeighbors_(const Configuration& config) const {
    return search_space_.generateNeighbors(config);
}

Configuration IteratedLocalSearch::iterativeFirstImprovement_(const Configuration& start_config) {
    Configuration current = search_space_.normalizeConfiguration(start_config);

    while (!terminationCriterionMet_()) {
        std::vector<Configuration> neighbors = generateNeighbors_(current);
        std::shuffle(neighbors.begin(), neighbors.end(), rng_);

        bool improved = false;
        for (const auto& neighbor : neighbors) {
            if (terminationCriterionMet_()) {
                break;
            }

            if (better_(neighbor, current)) {
                current = neighbor;
                improved = true;
                updateIncumbentIfNeeded_(current);
                break;
            }
        }

        if (!improved) {
            break;
        }
    }

    return current;
}

Configuration IteratedLocalSearch::perturb_(const Configuration& config) {
    Configuration perturbed = config;

    for (std::size_t i = 0; i < options_.perturbation_strength; ++i) {
        std::vector<Configuration> neighbors = generateNeighbors_(perturbed);
        if (neighbors.empty()) {
            break;
        }

        std::uniform_int_distribution<std::size_t> dist(0, neighbors.size() - 1);
        perturbed = neighbors[dist(rng_)];
    }

    return perturbed;
}

void IteratedLocalSearch::updateIncumbentIfNeeded_(const Configuration& candidate) {
    if (better_(candidate, incumbent_solution_)) {
        incumbent_solution_ = candidate;
        logger_.info("New incumbent found with objective value: ", evaluateConfiguration_(incumbent_solution_));
    }
}

bool IteratedLocalSearch::terminationCriterionMet_() const {
    return stop_condition_met_ || nb_evaluations_ >= options_.evaluation_budget;
}

void IteratedLocalSearch::updateStopConditionFromObjective_(double objective) {
    if (objective <= options_.acceptance_threshold) {
        stop_condition_met_ = true;
    }
}

void IteratedLocalSearch::run() {
    logger_.info("Starting local search phase...");

    run_start_time_ = std::chrono::steady_clock::now();

    createSearchSpace_();
    checkMipStartFile_();
    initializeFromSearchSpace_();
    injectInitialConfigurationIfAlreadyEvaluated_();
    computeInitialConfiguration_();
    computeRandomSampling_();

    current_configuration_ = iterativeFirstImprovement_(current_configuration_);
    updateIncumbentIfNeeded_(current_configuration_);

    std::bernoulli_distribution restart_distribution(options_.restart_probability);

    while (!terminationCriterionMet_()) {
        ++current_iteration_;
        Configuration candidate;

        if (restart_distribution(rng_)) {
            logger_.info("Restarting from a random configuration...");
            candidate = search_space_.sampleRandomConfiguration(rng_, options_.use_mip_starts);
        } else {
            logger_.info("Applying perturbation to current configuration...");
            candidate = perturb_(current_configuration_);
        }

        candidate = iterativeFirstImprovement_(candidate);

        if (better_(candidate, current_configuration_)) {
            current_configuration_ = candidate;
        }

        updateIncumbentIfNeeded_(candidate);
    }

    logger_.info("Local search phase completed.");
    logger_.info("Total number of new local-search evaluations: ", static_cast<int>(nb_evaluations_));

    if (memory_.getCachedObjectiveValue(incumbent_solution_).has_value()) {
        logger_.info("Final incumbent objective value: ",
                     memory_.getCachedObjectiveValue(incumbent_solution_).value());
    }
}