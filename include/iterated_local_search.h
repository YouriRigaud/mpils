// MPILS: Multi-Phase Iterated Local Search Tuner
//
// Author: Youri Rigaud
// License: GNU GPLv3

#ifndef ITERATED_LOCAL_SEARCH_H
#define ITERATED_LOCAL_SEARCH_H

#include "tuning_objective.h"
#include "solver_time_mode.h"
#include "working_directory.h"
#include "logger.h"
#include "tuner_memory.h"
#include "configuration.h"
#include "parameter_space.h"

#include <optional>
#include <random>
#include <unordered_map>
#include <vector>
#include <string>
#include <map>
#include <chrono>
#include <limits>

#ifdef USE_MPI
#include <mpi.h>
#endif

class LocalSearchMemory {
    private:
        Logger logger_;
        std::vector<EvaluationRecord> local_evaluations_;
        std::unordered_map<ConfigurationId, EvaluationRecord> cache_;
        std::unordered_map<ConfigurationId, Configuration> configurations_;

    public:
        explicit LocalSearchMemory(Logger logger): logger_(logger) {}

        void addEvaluation(const Configuration& config, const EvaluationRecord& record) {
            local_evaluations_.push_back(record);
            cache_[record.configuration_id] = record;
            configurations_[record.configuration_id] = config;
        }

        void seedCachedEvaluation(const Configuration& config, const EvaluationRecord& record) {
            cache_[record.configuration_id] = record;
            configurations_[record.configuration_id] = config;
        }

        void addSharedEvaluation(const Configuration& config, const EvaluationRecord& record) {
            cache_[record.configuration_id] = record;
            configurations_[record.configuration_id] = config;
        }

        bool hasEvaluation(const Configuration& config) const {
            return cache_.find(config.getConfigurationId()) != cache_.end();
        }

        std::optional<EvaluationRecord> getCachedEvaluation(const Configuration& config) const {
            auto it = cache_.find(config.getConfigurationId());
            if (it != cache_.end()) {
                return it->second;
            }
            return std::nullopt;
        }

        std::optional<double> getCachedObjectiveValue(const Configuration& config) const {
            auto it = cache_.find(config.getConfigurationId());
            if (it != cache_.end()) {
                return it->second.objective_value;
            }
            return std::nullopt;
        }

        const std::vector<EvaluationRecord>& getEvaluations() const {
            return local_evaluations_;
        }

        std::vector<std::pair<Configuration, EvaluationRecord>> getEvaluationsWithConfigurations() const {
            std::vector<std::pair<Configuration, EvaluationRecord>> result;
            result.reserve(local_evaluations_.size());

            for (const auto& record : local_evaluations_) {
                result.emplace_back(
                    configurations_.at(record.configuration_id),
                    record
                );
            }

            return result;
        }   
};

struct ConditionalActivation {
    std::string child_parameter;
    std::string parent_parameter;
    std::vector<Value> activating_values;
};

class ReplicatedObjectiveCache {
    private:
        Logger logger_;
        std::unordered_map<ConfigurationId, double> objective_cache_;

#ifdef USE_MPI
        struct ObjectiveUpdateMessage {
            ConfigurationId configuration_id;
            double objective_value;
        };

        struct PendingSend {
            ObjectiveUpdateMessage message;
            MPI_Request request = MPI_REQUEST_NULL;
            int destination = -1;
        };

        std::vector<PendingSend> pending_sends_;
        bool mpi_available_ = false;
        int world_rank_ = 0;
        int world_size_ = 1;
        static constexpr int kObjectiveCacheTag = 701;
#endif

        void updateObjective_(ConfigurationId configuration_id, double objective_value);

    public:
        explicit ReplicatedObjectiveCache(Logger logger): logger_(logger) {}

        void initialize();
        void seed(ConfigurationId configuration_id, double objective_value);
        void publish(ConfigurationId configuration_id, double objective_value);
        void pollIncoming();
        void flushPendingSends();
        void waitForPendingSends();

        std::optional<double> getObjective(ConfigurationId configuration_id) const {
            auto it = objective_cache_.find(configuration_id);
            if (it != objective_cache_.end()) {
                return it->second;
            }
            return std::nullopt;
        }
};

class LocalSearchSpace {
    private:
        Logger logger_;
        std::vector<Parameter> parameters_;
        std::unordered_map<std::string, std::size_t> parameter_index_;
        std::vector<std::vector<std::pair<std::string, Value>>> forbidden_tuples_;
        std::vector<ConditionalActivation> conditionals_;

        Configuration initial_configuration_;
        bool initial_configuration_evaluated_ = false;
        std::optional<double> initial_configuration_objective_ = std::nullopt;

        static std::string trim_(const std::string& s);
        static bool startsWith_(const std::string& s, const std::string& prefix);
        static std::vector<std::string> split_(const std::string& s, char delimiter);
        static std::string stripComment_(const std::string& line);

        Value parseValue_(const std::string& token) const;
        std::vector<Value> parseValueList_(const std::string& values_str) const;

        void parseParameterLine_(const std::string& line);
        void parseForbiddenTupleLine_(const std::string& line);
        void parseConditionalLine_(const std::string& line);
        void parseInfoLine_(const std::string& line);

        void registerParameter_(const Parameter& parameter);

        bool valueInDomain_(const std::string& parameter_name, const Value& value) const;
        const Parameter& getParameter_(const std::string& parameter_name) const;

        bool matchesForbiddenTuple_(const Configuration& config, const std::vector<std::pair<std::string, Value>>& tuple) const;
        bool conditionalSatisfied_(const Configuration& config, const ConditionalActivation& conditional) const;

    public:
        explicit LocalSearchSpace(Logger logger): logger_(logger) {}

        void clear();
        void loadFromFile(const std::string& filename);

        const std::vector<Parameter>& getParameters() const {
            return parameters_;
        }

        const std::vector<std::vector<std::pair<std::string, Value>>>& getForbiddenTuples() const {
            return forbidden_tuples_;
        }

        const std::vector<ConditionalActivation>& getConditionals() const {
            return conditionals_;
        }

        const Configuration& getInitialConfiguration() const {
            return initial_configuration_;
        }

        bool initialConfigurationAlreadyEvaluated() const {
            return initial_configuration_evaluated_;
        }

        std::optional<double> getInitialConfigurationObjective() const {
            return initial_configuration_objective_;
        }

        bool isKnownParameter(const std::string& parameter_name) const {
            return parameter_index_.find(parameter_name) != parameter_index_.end();
        }

        bool isParameterActive(const Configuration& config, const std::string& parameter_name) const;
        std::vector<std::string> getActiveParameters(const Configuration& config) const;

        Configuration normalizeConfiguration(const Configuration& config) const;
        bool isValidConfiguration(const Configuration& config) const;
        std::vector<Configuration> generateNeighbors(const Configuration& config) const;
        Configuration sampleRandomConfiguration(std::mt19937& rng, bool use_mip_start) const;
};

class IteratedLocalSearch {
    public:
        struct Options {
            std::string search_space_file;
            std::string instance_file;
            std::string log_file_solver;
            std::string working_directory = buildTunerPath("local_search");

            unsigned int random_seed = 0;
            std::size_t evaluation_budget = 20;
            std::size_t perturbation_strength = 3;
            std::size_t random_initial_samples = 0;
            double restart_probability = 0.1;
            bool accept_ties = false;
            double acceptance_threshold = 0.0;
            bool use_shared_cache = false;
            std::vector<CompactSharedCacheSeedEntry> shared_cache_seed_entries;

            bool use_mip_starts = false;
            std::optional<std::string> mip_start_file = std::nullopt;
            std::optional<MipStartId> used_mip_start_id = std::nullopt;
            bool produce_mip_starts = false;
            double best_mip_start_upper_bound = std::numeric_limits<double>::max();

            int nb_threads_solver = 2;
            double cutoff_solver_time = 15.0;
            SolverTimeMode solver_time_mode = SolverTimeMode::Seconds;
            TuningObjective tuning_objective = TuningObjective::Gap;
        };

    private:
        Logger logger_;
        LocalSearchMemory memory_;
        ReplicatedObjectiveCache shared_objective_cache_;
        LocalSearchSpace search_space_;
        Options options_;

        bool stop_condition_met_ = false;
        std::size_t nb_evaluations_ = 0;
        std::size_t next_evaluation_id_ = 1;
        std::size_t current_iteration_ = 0;

        std::mt19937 rng_;
        std::chrono::steady_clock::time_point run_start_time_;

        Configuration incumbent_solution_;
        Configuration current_configuration_;
        double best_mip_start_upper_bound_;

#ifdef USE_MPI
        struct PendingStopSignal {
            int payload = 1;
            MPI_Request request = MPI_REQUEST_NULL;
            int destination = -1;
        };

        std::vector<PendingStopSignal> pending_stop_signals_;
        bool mpi_available_ = false;
        int world_rank_ = 0;
        int world_size_ = 1;
        bool global_stop_published_ = false;
        static constexpr int kGlobalStopTag = 702;
#endif

        void createSearchSpace_();
        void seedSharedCache_();
        void checkMipStartFile_();
        void initializeFromSearchSpace_();
        void injectInitialConfigurationIfAlreadyEvaluated_();
        void computeInitialConfiguration_();
        void computeRandomSampling_();

        bool better_(const Configuration& new_config, const Configuration& current_config);

        double evaluateConfiguration_(const Configuration& config);
        EvaluationRecord getOrEvaluate_(const Configuration& config);

        EvaluationRecord runSolverAndCreateRecord_(const Configuration& config);
        bool wouldImproveMipStartUpperBound_(const std::optional<double>& upper_bound) const;

        EvaluationRecord createEvaluationRecord_(
            const Configuration& config,
            double objective_value,
            std::optional<double> gap = std::nullopt,
            std::optional<double> upper_bound = std::nullopt,
            std::optional<double> lower_bound = std::nullopt,
            bool produced_mip_start = false,
            const std::string& produced_mip_start_file = ""
        );
        EvaluationRecord createSharedEvaluationRecord_(const Configuration& config, double objective_value);

        std::vector<Configuration> generateNeighbors_(const Configuration& config) const;
        Configuration iterativeFirstImprovement_(const Configuration& start_config);
        Configuration perturb_(const Configuration& config);
        void updateIncumbentIfNeeded_(const Configuration& candidate);
        bool terminationCriterionMet_();
        void updateStopConditionFromGap_(const std::optional<double>& gap);
        int elapsedSeconds_() const;

        void initializeGlobalStop_();
        void publishGlobalStop_();
        void pollGlobalStop_();
        void flushPendingStopSignals_();
        void waitForPendingStopSignals_();

        std::string buildConfigFilePath_(const Configuration& config) const;
        std::string buildLogFilePath_(const Configuration& config) const;
        std::string buildMipStartCandidateFilePath_(const Configuration& config, EvaluationId evaluation_id) const;

    public:
        IteratedLocalSearch(
            Logger logger,
            const Options& options
        ): logger_(logger),
           memory_(logger),
           shared_objective_cache_(logger),
           search_space_(logger),
           options_(options),
           rng_(options.random_seed),
           best_mip_start_upper_bound_(options.best_mip_start_upper_bound)
        {}

        void run();

        std::vector<EvaluationRecord> getEvaluations() const {
            return memory_.getEvaluations();
        }

        std::vector<std::pair<Configuration, EvaluationRecord>> getEvaluationsWithConfigurations() const {
            return memory_.getEvaluationsWithConfigurations();
        }
};

#endif // ITERATED_LOCAL_SEARCH_H
