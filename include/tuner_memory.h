/**
 * @file tuner_memory.h
 * @brief Represents the memory component of the tuner.
 *
 * The memory module is responsible for storing all the configurations of the Tuner evaluated.
 * It has an evaluation history, it provides statistics about the configurations, and it keeps track of the best configuration found.
 * The memory also stores the history of MIP starts generated and used, to be able to analyze their impact on the tuning process.
 *
 * @author Youri Rigaud
 * @copyright Copyright 2026 Youri Rigaud. All rights reserved.
 *            This software is licensed under the GNU General Public License v3.0.
 *            See the accompanying LICENSE file for full details.
 */

#ifndef TUNER_MEMORY_H
#define TUNER_MEMORY_H

#include "configuration.h"
#include "parameter_space.h"
#include "logger.h"
#include "globaltimer.h"

#include <map>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <stdexcept>
#include <memory>
#include <fstream>
#include <optional>
#include <limits>
#include <algorithm>
#include <string>

static constexpr double kMaxObjective = std::numeric_limits<double>::max();

using EvaluationId = uint64_t;
using MipStartId = uint64_t;

/**
 * @brief Struct representing statistics about a configuration.
 */
struct ConfigurationStats {
    int nb_evaluations = 0;
    double best_objective = std::numeric_limits<double>::max();
    std::optional<EvaluationId> best_evaluation_id;
};

/**
 * @brief Struct representing a record of an evaluated configuration.
 */
struct EvaluationRecord {
    EvaluationId evaluation_id;
    double objective_value; ///< Tuning objective value for this evaluation (currently upper bound with fallback)
    std::optional<double> gap; ///< Solver gap recorded for this evaluation, if available
    std::optional<double> upper_bound; ///< Solver upper bound recorded for this evaluation, if available
    std::optional<double> lower_bound; ///< Solver lower bound recorded for this evaluation, if available
    int time_evaluated;
    ConfigurationId configuration_id;

    bool mip_start_used;
    std::optional<MipStartId> used_mip_start_id;
    std::optional<EvaluationId> mip_start_source_evaluation_id;
    
    bool produced_mip_start;
    std::optional<MipStartId> produced_mip_start_id;
    
    int worker_id;
    int iteration;
    int phase; // 0 for exploration, 1 for expansion
};

/**
 * @brief Struct representing a record of a MIP start generated.
 */
struct MipStartRecord {
    MipStartId mip_start_id;
    std::string mip_start_file;
    EvaluationId evaluation_id; // Evaluation that produced this MIP start
};

struct RecordEvaluationOptions {
    double objective_value = std::numeric_limits<double>::max(); /// Tuning objective value obtained from the evaluation.
    std::optional<double> gap = std::nullopt; /// Solver gap obtained from the evaluation, if available.
    std::optional<double> upper_bound = std::nullopt; /// Solver upper bound obtained from the evaluation, if available.
    std::optional<double> lower_bound = std::nullopt; /// Solver lower bound obtained from the evaluation, if available.
    int time_evaluated = -1;    /// Time when the configuration was evaluated (in seconds since tuning started).
    int worker_id = -1;         /// ID of the worker that performed the evaluation (for logging purposes).
    int iteration = -1;         /// Iteration number during which the evaluation was performed (for logging purposes).
    int phase = -1;             /// Phase of the tuning process during which the evaluation was performed (0 for exploration, 1 for expansion).
    
    bool mip_start_used = false;                                                /// Flag indicating if a MIP start was used for this evaluation.
    std::optional<MipStartId> used_mip_start_id = std::nullopt;                 /// Optional ID of the MIP start used, if any.
    std::optional<EvaluationId> mip_start_source_evaluation_id = std::nullopt;  /// Optional ID of the evaluation that produced the MIP start used, if any.
    bool produced_mip_start = false;                                            /// Flag indicating if this evaluation produced a MIP start for future evaluations.
    std::string mip_start_file;                                                 /// If produced_mip_start is true, the file path of the MIP start generated.
};

/**
 * @brief Class representing the memory of the tuner.
 * 
 * This class stores all the configurations tested by the tuner,
 * along with their objective values if evaluated.
 * It also keeps track of the best configuration found so far.
 */
class TunerMemory {
    private:
        Logger& logger_;

        std::unordered_map<ConfigurationId, Configuration> configurations_by_id_;                     ///< Map of all configurations stored, indexed by their hash ID
        std::unordered_map<ConfigurationId, ConfigurationStats> stats_by_id_;               ///< Map of configuration statistics, indexed by configuration hash ID
        std::vector<EvaluationRecord> evaluations_;                                                  ///< History of the evaluations performed, in chronological order
        
        std::unordered_map<MipStartId, MipStartRecord> mip_starts_by_id_;                   ///< Map of MIP start records indexed by their ID

        std::optional<uint64_t> best_evaluation_index_;                 ///< Index of the best evaluation in the evaluations_ vector
        double best_objective_ = std::numeric_limits<double>::max(); ///< Best objective value found so far

        std::optional<uint64_t> best_evaluation_without_mip_start_index_;                ///< Index of the best evaluation in the evaluations_ vector that did not use MIP start
        double best_objective_without_mip_start_ = std::numeric_limits<double>::max(); ///< Best objective value found so far among evaluations that did not use MIP start

        Configuration default_configuration_;       ///< Default configuration
        EvaluationId next_evaluation_id_ = 1;                ///< Counter for assigning unique evaluation IDs
        MipStartId next_mip_start_id_ = 1;                      ///< Counter for assigning unique MIP start IDs
    
        static double clampObjective_(double objective) { return std::min(objective, kMaxObjective); } ///< Clamp the objective value to a maximum threshold kMaxObjective

        ConfigurationId ensureConfigurationStored_(const Configuration& config); ///< Ensure that a configuration is stored in memory and return its ID. If the configuration already exists, return the existing ID.

        EvaluationRecord createEvaluationRecord_(const RecordEvaluationOptions& options, const ConfigurationId config_id, const EvaluationId eval_id); ///< Create an evaluation record from a configuration and the provided options.

        void updateBestEvaluation_(const EvaluationRecord& record); ///< Update the best evaluation found based on a new evaluation record.
        
        void updateStatsForConfiguration_(const EvaluationRecord& record); ///< Update the statistics for a configuration based on a new evaluation record.
    
        void updateMipStartRecords_(const EvaluationRecord& record, const std::string& mip_start_file); ///< Update the MIP start records based on a new evaluation record that has produced a MIP start.

        void logNewBestEvaluation_(const EvaluationRecord& record); ///< Print a log message when a new best evaluation is found, with details about the evaluation and the configuration.

    public:
        /** @brief Construct a TunerMemory object */
        TunerMemory(Logger& logger): logger_(logger) {}

        /** @brief Set the default configuration */
        void setDefaultConfiguration(const Configuration& config) {
            default_configuration_ = config;
        }

        /** @brief Get the default configuration */
        const Configuration& getDefaultConfiguration() const {
            return default_configuration_;
        }

        /**
         * @brief Add a configuration evaluation to the memory.
         * @param config Configuration that was evaluated.
         * @param options Struct containing optional parameters for the evaluation record. See RecordEvaluationOptions for details.
         * @return The unique ID assigned to this evaluation record.
         */
        EvaluationId recordEvaluation(const Configuration& config, const RecordEvaluationOptions& options = RecordEvaluationOptions());

        /**
         * @brief Set the file path of a MIP start generated from an evaluation.
         * @param mip_start_file Path to the MIP start file generated.
         * @param mip_start_id ID of the MIP start for which to set the file path.
         * @return True if the MIP start ID was found and the file path was set, false otherwise.
         */
        bool setMipStartFile(const std::string& mip_start_file, MipStartId mip_start_id);

        //TODO Change that 
        MipStartId getMipStartToUse() {
            return next_mip_start_id_ - 1; // The next MIP start ID to use is the last one that was generated, which is next_mip_start_id_ - 1
        }

        /** @brief Get the evaluation from its ID */
        const EvaluationRecord* getEvaluationById(EvaluationId eval_id) const {
            auto it = std::find_if(evaluations_.begin(), evaluations_.end(),
                                   [eval_id](const EvaluationRecord& record) { return record.evaluation_id == eval_id; });
            if (it == evaluations_.end()) {
                return nullptr;
            }
            return &(*it);
        }

        /** @brief Get the best evaluation record found so far, or nullptr if no evaluation has been recorded yet */
        const EvaluationRecord* getBestEvaluation() const {
            if (!best_evaluation_index_.has_value()) {
                return nullptr;
            }
            return &evaluations_[best_evaluation_index_.value()];
        }

        /** @brief Get the best evaluation record found so far that did not use MIP start, or nullptr if no such evaluation exists */
        const EvaluationRecord* getBestEvaluationWithoutMipStart() const {
            if (!best_evaluation_without_mip_start_index_.has_value()) {
                return nullptr;
            }
            return &evaluations_[best_evaluation_without_mip_start_index_.value()];
        }

        /** @brief Get the best configuration found so far, or nullptr if no configuration has been evaluated yet */
        const Configuration* getBestConfiguration() const {
            const EvaluationRecord* best_eval = getBestEvaluation();
            if (best_eval == nullptr) {
                return nullptr;
            }
            return &configurations_by_id_.at(best_eval->configuration_id);
        }

        /** @brief Get the best configuration found so far that did not use MIP start, or nullptr if no such configuration exists */
        const Configuration* getBestConfigurationWithoutMipStart() const {
            const EvaluationRecord* best_eval = getBestEvaluationWithoutMipStart();
            if (best_eval == nullptr) {
                return nullptr;
            }
            return &configurations_by_id_.at(best_eval->configuration_id);
        }

        /** @brief Get a configuration by its ID, throws an exception if the ID is not found */
        const Configuration& getConfigurationById(ConfigurationId config_id) const {
            auto it = configurations_by_id_.find(config_id);
            if (it == configurations_by_id_.end()) {
                throw std::runtime_error("Configuration ID not found in memory: " + std::to_string(config_id));
            }
            return it->second;
        }

        /** @brief Get the statistics of a configuration by its ID, throws an exception if the ID is not found */
        const ConfigurationStats& getConfigurationStatsById(ConfigurationId config_id) const {
            auto it = stats_by_id_.find(config_id);
            if (it == stats_by_id_.end()) {
                throw std::runtime_error("Configuration ID not found in memory for stats: " + std::to_string(config_id));
            }
            return it->second;
        }

        /** @brief Get the best objective value found so far */
        double getBestObjective() const {
            return best_objective_;
        }

        /** @brief Return true if any recorded evaluation has a gap at or below the given threshold */
        bool hasEvaluationAtOrBelowGap(double threshold) const {
            return std::any_of(evaluations_.begin(), evaluations_.end(),
                               [threshold](const EvaluationRecord& record) {
                                   return record.gap.has_value() && record.gap.value() <= threshold;
                               });
        }

        /** @brief Get the best objective value found so far among evaluations that did not use MIP start */
        double getBestObjectiveWithoutMipStart() const {
            return best_objective_without_mip_start_;
        }

        void exportEvaluationLogCSV(const std::string& filename) const {
            std::ofstream file(filename);
            if (!file.is_open()) {
                throw std::runtime_error("Could not open file to write evaluation log: " + filename);
            }
            // Write header
            file << "EvalID,TimeEvaluated,ObjectiveValue,Gap,UpperBound,LowerBound,ConfigID,MipStartID,WorkerID,Iteration,Phase\n";
            // Write records
            for (const auto& record : evaluations_) {
                file << record.evaluation_id << ","
                     << record.time_evaluated << ","
                     << record.objective_value << ","
                     << (record.gap.has_value() ? std::to_string(record.gap.value()) : "") << ","
                     << (record.upper_bound.has_value() ? std::to_string(record.upper_bound.value()) : "") << ","
                     << (record.lower_bound.has_value() ? std::to_string(record.lower_bound.value()) : "") << ","
                     << record.configuration_id << ",";
                if (record.mip_start_used) {
                    file << (record.used_mip_start_id.has_value() ? std::to_string(record.used_mip_start_id.value()) : "null");
                } else {
                    file << "null";
                }
                file << ","
                     << record.worker_id << ","
                     << record.iteration << ","
                     << record.phase << "\n";
            }
            file.close();
        }

        void exportUniqueConfigsCSV(const std::string& filename) const {
            std::ofstream file(filename);
            if (!file.is_open()) {
                throw std::runtime_error("Could not open file to write unique configurations: " + filename);
            }
            // Write header
            file << "ConfigID,UseMipStart,Configuration\n";
            // Write unique configurations
            for (const auto& [config_id, config] : configurations_by_id_) {
                file << config_id << "," << (config.useMipStart() ? "1" : "0") << ",";
                const auto& config_map = config.getConfigurationMap();
                size_t count = 0;
                for (const auto& pair : config_map) {
                    file << pair.first << "=" << pair.second.getString();
                    if (count < config_map.size() - 1) {
                        file << ";";
                    }
                    count++;
                }
                file << "\n";
            }
        }

        void exportUniqueMipStartsCSV(const std::string& filename) const {
            std::ofstream file(filename);
            if (!file.is_open()) {
                throw std::runtime_error("Could not open file to write unique MIP starts: " + filename);
            }
            // Write header
            file << "MipStartID,MipStartFile,SourceEvalID\n";
            // Write unique MIP starts
            for (const auto& [mip_start_id, mip_start] : mip_starts_by_id_) {
                file << mip_start_id << "," << mip_start.mip_start_file << ",";
                if (mip_start.evaluation_id > 0) {
                    file << mip_start.evaluation_id;
                } else {
                    file << "null";
                }
                file << "\n";
            }
            file.close();
        }

        /** @brief Get all the configurations stored in memory along with their best objective value */
        std::vector<std::pair<Configuration, double>> getAllConfigurationsWithObjectives() const {
            std::vector<std::pair<Configuration, double>> configs_with_objectives;
            for (const auto& [config_id, config] : configurations_by_id_) {
                const ConfigurationStats& stats = stats_by_id_.at(config_id);
                if (stats.nb_evaluations > 0) {
                    configs_with_objectives.emplace_back(config, stats.best_objective);
                }
            }
            return configs_with_objectives;
        }

        /** @brief Get the best MIP start file from the memory */
        std::optional<std::string> getBestMipStartFile() const {
            if (mip_starts_by_id_.empty()) {
                return std::nullopt;
            }
            // Return the last MIP start generated, which is the one with the highest ID (next_mip_start_id_ - 1)
            MipStartId best_mip_start_id = next_mip_start_id_ - 1;
            auto it = mip_starts_by_id_.find(best_mip_start_id);
            if (it != mip_starts_by_id_.end()) {
                return it->second.mip_start_file;
            }
            return std::nullopt;
        }
};

#endif // TUNER_MEMORY_H
