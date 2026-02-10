/**
 * @file tuner_memory.h
 * @brief Represents the memory component of the tuner.
 *
 * The memory module is responsible for storing all the configurations of the Tuner tested,
 * including their parameter values and objective value if evaluated by the solver.
 * It also keeps track of the best configuration found.
 *
 * @author Youri Rigaud
 * @copyright Copyright 2025 Youri Rigaud. All rights reserved.
 *            This software is licensed under the GNU General Public License v3.0.
 *            See the accompanying LICENSE file for full details.
 */

#ifndef TUNER_MEMORY_H
#define TUNER_MEMORY_H

#include "parameter.h"
#include "logger.h"
#include "globaltimer.h"

#include <map>
#include <unordered_set>
#include <stdexcept>
#include <memory>
#include <fstream>

/**
 * @brief Class representing a configuration.
 * 
 * This class stores a map of the parameter names to their values,
 * the objective value if evaluated, and a flag indicating if it has been evaluated.
 * Normally, the configuration is always evaluated pretty much right after being created.
 */
class Configuration {
    private:
        std::map<std::string, Value> configuration_; ///< Map of parameter names to their values
        double objective_;                           ///< Objective value if evaluated
        bool evaluated_;                             ///< Flag indicating if the configuration has been evaluated
        int time_evaluated_;                         ///< Time when the configuration was evaluated (in seconds since tuning started)
    
    public:
        /**
         * @brief Default constructor for Configuration.
         * Creates an unevaluated configuration.
         */
        Configuration() : evaluated_(false) {}

        /**
         * @brief Construct a Configuration object.
         * @param configuration Map of parameter names to their values.
         */
        Configuration(std::map<std::string, Value> configuration): configuration_(configuration), evaluated_(false) {}

        /**
         * @brief Construct a Configuration object with an objective value.
         * @param configuration Map of parameter names to their values.
         * @param objective     Objective value.
         */
        Configuration(
            std::map<std::string, Value> configuration,
            double objective
        ): configuration_(configuration), objective_(objective), evaluated_(true) {
            scaleObjective();
            time_evaluated_ = GlobalTimer::elapsedSeconds();
	    }

        /**
         * @brief Construct a Configuration object with an objective value.
         * @param configuration Map of parameter names to their values.
         * @param objective     Objective value.
         * @param time_evluated Time when the configuration was evaluated.
         */
        Configuration(
            std::map<std::string, Value> configuration,
            double objective,
            int time_evaluated
        ): configuration_(configuration), objective_(objective), evaluated_(true), time_evaluated_(time_evaluated) {
            scaleObjective();
	    }

        /** @brief Get the configuration map */
        std::map<std::string, Value> getConfiguration() const { return configuration_; }

        /** @brief Get the objective value */
        double getObjective() const { return objective_; }

        /** @brief Get the time when the configuration was evaluated */
        int getTime() const { return time_evaluated_; }

        /** @brief Check if the configuration has been evaluated */
        bool isEvaluated() const { return evaluated_; }

        /**
         * @brief Set the objective value and mark as evaluated.
         * @param objective Objective value to set.
         * @throws std::runtime_error if the configuration has already been evaluated.
         */
        void setObjective(double objective) {
            if (evaluated_) {
                throw std::runtime_error("Configuration has already been evaluated.");
            }
            objective_ = objective;
            scaleObjective();
            evaluated_ = true;
            time_evaluated_ = GlobalTimer::elapsedSeconds();
        }

        /**
         * @brief Set the objective value and mark as evaluated.
         * @param objective Objective value to set.
         * @param time_evaluated Time when the configuration was evaluated.
         * @throws std::runtime_error if the configuration has already been evaluated.
         */
        void setObjective(double objective, int time_evaluated) {
            if (evaluated_) {
                throw std::runtime_error("Configuration has already been evaluated.");
            }
            objective_ = objective;
            scaleObjective();
            evaluated_ = true;
            time_evaluated_ = time_evaluated;
        }

        /** @brief Scale the objective value if it exceeds 100 */
	    void scaleObjective() {
	        if (objective_ > 100) {
	            objective_ = 100;
	        }
	    }

        /** @brief Equality operator based on configuration map */
        bool operator==(const Configuration& other) const {
            return configuration_ == other.configuration_;
        }

        /** @brief Hash function for Configuration to be used in unordered_set */
        struct HashFunction {
            std::size_t operator()(const Configuration& config) const {
                std::size_t seed = 0;
                for (const auto& pair : config.getConfiguration()) {
                    seed ^= std::hash<std::string>()(pair.first) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
                    // Simple hash for Value based on its string representation
                    seed ^= std::hash<std::string>()(pair.second.getString()) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
                }
                return seed;
            }
        };

        void printConfiguration(std::ostream& out) const {
            out << "Configuration: ";
            for (const auto& pair : configuration_) {
                out << pair.first << "=" << pair.second.getString() << " ";
            }
            if (evaluated_) {
                out << "| Objective: " << objective_;
            } else {
                out << "| Objective: Not evaluated";
            }
            out << std::endl;
        }

        void generateConfigFile(const std::string& filename) const {
            std::ofstream file(filename);
            if (!file.is_open()) {
                throw std::runtime_error("Could not open file to write configuration: " + filename);
            }
            for (const auto& pair : configuration_) {
                file << pair.first << " " << pair.second.getString() << std::endl;
            }
            file.close();
        }

        std::size_t getConfigId() const {
            return HashFunction()(*this);
        }

};

/**
 * @brief Struct representing a record of an evaluated configuration.
 */
struct EvaluationRecord {
    int eval_id;
    int time_evaluated;
    double objective_value;
    std::size_t config_id;
    int worker_id;
    int iteration;
    int phase; // 0 for exploration, 1 for expansion
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

        std::unordered_set<Configuration, Configuration::HashFunction> configurations_;                     ///< Set of all configurations tested
        std::unordered_set<Configuration, Configuration::HashFunction>::const_iterator best_configuration_; ///< Iterator to the best configuration
        std::vector<EvaluationRecord> evaluation_history_;                                                  ///< History of evaluated configurations

        Configuration default_configuration_; ///< Default configuration (unevaluated)
        int next_eval_id_ = 1;                ///< Counter for assigning unique evaluation IDs

        /** @brief Print information about a new best configuration found */
        void printNewBestConfiguration(const Configuration& config) {
            logger_.info("***************");
            logger_.info("New best configuration found at time ", config.getTime(), "s, with objective ", config.getObjective());
            logger_.info("With config: ");
            config.printConfiguration(logger_.getOutputStream());
            logger_.info("***************");
        }
    
    public:
        /** @brief Construct a TunerMemory object */
        TunerMemory(Logger& logger): logger_(logger), best_configuration_(configurations_.end()) {}

        /**
         * @brief Add a configuration to the memory.
         * @param config Configuration to add.
         * @param worker_id ID of the worker that evaluated the configuration (for logging purposes).
         * @throws std::runtime_error if the configuration is not evaluated.
         * @note Only evaluated configurations can be added to the memory.
         * If the configuration already exists and the new one has a better objective, it updates the stored configuration.
         * If it is the best configuration, the best_configuration_ iterator will be updated.
         */
        void addConfiguration(const Configuration& config, int worker_id, int iteration, int phase) {
            EvaluationRecord r = {
                .eval_id = next_eval_id_++,
                .time_evaluated = config.getTime(),
                .objective_value = config.getObjective(),
                .config_id = config.getConfigId(),
                .worker_id = worker_id,
                .iteration = iteration,
                .phase = phase
            };
            evaluation_history_.push_back(r);

            if (!config.isEvaluated()) {
                throw std::runtime_error("Cannot add unevaluated configuration to memory.");
            }

            auto [it, inserted] = configurations_.insert(config);

            // If the configuration already exists but the new one has a better objective, update it
            if (!inserted) {
                if (config.getObjective() < it->getObjective()) {
                    configurations_.erase(it);
                    it = configurations_.insert(config).first;
                }
            }

            // Update best configuration if necessary
            if (best_configuration_ == configurations_.end() || config.getObjective() < best_configuration_->getObjective()) {
                best_configuration_ = it;
                printNewBestConfiguration(*best_configuration_);
            }

            logger_.debug("Added configuration to memory with objective: ", config.getObjective(), ". Total stored: ", configurations_.size());
        }

        /**
         * @brief Add multiple configurations to the memory.
         * @param configs Vector of configurations to add.
         * @param worker_id ID of the worker that evaluated the configurations (for logging purposes).
         * @note This method calls addConfiguration for each configuration in the vector.
         */
        void addConfigurations(const std::vector<Configuration>& configs, int worker_id = -1, int iteration = -1, int phase = -1) {
            for (const auto& config : configs) {
                addConfiguration(config, worker_id, iteration, phase);
            }
            logger_.info("Added ", configs.size(), " configurations to memory. Total stored: ", configurations_.size());
        }

        /** @brief Get the configurations stored */
        const std::unordered_set<Configuration, Configuration::HashFunction>& getConfigurations() const {
            return configurations_;
        }

        /** @brief Get a pointer to the best configuration */
        const Configuration* getBestConfiguration() const {
            if (best_configuration_ == configurations_.end()) {
                return nullptr;
            }
            return &(*best_configuration_);
        }

        /** @brief Get the default configuration (unevaluated) */
        const Configuration& getDefaultConfiguration() const {
            return default_configuration_;
        }

        /** @brief Set the default configuration (unevaluated) */
        void setDefaultConfiguration(const Configuration& config) {
            default_configuration_ = config;
        }

        /** @brief Print the trace of evaluated configs. */
        void exportEvaluationLogCSV(const std::string& filename) const {
            std::ofstream file(filename);
            if (!file.is_open()) {
                throw std::runtime_error("Could not open file to write evaluation log: " + filename);
            }
            // Write header
            file << "EvalID,TimeEvaluated,ObjectiveValue,ConfigID,WorkerID,Iteration,Phase\n";
            // Write records
            for (const auto& record : evaluation_history_) {
                file << record.eval_id << ","
                     << record.time_evaluated << ","
                     << record.objective_value << ","
                     << record.config_id << ","
                     << record.worker_id << ","
                     << record.iteration << ","
                     << record.phase << "\n";
            }
            file.close();
        }

        /** @brief Print all the configs from memory */
        void exportUniqueConfigsCSV(const std::string& filename) const {
            std::ofstream file(filename);
            if (!file.is_open()) {
                throw std::runtime_error("Could not open file to write unique configurations: " + filename);
            }
            // Write header
            file << "ConfigID,Configuration\n";
            // Write unique configurations
            for (const auto& config : configurations_) {
                file << config.getConfigId() << ",";
                for (const auto& pair : config.getConfiguration()) {
                    file << pair.first << "=" << pair.second.getString() << " ";
                }
                file << "\n";
            }
            file.close();
        }
};

#endif // TUNER_MEMORY_H
