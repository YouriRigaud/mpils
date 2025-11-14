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

#include <map>
#include <unordered_set>
#include <stdexcept>
#include <memory>

/**
 * @brief Class representing a configuration.
 * 
 * This class stores a map of the parameter names to their values,
 * the objective value if evaluated, and a flag indicating if it has been evaluated.
 * Normally, the configuration is always evaluated pretty much right after being created.
 */
class Configuration {
    private:
        const std::map<std::string, Value> configuration_; ///< Map of parameter names to their values
        double objective_;                                 ///< Objective value if evaluated
        bool evaluated_;                                   ///< Flag indicating if the configuration has been evaluated
    
    public:
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
        ): configuration_(configuration), objective_(objective), evaluated_(true) {}

        /** @brief Get the configuration map */
        std::map<std::string, Value> getConfiguration() const { return configuration_; }

        /** @brief Get the objective value */
        double getObjective() const { return objective_; }

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
            evaluated_ = true;
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
        std::unordered_set<Configuration, Configuration::HashFunction> configurations_;                     ///< Set of all configurations tested
        std::unordered_set<Configuration, Configuration::HashFunction>::const_iterator best_configuration_; ///< Iterator to the best configuration
    
    public:
        /** @brief Construct a TunerMemory object */
        TunerMemory(): best_configuration_(configurations_.end()) {}

        /**
         * @brief Add a configuration to the memory.
         * @param config Configuration to add.
         * @throws std::runtime_error if the configuration is not evaluated.
         * @note Only evaluated configurations can be added to the memory.
         * If the configuration already exists and the new one has a better objective, it updates the stored configuration.
         * If it is the best configuration, the best_configuration_ iterator will be updated.
        */
        void addConfiguration(const Configuration& config) {
            if (!config.isEvaluated()) {
                throw std::runtime_error("Cannot add unevaluated configuration to memory.");
            }

            auto [it, inserted] = configurations_.insert(config);

            // If the configuration already exists but the new one has a better objective, update it
            if (!inserted) {
                if (config.getObjective() < it->getObjective()) {
                    configurations_.erase(it);
                    auto [new_it, _] = configurations_.insert(config);
                    it = new_it;
                }
            }

            // Update best configuration if necessary
            if (best_configuration_ == configurations_.end() || config.getObjective() < best_configuration_->getObjective()) {
                best_configuration_ = it;
            }
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
};

#endif // TUNER_MEMORY_H