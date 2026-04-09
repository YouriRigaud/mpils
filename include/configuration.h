/**
 * @file configuration.h
 * @brief Represents a configuration of parameters for the solver.
 *
 * The Configuration class encapsulates a specific set of parameter values for the solver, along with a flag indicating whether to use MIP start or not.
 *
 * @author Youri Rigaud
 * @copyright Copyright 2026 Youri Rigaud. All rights reserved.
 *            This software is licensed under the GNU General Public License v3.0.
 *            See the accompanying LICENSE file for full details.
 */

#ifndef CONFIGURATION_H
#define CONFIGURATION_H

#include "parameter_space.h"
#include "filesystem_utils.h"

#include <map>
#include <string>
#include <fstream>
#include <cstdint>

using ConfigurationId = uint64_t;

/**
 * @brief Class representing a configuration.
 * 
 * This class stores a map of the parameter names to their values, and a flag indicating if the configuration use mip start.
 * @note The use_mip_start_ flag is also used to distinguish configurations. (See operator== and HashFunction)
 */
class Configuration {
    private:
        std::map<std::string, Value> configuration_; ///< Map of parameter names to their values
        bool use_mip_start_;                         ///< Flag indicating if this configuration use mip start
    
    public:
        /**
         * @brief Default constructor for Configuration.
         */
        Configuration() : use_mip_start_(false) {}

        /**
         * @brief Construct a Configuration object.
         * @param configuration Map of parameter names to their values.
         * @param use_mip_start Flag indicating if this configuration use mip start or not.
         */
        Configuration(
            std::map<std::string, Value> configuration,
            bool use_mip_start = false
        ): configuration_(configuration), use_mip_start_(use_mip_start) {}

        /** @brief Get the configuration map */
        const std::map<std::string, Value>& getConfigurationMap() const { return configuration_; }

        /** @brief Check if the configuration use mip start */
        bool useMipStart() const { return use_mip_start_; }

        /** @brief Equality operator based on configuration map amd mip start flag */
        bool operator==(const Configuration& other) const {
            return configuration_ == other.configuration_ && use_mip_start_ == other.use_mip_start_;
        }

        /** @brief Hash function for Configuration to be used in unordered_set */
        struct HashFunction {
            uint64_t operator()(const Configuration& config) const {
                uint64_t seed = 0;
                for (const auto& pair : config.getConfigurationMap()) {
                    seed ^= std::hash<std::string>()(pair.first) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
                    // Simple hash for Value based on its string representation
                    seed ^= std::hash<std::string>()(pair.second.getString()) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
                }
                // Include mip start flag in hash
                seed ^= std::hash<bool>()(config.useMipStart()) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
                return seed;
            }
        };

        void printConfiguration(std::ostream& out) const {
            out << "Configuration: ";
            for (const auto& pair : configuration_) {
                out << pair.first << "=" << pair.second.getString() << " ";
            }
            if (use_mip_start_) {
                out << "| MIP Start: Yes ";
            } else {
                out << "| MIP Start: No ";
            }
            out << std::endl;
        }

        void generateConfigFile(const std::string& filename) const {
            ensureParentDirectoryForFile(filename);
            std::ofstream file(filename);
            if (!file.is_open()) {
                throw std::runtime_error("Could not open file to write configuration: " + filename);
            }
            for (const auto& pair : configuration_) {
                file << pair.first << " " << pair.second.getString() << std::endl;
            }
            file.close();
        }

        ConfigurationId getConfigurationId() const {
            return HashFunction()(*this);
        }

};

#endif // CONFIGURATION_H
