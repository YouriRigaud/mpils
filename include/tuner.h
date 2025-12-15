// MPILS: Multi-Phase Iterated Local Search Tuner
//
// Author: Youri Rigaud
// License: GNU GPLv3

#ifndef TUNER_H
#define TUNER_H

#include "parameter.h"
#include "logger.h"
#include "tuner_memory.h"
#include "exploration.h"
#include "parameter_space.h"
#include "expansion.h"

#include <vector>
#include <string>
#include <fstream>

class Tuner {
    private:
        Logger logger_;

        const std::string tuner_dir_;              ///< Directory where the tuner stores its files
        const std::string parameters_file_;        ///< File containing the list of parameters
        const std::string instance_file_;        ///< File containing the problem instance
        const std::string param_ils_instance_file_;          ///< File containing the problem instance
        const std::string solver_log_file_;        ///< File for solver logs
        const int nb_initial_selected_parameters_; ///< Number of parameters to select initially
        const int nb_threads_solver_;              ///< Number of threads for the solver
        const double cutoff_solver_time_;          ///< Cutoff time for each solver run
        TunerMemory memory_;                       ///< Memory to store configurations tested
        ParameterSpace parameter_space_;           ///< Parameter space
        Exploration exploration_;                  ///< Exploration component
        Expansion expansion_;                      ///< Expansion component

        int iteration_ = 1;                          ///< Current iteration of the tuning process

        /** @brief Print the list of parameters for debugging */
        void printParameters(const std::vector<Parameter>& parameters);
        /** @brief Read and return parameters from the file */
        std::vector<Parameter> getParameters();
        /** @brief Set all parameters vectors (tuned, residual, selected, discarded) */
        void setAllParametersFlags();
        /** @brief Set the default configuration in tuner memory based on default parameter values */
        void setDefaultConfiguration();

        bool stopConditionMet(); // Check if stopping condition is met

    public:
        Tuner(
            Verbosity level,
            std::ostream& out,
            const std::string& tuner_dir,
            const std::string& parameters_file,
            const std::string& instance_file,
            const std::string& param_ils_instance_file,
            const std::string& solver_log_file,
            int nb_initial_selected_parameters,
            int nb_threads_solver,
            double cutoff_solver_time
        ):  logger_(level, out),
            tuner_dir_(tuner_dir),
            parameters_file_(parameters_file),
            instance_file_(instance_file),
            param_ils_instance_file_(param_ils_instance_file),
            solver_log_file_(solver_log_file),
            nb_initial_selected_parameters_(nb_initial_selected_parameters),
            nb_threads_solver_(nb_threads_solver),
            cutoff_solver_time_(cutoff_solver_time),
            memory_(TunerMemory(logger_)),
            parameter_space_(ParameterSpace(getParameters())),
            exploration_(memory_, parameter_space_, logger_, iteration_, param_ils_instance_file_, solver_log_file_, nb_threads_solver_, cutoff_solver_time_),
            expansion_(logger_, memory_, parameter_space_, instance_file_, solver_log_file_, iteration_, 5, nb_threads_solver_, cutoff_solver_time_) // Evaluation budget set to 10 as placeholder
        {}

        void setup(); // Setup the tuner

        void run(); // Run the tuning process

        /** @brief Get the best configuration found */
        const Configuration& getBestConfiguration() const {
            const Configuration* best_config = memory_.getBestConfiguration();
            if (best_config == nullptr) {
                throw std::runtime_error("No best configuration found in memory.");
            }
            return *best_config;
        }

};

#endif // TUNER_H