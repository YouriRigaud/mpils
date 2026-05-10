// MPILS: Multi-Phase Iterated Local Search Tuner
//
// Author: Youri Rigaud
// License: GNU GPLv3

#ifndef TUNER_H
#define TUNER_H

#include "tuning_objective.h"
#include "solver_time_mode.h"
#include "local_search_backend.h"
#include "logger.h"
#include "tuner_memory.h"
#include "exploration.h"
#include "parameter_space.h"
#include "expansion.h"
#include "pruning.h"

#include <vector>
#include <string>
#include <fstream>
#include <cstdint>
#include <limits>
#include <optional>

class Tuner {
    private:
        int iteration_ = 1;                          ///< Current iteration of the tuning process

        Logger logger_;

        const std::string tuner_dir_;              ///< Directory where the tuner stores its files
        const std::string parameters_file_;        ///< File containing the list of parameters
        const std::string instance_file_;        ///< File containing the problem instance
        const std::string param_ils_instance_file_;          ///< File containing the problem instance
        const std::string solver_log_file_;        ///< File for solver logs
        const int nb_initial_selected_parameters_; ///< Number of parameters to select initially
        const int nb_threads_solver_;              ///< Number of threads for the solver
        const double cutoff_solver_time_;          ///< Cutoff time for each solver run
        const SolverTimeMode solver_time_mode_;    ///< Whether solver time budget is in seconds or deterministic ticks
        const SolverWatchdogOptions solver_watchdog_options_; ///< Cooperative wall-clock watchdog options for solver runs
        const int nb_workers_;                     ///< Number of worker processes for parallel execution
        const bool use_shared_cache_;              ///< Whether ILS workers should share cached objectives
        const bool exploration_only_;             ///< Whether to stop after one exploration phase
        const LocalSearchBackend local_search_backend_; ///< Local search backend to use during exploration
        const std::uint32_t base_seed_;           ///< Base seed for exploration-local-search reproducibility
        const TuningObjective tuning_objective_; ///< Objective used by direct CPLEX paths
        const int max_iterations_;               ///< Maximum number of tuner iterations
        const bool enable_mip_starts_;           ///< Whether exploration can use MIP starts
        const bool random_worker_initial_configs_; ///< Whether MPI ILS workers use per-rank initial configs
        const MipStartInitialConfigPolicy mip_start_initial_config_policy_; ///< Initial configuration policy for the MIP-start worker
        const ExpansionSelectRule expansion_select_rule_; ///< Comparison rule used by expansion selection
        const ExpansionValueStrategy expansion_value_strategy_; ///< Value selection strategy used by expansion
        const double expansion_max_deviation_;   ///< Maximum allowed deviation for expansion classification
        const bool expansion_enable_early_stop_; ///< Whether expansion may stop early on improvement
        const std::optional<double> paramils_wall_time_; ///< Optional wall-clock time budget for ParamILS
        TunerMemory memory_;                       ///< Memory to store configurations tested
        ParameterSpace parameter_space_;           ///< Parameter space
        Exploration exploration_;                  ///< Exploration component
        Expansion expansion_;                      ///< Expansion component
        Pruning pruning_;                        ///< Pruning component

        /** @brief Print the list of parameters for debugging */
        void printParameters(const std::vector<Parameter>& parameters);
        /** @brief Read and return parameters from the file */
        std::vector<Parameter> getParameters();
        /** @brief Set all parameters vectors (tuned, residual, selected, discarded) */
        void setAllParametersFlags();
        /** @brief Set the default configuration in tuner memory based on default parameter values */
        void setDefaultConfiguration();
        /** @brief Create the working directories needed for a fresh tuner run */
        void createWorkingDirectories();

        bool stopConditionMet(); // Check if stopping condition is met

        void writeParametersIdToFile(const Configuration& config, const std::string& filepath); // Write parameter IDs of a configuration to a file

#ifdef USE_MPI
        void sendStopOrderToWorkers();
#endif

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
            int nb_parameter_to_evaluate_expansion,
            int nb_threads_solver,
            double cutoff_solver_time,
            SolverTimeMode solver_time_mode,
            SolverWatchdogOptions solver_watchdog_options,
            int nb_workers,
            bool use_shared_cache,
            bool exploration_only,
            LocalSearchBackend local_search_backend,
            std::uint32_t base_seed,
            TuningObjective tuning_objective,
            std::optional<int> number_of_evaluations = std::nullopt,
            std::optional<int> exploration_budget_divisor = std::nullopt,
            int max_iterations = 15,
            bool enable_mip_starts = true,
            bool random_worker_initial_configs = true,
            MipStartInitialConfigPolicy mip_start_initial_config_policy = MipStartInitialConfigPolicy::ProducerConfig,
            ExpansionSelectRule expansion_select_rule = ExpansionSelectRule::Strict,
            ExpansionValueStrategy expansion_value_strategy = ExpansionValueStrategy::FirstLast,
            double expansion_max_deviation = std::numeric_limits<double>::max(),
            bool expansion_enable_early_stop = true,
            std::optional<double> paramils_wall_time = std::nullopt
        ):  logger_(level, out),
            tuner_dir_(tuner_dir),
            parameters_file_(parameters_file),
            instance_file_(instance_file),
            param_ils_instance_file_(param_ils_instance_file),
            solver_log_file_(solver_log_file),
            nb_initial_selected_parameters_(nb_initial_selected_parameters),
            nb_threads_solver_(nb_threads_solver),
            cutoff_solver_time_(cutoff_solver_time),
            solver_time_mode_(solver_time_mode),
            solver_watchdog_options_(solver_watchdog_options),
            nb_workers_(nb_workers),
            use_shared_cache_(use_shared_cache),
            exploration_only_(exploration_only),
            local_search_backend_(local_search_backend),
            base_seed_(base_seed),
            tuning_objective_(tuning_objective),
            max_iterations_(max_iterations),
            enable_mip_starts_(enable_mip_starts),
            random_worker_initial_configs_(random_worker_initial_configs),
            mip_start_initial_config_policy_(mip_start_initial_config_policy),
            expansion_select_rule_(expansion_select_rule),
            expansion_value_strategy_(expansion_value_strategy),
            expansion_max_deviation_(expansion_max_deviation),
            expansion_enable_early_stop_(expansion_enable_early_stop),
            paramils_wall_time_(paramils_wall_time),
            memory_(TunerMemory(logger_)),
            parameter_space_(ParameterSpace(getParameters())),
            exploration_(memory_, parameter_space_, logger_, iteration_, instance_file_, param_ils_instance_file_, solver_log_file_, nb_threads_solver_, cutoff_solver_time_, solver_time_mode_, solver_watchdog_options_, nb_workers_, use_shared_cache_, local_search_backend_, base_seed_, tuning_objective_, number_of_evaluations, exploration_budget_divisor, enable_mip_starts_, random_worker_initial_configs_, mip_start_initial_config_policy_, paramils_wall_time_),
            expansion_(logger_, memory_, parameter_space_, tuner_dir_, instance_file_, solver_log_file_, iteration_, nb_parameter_to_evaluate_expansion, nb_threads_solver_, cutoff_solver_time_, solver_time_mode_, solver_watchdog_options_, tuning_objective_, expansion_select_rule_, expansion_value_strategy_, expansion_max_deviation_, expansion_enable_early_stop_),
            pruning_(logger_, memory_, parameter_space_, iteration_)
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

        /** @brief Get the objective value of the best evaluation */
        double getBestObjective() const {
            return memory_.getBestObjective();
        }

        /** @brief Write the history of evaluated configurations to some files for analysis */
        void writeConfigurationsHistoryToFiles(const std::string& filename) const {
            memory_.exportEvaluationLogCSV(filename + "_evaluation_log.csv");
            memory_.exportUniqueConfigsCSV(filename + "_unique_configs.csv");
            memory_.exportUniqueMipStartsCSV(filename + "_unique_mip_starts.csv");
        }
};

#ifdef USE_MPI
struct WorkerOrder {
    int step;
    int iteration;
    int nb_evaluations; // Number of evaluations to perform in the local search phase, relevant for step 1 (exploration)
    double expansion_best_objective_value;
    int expansion_enable_early_stop;
};

class Worker {
    private:
        int worker_id_;
        int worker_step_;                     ///< Current step of the worker (0: waiting order, 1: exploration, 2: expansion, 3: finished)
        int iteration_;                        ///< Current iteration of the tuning process
        std::ofstream worker_log_file_;
        Logger worker_logger_;
        std::string instance_file_;
        std::string solver_log_file_;
        int nb_threads_solver_;
        double cutoff_solver_time_;
        SolverTimeMode solver_time_mode_;
        SolverWatchdogOptions solver_watchdog_options_;
        bool use_shared_cache_;
        LocalSearchBackend local_search_backend_;
        std::uint32_t base_seed_;
        TuningObjective tuning_objective_;
        bool enable_mip_starts_;
        bool random_worker_initial_configs_;
        int nb_evaluations_ = 0;  ///< Number of evaluations to perform in the local search phase
        double expansion_best_objective_value_ = 0.0;
        bool expansion_enable_early_stop_ = false;

        std::unique_ptr<LocalSearchWorker> local_search_worker_ = nullptr;
        std::unique_ptr<ExpansionWorker> expansion_worker_ = nullptr;

        static std::ofstream openWorkerLogFile(int worker_id);

        void setLocalSearchWorker(std::unique_ptr<LocalSearchWorker> worker) {
            local_search_worker_ = std::move(worker);
        }

        void setExpansionWorker(std::unique_ptr<ExpansionWorker> worker) {
            expansion_worker_ = std::move(worker);
        }

        bool stopConditionMet() {
            return worker_step_ == 3;
        }

        void receiveOrderFromMaster();

        void runExplorationPhase();

        void runExpansionPhase();

    public:
        Worker(int worker_id, const std::string& instance_file, const std::string& solver_log_file, int nb_threads_solver, double cutoff_solver_time, SolverTimeMode solver_time_mode, SolverWatchdogOptions solver_watchdog_options, bool use_shared_cache, LocalSearchBackend local_search_backend, std::uint32_t base_seed, TuningObjective tuning_objective, bool enable_mip_starts, bool random_worker_initial_configs)
            : worker_id_(worker_id),
              worker_step_(0),
              iteration_(1),
              worker_log_file_(openWorkerLogFile(worker_id)),
              worker_logger_(Verbosity::Debug, worker_log_file_),
              instance_file_(instance_file),
              solver_log_file_(solver_log_file),
              nb_threads_solver_(nb_threads_solver),
              cutoff_solver_time_(cutoff_solver_time),
              solver_time_mode_(solver_time_mode),
              solver_watchdog_options_(solver_watchdog_options),
              use_shared_cache_(use_shared_cache),
              local_search_backend_(local_search_backend),
              base_seed_(base_seed),
              tuning_objective_(tuning_objective),
              enable_mip_starts_(enable_mip_starts),
              random_worker_initial_configs_(random_worker_initial_configs)
        {}

        void run(); // Run the worker process
};
#endif // USE_MPI

#endif // TUNER_H
