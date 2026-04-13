// MPILS: Multi-Phase Iterated Local Search Tuner
//
// Author: Youri Rigaud
// License: GNU GPLv3

#ifndef SOLVER_H
#define SOLVER_H

#include "logger.h"
#include "solver_time_mode.h"
#include "tuning_objective.h"

#include <limits>
#include <optional>
#include <string>

class Solver {
    protected:
        Logger& logger_;
        std::string instance_file_; // Problem instance file
        std::string config_file_path_; // Path to the solver configuration file
        std::string log_file_;      // Log file for solver output
        int nb_threads_;            // Number of threads for the solver
        double cutoff_solver_time_;   // Cutoff time for the solver
        SolverTimeMode solver_time_mode_;
        std::string mip_start_from_file_; // mip start file for the solver (optional)
        std::string produce_mip_start_file_; // mip start file produced by the solver (optional)
        TuningObjective tuning_objective_;

    public:
        Solver(Logger& logger, const std::string& instance_file, const std::string& config_file_path, const std::string& log_file, int nb_threads, double cutoff_solver_time, SolverTimeMode solver_time_mode = SolverTimeMode::Seconds, TuningObjective tuning_objective = TuningObjective::Gap)
            : logger_(logger),
              instance_file_(instance_file),
              config_file_path_(config_file_path),
              log_file_(log_file),
              nb_threads_(nb_threads),
              cutoff_solver_time_(cutoff_solver_time),
              solver_time_mode_(solver_time_mode),
              mip_start_from_file_(""),
              produce_mip_start_file_(""),
              tuning_objective_(tuning_objective)
        {}

        Solver(Logger& logger, const std::string& instance_file, const std::string& config_file_path, const std::string& log_file, int nb_threads, double cutoff_solver_time, SolverTimeMode solver_time_mode, std::string& mip_start_from_file, TuningObjective tuning_objective = TuningObjective::Gap)
            : logger_(logger),
              instance_file_(instance_file),
              config_file_path_(config_file_path),
              log_file_(log_file),
              nb_threads_(nb_threads),
              cutoff_solver_time_(cutoff_solver_time),
              solver_time_mode_(solver_time_mode),
              mip_start_from_file_(mip_start_from_file),
              produce_mip_start_file_(""),
              tuning_objective_(tuning_objective)
        {}

        Solver(Logger& logger, const std::string& instance_file, const std::string& config_file_path, const std::string& log_file, int nb_threads, double cutoff_solver_time, SolverTimeMode solver_time_mode, std::string& mip_start_from_file, std::string& produce_mip_start_file, TuningObjective tuning_objective = TuningObjective::Gap)
            : logger_(logger),
              instance_file_(instance_file),
              config_file_path_(config_file_path),
              log_file_(log_file),
              nb_threads_(nb_threads),
              cutoff_solver_time_(cutoff_solver_time),
              solver_time_mode_(solver_time_mode),
              mip_start_from_file_(mip_start_from_file),
              produce_mip_start_file_(produce_mip_start_file),
              tuning_objective_(tuning_objective)
        {}

        virtual ~Solver() = default; // Virtual destructor
        
        virtual void solve() = 0; // Pure virtual function to solve the problem

        virtual double getObjectiveValue() = 0; // Pure virtual function to get the objective value
        virtual std::optional<double> getGap() = 0; // Pure virtual function to get the raw gap value
        virtual std::optional<double> getUpperBound() = 0; // Pure virtual function to get the best solution value
        virtual std::optional<double> getLowerBound() = 0; // Pure virtual function to get the best bound value
};


class CPLEXSolver : public Solver {
    private:
        double gap_;
        double time_sec_;
        std::optional<double> upper_bound_;
        std::optional<double> lower_bound_;

    public:
        CPLEXSolver(
            Logger& logger,
            const std::string& instance_file,
            const std::string& config_file_path,
            const std::string& log_file,
            int nb_threads,
            double cutoff_solver_time,
            SolverTimeMode solver_time_mode = SolverTimeMode::Seconds,
            TuningObjective tuning_objective = TuningObjective::Gap
        ): Solver(logger, instance_file, config_file_path, log_file, nb_threads, cutoff_solver_time, solver_time_mode, tuning_objective),
           gap_(std::numeric_limits<double>::max()),
           time_sec_(std::numeric_limits<double>::max()),
           upper_bound_(std::nullopt),
           lower_bound_(std::nullopt)
        {}

        CPLEXSolver(
            Logger& logger,
            const std::string& instance_file,
            const std::string& config_file_path,
            const std::string& log_file,
            int nb_threads,
            double cutoff_solver_time,
            SolverTimeMode solver_time_mode,
            std::string& mip_start_from_file,
            TuningObjective tuning_objective = TuningObjective::Gap
        ): Solver(logger, instance_file, config_file_path, log_file, nb_threads, cutoff_solver_time, solver_time_mode, mip_start_from_file, tuning_objective),
           gap_(std::numeric_limits<double>::max()),
           time_sec_(std::numeric_limits<double>::max()),
           upper_bound_(std::nullopt),
           lower_bound_(std::nullopt)
        {}

        CPLEXSolver(
            Logger& logger,
            const std::string& instance_file,
            const std::string& config_file_path,
            const std::string& log_file,
            int nb_threads,
            double cutoff_solver_time,
            SolverTimeMode solver_time_mode,
            std::string& mip_start_from_file,
            std::string& produce_mip_start_file,
            TuningObjective tuning_objective = TuningObjective::Gap
        ): Solver(logger, instance_file, config_file_path, log_file, nb_threads, cutoff_solver_time, solver_time_mode, mip_start_from_file, produce_mip_start_file, tuning_objective),
           gap_(std::numeric_limits<double>::max()),
           time_sec_(std::numeric_limits<double>::max()),
           upper_bound_(std::nullopt),
           lower_bound_(std::nullopt)
        {}

        void solve() override;
        double getObjectiveValue() override;
        std::optional<double> getGap() override;
        std::optional<double> getUpperBound() override;
        std::optional<double> getLowerBound() override;
};


#endif // SOLVER_H
