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

enum class SolverTerminationStatus {
    Normal,
    WatchdogAbortRequested,
    CplexException,
    UnknownException
};

inline std::string solverTerminationStatusToString(SolverTerminationStatus status) {
    switch (status) {
        case SolverTerminationStatus::Normal:
            return "Normal";
        case SolverTerminationStatus::WatchdogAbortRequested:
            return "WatchdogAbortRequested";
        case SolverTerminationStatus::CplexException:
            return "CplexException";
        case SolverTerminationStatus::UnknownException:
            return "UnknownException";
    }
    return "UnknownException";
}

inline SolverTerminationStatus parseSolverTerminationStatus(const std::string& value) {
    if (value == "Normal") {
        return SolverTerminationStatus::Normal;
    }
    if (value == "WatchdogAbortRequested") {
        return SolverTerminationStatus::WatchdogAbortRequested;
    }
    if (value == "CplexException") {
        return SolverTerminationStatus::CplexException;
    }
    if (value == "UnknownException") {
        return SolverTerminationStatus::UnknownException;
    }
    return SolverTerminationStatus::UnknownException;
}

struct SolverWatchdogOptions {
    bool enabled = true;
    double wall_time_factor = 1.25;
    double wall_time_grace_seconds = 0.0;
};

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
        std::optional<double> mip_start_write_upper_bound_threshold_; // only write produced MIP start if upper bound improves this threshold
        TuningObjective tuning_objective_;
        SolverWatchdogOptions watchdog_options_;

    public:
        Solver(Logger& logger, const std::string& instance_file, const std::string& config_file_path, const std::string& log_file, int nb_threads, double cutoff_solver_time, SolverTimeMode solver_time_mode = SolverTimeMode::Seconds, TuningObjective tuning_objective = TuningObjective::Gap, SolverWatchdogOptions watchdog_options = SolverWatchdogOptions())
            : logger_(logger),
              instance_file_(instance_file),
              config_file_path_(config_file_path),
              log_file_(log_file),
              nb_threads_(nb_threads),
              cutoff_solver_time_(cutoff_solver_time),
              solver_time_mode_(solver_time_mode),
              mip_start_from_file_(""),
              produce_mip_start_file_(""),
              mip_start_write_upper_bound_threshold_(std::nullopt),
              tuning_objective_(tuning_objective),
              watchdog_options_(watchdog_options)
        {}

        Solver(Logger& logger, const std::string& instance_file, const std::string& config_file_path, const std::string& log_file, int nb_threads, double cutoff_solver_time, SolverTimeMode solver_time_mode, std::string& mip_start_from_file, TuningObjective tuning_objective = TuningObjective::Gap, SolverWatchdogOptions watchdog_options = SolverWatchdogOptions())
            : logger_(logger),
              instance_file_(instance_file),
              config_file_path_(config_file_path),
              log_file_(log_file),
              nb_threads_(nb_threads),
              cutoff_solver_time_(cutoff_solver_time),
              solver_time_mode_(solver_time_mode),
              mip_start_from_file_(mip_start_from_file),
              produce_mip_start_file_(""),
              mip_start_write_upper_bound_threshold_(std::nullopt),
              tuning_objective_(tuning_objective),
              watchdog_options_(watchdog_options)
        {}

        Solver(Logger& logger, const std::string& instance_file, const std::string& config_file_path, const std::string& log_file, int nb_threads, double cutoff_solver_time, SolverTimeMode solver_time_mode, std::string& mip_start_from_file, std::string& produce_mip_start_file, TuningObjective tuning_objective = TuningObjective::Gap, std::optional<double> mip_start_write_upper_bound_threshold = std::nullopt, SolverWatchdogOptions watchdog_options = SolverWatchdogOptions())
            : logger_(logger),
              instance_file_(instance_file),
              config_file_path_(config_file_path),
              log_file_(log_file),
              nb_threads_(nb_threads),
              cutoff_solver_time_(cutoff_solver_time),
              solver_time_mode_(solver_time_mode),
              mip_start_from_file_(mip_start_from_file),
              produce_mip_start_file_(produce_mip_start_file),
              mip_start_write_upper_bound_threshold_(mip_start_write_upper_bound_threshold),
              tuning_objective_(tuning_objective),
              watchdog_options_(watchdog_options)
        {}

        virtual ~Solver() = default; // Virtual destructor
        
        virtual void solve() = 0; // Pure virtual function to solve the problem

        virtual double getObjectiveValue() = 0; // Pure virtual function to get the objective value
        virtual std::optional<double> getGap() = 0; // Pure virtual function to get the raw gap value
        virtual std::optional<double> getUpperBound() = 0; // Pure virtual function to get the best solution value
        virtual std::optional<double> getLowerBound() = 0; // Pure virtual function to get the best bound value
        virtual double getSolveTimeSeconds() const = 0; // Pure virtual function to get the solver runtime in seconds
        virtual SolverTerminationStatus getTerminationStatus() const = 0;
};


class CPLEXSolver : public Solver {
    private:
        double gap_;
        double time_sec_;
        std::optional<double> upper_bound_;
        std::optional<double> lower_bound_;
        SolverTerminationStatus termination_status_;

    public:
        CPLEXSolver(
            Logger& logger,
            const std::string& instance_file,
            const std::string& config_file_path,
            const std::string& log_file,
            int nb_threads,
            double cutoff_solver_time,
            SolverTimeMode solver_time_mode = SolverTimeMode::Seconds,
            TuningObjective tuning_objective = TuningObjective::Gap,
            SolverWatchdogOptions watchdog_options = SolverWatchdogOptions()
        ): Solver(logger, instance_file, config_file_path, log_file, nb_threads, cutoff_solver_time, solver_time_mode, tuning_objective, watchdog_options),
           gap_(std::numeric_limits<double>::max()),
           time_sec_(std::numeric_limits<double>::max()),
           upper_bound_(std::nullopt),
           lower_bound_(std::nullopt),
           termination_status_(SolverTerminationStatus::Normal)
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
            TuningObjective tuning_objective = TuningObjective::Gap,
            SolverWatchdogOptions watchdog_options = SolverWatchdogOptions()
        ): Solver(logger, instance_file, config_file_path, log_file, nb_threads, cutoff_solver_time, solver_time_mode, mip_start_from_file, tuning_objective, watchdog_options),
           gap_(std::numeric_limits<double>::max()),
           time_sec_(std::numeric_limits<double>::max()),
           upper_bound_(std::nullopt),
           lower_bound_(std::nullopt),
           termination_status_(SolverTerminationStatus::Normal)
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
            TuningObjective tuning_objective = TuningObjective::Gap,
            std::optional<double> mip_start_write_upper_bound_threshold = std::nullopt,
            SolverWatchdogOptions watchdog_options = SolverWatchdogOptions()
        ): Solver(logger, instance_file, config_file_path, log_file, nb_threads, cutoff_solver_time, solver_time_mode, mip_start_from_file, produce_mip_start_file, tuning_objective, mip_start_write_upper_bound_threshold, watchdog_options),
           gap_(std::numeric_limits<double>::max()),
           time_sec_(std::numeric_limits<double>::max()),
           upper_bound_(std::nullopt),
           lower_bound_(std::nullopt),
           termination_status_(SolverTerminationStatus::Normal)
        {}

        void solve() override;
        double getObjectiveValue() override;
        std::optional<double> getGap() override;
        std::optional<double> getUpperBound() override;
        std::optional<double> getLowerBound() override;
        double getSolveTimeSeconds() const override;
        SolverTerminationStatus getTerminationStatus() const override;
};


#endif // SOLVER_H
