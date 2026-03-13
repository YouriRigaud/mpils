// MPILS: Multi-Phase Iterated Local Search Tuner
//
// Author: Youri Rigaud
// License: GNU GPLv3

#ifndef SOLVER_H
#define SOLVER_H

#include "logger.h"

#include <string>

class Solver {
    protected:
        Logger& logger_;
        std::string instance_file_; // Problem instance file
        std::string config_file_path_; // Path to the solver configuration file
        std::string log_file_;      // Log file for solver output
        int nb_threads_;            // Number of threads for the solver
        double cutoff_solver_time_;   // Cutoff time for the solver
        std::string mip_start_from_file_; // mip start file for the solver (optional)
        std::string produce_mip_start_file_; // mip start file produced by the solver (optional)

    public:
        Solver(Logger& logger, const std::string& instance_file, const std::string& config_file_path, const std::string& log_file, int nb_threads, double cutoff_solver_time)
            : logger_(logger),
              instance_file_(instance_file),
              config_file_path_(config_file_path),
              log_file_(log_file),
              nb_threads_(nb_threads),
              cutoff_solver_time_(cutoff_solver_time),
              mip_start_from_file_(""),
              produce_mip_start_file_("")
        {}

        Solver(Logger& logger, const std::string& instance_file, const std::string& config_file_path, const std::string& log_file, int nb_threads, double cutoff_solver_time, std::string& mip_start_from_file)
            : logger_(logger),
              instance_file_(instance_file),
              config_file_path_(config_file_path),
              log_file_(log_file),
              nb_threads_(nb_threads),
              cutoff_solver_time_(cutoff_solver_time),
              mip_start_from_file_(mip_start_from_file),
              produce_mip_start_file_("")
        {}

        Solver(Logger& logger, const std::string& instance_file, const std::string& config_file_path, const std::string& log_file, int nb_threads, double cutoff_solver_time, std::string& mip_start_from_file, std::string& produce_mip_start_file)
            : logger_(logger),
              instance_file_(instance_file),
              config_file_path_(config_file_path),
              log_file_(log_file),
              nb_threads_(nb_threads),
              cutoff_solver_time_(cutoff_solver_time),
              mip_start_from_file_(mip_start_from_file),
              produce_mip_start_file_(produce_mip_start_file)
        {}

        virtual ~Solver() = default; // Virtual destructor
        
        virtual void solve() = 0; // Pure virtual function to solve the problem

        virtual double getObjectiveValue() = 0; // Pure virtual function to get the objective value
};


class CPLEXSolver : public Solver {
    private:
        double gap_;
        double time_sec_;

    public:
        CPLEXSolver(
            Logger& logger,
            const std::string& instance_file,
            const std::string& config_file_path,
            const std::string& log_file,
            int nb_threads,
            double cutoff_solver_time        
        ): Solver(logger, instance_file, config_file_path, log_file, nb_threads, cutoff_solver_time),
           gap_(std::numeric_limits<double>::max()),
           time_sec_(std::numeric_limits<double>::max())
        {}

        CPLEXSolver(
            Logger& logger,
            const std::string& instance_file,
            const std::string& config_file_path,
            const std::string& log_file,
            int nb_threads,
            double cutoff_solver_time,
            std::string& mip_start_from_file
        ): Solver(logger, instance_file, config_file_path, log_file, nb_threads, cutoff_solver_time, mip_start_from_file),
           gap_(std::numeric_limits<double>::max()),
           time_sec_(std::numeric_limits<double>::max())
        {}

        CPLEXSolver(
            Logger& logger,
            const std::string& instance_file,
            const std::string& config_file_path,
            const std::string& log_file,
            int nb_threads,
            double cutoff_solver_time,
            std::string& mip_start_from_file,
            std::string& produce_mip_start_file
        ): Solver(logger, instance_file, config_file_path, log_file, nb_threads, cutoff_solver_time, mip_start_from_file, produce_mip_start_file),
           gap_(std::numeric_limits<double>::max()),
           time_sec_(std::numeric_limits<double>::max())
        {}

        void solve() override;
        double getObjectiveValue() override;
};


#endif // SOLVER_H