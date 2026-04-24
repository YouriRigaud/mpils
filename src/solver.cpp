// MPILS: Multi-Phase Iterated Local Search Tuner
//
// Author: Youri Rigaud
// License: GNU GPLv3

#include "../include/solver.h"
#include "../include/filesystem_utils.h"

#include <ilcplex/ilocplex.h>


void CPLEXSolver::solve() {
    logger_.info("Starting CPLEX solver on instance: " + instance_file_ + " with config: " + config_file_path_);
    IloEnv env;
    gap_ = std::numeric_limits<double>::max();
    upper_bound_ = std::nullopt;
    lower_bound_ = std::nullopt;
    ensureParentDirectoryForFile(log_file_);
    std::ofstream logStream(log_file_, std::ios::app);
    try {
        IloModel model(env);
        IloCplex cplex(model);
        
        // Set log file
        logStream << "\n===== New CPLEX run =====\n";
        logStream << "Instance: " << instance_file_ << "\n";
        logStream << "Config: " << config_file_path_ << "\n";
        logStream << "========================\n";

        cplex.setOut(logStream);
        cplex.setWarning(logStream);
        cplex.setError(logStream);
    
        // Load model from instance file
        cplex.importModel(model, instance_file_.c_str());
    
        // Set parameters from config file
        cplex.readParam(config_file_path_.c_str());
    
        // Set number of threads
        cplex.setParam(IloCplex::Param::Threads, nb_threads_);
        // Set the selected time-budget mode.
        if (solver_time_mode_ == SolverTimeMode::Ticks) {
            cplex.setParam(IloCplex::Param::DetTimeLimit, cutoff_solver_time_);
        } else {
            cplex.setParam(IloCplex::Param::TimeLimit, cutoff_solver_time_);
        }
       
        // Set mip start
        if (!mip_start_from_file_.empty()) {
            cplex.readMIPStarts(mip_start_from_file_.c_str());
        }
        // Solve the model
        double startTime = cplex.getCplexTime();
        cplex.solve();
        double endTime = cplex.getCplexTime();
        time_sec_ = endTime - startTime;

        if (cplex.isPrimalFeasible()) {
            upper_bound_ = cplex.getObjValue();
            gap_ = cplex.getMIPRelativeGap() * 100.0; // Convert to percentage
            gap_ = std::round(gap_ * 100.0) / 100.0; // Round to 1e-2
        } else {
            logger_.warn("CPLEX finished without a primal feasible solution for config ", config_file_path_, ".");
        }
        try {
            lower_bound_ = cplex.getBestObjValue();
        } catch (IloException&) {
            lower_bound_ = std::nullopt;
        }

        const std::string upper_bound_text = upper_bound_.has_value() ? std::to_string(upper_bound_.value()) : "null";
        const std::string lower_bound_text = lower_bound_.has_value() ? std::to_string(lower_bound_.value()) : "null";
        const std::string gap_text = gap_ == std::numeric_limits<double>::max() ? "unavailable" : std::to_string(gap_);

        logger_.info("CPLEX results. Gap: ", gap_text, ", upper bound: ", upper_bound_text, ", lower bound: ", lower_bound_text);
        if (upper_bound_.has_value() && lower_bound_.has_value() && upper_bound_.value() < lower_bound_.value()) {
            logger_.warn("Inconsistent solver bounds for config ", config_file_path_,
                         ": upper bound ", upper_bound_.value(),
                         " is smaller than lower bound ", lower_bound_.value(), ".");
        }
        
        // Log the gap in the cplex log file
        if (gap_ == std::numeric_limits<double>::max()) {
            logStream << "Gap: unavailable\n";
        } else {
            logStream << "Gap: " << gap_ << "%\n";
        }
        if (upper_bound_.has_value()) {
            logStream << "Upper bound: " << upper_bound_.value() << "\n";
        }
        if (lower_bound_.has_value()) {
            logStream << "Lower bound: " << lower_bound_.value() << "\n";
        }
        logStream << "End of CPLEX run. Time: " << time_sec_ << " seconds\n";
        logStream.flush();

        // write mip start file when this solve found a feasible incumbent that improves the requested threshold
        const bool should_write_mip_start =
            !produce_mip_start_file_.empty() &&
            upper_bound_.has_value() &&
            (!mip_start_write_upper_bound_threshold_.has_value() ||
             upper_bound_.value() < mip_start_write_upper_bound_threshold_.value());
        if (should_write_mip_start) {
            ensureParentDirectoryForFile(produce_mip_start_file_);
            cplex.writeMIPStarts(produce_mip_start_file_.c_str());
            logger_.info("Writing MIP start file: " + produce_mip_start_file_);
        }
    } catch (IloException& e) {
        logStream << "CPLEX Exception: " << e << "\n";
        logStream << "End of CPLEX run due to exception.\n";
        logStream.flush();
        env.error() << "CPLEX Exception: " << e << std::endl;
    } catch (...) {
        logStream << "Unknown exception caught.\n";
        logStream << "End of CPLEX run due to exception.\n";
        logStream.flush();
        env.error() << "Unknown exception caught." << std::endl;
    }
    logStream.close();
    env.end();
    logger_.info("CPLEX solver finished. Gap: " + std::to_string(gap_));
}

double CPLEXSolver::getObjectiveValue() {
    switch (tuning_objective_) {
        case TuningObjective::Gap:
            return gap_;
        case TuningObjective::UpperBound:
            return upper_bound_.value_or(std::numeric_limits<double>::max());
    }

    return gap_;
}

std::optional<double> CPLEXSolver::getGap() {
    if (gap_ == std::numeric_limits<double>::max()) {
        return std::nullopt;
    }
    return gap_;
}

std::optional<double> CPLEXSolver::getUpperBound() {
    return upper_bound_;
}

std::optional<double> CPLEXSolver::getLowerBound() {
    return lower_bound_;
}
