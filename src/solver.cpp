// MPILS: Multi-Phase Iterated Local Search Tuner
//
// Author: Youri Rigaud
// License: GNU GPLv3

#include "../include/solver.h"

#include <ilcplex/ilocplex.h>


void CPLEXSolver::solve() {
    logger_.info("Starting CPLEX solver on instance: " + instance_file_ + " with config: " + config_file_path_);
    IloEnv env;
    try {
        IloModel model(env);
        IloCplex cplex(model);
        // Load model from instance file
        cplex.importModel(model, instance_file_.c_str());
        // Set number of threads
        //cplex.setParam(IloCplex::Param::Threads, nb_threads_);
        
        // Set log file
        std::ofstream logStream(log_file_, std::ios::app);
        
        logStream << "\n===== New CPLEX run =====\n";
        logStream << "Instance: " << instance_file_ << "\n";
        logStream << "Config: " << config_file_path_ << "\n";
        logStream << "========================\n";

        cplex.setOut(logStream);
        cplex.setWarning(logStream);
        cplex.setError(logStream);
        

        // Set parameters from config file
        cplex.readParam(config_file_path_.c_str());
        // Set time limit
        cplex.setParam(IloCplex::Param::TimeLimit, cutoff_solver_time_);
        // Set mip start
        //if (!mip_start_file_.empty()) {
        //    cplex.readMIPStart(mip_start_file_.c_str());
        //}
        // Solve the model
        cplex.solve();
        // Get gap and time
        gap_ = cplex.getMIPRelativeGap() * 100.0; // Convert to percentage
        // change number of floating point precision to 1e-2
        gap_ = std::round(gap_ * 100.0) / 100.0;
        time_sec_ = cplex.getTime();
        // write mip start file
        if (!mip_start_from_file_.empty()) {
            cplex.writeMIPStarts(mip_start_from_file_.c_str());
            logger_.info("Writing MIP start file: " + mip_start_from_file_);
        }
    } catch (IloException& e) {
        env.error() << "CPLEX Exception: " << e << std::endl;
    } catch (...) {
        env.error() << "Unknown exception caught." << std::endl;
    }
    env.end();
    logger_.info("CPLEX solver finished. Gap: " + std::to_string(gap_));
}

double CPLEXSolver::getObjectiveValue() {
    return gap_; // Return gap as objective value
}
