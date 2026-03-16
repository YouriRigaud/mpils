// MPILS: Multi-Phase Iterated Local Search Tuner
//
// Author: Youri Rigaud
// License: GNU GPLv3

#include "../include/tuner.h"
#include "../include/tuner_memory.h"
#include "../include/globaltimer.h"
#include "../include/tuning_objective.h"

#ifdef USE_MPI
#include <mpi.h>
#endif

#include <cstring>
#include <stdexcept>
#include <iostream>
#include <string>
#include <fstream>

// Use this struct to tune the options of the tuner
struct TunerOptions {
    std::string tuner_dir = "./tuner_working_dir/";
    std::string parameters_file = "./cplex/params_12_cpx.txt";
    std::string instance_file = "./cplex/30n20b8.mps";
    std::string param_ils_instance_file = "./cplex/instances.txt";
    std::string solver_log_file = "./tuner_working_dir/solver/cplex.log";
    int nb_initial_selected_parameters = 12;
    int nb_parameter_to_evaluate_expansion = 20;
    int nb_threads_solver = 2;
    double cutoff_solver_time = 15.0;
    int nb_workers = 1;
    bool use_shared_cache = false;
    TuningObjective tuning_objective = TuningObjective::Gap;
};

void getTunerOptions(int argc, char** argv, TunerOptions& options) {
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--shared-cache") == 0) {
            options.use_shared_cache = true;
        } else if (std::strcmp(argv[i], "--no-shared-cache") == 0) {
            options.use_shared_cache = false;
        } else if (std::strcmp(argv[i], "--solver-threads") == 0) {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for --solver-threads");
            }
            options.nb_threads_solver = std::stoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--solver-time") == 0) {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for --solver-time");
            }
            options.cutoff_solver_time = std::stod(argv[++i]);
        } else if (std::strcmp(argv[i], "--tuning-objective") == 0) {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for --tuning-objective");
            }
            options.tuning_objective = parseTuningObjective(argv[++i]);
        } else if (argv[i][0] != '-') {
            options.instance_file = std::string(argv[i]);
        } else {
            throw std::runtime_error("Unknown command line option: " + std::string(argv[i]));
        }
    }
}

void writeParamILSInstanceFile(const std::string& filepath, const std::string& instance_file) {
    std::ofstream myfile;
    myfile.open(filepath);
    myfile << instance_file << std::endl;
    myfile.close();
}

void masterProcess(int argc, char** argv, TunerOptions options) {
    std::cout << "Welcome to the MPILS tuner!" << std::endl;
    std::cout << "Tuning instance: " << options.instance_file << std::endl;
    std::cout << "ILS shared cache: " << (options.use_shared_cache ? "enabled" : "disabled") << std::endl;
    std::cout << "Tuning objective: " << tuningObjectiveToString(options.tuning_objective) << std::endl;

    // init a clock to measure tuning time
    GlobalTimer::start();

    writeParamILSInstanceFile(options.param_ils_instance_file, options.instance_file);

    std::string log_file_path = options.tuner_dir + "tuner.log";
    std::ofstream log_file(log_file_path);
    Tuner tuner(
        Verbosity::Debug,
        log_file,
        options.tuner_dir,
        options.parameters_file,
        options.instance_file,
        options.param_ils_instance_file,
        options.solver_log_file,
        options.nb_initial_selected_parameters,
        options.nb_parameter_to_evaluate_expansion,
        options.nb_threads_solver,
        options.cutoff_solver_time,
        options.nb_workers,
        options.use_shared_cache,
        options.tuning_objective
    );
    
    tuner.setup();

    tuner.run();

    std::cout << "Best configuration found:" << std::endl;
    std::cout << "Objective: " << tuner.getBestObjective() << std::endl;

    tuner.getBestConfiguration().printConfiguration(std::cout);
    log_file << "Objective: " << tuner.getBestObjective() << std::endl;

    tuner.getBestConfiguration().generateConfigFile(options.tuner_dir + "best_configuration.prm");
    
    std::cout << "Total tuning time: " << GlobalTimer::elapsedSeconds() << " seconds." << std::endl;

    log_file << "Total tuning time: " << GlobalTimer::elapsedSeconds() << " seconds." << std::endl;

    tuner.writeConfigurationsHistoryToFiles(options.tuner_dir + "tuner_history");

    log_file.close();
}

#ifdef USE_MPI
void workerProcess(int argc, char** argv, int world_rank, TunerOptions options) {
    // init a clock to measure total tuning time
    GlobalTimer::start();
    std::cout << "Worker process " << world_rank << " started." << std::endl;
    Worker worker(world_rank, options.instance_file, options.solver_log_file, options.nb_threads_solver, options.cutoff_solver_time, options.use_shared_cache, options.tuning_objective);
    worker.run();
    std::cout << "Worker process " << world_rank << " finished." << std::endl;
}
#endif

int main(int argc, char** argv) {

    TunerOptions options;
    getTunerOptions(argc, argv, options);

#ifdef USE_MPI
    MPI_Init(&argc, &argv);
    int world_rank;
    int world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    options.nb_workers = world_size;
    if (world_rank == 0) {
        masterProcess(argc, argv, options);
    } else {
        workerProcess(argc, argv, world_rank, options);
    }
    MPI_Finalize();
#else
    masterProcess(argc, argv, options);
#endif

    return 0;
}
