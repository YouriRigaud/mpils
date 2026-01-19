// MPILS: Multi-Phase Iterated Local Search Tuner
//
// Author: Youri Rigaud
// License: GNU GPLv3

#include "../include/parameter.h"
#include "../include/tuner.h"
#include "../include/tuner_memory.h"


#include <iostream>
#include <string>
#include <fstream>

int main(int argc, char** argv) {
    // If no arguments, use default instance file
    std::string instance_file = "./cplex/30n20b8.mps";
    // If argument provided, use it as instance file
    if (argc > 1) {
        instance_file = std::string(argv[1]);
    }

    std::cout << "Welcome to the MPILS tuner!" << std::endl;
    std::cout << "Tuning instance: " << instance_file << std::endl;

    // init a clock to measure total tuning time
    auto start_time = std::chrono::high_resolution_clock::now();

    std::ofstream log_file("./tuner_working_dir/tuner.log");
    Tuner tuner(
        Verbosity::Debug,
        log_file,
        "./tuner_working_dir/",
        "./cplex/params_12_cpx.txt",
        instance_file,
        "./cplex/instances.txt",
        "./tuner_working_dir/solver/cplex.log",
        10,      // Number of initial selected parameters
        2,      // Number of threads for the solver
        30.0   // Cutoff time for the solver
    );
    
    tuner.setup();

    tuner.run();

    // Calculate and print total tuning time
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();

    std::cout << "Best configuration found:" << std::endl;
    std::cout << "Objective: " << tuner.getBestConfiguration().getObjective() << std::endl;

    tuner.getBestConfiguration().printConfiguration(std::cout);
    log_file << "Objective: " << tuner.getBestConfiguration().getObjective() << std::endl;

    tuner.getBestConfiguration().generateConfigFile("./tuner_working_dir/best_configuration.prm");
    
    std::cout << "Total tuning time: " << duration << " seconds." << std::endl;

    log_file << "Total tuning time: " << duration << " seconds." << std::endl;

    log_file.close();

    return 0;
}
