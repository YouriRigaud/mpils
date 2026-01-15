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

int main() {
    std::cout << "Welcome to the MPILS tuner!" << std::endl;
    
    std::ofstream log_file("./tuner_working_dir/tuner.log");
    Tuner tuner(
        Verbosity::Debug,
        log_file,
        "./tuner_working_dir/",
        "./cplex/params_12_cpx.txt",
        "./cplex/30n20b8.mps",
        "./cplex/instances.txt",
        "./tuner_working_dir/solver/cplex.log",
        10,      // Number of initial selected parameters
        2,      // Number of threads for the solver
        2.0   // Cutoff time for the solver
    );
    
    tuner.setup();

    tuner.run();

    tuner.getBestConfiguration().printConfiguration(std::cout);

    tuner.getBestConfiguration().generateConfigFile("./tuner_working_dir/best_configuration.prm");
    
    log_file.close();

    return 0;
}
