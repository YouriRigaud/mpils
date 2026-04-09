// MPILS: Multi-Phase Iterated Local Search Tuner
//
// Author: Youri Rigaud
// License: GNU GPLv3

#include "../include/tuner.h"
#include "../include/tuner_memory.h"
#include "../include/globaltimer.h"
#include "../include/tuning_objective.h"
#include "../include/filesystem_utils.h"
#include "../include/working_directory.h"

#ifdef USE_MPI
#include <mpi.h>
#endif

#include <cstring>
#include <limits>
#include <stdexcept>
#include <iostream>
#include <string>
#include <fstream>
#include <optional>

// Use this struct to tune the options of the tuner
struct TunerOptions {
    std::string tuner_dir = "./tuner_working_dir/";
    std::string parameters_file = "./cplex/params_12_cpx.txt";
    std::string instance_file = "./cplex/30n20b8.mps";
    std::string param_ils_instance_file = "./cplex/instances.txt";
    std::string solver_log_file = "./tuner_working_dir/solver/cplex.log";
    int nb_initial_selected_parameters = 10;
    int nb_parameter_to_evaluate_expansion = 10;
    int nb_threads_solver = 2;
    double cutoff_solver_time = 15.0;
    int nb_workers = 1;
    bool use_shared_cache = false;
    bool exploration_only = false;
    LocalSearchBackend local_search_backend = LocalSearchBackend::IteratedLocalSearch;
    std::uint32_t seed = 0;
    TuningObjective tuning_objective = TuningObjective::Gap;
    std::optional<int> number_of_evaluations = std::nullopt;
    int max_iterations = 15;
    bool enable_mip_starts = true;
    ExpansionSelectRule expansion_select_rule = ExpansionSelectRule::Strict;
    ExpansionValueStrategy expansion_value_strategy = ExpansionValueStrategy::FirstLast;
};

void printHelp(const char* program_name) {
    std::cout << "Usage: " << program_name << " [options] [instance_file]" << std::endl;
    std::cout << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  --help                          Show this help message and exit" << std::endl;
    std::cout << "  --working-dir PATH              Set the tuner working directory" << std::endl;
    std::cout << "  --parameters-file PATH          Set the parameter definition file" << std::endl;
    std::cout << "  --paramils-instance-file PATH   Set the ParamILS instance list file" << std::endl;
    std::cout << "  --initial-selected-parameters N Set the number of initially selected parameters" << std::endl;
    std::cout << "  --expansion-parameter-budget N  Set the number of residual parameters evaluated in expansion" << std::endl;
    std::cout << "  --solver-threads N              Set the number of solver threads" << std::endl;
    std::cout << "  --solver-time SECONDS           Set the cutoff time for each solver run" << std::endl;
    std::cout << "  --local-search-engine NAME      Set the exploration backend (iterated_local_search or paramils)" << std::endl;
    std::cout << "  --seed N                        Set the base random seed" << std::endl;
    std::cout << "  --tuning-objective NAME         Set the tuning objective (gap or upper_bound)" << std::endl;
    std::cout << "  --number-of-evaluations N       Override the exploration evaluation budget" << std::endl;
    std::cout << "  --max-iterations N              Set the maximum number of tuner iterations" << std::endl;
    std::cout << "  --expansion-select-rule MODE    Set expansion selection comparison (strict or inclusive)" << std::endl;
    std::cout << "  --expansion-value-strategy MODE Set expansion value evaluation strategy (all or first_last)" << std::endl;
    std::cout << "  --shared-cache                  Enable shared cache for ILS workers" << std::endl;
    std::cout << "  --no-shared-cache               Disable shared cache for ILS workers" << std::endl;
    std::cout << "  --exploration-only              Stop after the exploration phase" << std::endl;
    std::cout << "  --enable-mip-starts             Enable MIP starts during exploration when applicable" << std::endl;
    std::cout << "  --disable-mip-starts            Disable MIP starts during exploration" << std::endl;
    std::cout << std::endl;
    std::cout << "If instance_file is provided as a positional argument, it overrides the default instance path." << std::endl;
}

void getTunerOptions(int argc, char** argv, TunerOptions& options) {
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--help") == 0) {
            printHelp(argv[0]);
            std::exit(0);
        } else if (std::strcmp(argv[i], "--shared-cache") == 0) {
            options.use_shared_cache = true;
        } else if (std::strcmp(argv[i], "--no-shared-cache") == 0) {
            options.use_shared_cache = false;
        } else if (std::strcmp(argv[i], "--exploration-only") == 0) {
            options.exploration_only = true;
        } else if (std::strcmp(argv[i], "--parameters-file") == 0) {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for --parameters-file");
            }
            options.parameters_file = argv[++i];
        } else if (std::strcmp(argv[i], "--paramils-instance-file") == 0) {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for --paramils-instance-file");
            }
            options.param_ils_instance_file = argv[++i];
        } else if (std::strcmp(argv[i], "--initial-selected-parameters") == 0) {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for --initial-selected-parameters");
            }
            options.nb_initial_selected_parameters = std::stoi(argv[++i]);
            if (options.nb_initial_selected_parameters <= 0) {
                throw std::runtime_error("--initial-selected-parameters must be greater than 0");
            }
        } else if (std::strcmp(argv[i], "--expansion-parameter-budget") == 0) {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for --expansion-parameter-budget");
            }
            options.nb_parameter_to_evaluate_expansion = std::stoi(argv[++i]);
            if (options.nb_parameter_to_evaluate_expansion <= 0) {
                throw std::runtime_error("--expansion-parameter-budget must be greater than 0");
            }
        } else if (std::strcmp(argv[i], "--local-search-engine") == 0) {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for --local-search-engine");
            }
            options.local_search_backend = parseLocalSearchBackend(argv[++i]);
        } else if (std::strcmp(argv[i], "--seed") == 0) {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for --seed");
            }
            const unsigned long long parsed_seed = std::stoull(argv[++i]);
            if (parsed_seed > std::numeric_limits<std::uint32_t>::max()) {
                throw std::runtime_error("Seed value is too large for --seed");
            }
            options.seed = static_cast<std::uint32_t>(parsed_seed);
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
        } else if (std::strcmp(argv[i], "--number-of-evaluations") == 0) {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for --number-of-evaluations");
            }
            const int parsed_number_of_evaluations = std::stoi(argv[++i]);
            if (parsed_number_of_evaluations <= 0) {
                throw std::runtime_error("--number-of-evaluations must be greater than 0");
            }
            options.number_of_evaluations = parsed_number_of_evaluations;
        } else if (std::strcmp(argv[i], "--max-iterations") == 0) {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for --max-iterations");
            }
            options.max_iterations = std::stoi(argv[++i]);
            if (options.max_iterations <= 0) {
                throw std::runtime_error("--max-iterations must be greater than 0");
            }
        } else if (std::strcmp(argv[i], "--expansion-select-rule") == 0) {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for --expansion-select-rule");
            }
            options.expansion_select_rule = parseExpansionSelectRule(argv[++i]);
        } else if (std::strcmp(argv[i], "--expansion-value-strategy") == 0) {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for --expansion-value-strategy");
            }
            options.expansion_value_strategy = parseExpansionValueStrategy(argv[++i]);
        } else if (std::strcmp(argv[i], "--enable-mip-starts") == 0) {
            options.enable_mip_starts = true;
        } else if (std::strcmp(argv[i], "--disable-mip-starts") == 0) {
            options.enable_mip_starts = false;
        } else if (std::strcmp(argv[i], "--working-dir") == 0) {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for --working-dir");
            }
            options.tuner_dir = normalizeTunerWorkingDirectory(argv[++i]);
            options.solver_log_file = options.tuner_dir + "solver/cplex.log";
        } else if (argv[i][0] != '-') {
            options.instance_file = std::string(argv[i]);
        } else {
            throw std::runtime_error("Unknown command line option: " + std::string(argv[i]));
        }
    }
}

void writeParamILSInstanceFile(const std::string& filepath, const std::string& instance_file) {
    ensureParentDirectoryForFile(filepath);
    std::ofstream myfile;
    myfile.open(filepath);
    myfile << instance_file << std::endl;
    myfile.close();
}

void masterProcess(int argc, char** argv, TunerOptions options) {
    std::cout << "Welcome to the MPILS tuner!" << std::endl;
    std::cout << "Tuning instance: " << options.instance_file << std::endl;
    std::cout << "ILS shared cache: " << (options.use_shared_cache ? "enabled" : "disabled") << std::endl;
    std::cout << "Local search engine: " << localSearchBackendToString(options.local_search_backend) << std::endl;
    std::cout << "Exploration only: " << (options.exploration_only ? "enabled" : "disabled") << std::endl;
    std::cout << "Seed: " << options.seed << std::endl;
    std::cout << "Tuning objective: " << tuningObjectiveToString(options.tuning_objective) << std::endl;
    std::cout << "Parameters file: " << options.parameters_file << std::endl;
    std::cout << "ParamILS instance file: " << options.param_ils_instance_file << std::endl;
    std::cout << "Initial selected parameters: " << options.nb_initial_selected_parameters << std::endl;
    std::cout << "Expansion parameter budget: " << options.nb_parameter_to_evaluate_expansion << std::endl;
    std::cout << "Expansion select rule: " << expansionSelectRuleToString(options.expansion_select_rule) << std::endl;
    std::cout << "Expansion value strategy: " << expansionValueStrategyToString(options.expansion_value_strategy) << std::endl;
    std::cout << "Max iterations: " << options.max_iterations << std::endl;
    std::cout << "MIP starts: " << (options.enable_mip_starts ? "enabled" : "disabled") << std::endl;
    std::cout << "Number of evaluations: ";
    if (options.number_of_evaluations.has_value()) {
        std::cout << options.number_of_evaluations.value() << std::endl;
    } else {
        std::cout << "auto" << std::endl;
    }

    // init a clock to measure tuning time
    GlobalTimer::start();

    writeParamILSInstanceFile(options.param_ils_instance_file, options.instance_file);

    std::string log_file_path = options.tuner_dir + "tuner.log";
    ensureParentDirectoryForFile(log_file_path);
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
        options.exploration_only,
        options.local_search_backend,
        options.seed,
        options.tuning_objective,
        options.number_of_evaluations,
        options.max_iterations,
        options.enable_mip_starts,
        options.expansion_select_rule,
        options.expansion_value_strategy
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
    Worker worker(world_rank, options.instance_file, options.solver_log_file, options.nb_threads_solver, options.cutoff_solver_time, options.use_shared_cache, options.local_search_backend, options.seed, options.tuning_objective, options.enable_mip_starts);
    worker.run();
    std::cout << "Worker process " << world_rank << " finished." << std::endl;
}
#endif

int main(int argc, char** argv) {

    TunerOptions options;
    getTunerOptions(argc, argv, options);
    setTunerWorkingDirectory(options.tuner_dir);

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
