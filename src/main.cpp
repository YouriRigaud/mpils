// MPILS: Multi-Phase Iterated Local Search Tuner
//
// Author: Youri Rigaud
// License: GNU GPLv3

#include "../include/tuner.h"
#include "../include/tuner_memory.h"
#include "../include/globaltimer.h"
#include "../include/solver_time_mode.h"
#include "../include/tuning_objective.h"
#include "../include/filesystem_utils.h"
#include "../include/working_directory.h"

#ifdef USE_MPI
#include <mpi.h>
#endif

#include <cstring>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <optional>

// Use this struct to tune the options of the tuner
struct TunerOptions {
    std::string tuner_dir = "./tuner_working_dir/";
    std::string project_dir = "./";
    std::string parameters_file = "./cplex/params_12_cpx.txt";
    std::string instance_file = "./cplex/30n20b8.mps";
    std::string param_ils_instance_file = "";
    std::string solver_log_file = "./tuner_working_dir/solver/cplex.log";
    int nb_initial_selected_parameters = 10;
    int nb_parameter_to_evaluate_expansion = 10;
    int nb_threads_solver = 2;
    double cutoff_solver_time = 15.0;
    SolverTimeMode solver_time_mode = SolverTimeMode::Seconds;
    SolverWatchdogOptions solver_watchdog_options;
    int nb_workers = 1;
    bool use_shared_cache = false;
    bool exploration_only = false;
    LocalSearchBackend local_search_backend = LocalSearchBackend::IteratedLocalSearch;
    std::uint32_t seed = 0;
    TuningObjective tuning_objective = TuningObjective::Gap;
    std::optional<int> number_of_evaluations = std::nullopt;
    std::optional<int> exploration_budget_divisor = std::nullopt;
    std::vector<int> exploration_budget_factors = {5};
    int max_iterations = 15;
    int mpi_procs_per_ils = 1;
    std::optional<double> paramils_wall_time = std::nullopt;
    bool mip_worker_strategy = false;
    bool enable_mip_starts = true;
    bool random_worker_initial_configs = true;
    MipStartInitialConfigPolicy mip_start_initial_config_policy = MipStartInitialConfigPolicy::ProducerConfig;
    ExpansionSelectRule expansion_select_rule = ExpansionSelectRule::Strict;
    ExpansionValueStrategy expansion_value_strategy = ExpansionValueStrategy::FirstLast;
    double expansion_max_deviation = std::numeric_limits<double>::max();
    bool expansion_enable_early_stop = true;
    bool clean_working_dir = false;
};

void printHelp(const char* program_name) {
    std::cout << "Usage: " << program_name << " [options] [instance_file]" << std::endl;
    std::cout << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  --help                          Show this help message and exit" << std::endl;
    std::cout << "  --working-dir PATH              Set the tuner working directory" << std::endl;
    std::cout << "  --project-dir PATH              Set the MPILS project root directory (must contain param_ils/)" << std::endl;
    std::cout << "  --clean-working-dir             Remove generated subdirectories and transient files after tuning" << std::endl;
    std::cout << "  --no-clean-working-dir          Keep the full working directory after tuning (default)" << std::endl;
    std::cout << "  --parameters-file PATH          Set the parameter definition file" << std::endl;
    std::cout << "  --paramils-instance-file PATH   Set the ParamILS instance list file (default: <working-dir>/param_ils/instances.txt)" << std::endl;
    std::cout << "  --initial-selected-parameters N Set the number of initially selected parameters" << std::endl;
    std::cout << "  --expansion-parameter-budget N  Set the number of residual parameters evaluated in expansion" << std::endl;
    std::cout << "  --solver-threads N              Set the number of solver threads" << std::endl;
    std::cout << "  --solver-time SECONDS           Set the cutoff time for each solver run" << std::endl;
    std::cout << "  --solver-time-mode MODE         Set solver time budget mode (seconds or ticks)" << std::endl;
    std::cout << "  --solver-wall-watchdog          Enable cooperative wall-clock watchdog for seconds mode" << std::endl;
    std::cout << "  --no-solver-wall-watchdog       Disable cooperative wall-clock watchdog" << std::endl;
    std::cout << "  --solver-wall-factor VALUE      Set watchdog wall limit factor (default 1.25)" << std::endl;
    std::cout << "  --solver-wall-grace SECONDS     Set watchdog wall limit grace seconds (default 5.0)" << std::endl;
    std::cout << "  --local-search-engine NAME      Set the exploration backend (iterated_local_search or paramils)" << std::endl;
    std::cout << "  --seed N                        Set the base random seed" << std::endl;
    std::cout << "  --tuning-objective NAME         Set the tuning objective (gap or upper_bound)" << std::endl;
    std::cout << "  --number-of-evaluations N       Override the exploration evaluation budget" << std::endl;
    std::cout << "  --exploration-budget-factor K[,K2,K3,...]  Set the exploration budget factor per iteration (budget = K_i * nb_selected_params). Single value applies to all iterations; list uses K_i for iteration i and last value for subsequent ones. Default: 5" << std::endl;
    std::cout << "  --divide-exploration-budget N   Divide the exploration evaluation budget by N" << std::endl;
    std::cout << "  --max-iterations N              Set the maximum number of tuner iterations" << std::endl;
    std::cout << "  --expansion-select-rule MODE    Set expansion selection comparison (strict or inclusive)" << std::endl;
    std::cout << "  --expansion-value-strategy MODE Set expansion value evaluation strategy (all or first_last)" << std::endl;
    std::cout << "  --expansion-max-deviation VALUE Set the maximum allowed RMS deviation during expansion classification" << std::endl;
    std::cout << "  --expansion-early-stop          Enable early stop in expansion" << std::endl;
    std::cout << "  --no-expansion-early-stop       Disable early stop in expansion" << std::endl;
    std::cout << "  --shared-cache                  Enable shared cache for ILS workers" << std::endl;
    std::cout << "  --no-shared-cache               Disable shared cache for ILS workers" << std::endl;
    std::cout << "  --exploration-only              Stop after the exploration phase" << std::endl;
    std::cout << "  --enable-mip-starts             Enable MIP starts during exploration when applicable" << std::endl;
    std::cout << "  --disable-mip-starts            Disable MIP starts during exploration" << std::endl;
    std::cout << "  --mip-start-initial-config MODE Set MIP-start worker initial config (producer_config or best_config)" << std::endl;
    std::cout << "  --random-worker-initial-configs Enable per-worker random initial configs for MPI ILS exploration" << std::endl;
    std::cout << "  --no-random-worker-initial-configs Disable per-worker random initial configs for MPI ILS exploration" << std::endl;
    std::cout << "  --mpi-procs-per-ils N           Number of MPI processes per ILS instance for parallel neighbor eval (default: 1)" << std::endl;
    std::cout << "  --paramils-wall-time SECONDS     Set wall-clock time budget for the ParamILS run (overrides cutoff_time * maxEvals)" << std::endl;
    std::cout << "  --mip-worker-strategy            Enable 4-role MIP worker strategy (M1 cold, M2 MIP chainer, M3 MIP exploiter, M4 cold explorer)" << std::endl;
    std::cout << "  --no-mip-worker-strategy         Disable 4-role MIP worker strategy (default)" << std::endl;
    std::cout << std::endl;
    std::cout << "If instance_file is provided as a positional argument, it overrides the default instance path." << std::endl;
}

void getTunerOptions(int argc, char** argv, TunerOptions& options) {
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--help") == 0) {
            printHelp(argv[0]);
            std::exit(0);
        } else if (std::strcmp(argv[i], "--clean-working-dir") == 0) {
            options.clean_working_dir = true;
        } else if (std::strcmp(argv[i], "--no-clean-working-dir") == 0) {
            options.clean_working_dir = false;
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
        } else if (std::strcmp(argv[i], "--solver-time-mode") == 0) {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for --solver-time-mode");
            }
            options.solver_time_mode = parseSolverTimeMode(argv[++i]);
        } else if (std::strcmp(argv[i], "--solver-wall-watchdog") == 0) {
            options.solver_watchdog_options.enabled = true;
        } else if (std::strcmp(argv[i], "--no-solver-wall-watchdog") == 0) {
            options.solver_watchdog_options.enabled = false;
        } else if (std::strcmp(argv[i], "--solver-wall-factor") == 0) {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for --solver-wall-factor");
            }
            options.solver_watchdog_options.wall_time_factor = std::stod(argv[++i]);
        } else if (std::strcmp(argv[i], "--solver-wall-grace") == 0) {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for --solver-wall-grace");
            }
            options.solver_watchdog_options.wall_time_grace_seconds = std::stod(argv[++i]);
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
        } else if (std::strcmp(argv[i], "--divide-exploration-budget") == 0) {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for --divide-exploration-budget");
            }
            const int parsed_exploration_budget_divisor = std::stoi(argv[++i]);
            if (parsed_exploration_budget_divisor <= 0) {
                throw std::runtime_error("--divide-exploration-budget must be greater than 0");
            }
            options.exploration_budget_divisor = parsed_exploration_budget_divisor;
        } else if (std::strcmp(argv[i], "--exploration-budget-factor") == 0) {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for --exploration-budget-factor");
            }
            std::string arg = argv[++i];
            options.exploration_budget_factors.clear();
            std::size_t pos = 0;
            while (true) {
                std::size_t comma = arg.find(',', pos);
                std::string token = arg.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
                int val = std::stoi(token);
                if (val <= 0) {
                    throw std::runtime_error("--exploration-budget-factor values must be greater than 0");
                }
                options.exploration_budget_factors.push_back(val);
                if (comma == std::string::npos) break;
                pos = comma + 1;
            }
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
        } else if (std::strcmp(argv[i], "--expansion-max-deviation") == 0) {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for --expansion-max-deviation");
            }
            options.expansion_max_deviation = std::stod(argv[++i]);
            if (options.expansion_max_deviation < 0.0) {
                throw std::runtime_error("--expansion-max-deviation must be non-negative");
            }
        } else if (std::strcmp(argv[i], "--expansion-early-stop") == 0) {
            options.expansion_enable_early_stop = true;
        } else if (std::strcmp(argv[i], "--no-expansion-early-stop") == 0) {
            options.expansion_enable_early_stop = false;
        } else if (std::strcmp(argv[i], "--enable-mip-starts") == 0) {
            options.enable_mip_starts = true;
        } else if (std::strcmp(argv[i], "--disable-mip-starts") == 0) {
            options.enable_mip_starts = false;
        } else if (std::strcmp(argv[i], "--mip-start-initial-config") == 0) {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for --mip-start-initial-config");
            }
            options.mip_start_initial_config_policy = parseMipStartInitialConfigPolicy(argv[++i]);
        } else if (std::strcmp(argv[i], "--random-worker-initial-configs") == 0) {
            options.random_worker_initial_configs = true;
        } else if (std::strcmp(argv[i], "--no-random-worker-initial-configs") == 0) {
            options.random_worker_initial_configs = false;
        } else if (std::strcmp(argv[i], "--working-dir") == 0) {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for --working-dir");
            }
            options.tuner_dir = normalizeTunerWorkingDirectory(argv[++i]);
            options.solver_log_file = options.tuner_dir + "solver/cplex.log";
        } else if (std::strcmp(argv[i], "--project-dir") == 0) {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for --project-dir");
            }
            options.project_dir = argv[++i];
        } else if (std::strcmp(argv[i], "--mip-worker-strategy") == 0) {
            options.mip_worker_strategy = true;
            options.enable_mip_starts = true;
        } else if (std::strcmp(argv[i], "--no-mip-worker-strategy") == 0) {
            options.mip_worker_strategy = false;
        } else if (std::strcmp(argv[i], "--paramils-wall-time") == 0) {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for --paramils-wall-time");
            }
            const double val = std::stod(argv[++i]);
            if (val <= 0.0) {
                throw std::runtime_error("--paramils-wall-time must be greater than 0");
            }
            options.paramils_wall_time = val;
        } else if (std::strcmp(argv[i], "--mpi-procs-per-ils") == 0) {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for --mpi-procs-per-ils");
            }
            options.mpi_procs_per_ils = std::stoi(argv[++i]);
            if (options.mpi_procs_per_ils <= 0) {
                throw std::runtime_error("--mpi-procs-per-ils must be greater than 0");
            }
        } else if (argv[i][0] != '-') {
            options.instance_file = std::string(argv[i]);
        } else {
            throw std::runtime_error("Unknown command line option: " + std::string(argv[i]));
        }
    }
    if (options.param_ils_instance_file.empty()) {
        options.param_ils_instance_file = options.tuner_dir + "param_ils/instances.txt";
    }
}

void cleanTunerWorkingDirectory(const std::string& tuner_dir) {
    namespace fs = std::filesystem;

    const fs::path working_dir(tuner_dir);
    if (!fs::exists(working_dir) || !fs::is_directory(working_dir)) {
        return;
    }

    for (const auto& entry : fs::directory_iterator(working_dir)) {
        const fs::path& path = entry.path();
        const std::string filename = path.filename().string();

        if (entry.is_directory()) {
            if (filename == "worker_logs") {
                continue;
            }
            fs::remove_all(path);
            continue;
        }

        const bool keep_file = filename == "tuner.log" ||
                               filename == "run.log" ||
                               filename == "best_configuration.prm" ||
                               path.extension() == ".csv";

        if (!keep_file) {
            fs::remove(path);
        }
    }
}

void validateTunerOptions(const TunerOptions& options) {
    if (options.solver_watchdog_options.wall_time_factor < 0.0) {
        throw std::runtime_error("--solver-wall-factor must be non-negative");
    }
    if (options.solver_watchdog_options.wall_time_grace_seconds < 0.0) {
        throw std::runtime_error("--solver-wall-grace must be non-negative");
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
    std::cout << "Project directory: " << options.project_dir << std::endl;
    std::cout << "ParamILS instance file: " << options.param_ils_instance_file << std::endl;
    std::cout << "Initial selected parameters: " << options.nb_initial_selected_parameters << std::endl;
    std::cout << "Expansion parameter budget: " << options.nb_parameter_to_evaluate_expansion << std::endl;
    std::cout << "Expansion select rule: " << expansionSelectRuleToString(options.expansion_select_rule) << std::endl;
    std::cout << "Expansion value strategy: " << expansionValueStrategyToString(options.expansion_value_strategy) << std::endl;
    std::cout << "Expansion max deviation: " << options.expansion_max_deviation << std::endl;
    std::cout << "Expansion early stop: " << (options.expansion_enable_early_stop ? "enabled" : "disabled") << std::endl;
    std::cout << "Max iterations: " << options.max_iterations << std::endl;
    std::cout << "MIP starts: " << (options.enable_mip_starts ? "enabled" : "disabled") << std::endl;
    std::cout << "MIP worker strategy: " << (options.mip_worker_strategy ? "enabled" : "disabled") << std::endl;
    std::cout << "MIP-start initial config policy: " << mipStartInitialConfigPolicyToString(options.mip_start_initial_config_policy) << std::endl;
    std::cout << "Random worker initial configs: " << (options.random_worker_initial_configs ? "enabled" : "disabled") << std::endl;
    std::cout << "Solver time mode: " << solverTimeModeToString(options.solver_time_mode) << std::endl;
    std::cout << "Solver wall watchdog: "
              << (options.solver_watchdog_options.enabled && options.solver_time_mode == SolverTimeMode::Seconds ? "enabled" : "disabled")
              << " | factor: " << options.solver_watchdog_options.wall_time_factor
              << " | grace: " << options.solver_watchdog_options.wall_time_grace_seconds << "s"
              << std::endl;
    std::cout << "Clean working directory: " << (options.clean_working_dir ? "enabled" : "disabled") << std::endl;
    std::cout << "Number of evaluations: ";
    if (options.number_of_evaluations.has_value()) {
        std::cout << options.number_of_evaluations.value() << std::endl;
    } else {
        std::cout << "auto" << std::endl;
    }
    std::cout << "Exploration budget factors: ";
    for (std::size_t j = 0; j < options.exploration_budget_factors.size(); ++j) {
        if (j > 0) std::cout << ",";
        std::cout << options.exploration_budget_factors[j];
    }
    std::cout << std::endl;
    std::cout << "Exploration budget divisor: ";
    if (options.exploration_budget_divisor.has_value()) {
        std::cout << options.exploration_budget_divisor.value() << std::endl;
    } else {
        std::cout << "none" << std::endl;
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
        options.solver_time_mode,
        options.solver_watchdog_options,
        options.nb_workers,
        options.use_shared_cache,
        options.exploration_only,
        options.local_search_backend,
        options.seed,
        options.tuning_objective,
        options.number_of_evaluations,
        options.exploration_budget_divisor,
        options.exploration_budget_factors,
        options.max_iterations,
        options.enable_mip_starts,
        options.random_worker_initial_configs,
        options.mip_start_initial_config_policy,
        options.expansion_select_rule,
        options.expansion_value_strategy,
        options.expansion_max_deviation,
        options.expansion_enable_early_stop,
        options.paramils_wall_time,
        options.mip_worker_strategy,
        options.clean_working_dir
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

    if (options.clean_working_dir) {
        cleanTunerWorkingDirectory(options.tuner_dir);
    }
}

#ifdef USE_MPI
void workerProcess(int world_rank, TunerOptions options) {
    // init a clock to measure total tuning time
    GlobalTimer::start();
    Worker worker(world_rank, options.instance_file, options.solver_log_file, options.nb_threads_solver, options.cutoff_solver_time, options.solver_time_mode, options.solver_watchdog_options, options.use_shared_cache, options.local_search_backend, options.seed, options.tuning_objective, options.enable_mip_starts, options.random_worker_initial_configs, options.mip_worker_strategy);
    worker.run();
}
#endif

int main(int argc, char** argv) {
    try {
        TunerOptions options;
        getTunerOptions(argc, argv, options);
        validateTunerOptions(options);
        setTunerWorkingDirectory(options.tuner_dir);
        setParamILSSourceDir(options.project_dir);

#ifdef USE_MPI
        MPI_Init(&argc, &argv);
        int world_rank;
        int world_size;
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
        options.nb_workers = world_size;

        if (world_size % options.mpi_procs_per_ils != 0) {
            throw std::runtime_error(
                "--mpi-procs-per-ils (" + std::to_string(options.mpi_procs_per_ils) +
                ") must divide world_size (" + std::to_string(world_size) + ") evenly"
            );
        }

        {
            const int procs_per_ils  = options.mpi_procs_per_ils;
            const int ils_group_id   = world_rank / procs_per_ils;
            const int ils_group_rank = world_rank % procs_per_ils;
            MPI_Comm ils_comm;
            MPI_Comm_split(MPI_COMM_WORLD, ils_group_id, world_rank, &ils_comm);
            setParallelILSInfo(ils_comm, ils_group_rank, procs_per_ils);
        }

        if (world_rank == 0) {
            masterProcess(argc, argv, options);
        } else {
            workerProcess(world_rank, options);
        }
        MPI_Finalize();
#else
        masterProcess(argc, argv, options);
#endif

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
