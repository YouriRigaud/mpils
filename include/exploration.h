// MPILS: Multi-Phase Iterated Local Search Tuner
//
// Author: Youri Rigaud
// License: GNU GPLv3

#ifndef EXPLORATION_H
#define EXPLORATION_H

#include "parameter.h"
#include "tuner_memory.h"
#include "logger.h"
#include "parameter_space.h"

#include <string>

class LocalSearchEngine {
    protected:
        Logger& logger_;
        const std::vector<Configuration>& initial_configurations_;
        ParameterSpace& parameter_space_;
        const std::string instance_file_;
        const std::string solver_log_file_;
        int max_evaluations_;
        int& iteration_;
        int nb_threads_solver_;
        double cutoff_solver_time_;
        int nb_workers_;

        const std::vector<Configuration> parseCplexResultsFromLogFile(int run_obj, int worker_id);

#ifdef USE_MPI
        void launchLocalSearchWorkers();
        void waitLocalSearchWorkers();
#endif
        
    public:
        LocalSearchEngine(
            Logger& logger,
            const std::vector<Configuration>& initial_configurations,
            ParameterSpace& parameter_space,
            const std::string& instance_file,
            const std::string& solver_log_file,
            int max_evaluations,
            int& iteration,
            int nb_threads_solver,
            double cutoff_solver_time,
            int nb_workers
        ): logger_(logger),
           initial_configurations_(initial_configurations),
           parameter_space_(parameter_space),
           instance_file_(instance_file),
           solver_log_file_(solver_log_file),
           max_evaluations_(max_evaluations),
           iteration_(iteration),
           nb_threads_solver_(nb_threads_solver),
           cutoff_solver_time_(cutoff_solver_time),
           nb_workers_(nb_workers)
        {}

        virtual ~LocalSearchEngine() = default;

        virtual std::vector<Configuration> run() = 0;
};

class ParamILSEngine : public LocalSearchEngine {
    private:
        const std::string param_ils_dir_ = "param_ils/";
        const std::string param_ils_executable_ = "param_ils_2_3_run.rb";
        const std::string param_ils_working_dir_ = "tuner_working_dir/param_ils/";
        std::string parameter_file_;
        std::string scenario_file_;

        void writeParamILSParameterFiles();

        void writeParamILSScenarioFiles();

        void writeParameterOptionsToFile(std::ofstream& myfile, int worker_id);

        void writeForbiddenOptionsToFile(std::ofstream& myfile, int worker_id);

        void writeConditionalCplexOptionsToFile(std::ofstream& myfile);

        void callParamILS();

        const std::vector<Configuration> getParamILSResults();

    public:
        ParamILSEngine(
            Logger& logger,
            const std::vector<Configuration>& initial_configurations,
            ParameterSpace& parameter_space,
            const std::string& instance_file,
            const std::string& solver_log_file,
            int max_evaluations,
            int& iteration,
            int nb_threads_solver,
            double cutoff_solver_time,
            int nb_workers
        ): LocalSearchEngine(logger, initial_configurations, parameter_space, instance_file, solver_log_file, max_evaluations, iteration, nb_threads_solver, cutoff_solver_time, nb_workers)
        {}

        std::vector<Configuration> run() override;

};

class RandomLocalSearch : public LocalSearchEngine {
    private:
        std::vector<Configuration> generateRandomNeighbors() {
            // Implementation of random neighbor generation
            return {};
        }

    public:
        RandomLocalSearch(
            Logger& logger,
            const std::vector<Configuration>& initial_configurations,
            ParameterSpace& parameter_space,
            const std::string& instance_file,
            const std::string& solver_log_file,
            int max_evaluations,
            int& iteration,
            int nb_threads_solver,
            double cutoff_solver_time,
            int nb_workers
        ): LocalSearchEngine(logger, initial_configurations, parameter_space, instance_file, solver_log_file, max_evaluations, iteration, nb_threads_solver, cutoff_solver_time, nb_workers)
        {}

        std::vector<Configuration> run() override {
            return generateRandomNeighbors();
        }
};

class Exploration {
    private:
        TunerMemory& memory_;                       ///< Memory to store configurations tested
        ParameterSpace& parameter_space_;
        Logger& logger_;
        int& iteration_;
        const std::string instance_file_;
        const std::string solver_log_file_;
        int nb_threads_solver_;
        double cutoff_solver_time_;
        int nb_workers_;

        std::unique_ptr<LocalSearchEngine> engine_ = nullptr;


        void updateTunedParameters();
        std::vector<Configuration> selectInitialConfigurations();
        int selectNumberOfEvaluations();

    public:
        Exploration(
            TunerMemory& memory,
            ParameterSpace& parameter_space,
            Logger& logger,
            int& iteration,
            const std::string& instance_file,
            const std::string& solver_log_file,
            int nb_threads_solver,
            double cutoff_solver_time,
            int nb_workers
        ): memory_(memory),
           parameter_space_(parameter_space),
           logger_(logger),
           iteration_(iteration),
           instance_file_(instance_file),
           solver_log_file_(solver_log_file),
           nb_threads_solver_(nb_threads_solver),
           cutoff_solver_time_(cutoff_solver_time),
           nb_workers_(nb_workers)
        {}

        void setEngine(std::unique_ptr<LocalSearchEngine> engine) {
            engine_ = std::move(engine);
        }


        void run();
};

#ifdef USE_MPI
class LocalSearchWorker {
    protected:
        int worker_id_;
        int iteration_;
        
    public:
        LocalSearchWorker(int worker_id, int iteration): worker_id_(worker_id), iteration_(iteration) {}
        
        virtual void run() = 0;
};

class ParamILSWorker : public LocalSearchWorker {
    private:
        const std::string param_ils_dir_ = "param_ils/";
        const std::string param_ils_executable_ = "param_ils_2_3_run.rb";
        const std::string param_ils_working_dir_ = "tuner_working_dir/param_ils/";

        void callParamILS() {
            std::string scenario_file_path = param_ils_working_dir_ + "scenario/scenario_file_" + std::to_string(iteration_) + "_worker_" + std::to_string(worker_id_) + ".txt";
            std::string command = "ruby " + param_ils_dir_ + param_ils_executable_ + " -numRun 0 -scenariofile " + scenario_file_path;
            int ret = system(command.c_str());
            if (ret != 0) {
                std::cout << "Error calling ParamILS executable for worker " << worker_id_ << " at iteration " << iteration_ << std::endl;
            }
        }
    
    public:
        ParamILSWorker(
            int worker_id,
            int iteration
        ): LocalSearchWorker(worker_id, iteration)
        {}

        void run() override {
            callParamILS();
        }
};
#endif // USE_MPI

#endif // EXPLORATION_H