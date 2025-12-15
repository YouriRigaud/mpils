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
        const Configuration& initial_configuration_;
        ParameterSpace& parameter_space_;
        const std::string instance_file_;
        const std::string solver_log_file_;
        int max_evaluations_;
        int& iteration_;
        int nb_threads_solver_;
        double cutoff_solver_time_;

        const std::vector<Configuration> parseCplexResultsFromLogFile(int run_obj);
        
    public:
        LocalSearchEngine(
            Logger& logger,
            const Configuration& initial_configuration,
            ParameterSpace& parameter_space,
            const std::string& instance_file,
            const std::string& solver_log_file,
            int max_evaluations,
            int& iteration,
            int nb_threads_solver,
            double cutoff_solver_time
        ): logger_(logger),
           initial_configuration_(initial_configuration),
           parameter_space_(parameter_space),
           instance_file_(instance_file),
           solver_log_file_(solver_log_file),
           max_evaluations_(max_evaluations),
           iteration_(iteration),
              nb_threads_solver_(nb_threads_solver),
           cutoff_solver_time_(cutoff_solver_time)
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

        void writeParamILSParameterFile();

        void writeParamILSScenarioFile();

        void writeParameterOptionsToFile(std::ofstream& myfile);

        void writeForbiddenOptionsToFile(std::ofstream& myfile);

        void writeConditionalCplexOptionsToFile(std::ofstream& myfile);

        void callParamILS();

        const std::vector<Configuration> getParamILSResults();

    public:
        ParamILSEngine(
            Logger& logger,
            const Configuration& initial_configuration,
            ParameterSpace& parameter_space,
            const std::string& instance_file,
            const std::string& solver_log_file,
            int max_evaluations,
            int& iteration,
            int nb_threads_solver,
            double cutoff_solver_time
        ): LocalSearchEngine(logger, initial_configuration, parameter_space, instance_file, solver_log_file, max_evaluations, iteration, nb_threads_solver, cutoff_solver_time)
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
            const Configuration& initial_configuration,
            ParameterSpace& parameter_space,
            const std::string& instance_file,
            const std::string& solver_log_file,
            int max_evaluations,
            int& iteration,
            int nb_threads_solver,
            double cutoff_solver_time
        ): LocalSearchEngine(logger, initial_configuration, parameter_space, instance_file, solver_log_file, max_evaluations, iteration, nb_threads_solver, cutoff_solver_time)
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
            double cutoff_solver_time
        ): memory_(memory),
           parameter_space_(parameter_space),
           logger_(logger),
           iteration_(iteration),
           instance_file_(instance_file),
           solver_log_file_(solver_log_file),
              nb_threads_solver_(nb_threads_solver),
              cutoff_solver_time_(cutoff_solver_time)
        {}

        void setEngine(std::unique_ptr<LocalSearchEngine> engine) {
            engine_ = std::move(engine);
        }


        void run();
};


#endif // EXPLORATION_H