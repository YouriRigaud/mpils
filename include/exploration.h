// MPILS: Multi-Phase Iterated Local Search Tuner
//
// Author: Youri Rigaud
// License: GNU GPLv3

#ifndef EXPLORATION_H
#define EXPLORATION_H

#include "tuning_objective.h"
#include "tuner_memory.h"
#include "logger.h"
#include "parameter_space.h"
#include "configuration.h"
#include "iterated_local_search.h"

#include <string>

class LocalSearchEngine {
    protected:
        TunerMemory& memory_;
        Logger& logger_;
        const std::vector<Configuration>& initial_configurations_;
        ParameterSpace& parameter_space_;
        const std::string instance_file_;
        const std::string param_ils_instance_file_;
        const std::string solver_log_file_;
        int max_evaluations_;
        int& iteration_;
        int nb_threads_solver_;
        double cutoff_solver_time_;
        int nb_workers_;
        int local_search_start_time_;
        bool mip_start_;
        TuningObjective tuning_objective_;
        MipStartId used_mip_start_id_;
        std::string mip_start_file_;

        const std::vector<EvaluationRecord> parseCplexResultsFromLogFile(int run_obj, int worker_id);

        void setMipStartFile();

#ifdef USE_MPI
        void launchLocalSearchWorkers();
        void waitLocalSearchWorkers();
#endif
        
    public:
        LocalSearchEngine(
            TunerMemory& memory,
            Logger& logger,
            const std::vector<Configuration>& initial_configurations,
            ParameterSpace& parameter_space,
            const std::string& instance_file,
            const std::string& param_ils_instance_file,
            const std::string& solver_log_file,
            int max_evaluations,
            int& iteration,
            int nb_threads_solver,
            double cutoff_solver_time,
            int nb_workers,
            TuningObjective tuning_objective,
            bool mip_start = false
        ): memory_(memory),
           logger_(logger),
           initial_configurations_(initial_configurations),
           parameter_space_(parameter_space),
           instance_file_(instance_file),
           param_ils_instance_file_(param_ils_instance_file),
           solver_log_file_(solver_log_file),
           max_evaluations_(max_evaluations),
           iteration_(iteration),
           nb_threads_solver_(nb_threads_solver),
           cutoff_solver_time_(cutoff_solver_time),
           nb_workers_(nb_workers),
           mip_start_(mip_start),
           tuning_objective_(tuning_objective)
        {}

        virtual ~LocalSearchEngine() = default;

        virtual std::vector<std::pair<int, std::vector<EvaluationRecord>>> run() = 0;
};

class IteratedLocalSearchEngine : public LocalSearchEngine {
    private:
        std::unique_ptr<IteratedLocalSearch> ils_;
        bool use_shared_cache_;

        const std::string ils_working_dir_ = "tuner_working_dir/iterated_local_search/";
        std::string search_space_file_;

        void writeILSSearchSpaceFile();
        void writeILSParameterOptionsToFile(std::ofstream& myfile);
        void writeILSForbiddenOptionsToFile(std::ofstream& myfile);
        void writeILSConditionalCplexOptionsToFile(std::ofstream& myfile);
        void writeILSInfoToFile(std::ofstream& myfile);

        std::optional<double> getKnownInitialObjective_() const;

        std::vector<EvaluationRecord> syncILSResultsToGlobalMemory_(
            int worker_id,
            const std::vector<std::pair<Configuration, EvaluationRecord>>& local_results
        );

        std::vector<std::pair<Configuration, EvaluationRecord>> readLocalResultsFromFile_(int worker_id);

    public:
        IteratedLocalSearchEngine(
            TunerMemory& memory,
            Logger& logger,
            const std::vector<Configuration>& initial_configurations,
            ParameterSpace& parameter_space,
            const std::string& instance_file,
            const std::string& param_ils_instance_file,
            const std::string& solver_log_file,
            int max_evaluations,
            int& iteration,
            int nb_threads_solver,
            double cutoff_solver_time,
            int nb_workers,
            bool use_shared_cache,
            TuningObjective tuning_objective,
            bool mip_start = false
        ): LocalSearchEngine(memory, logger, initial_configurations, parameter_space, instance_file, param_ils_instance_file, solver_log_file, max_evaluations, iteration, nb_threads_solver, cutoff_solver_time, nb_workers, tuning_objective, mip_start),
           use_shared_cache_(use_shared_cache)
        {}

        std::vector<std::pair<int, std::vector<EvaluationRecord>>> run() override;
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

        const std::vector<std::pair<int, std::vector<EvaluationRecord>>> getParamILSResults();

    public:
        ParamILSEngine(
            TunerMemory& memory,
            Logger& logger,
            const std::vector<Configuration>& initial_configurations,
            ParameterSpace& parameter_space,
            const std::string& instance_file,
            const std::string& param_ils_instance_file,
            const std::string& solver_log_file,
            int max_evaluations,
            int& iteration,
            int nb_threads_solver,
            double cutoff_solver_time,
            int nb_workers,
            TuningObjective tuning_objective,
            bool mip_start = false
        ): LocalSearchEngine(memory, logger, initial_configurations, parameter_space, instance_file, param_ils_instance_file, solver_log_file, max_evaluations, iteration, nb_threads_solver, cutoff_solver_time, nb_workers, tuning_objective, mip_start)
        {}

        std::vector<std::pair<int, std::vector<EvaluationRecord>>> run() override;

};

class RandomLocalSearch : public LocalSearchEngine {
    private:
        std::vector<std::pair<int, std::vector<EvaluationRecord>>> generateRandomNeighbors() {
            // Implementation of random neighbor generation
            return {};
        }

    public:
        RandomLocalSearch(
            TunerMemory& memory,
            Logger& logger,
            const std::vector<Configuration>& initial_configurations,
            ParameterSpace& parameter_space,
            const std::string& instance_file,
            const std::string& param_ils_instance_file,
            const std::string& solver_log_file,
            int max_evaluations,
            int& iteration,
            int nb_threads_solver,
            double cutoff_solver_time,
            int nb_workers,
            TuningObjective tuning_objective,
            bool mip_start = false
        ): LocalSearchEngine(memory, logger, initial_configurations, parameter_space, instance_file, param_ils_instance_file, solver_log_file, max_evaluations, iteration, nb_threads_solver, cutoff_solver_time, nb_workers, tuning_objective, mip_start)
        {}

        std::vector<std::pair<int, std::vector<EvaluationRecord>>> run() override {
            return generateRandomNeighbors();
        }
};

class Exploration {
    private:
        TunerMemory& memory_;                       ///< Memory to store evaluations
        ParameterSpace& parameter_space_;
        Logger& logger_;
        int& iteration_;
        const std::string instance_file_;
        const std::string param_ils_instance_file_;
        const std::string solver_log_file_;
        int nb_threads_solver_;
        double cutoff_solver_time_;
        int nb_workers_;
        bool use_shared_cache_;
        TuningObjective tuning_objective_;

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
            const std::string& param_ils_instance_file,
            const std::string& solver_log_file,
            int nb_threads_solver,
            double cutoff_solver_time,
            int nb_workers,
            bool use_shared_cache,
            TuningObjective tuning_objective
        ): memory_(memory),
           parameter_space_(parameter_space),
           logger_(logger),
           iteration_(iteration),
           instance_file_(instance_file),
           param_ils_instance_file_(param_ils_instance_file),
           solver_log_file_(solver_log_file),
           nb_threads_solver_(nb_threads_solver),
           cutoff_solver_time_(cutoff_solver_time),
           nb_workers_(nb_workers),
           use_shared_cache_(use_shared_cache),
           tuning_objective_(tuning_objective)
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
        bool mip_start_;
        int nb_threads_solver_;
        double cutoff_solver_time_;
        bool use_shared_cache_;
        TuningObjective tuning_objective_;
        
    public:
        LocalSearchWorker(int worker_id, int iteration, TuningObjective tuning_objective, bool mip_start = false, bool use_shared_cache = false): worker_id_(worker_id), iteration_(iteration), mip_start_(mip_start), use_shared_cache_(use_shared_cache), tuning_objective_(tuning_objective) {}

        LocalSearchWorker(
            int worker_id,
            int iteration,
            int nb_threads_solver,
            double cutoff_solver_time,
            TuningObjective tuning_objective,
            bool mip_start = false,
            bool use_shared_cache = false
        ): worker_id_(worker_id), iteration_(iteration), mip_start_(mip_start), nb_threads_solver_(nb_threads_solver), cutoff_solver_time_(cutoff_solver_time), use_shared_cache_(use_shared_cache), tuning_objective_(tuning_objective)
        {}

        virtual ~LocalSearchWorker() = default;

        virtual void run() = 0;
};

class ParamILSWorker : public LocalSearchWorker {
    private:
        const std::string param_ils_dir_ = "param_ils/";
        const std::string param_ils_executable_ = "param_ils_2_3_run.rb";
        const std::string param_ils_working_dir_ = "tuner_working_dir/param_ils/";

        void callParamILS();
    
    public:
        ParamILSWorker(
            int worker_id,
            int iteration,
            TuningObjective tuning_objective,
            bool mip_start = false,
            bool use_shared_cache = false
        ): LocalSearchWorker(worker_id, iteration, tuning_objective, mip_start, use_shared_cache)
        {}

        void run() override {
            callParamILS();
        }
};

class IteratedLocalSearchWorker : public LocalSearchWorker {
    private:
        const std::string ils_working_dir_ = "tuner_working_dir/iterated_local_search/";
        int max_evaluations_;
        std::string instance_file_;
        std::string solver_log_file_;
        std::string mip_start_file_;

        void callIteratedLocalSearch();
    
    public:
        IteratedLocalSearchWorker(
            int worker_id,
            int iteration,
            int max_evaluations,
            int nb_threads_solver,
            double cutoff_solver_time,
            std::string instance_file,
            std::string solver_log_file,
            TuningObjective tuning_objective,
            bool use_shared_cache,
            bool mip_start = false
        ): LocalSearchWorker(worker_id, iteration, nb_threads_solver, cutoff_solver_time, tuning_objective, mip_start, use_shared_cache), max_evaluations_(max_evaluations), instance_file_(instance_file), solver_log_file_(solver_log_file)
        {}

        void run() override {
            callIteratedLocalSearch();
        }
};
#endif // USE_MPI

#endif // EXPLORATION_H
