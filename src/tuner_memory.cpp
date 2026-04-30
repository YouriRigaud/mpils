// MPILS: Multi-Phase Iterated Local Search Tuner
//
// Author: Youri Rigaud
// License: GNU GPLv3

#include "../include/tuner_memory.h"

ConfigurationId TunerMemory::ensureConfigurationStored_(const Configuration& config) {
    ConfigurationId config_id = config.getConfigurationId();
    auto it = configurations_by_id_.find(config_id);
    if (it != configurations_by_id_.end()) {
        if (!(it->second == config)) {
            logger_.warn("Hash collision detected for configuration ID: ", config_id, ". This should be very rare, but it means that two different configurations have the same hash. Consider improving the hash function to reduce the probability of collisions.");
        }
        return it->first; // Configuration already stored, return its ID
    } else {
        // Store the new configuration, create its stats entry, and return its ID
        configurations_by_id_[config_id] = config;
        stats_by_id_[config_id] = ConfigurationStats(); // Initialize stats for this configuration
        return config_id;
    }
}

EvaluationRecord TunerMemory::createEvaluationRecord_(const RecordEvaluationOptions& options, const ConfigurationId config_id, const EvaluationId eval_id) {
    EvaluationRecord record{};
    record.evaluation_id = eval_id;
    record.configuration_id = config_id;

    // Compute the objective value, clamping it if necessary
    record.objective_value = clampObjective_(options.objective_value);
    record.gap = options.gap;
    record.upper_bound = options.upper_bound;
    record.lower_bound = options.lower_bound;
    record.solver_runtime_seconds = options.solver_runtime_seconds;

    // Compute the time evaluated (in seconds since tuning started)
    if (options.time_evaluated >= 0) {
        record.time_evaluated = options.time_evaluated;
    } else {
        record.time_evaluated = GlobalTimer::elapsedSeconds();
    }

    record.worker_id = options.worker_id;
    record.iteration = options.iteration;
    record.phase = options.phase;

    record.mip_start_used = options.mip_start_used;
    record.used_mip_start_id = options.used_mip_start_id;
    if (options.mip_start_used && !options.used_mip_start_id.has_value()) {
        logger_.warn("Creating an evaluation record that used a MIP start but no MIP start ID was provided. EvaluationId: ", record.evaluation_id);
    }
    record.mip_start_source_evaluation_id = options.mip_start_source_evaluation_id;
    if (options.mip_start_used && !options.mip_start_source_evaluation_id.has_value()) {
        logger_.debug("Creating an evaluation record that used a MIP start but no source evaluation ID was provided. EvaluationId: ", record.evaluation_id);
        // Find the evaluation that produced the MIP start used, if possible, to fill the mip_start_source_evaluation_id field
        if (options.used_mip_start_id.has_value()) {
            auto mip_start_it = mip_starts_by_id_.find(*options.used_mip_start_id);
            if (mip_start_it != mip_starts_by_id_.end()) {
                record.mip_start_source_evaluation_id = mip_start_it->second.evaluation_id;
            } else {
                logger_.debug("The MIP start ID used in this evaluation record was not found in memory. EvaluationId: ", record.evaluation_id, ", Used MIP Start ID: ", *options.used_mip_start_id);
            }
        }
    }

    record.produced_mip_start = options.produced_mip_start;
    record.produced_mip_start_id = std::nullopt;
    record.produced_mip_start_file = options.produced_mip_start ? options.mip_start_file : "";
    return record;
}

void TunerMemory::updateBestEvaluation_(const EvaluationRecord& record) {
    bool is_new_best = false;
    if (record.objective_value < best_objective_) {
        best_objective_ = record.objective_value;
        best_evaluation_index_ = evaluations_.size() - 1; // The new record is the last one in the evaluations vector
        is_new_best = true;
    }
    if (!record.mip_start_used && record.objective_value < best_objective_without_mip_start_) {
        best_objective_without_mip_start_ = record.objective_value;
        best_evaluation_without_mip_start_index_ = evaluations_.size() - 1;
        is_new_best = true;
    }
    if (is_new_best) {
        logNewBestEvaluation_(record);
    }
}

void TunerMemory::updateStatsForConfiguration_(const EvaluationRecord& record) {
    ConfigurationId config_id = record.configuration_id;
    ConfigurationStats& stats = stats_by_id_[config_id];
    stats.nb_evaluations++;
    if (record.objective_value < stats.best_objective) {
        stats.best_objective = record.objective_value;
        stats.best_evaluation_id = record.evaluation_id;
    }
}

void TunerMemory::updateMipStartRecords_(EvaluationRecord& record, const std::string& mip_start_file) {
    if (!record.produced_mip_start) {
        return;
    }
    if (!record.upper_bound.has_value()) {
        logger_.warn("Evaluation marked as producing a MIP start but no upper bound is available. EvalId: ", record.evaluation_id);
        record.produced_mip_start = false;
        record.produced_mip_start_file.clear();
        return;
    }
    if (!wouldImproveBestMipStartUpperBound(record.upper_bound.value())) {
        logger_.debug("Ignoring produced MIP start because upper bound ", record.upper_bound.value(),
                      " does not improve current best MIP-start upper bound ", best_mip_start_upper_bound_,
                      ". EvalId: ", record.evaluation_id);
        record.produced_mip_start = false;
        record.produced_mip_start_file.clear();
        return;
    }
    record.produced_mip_start_id = next_mip_start_id_++;
    record.produced_mip_start_file = mip_start_file;

    MipStartRecord mip_start_record{};
    mip_start_record.mip_start_id = *record.produced_mip_start_id;
    mip_start_record.evaluation_id = record.evaluation_id;
    mip_start_record.mip_start_file = mip_start_file;
    mip_start_record.upper_bound = record.upper_bound.value();

    if (mip_start_record.mip_start_file.empty()) {
        logger_.warn("Produced mip start but mip_start_file is empty. EvalId: ", record.evaluation_id, " MipStartId: ", mip_start_record.mip_start_id);
    }

    best_mip_start_id_ = mip_start_record.mip_start_id;
    best_mip_start_upper_bound_ = mip_start_record.upper_bound;
    mip_starts_by_id_[mip_start_record.mip_start_id] = std::move(mip_start_record);
}

EvaluationId TunerMemory::recordEvaluation(const Configuration& config, const RecordEvaluationOptions& options) {
    logger_.info("recordEvaluation produced_mip_start=", options.produced_mip_start, " mip_start_file='", options.mip_start_file, "'");
    EvaluationId eval_id = next_evaluation_id_++;
    ConfigurationId config_id = ensureConfigurationStored_(config);
    EvaluationRecord record = createEvaluationRecord_(options, config_id, eval_id);
    evaluations_.push_back(record);
    EvaluationRecord& stored_record = evaluations_.back();
    updateBestEvaluation_(stored_record);
    updateStatsForConfiguration_(stored_record);
    updateMipStartRecords_(stored_record, options.mip_start_file);
    logger_.debug("Added evaluation to memory with objective: ", stored_record.objective_value, ", ConfigId: ", stored_record.configuration_id, ", EvalId: ", stored_record.evaluation_id, ". Total evaluations stored: ", evaluations_.size());
    return eval_id;
}

bool TunerMemory::setMipStartFile(const std::string& mip_start_file, MipStartId mip_start_id) {
    auto it = mip_starts_by_id_.find(mip_start_id);
    if (it == mip_starts_by_id_.end()) {
        logger_.warn("Trying to set MIP start file for unknown MipStartId: ", mip_start_id);
        return false;
    }
    it->second.mip_start_file = mip_start_file;
    return true;
}

void TunerMemory::logNewBestEvaluation_(const EvaluationRecord& record) {
    Configuration config = configurations_by_id_.at(record.configuration_id);
    logger_.info("***************");
    if (record.mip_start_used) {
        logger_.info("New best evaluation found at time ", record.time_evaluated, "s, with objective ", record.objective_value, " using a MIP start from evaluation ", record.mip_start_source_evaluation_id.value_or(-1), " and configuration ID ", record.configuration_id);
    } else {
        logger_.info("New best evaluation without MIP start found at time ", record.time_evaluated, "s, with objective ", record.objective_value, " and configuration ID ", record.configuration_id);
    }
    logger_.info("With config: ", record.configuration_id, " -> ");
    config.printConfiguration(logger_.getOutputStream());
    logger_.info("***************");
}
