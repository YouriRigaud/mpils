// MPILS: Multi-Phase Iterated Local Search Tuner
//
// Author: Youri Rigaud
// License: GNU GPLv3

#include "../include/pruning.h"
#include "../include/learner.h"

void Pruning::run() {
    logger_.info("Starting pruning phase...");

    std::vector<std::vector<std::pair<std::string, Value>>> forbidden_tuples = extractForbiddenTuples();

    applyPruning(forbidden_tuples);
    

    logger_.info("Pruning phase completed.");
}

void Pruning::writeLearnerFile(const std::string& learner_file) {
    // Implementation to write learner file for pruning phase
    logger_.info("Writing learner file for pruning phase: ", learner_file);
    std::ofstream file(learner_file);
    if (!file.is_open()) {
        logger_.info("Could not open learner file for writing: ", learner_file);
        throw std::runtime_error("Could not open learner file for writing: " + learner_file);
    }

    // Write configurations and their objective values from memory
    for (const auto& config : memory_.getConfigurations()) {
        for (const auto& pair : config.getConfiguration()) {
            file << pair.second.getString() << ", ";
        }
        file << config.getObjective() << "\n";
    }


    file.close();
    logger_.info("Learner file written successfully: ", learner_file);
}

std::vector<std::vector<std::pair<std::string, Value>>> Pruning::extractForbiddenTuples() {
    const std::string learner_file = "tuner_working_dir/pruning/input/learner_file_" + std::to_string(iteration_) + ".txt";
    writeLearnerFile(learner_file);

    logger_.info("Extracting forbidden tuples from memory...");

    double threshold = 0.1;

    std::vector<std::vector<std::pair<std::string, Value>>> forbidden_tuples;

    // Appel du learner
    Options options_learner;
    options_learner.use_lhs = false;
    options_learner.algorithm = "multinomial";

    Learner learner;
    std::string table_name = "iteration_" + std::to_string(iteration_) + "_";
    learner.call_learning_model(learner_file, table_name, options_learner);

    // Fichiers de sortie
    std::vector<std::string> output_files = {
        "tuner_working_dir/pruning/output/" + table_name + "1Option.txt",
        "tuner_working_dir/pruning/output/" + table_name + "2Option.txt"
    };

    for (const auto& file_path : output_files) {
        std::ifstream file(file_path);
        if (!file.is_open()) {
            logger_.info("Failed to open learner output file: ", file_path);
            continue;
        }

        std::string line;
        while (std::getline(file, line)) {
            // Ignorer les lignes vides
            if (line.empty()) continue;

            // Découper la ligne par '#'
            std::vector<std::string> tokens;
            std::stringstream ss(line);
            std::string token;
            while (std::getline(ss, token, '#')) {
                tokens.push_back(token);
            }

            try {
                // Identifier le score à la fin de la ligne (toujours avant le dernier élément ?)
                double score = 0.0;
                if (tokens.size() % 4 == 1) {
                    // Format avec 1 paramètre : Param#ID#Value#Other#Score
                    score = std::stod(tokens[tokens.size() - 1]);
                } else if (tokens.size() % 6 == 2) {
                    // Format avec 2 paramètres : Param1#ID1#Value1#Param2#ID2#Value2#Other#Score
                    score = std::stod(tokens[tokens.size() - 1]);
                } else {
                    // Format non reconnu, ignorer
                    continue;
                }

                if (score > threshold) {
                    std::vector<std::pair<std::string, Value>> tuple;
                    // Extraire tous les paires Param/Value
                    for (size_t i = 0; i + 2 < tokens.size(); i += 3) {
                        const std::string& param_name = tokens[i];
                        Value val = std::stoi(tokens[i + 2]); // adapter selon le type Value
                        tuple.emplace_back(param_name, val);
                    }
                    forbidden_tuples.push_back(tuple);
                }
            } catch (const std::exception& e) {
                logger_.debug("Skipping malformed line in learner output: ", line, " (", e.what(), ")");
                continue;
            }
        }

        file.close();
    }

    logger_.info("Forbidden tuples extracted: ", forbidden_tuples.size());
    return forbidden_tuples;
}


void Pruning::applyPruning(std::vector<std::vector<std::pair<std::string, Value>>>& forbidden_tuples) {
    logger_.info("Applying pruning with ", forbidden_tuples.size(), " forbidden tuples...");

    for (const auto& tuple : forbidden_tuples) {
        if (tuple.size() == 0) {
            continue;
        }

        if (tuple.size() == 1) {
            logger_.debug("Adding forbidden value from single-element tuple: ", tuple[0].first, "=", tuple[0].second.getString());
            parameter_space_.addForbiddenValue(tuple[0].first, tuple[0].second);
        } else {
            logger_.debug("Adding forbidden tuple of size ", tuple.size(), ": ");
            for (const auto& pair : tuple) {
                logger_.debug("  ", pair.first, "=", pair.second.getString());
            }
            parameter_space_.addForbiddenTuple(tuple);
        }

    }

    logger_.info("Pruning applied successfully.");
}