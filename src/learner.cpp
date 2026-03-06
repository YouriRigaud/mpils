#include "../include/learner.h"

#include <cassert>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <set>
#include <map>
#include <cmath>
#include <limits>
#include <random>
#include <mlpack.hpp>
#include <armadillo>
#include <variant>
#include <string>
#include <ilcplex/ilocplex.h>

/**
 * @brief Reads a parameter file and stores parameter names and IDs.
 *
 * @param filename Path to the parameter file to read.
 * @throws std::runtime_error If the file cannot be opened.
 */
void Learner::read_parameter_file(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Unable to open parameter file: " + filename);
    }

    parameter_names.clear();
    parameter_ids.clear();

    std::string line;
    int i = 0;
    while (std::getline(file, line)) {
        if (i == 0) {
            // Skip the first line (file header)
            ++i;
            continue;
        }
        std::stringstream ss(line);
        std::string name, id_str;
        // Try to read <name> <TAB> <id> from the line, skip line if format isn't as expected
        if (!std::getline(ss, name, '\t') || !std::getline(ss, id_str)) {
            continue;
        }
        parameter_names.push_back(name);
        parameter_ids.push_back(id_str);
    }
    file.close();
}

/**
 * @brief Reads a dataset file, cleans it, and updates internal matrices and parameters.
 *
 * @param filename Path to the CSV file to read.
 * @throws std::runtime_error If the file cannot be opened or loaded.
 */
void Learner::read_data_file(const std::string& filename) {
    arma::mat raw_data;
    if (!raw_data.load(filename, arma::csv_ascii)) {
        throw std::runtime_error("Unable to open data file: " + filename);
    }

    // Separate X and y
    x_mat = raw_data.cols(0, raw_data.n_cols - 2);
    y_vec = raw_data.col(raw_data.n_cols - 1);

    remove_duplicates();

    // Remove constant columns
    arma::uvec variable_cols;
    for (size_t j = 0; j < x_mat.n_cols; ++j) {
        arma::rowvec col = x_mat.col(j).t();
        if (arma::any(col != col(0))) {
            variable_cols.insert_rows(variable_cols.n_elem, arma::uvec{j});
        }
    }

    // Store filtered X and y
    x_mat = x_mat.cols(variable_cols);

    // Update parameter names & ids
    std::vector<std::string> filtered_names, filtered_ids;
    for (auto idx : variable_cols) {
        filtered_names.push_back(parameter_names[idx]);
        filtered_ids.push_back(parameter_ids[idx]);
    }
    parameter_names = filtered_names;
    parameter_ids = filtered_ids;
}

/**
 * @brief Removes duplicate rows from the dataset.
 *
 */
void Learner::remove_duplicates() {
    arma::mat unique_x;
    arma::vec unique_y;

    for (size_t i = 0; i < x_mat.n_rows; ++i)
    {
        arma::rowvec current_row = x_mat.row(i);
        double current_y = y_vec(i);
        bool is_duplicate = false;

        // Compare against all previously stored unique rows
        for (size_t j = 0; j < unique_x.n_rows; ++j)
        {
            // Check if features match AND labels match
            if (arma::all(arma::vectorise(unique_x.row(j) == current_row)) && unique_y(j) == current_y) {
                is_duplicate = true;
                break; // Found a duplicate, no need to check further
            }
        }

        // If no duplicate found, add this row to the unique dataset
        if (!is_duplicate)
        {
            unique_x.insert_rows(unique_x.n_rows, current_row);
            unique_y.insert_rows(unique_y.n_rows, arma::vec({current_y}));
        }
    }
    x_mat = unique_x;
    y_vec = unique_y;
}

/**
 * @brief Executes the full learning pipeline on solver parameter data.
 *
 *   1. Reads parameter and dataset files.
 *   2. Performs ANOVA filtering.
 *   3. (Optional) Generates Latin Hypercube Samples (LHS) and applies CPLEX.
 *   4. Trains either a multinomial logistic regression or a Gaussian Bayesian model.
 *   5. Selects statistically significant coefficients.
 *   6. Saves results to output files.
 *
 * @param filepath   Path to the dataset file (CSV).
 * @param tablepath  Path to the directory or table file where results should be written.
 * @param options    Struct containing algorithm options and hyperparameters.
 *
 * @note Debugging information is printed to stdout at each stage of the pipeline.
 */
void Learner::call_learning_model(const std::string& filepath, const std::string& tablepath, Options options) {
    // Step 1: Read parameter definitions (names & IDs), and read dataset and preprocess (duplicates, constant columns)
    read_parameter_file("tuner_working_dir/parameter_ids.txt");
    std::cout << "read_parameter_file done" << std::endl;

    read_data_file(filepath);
    std::cout << "read_data_file done" << std::endl;

    // Step 2: Perform ANOVA filtering to reduce parameter space
    perform_two_way_anova_filtering(options.strict_p_value, options.loose_p_value);
    std::cout << "perform_two_way_anova_filtering done" << std::endl;

    reset_feature_metadata();

    // Step 3 (optional): Latin Hypercube Sampling & CPLEX evaluation
    if (options.use_lhs) {
       generate_lhs_samples(y_vec.n_elem);
       std::cout << "generate_lhs_samples done" << std::endl;

       apply_cplex_to_lhs(tablepath);
       std::cout << "apply_cplex_to_lhs done" << std::endl;
    }

    // Step 4: Train the selected learning model
    if (options.algorithm == "multinomial") {
        if (x_mat.n_cols == 0) {
            std::cerr << "[FATAL] No features left after preprocessing. Aborting learning." << std::endl;
            return;
        }
        
        train_multinomial_logistic_regression(options.class_count);
        std::cout << "train_multinomial_logistic_regression done" << std::endl;

        if (x_mat.n_cols >= 2) {
            train_multinomial_logistic_regression_with_interaction(options.class_count, options.max_interactions_count);
            std::cout << "train_multinomial_logistic_regression_with_interaction done" << std::endl;
        } else {
            std::cerr << "[INFO] Skipping interaction model (not enough features)." << std::endl;
        }
    } else {
        train_gaussian_bayesian_model();
        std::cout << "train_gaussian_bayesian_model done" << std::endl;
        train_gaussian_bayesian_model_with_interaction(options.max_interactions_count);
        std::cout << "train_gaussian_bayesian_model_with_interaction done" << std::endl;
    }

    // Step 5: Select significant coefficients from the trained models
    select_significant_coefficients(options.significance_threshold, options.algorithm);
    select_significant_coefficients(options.significance_threshold, options.algorithm, true);
    std::cout << "select_significant_coefficients done" << std::endl;

    // Step 7: Save results (individual and interaction terms) for external analysis
    save_results(tablepath, "1Option.txt", one_hot_values, "tuner_working_dir/pruning/output/");
    save_results(tablepath, "2Option.txt", interaction_pairs, "tuner_working_dir/pruning/output/");
}

/**
 * @brief Performs two-way ANOVA filtering on parameters to reduce feature space.
 *
 * @param strict_p_value Threshold F value for strong significance (p <= 0.01).
 * @param loose_p_value  Threshold F value for weaker significance (p <= 0.05).
 */
void Learner::perform_two_way_anova_filtering(double strict_p_value, double loose_p_value) {
    size_t num_samples = x_mat.n_rows;
    size_t num_params = x_mat.n_cols;

    std::vector<double> f_statistics(num_params, 0.0);

    double overall_mean = arma::mean(y_vec);

    // Loop through each parameter column
    for (size_t param_idx = 0; param_idx < num_params; ++param_idx) {
        std::unordered_map<int, std::vector<double>> groups;

        // Group y values by parameter value
        for (size_t i = 0; i < num_samples; ++i) {
            int param_val = static_cast<int>(x_mat(i, param_idx));
            groups[param_val].push_back(y_vec(i));
        }

        size_t df_between = groups.size() - 1;
        size_t df_within = num_samples - groups.size();

        if (df_between <= 0 || df_within <= 0) {
            f_statistics[param_idx] = 0.0;
            continue;
        }

        // Compute between-group and within-group sum of squares
        double ssb = 0.0;
        double ssw = 0.0;

        for (const auto& [val, samples] : groups) {
            double group_mean = std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();
            ssb += samples.size() * std::pow(group_mean - overall_mean, 2);

            for (const auto& sample : samples) {
                ssw += std::pow(sample - group_mean, 2);
            }
        }

        // Compute F statistic = MSB / MSW
        double msb = ssb / df_between;
        double msw = ssw / df_within;
        f_statistics[param_idx] = (msw == 0.0) ? 0.0 : msb / msw;
    }

    // Default thresholds (if not provided externally)
    double threshold_strict = 6.63; // p <= 0.01
    double threshold_loose = 3.94;  // p <= 0.05

    std::vector<size_t> selected_indices;
    size_t num_above_strict = 0;
    size_t num_above_loose = 0;
    std::vector<double> max_fs = {0, 0};
    double threshold = threshold_strict;

    // Count how many parameters exceed strict/loose thresholds
    for (size_t i = 0; i < num_params; ++i) {
        if (f_statistics[i] >= threshold_strict) {
            ++num_above_strict;
            ++num_above_loose;
            if (num_above_strict == 3) {
                break;
            }
        }
        else if (f_statistics[i] >= threshold_loose) {
            ++num_above_loose;
        }
        // Keep track of top 2 maximum F-statistics
        for (size_t j = 0; j < 2; ++j) {
            if (f_statistics[i] >= max_fs[j]) {
                max_fs.pop_back();
                max_fs.insert(max_fs.begin() + j, f_statistics[i]);
                break;
            }
        }
    }

    // Relax threshold if not enough parameters meet strict criteria
    if (num_above_strict < 3) {
        threshold = threshold_loose;
    }
    if (num_above_loose < 2) {
        threshold = max_fs[1];
    }

    // Select final parameter indices based on computed threshold
    for (size_t i = 0; i < num_params; ++i) {
        if (f_statistics[i] >= threshold) {
            selected_indices.push_back(i);
        }
    }

    if (selected_indices.empty()) {
        std::cerr << "[WARNING] ANOVA removed all parameters. Keeping original feature set." << std::endl;
        return;
    }


    // Build new filtered x_mat
    arma::mat new_x(num_samples, selected_indices.size());
    for (size_t c = 0; c < selected_indices.size(); ++c) {
        new_x.col(c) = x_mat.col(selected_indices[c]);
    }
    x_mat = new_x;

    // Update names and IDs
    std::vector<std::string> filtered_names;
    std::vector<std::string> filtered_ids;
    for (auto idx : selected_indices) {
        filtered_names.push_back(parameter_names[idx]);
        filtered_ids.push_back(parameter_ids[idx]);
    }
    parameter_names = filtered_names;
    parameter_ids = filtered_ids;
    remove_duplicates();
}

/**
 * @brief Generate Latin Hypercube Sampling (LHS) samples from the parameter space.
 *
 * @param n_samples Number of samples to generate.
 *
 * @note The generated samples are stored in `lhs_samples_mat`.
 */
void Learner::generate_lhs_samples(size_t n_samples) {
   size_t num_params = x_mat.n_cols;
   double n_combinations = pow(4, num_params);

    // Ensure we donâ€™t request more samples than possible combinations
   if (n_combinations < n_samples) n_samples = n_combinations;
   lhs_samples_mat.set_size(n_samples, num_params);

   std::random_device rd;
   std::mt19937 gen(rd());
   std::uniform_real_distribution<> dis(0.0, 1.0);

   // Generate samples for each parameter
   for (size_t param_idx = 0; param_idx < num_params; ++param_idx) {
       // Identify unique categorical options for this parameter
       std::set<int> unique_vals_set;
       for (size_t i = 0; i < x_mat.n_rows; ++i) {
           unique_vals_set.insert(static_cast<int>(x_mat(i, param_idx)));
       }
       std::vector<int> options(unique_vals_set.begin(), unique_vals_set.end());
       size_t num_options = options.size();

       // Generate stratified samples in [0,1)
       std::vector<double> stratified_samples(n_samples);
       double interval = 1.0 / static_cast<double>(n_samples);
       for (size_t i = 0; i < n_samples; ++i) {
           double lower = i * interval;
           stratified_samples[i] = lower + dis(gen) * interval;
       }

       // Shuffle:
       std::shuffle(stratified_samples.begin(), stratified_samples.end(), gen);

       // Map to categorical values:
       for (size_t i = 0; i < n_samples; ++i) {
           int idx = static_cast<int>(stratified_samples[i] * num_options);
           if (idx >= static_cast<int>(num_options)) idx = static_cast<int>(num_options) - 1;
           lhs_samples_mat(i, param_idx) = static_cast<double>(options[idx]);
       }
   }
}

/**
 * @brief Generate a CPLEX .prm configuration file from a given parameter configuration.
 *
 * @param config          A row vector representing parameter values.
 * @param parameter_names Names of the parameters corresponding to values in config.
 * @param prm_filename    Path to the output .prm file.
 *
 * @throws std::runtime_error If the file cannot be opened for writing.
 *
 * @note Adds two fixed parameters:
 *       - CPXPARAM_Threads = 4
 *       - CPXPARAM_TimeLimit = 30
 */
void Learner::generate_prm_file(const arma::rowvec& config,
                      const std::vector<std::string>& parameter_names,
                      const std::string& prm_filename) {
   std::ofstream prm_file(prm_filename, std::ios::trunc);
   if (!prm_file.is_open()) {
       throw std::runtime_error("Failed to open prm file for writing.");
   }

   // Write parameter values line by line
   for (size_t i = 0; i < config.n_elem; ++i) {
       prm_file << parameter_names[i] << " " << config(i) << "\n";
   }

   prm_file << "CPXPARAM_Threads 4\n";
   prm_file << "CPXPARAM_TimeLimit 30\n";

   prm_file.close();
}

/**
 * @brief Apply CPLEX solver to each configuration in the LHS sample matrix.
 *
 * @param tablepath Path containing information used to locate the instance file.
 *
 * @note If CPLEX fails, a fallback gap value of 100 is recorded.
 * @note Uses hardcoded paths for problem instances and temporary prm files.
 */
void Learner::apply_cplex_to_lhs(const std::string& tablepath) {
    // Extract problem name from tablepath using underscores
   size_t first = tablepath.find('_');
   size_t last = tablepath.rfind('_');
   std::string problem_name = "/home/rigayour/MPILS/Instances/" + tablepath.substr(first + 1, last - first - 1);

   // Iterate over each LHS configuration
   for (size_t i = 0; i < lhs_samples_mat.n_rows; ++i) {
       IloEnv env;
       try {
           IloModel model(env);
           IloCplex cplex(model);
           cplex.setOut(env.getNullStream());
           cplex.setWarning(env.getNullStream());
           IloObjective obj(env);
           IloNumVarArray x(env);
           IloRangeArray cons(env);

           std::cout << "\n=== Running config " << i << " ===" << std::endl;

           cplex.importModel(model, problem_name.c_str(), obj, x, cons);

           arma::rowvec config = lhs_samples_mat.row(i);

           generate_prm_file(config, parameter_names);
           cplex.readParam("/home/rigayour/MPILS/param_files/temp_config.prm");

           double gap;
           if (!cplex.solve()) {
               env.error() << "Failed to optimize LP.\n";
               gap = 1.0; // fallback high gap
           } else {
               double obj_val = cplex.getObjValue();
               double best_bound = cplex.getBestObjValue();
               gap = std::abs(obj_val - best_bound) / std::max(std::abs(obj_val), 1e-6);
               env.out() << "Gap: " << gap << "\n";
           }

           // Store configuration and corresponding gap (as percentage)
           x_mat.insert_rows(x_mat.n_rows, config);
           y_vec.insert_rows(y_vec.n_elem, arma::vec(1).fill(gap*100));
       } catch (IloException& e) {
           std::cerr << "Concert exception: " << e << std::endl;
           x_mat.insert_rows(x_mat.n_rows, lhs_samples_mat.row(i));
           y_vec.insert_rows(y_vec.n_elem, arma::vec(1).fill(100));
       } catch (...) {
           std::cerr << "Unknown exception occurred.\n";
           x_mat.insert_rows(x_mat.n_rows, lhs_samples_mat.row(i));
           y_vec.insert_rows(y_vec.n_elem, arma::vec(1).fill(100));
       }
       env.end();
   }
}

/**
 * @brief Discretize continuous labels into bins (classes).
 *
 * @param y            Vector of continuous label values.
 * @param class_count  Number of bins (classes) to split labels into.
 * @return arma::uvec  Vector of class indices in range [0, class_count-1].
 */
arma::uvec Learner::bin_labels(const arma::vec& y, int class_count) {
    // Sort labels to compute quantile thresholds
    arma::vec y_sorted = arma::sort(y);
    size_t n = y.n_elem;

    // Store class thresholds (quantiles)
    std::vector<double> class_thresholds(class_count - 1);

    for (size_t i = 0; i < class_count - 1; ++i) {
        // Compute threshold at (i+1)/class_count quantile
        size_t idx = static_cast<size_t>(std::floor((i + 1) * n / static_cast<double>(class_count)));
        idx = std::min(idx, n - 1);
        class_thresholds[i] = y_sorted(idx);
    }

    arma::uvec classes(n);

    // Assign each value to a class based on thresholds
    for (size_t i = 0; i < n; ++i) {
        double val = y(i);
        for (size_t j = 0; j < class_count - 1; ++j) {
            if (val < class_thresholds[j]) {
                classes(i) = j;
                break;
            }
        }

        // If value is above all thresholds, assign to last class
        if (val >= class_thresholds[class_count - 2]) {
            classes(i) = class_count - 1;
        }
    }
    return classes;
}

/**
 * @brief Perform one-hot encoding of categorical parameters.
 *
 * @param x_matrix Matrix of parameter values (rows = samples, cols = parameters).
 * @param init     If true, store one-hot mapping information into `one_hot_values`.
 * @return arma::mat  One-hot encoded matrix of shape (#categories-#samples, #samples).
 */
arma::mat Learner::one_hot_encode(const arma::mat& x_matrix, bool init) {
    if (init) {
        one_hot_values.clear();
    }
    size_t num_samples = x_matrix.n_rows;
    size_t num_params = x_matrix.n_cols;

    size_t total_one_hot_cols = 0;
    std::vector<std::vector<double>> unique_vals_per_param(num_params);

    // Identify unique values per parameter and count total one-hot columns
    for (size_t p = 0; p < num_params; ++p) {
        arma::colvec param_col = x_matrix.col(p);
        arma::vec unique_vals = arma::unique(param_col);
        std::vector<double> sorted_unique = arma::conv_to<std::vector<double>>::from(arma::sort(unique_vals));
        unique_vals_per_param[p] = sorted_unique;

        // For one-hot encoding, we drop the first category -> (size - 1) columns
        total_one_hot_cols += sorted_unique.size() - 1;
    }

    // Allocate result matrix (rows = one-hot features, cols = samples)
    arma::mat x_one_hot(total_one_hot_cols, num_samples, arma::fill::zeros);

    size_t col_offset = 0;
    // Generate one-hot features
    for (size_t p = 0; p < num_params; ++p) {
        const auto& unique_vals = unique_vals_per_param[p];

        for (size_t k = 1; k < unique_vals.size(); ++k) { // start from 1 to drop first
            double category = unique_vals[k];

            // Assign 1 if sample belongs to this category
            for (size_t s = 0; s < num_samples; ++s) {
                if (x_matrix(s, p) == category) {
                    x_one_hot(col_offset + k - 1, s) = 1.0;
                }
            }

            // If init == true, store mapping info (param name, id, category value)
            if (init) {
                one_hot_values.push_back({parameter_names[p], parameter_ids[p], std::to_string(static_cast<int>(category))});
            }
        }
        col_offset += unique_vals.size() - 1;
    }

    if (init) {
        assert(x_one_hot.n_rows == one_hot_values.size());
    }

    return x_one_hot;
}

/**
 * @brief Generate pairwise interaction features between one-hot encoded features.
 *
 * @param x_train                 One-hot encoded feature matrix
 *                                (rows = features, cols = samples).
 * @param max_interactions_count  Maximum number of interactions to generate.
 * @return arma::mat              Matrix of interaction features
 *                                (rows = interactions, cols = samples).
 */
arma::mat Learner::create_interaction(const arma::mat& x_train, int max_interactions_count) {
    interaction_pairs.clear();  // IMPORTANT : vide avant de remplir

    arma::mat x_train_interactions; // [num_interaction_features x num_samples]

    size_t num_features = x_train.n_rows;
    size_t num_samples = x_train.n_cols;

    assert(x_train.n_rows == one_hot_values.size());

    // Loop through all pairs of features (no repeats)
    for (size_t i = 0; i < num_features - 1; ++i) {
        for (size_t j = i + 1; j < num_features; ++j) {
            if (i >= one_hot_values.size() || j >= one_hot_values.size()) {
                continue;
            }

            // Skip if features come from the same parameter (same parameter_id)
            if (one_hot_values[i][1] == one_hot_values[j][1]) {
                continue;
            }

            // Compute interaction term as elementwise product of the two feature rows
            arma::rowvec interaction = x_train.row(i) % x_train.row(j);

            // Append to interaction matrix
            x_train_interactions.insert_rows(x_train_interactions.n_rows, interaction);

            // Record interaction details (names, ids, values for both features)
            std::vector<std::string> interaction_pair;
            interaction_pair.insert(interaction_pair.end(), one_hot_values[i].begin(), one_hot_values[i].end());
            interaction_pair.insert(interaction_pair.end(), one_hot_values[j].begin(), one_hot_values[j].end());
            interaction_pairs.push_back(interaction_pair);

            // Stop if weâ€™ve reached the interaction count limit
            if (interaction_pairs.size() == max_interactions_count) {
                return x_train_interactions;
            }
        }
    }
    return x_train_interactions;
}

/**
 * @brief Train a multinomial logistic regression model on one-hot encoded features.
 *
 * @param class_count Number of bins (classes) for discretization of labels.
 */
void Learner::train_multinomial_logistic_regression(int class_count) {
    // Convertir y_vec en classes discrètes
    arma::Row<size_t> y_classes = arma::conv_to<arma::Row<size_t>>::from(bin_labels(y_vec, class_count));

    // One-hot encoding
    arma::mat x_train_full = one_hot_encode(x_mat, true); // [features x samples]
    assert(x_train_full.n_rows == one_hot_values.size());

    // ===============================
    // Filtrage automatique des features constantes
    // ===============================
    arma::uvec active_feature_indices;

    for (size_t i = 0; i < x_train_full.n_rows; ++i)
    {
        arma::rowvec feature_row = x_train_full.row(i);
        arma::vec unique_vals = arma::unique(feature_row.t());
        if (unique_vals.n_elem > 1)
        {
            active_feature_indices.insert_rows(active_feature_indices.n_elem, arma::uvec({i}));
        }
    }

    arma::mat x_train = x_train_full.rows(active_feature_indices);
    std::vector<std::vector<std::string>> active_one_hot_values;
    for (size_t idx = 0; idx < active_feature_indices.n_elem; ++idx)
    {
        active_one_hot_values.push_back(one_hot_values[active_feature_indices[idx]]);
    }   

    std::cerr << "[DEBUG] x_train.n_rows = " << x_train.n_rows
              << ", x_train.n_cols = " << x_train.n_cols << std::endl;
    std::cerr << "[DEBUG] y.n_elem = " << y_classes.n_elem << std::endl;

    size_t num_classes;

    // ======== Ici on ajoute le remapping obligatoire ========
    arma::Row<size_t> unique = arma::unique(y_classes);
    std::unordered_map<size_t, size_t> remap;
    for (size_t i = 0; i < unique.n_elem; ++i)
        remap[unique[i]] = i;

    for (arma::uword i = 0; i < y_classes.n_elem; ++i)
        y_classes[i] = remap[y_classes[i]];

    num_classes = unique.n_elem;

    std::cerr << "[DEBUG] Labels remappés = " << y_classes << std::endl;
    std::cerr << "[DEBUG] NumClasses = " << num_classes << std::endl;
    // ========================================================

    std::cerr << "[DEBUG] y_classes min/max = "
          << y_classes.min() << " / "
          << y_classes.max() << std::endl;

    std::cerr << "[DEBUG] labels = " << y_classes;


    assert(y_classes.min() == 0);
    assert(y_classes.max() < num_classes);

    mlr_model = mlpack::regression::SoftmaxRegression();

    // Vérification avant entraînement
    assert(x_train.n_cols == y_classes.n_elem);
    if (x_train.n_rows != active_one_hot_values.size())
    {
        std::cerr << "[FATAL] Feature matrix mismatch after filtering: "
                  << x_train.n_rows << " vs metadata " << active_one_hot_values.size() << std::endl;
        std::abort();
    }

    std::cout << "data size = "
          << x_train.n_rows << " x " << x_train.n_cols << std::endl;

    std::cout << "labels size = "
          << y_classes.n_elem << std::endl;

    std::cout << "numClasses = "
          << num_classes << std::endl;

    std::cout << "labels min/max = "
          << y_classes.min() << " / "
          << y_classes.max() << std::endl;

    mlr_model.Train(x_train, y_classes, num_classes);

    // Remplacer le mapping one_hot_values par la version filtrée
    one_hot_values = active_one_hot_values;
}

/**
 * @brief Train a multinomial logistic regression model with feature interactions.
 *
 * @param class_count            Number of bins (classes) for discretization of labels.
 * @param max_interactions_count Maximum number of interaction features to include.
 */
void Learner::train_multinomial_logistic_regression_with_interaction(int class_count, int max_interactions_count) {
    // Convertir y_vec en classes discrètes
    arma::Row<size_t> y_classes = arma::conv_to<arma::Row<size_t>>::from(bin_labels(y_vec, class_count));

    // One-hot encoding complet
    arma::mat x_train_full = one_hot_encode(x_mat, true); // [features x samples]
    assert(x_train_full.n_rows == one_hot_values.size());

    // ===============================
    // Filtrage automatique des features constantes avant interaction
    // ===============================
    arma::uvec active_feature_indices;

    for (size_t i = 0; i < x_train_full.n_rows; ++i)
    {
        arma::rowvec feature_row = x_train_full.row(i);
        arma::vec unique_vals = arma::unique(feature_row.t());
        if (unique_vals.n_elem > 1)
        {
            active_feature_indices.insert_rows(active_feature_indices.n_elem, arma::uvec({i}));
        }
    }

    arma::mat x_train = x_train_full.rows(active_feature_indices);

    std::vector<std::vector<std::string>> active_one_hot_values;
    for (size_t idx = 0; idx < active_feature_indices.n_elem; ++idx)
    {
        active_one_hot_values.push_back(one_hot_values[active_feature_indices[idx]]);
    }

    // Mettre à jour one_hot_values pour interaction
    std::vector<std::vector<std::string>> backup_one_hot_values = one_hot_values;
    one_hot_values = active_one_hot_values;

    // ===============================
    // Création des interactions
    // ===============================
    arma::mat x_train_interactions = create_interaction(x_train, max_interactions_count);
    
    if (x_train_interactions.n_rows == 0 || x_train_interactions.n_cols == 0) {
        std::cerr << "[INFO] No valid interaction features generated. "
                  << "Skipping interaction model training." << std::endl;

        // IMPORTANT : restaurer le mapping
        one_hot_values = backup_one_hot_values;
        return;
    }

    assert(x_train_interactions.n_rows == interaction_pairs.size());

    size_t num_classes;
    // ======== Ici on ajoute le remapping obligatoire ========
    arma::Row<size_t> unique = arma::unique(y_classes);
    std::unordered_map<size_t, size_t> remap;
    for (size_t i = 0; i < unique.n_elem; ++i)
        remap[unique[i]] = i;

    for (arma::uword i = 0; i < y_classes.n_elem; ++i)
        y_classes[i] = remap[y_classes[i]];

    num_classes = unique.n_elem;

    std::cerr << "[DEBUG] Labels remappés = " << y_classes << std::endl;
    std::cerr << "[DEBUG] NumClasses = " << num_classes << std::endl;
    // ========================================================


    assert(y_classes.min() == 0);
    assert(y_classes.max() < num_classes);

    mlr_model_with_interaction = mlpack::regression::SoftmaxRegression();

    std::cerr << "[DEBUG] X_interactions.n_rows = " << x_train_interactions.n_rows
              << ", X_interactions.n_cols = " << x_train_interactions.n_cols << std::endl;
    std::cerr << "[DEBUG] y.n_elem = " << y_classes.n_elem << std::endl;

    // Vérification avant entraînement
    assert(x_train_interactions.n_cols == y_classes.n_elem);
    if (x_train_interactions.n_rows != interaction_pairs.size())
    {
        std::cerr << "[FATAL] Interaction feature matrix mismatch: "
                  << x_train_interactions.n_rows << " vs metadata " << interaction_pairs.size() << std::endl;
        std::abort();
    }


    mlr_model_with_interaction.Train(x_train_interactions, y_classes, num_classes);

    // Restaurer le mapping complet pour référence future
    one_hot_values = backup_one_hot_values;
}

/**
 * @brief Train a Gaussian Bayesian (Bayesian Linear Regression) model.
 */
void Learner::train_gaussian_bayesian_model() {
    arma::rowvec y_train = arma::conv_to<arma::rowvec>::from(y_vec);

    // Transpose x_matrix to [features, samples] as mlpack expects columns = samples
    arma::mat x_train = one_hot_encode(x_mat, true);  // [features, samples]

    // By default, mlpack assumes Gaussian distributions for continuous features in NaiveBayesClassifier
    mlpack::regression::BayesianLinearRegression model;
    model.Train(x_train, y_train);
    gbm_model = model;
}

/**
 * @brief Train a Gaussian Bayesian (Bayesian Linear Regression) model with interactions.
 *
 * @param max_interactions_count Maximum number of interaction features to include.
 */
void Learner::train_gaussian_bayesian_model_with_interaction(int max_interactions_count) {
    arma::rowvec y_train = arma::conv_to<arma::rowvec>::from(y_vec);

    // Transpose x_matrix to [features, samples] as mlpack expects columns = samples
    arma::mat x_train = one_hot_encode(x_mat);  // [features, samples]

    arma::mat x_train_interactions = create_interaction(x_train, max_interactions_count);
   
    // By default, mlpack assumes Gaussian distributions for continuous features in NaiveBayesClassifier
    mlpack::regression::BayesianLinearRegression model;
    model.Train(x_train_interactions, y_train);
    gbm_model_with_interaction = model;
}

/**
 * @brief Select significant coefficients from a trained model.
 *
 * @param significance_threshold  Fraction of the maximum coefficient to use as cutoff.
 * @param algorithm               Name of the algorithm ("bayesian" or "multinomial").
 * @param is_interaction          If true, filter interaction features; else filter base features.
 */
void Learner::select_significant_coefficients( double significance_threshold, const std::string& algorithm, const bool& is_interaction) {
    assert(!algorithm.empty());

    arma::vec coefs;

    // === Extract coefficients
    if (algorithm == "bayesian") {
        coefs = is_interaction
            ? arma::vec(gbm_model_with_interaction.Omega())
            : arma::vec(gbm_model.Omega());
    } else {
        arma::mat params = is_interaction
            ? arma::mat(mlr_model_with_interaction.Parameters())
            : arma::mat(mlr_model.Parameters());

        if (params.n_rows == 0 || params.n_cols <= 1)
            return;

        // mlpack: (numClasses - 1) x (numFeatures + 1)
        // → on transpose
        arma::mat P = params.t();

        // enlever le biais (dernière ligne)
        arma::mat W = P.rows(0, P.n_rows - 2);

        // DEBUG
        std::cerr << "[DEBUG] W size = "
                  << W.n_rows << " x " << W.n_cols << std::endl;

        // importance par feature = max |coef| sur les classes
        coefs = arma::max(arma::abs(W), 1);

        std::cerr << "[DEBUG] coefs.n_elem = "
                  << coefs.n_elem << std::endl;


    }

    if (coefs.is_empty())
        return;

    auto& target = is_interaction ? interaction_pairs : one_hot_values;

    // === SAFETY: align sizes
    size_t safe_size = std::min(static_cast<size_t>(coefs.n_elem), target.size());
    if (safe_size == 0)
        return;

    if (coefs.n_elem != target.size()) {
        std::cerr << "[WARNING] Coefs/metadata size mismatch: "
                  << coefs.n_elem << " vs " << target.size()
                  << " (using first " << safe_size << ")" << std::endl;
    }

    arma::vec safe_coefs = coefs.head(safe_size);

    // === Significance filtering
    double max_abs = arma::abs(safe_coefs).max();
    if (max_abs <= 0.0)
        return;

    double threshold = significance_threshold * max_abs;
    arma::uvec indices = arma::find(arma::abs(safe_coefs) >= threshold);

    if (indices.is_empty())
        return;

    // === Build filtered metadata
    std::vector<std::vector<std::string>> filtered;
    filtered.reserve(indices.n_elem);

    for (arma::uword k = 0; k < indices.n_elem; ++k) {
        size_t idx = indices[k];
        if (idx >= safe_size)
            continue;

        auto entry = target[idx];
        entry.emplace_back(std::to_string(safe_coefs[idx]));
        entry.emplace_back(std::to_string(safe_coefs[idx] / max_abs));
        filtered.emplace_back(std::move(entry));
    }

    target = std::move(filtered);
}


/**
 * @brief Save analysis results to a file.
 *
 * @param tablepath  Path or subfolder name.
 * @param filename   Name of the output file.
 * @param results    Results to save.
 * @param basePath   Base path where output files will be stored.
 *
 * @note If the file cannot be opened, an error is printed and the function returns.
 */
void Learner::save_results(const std::string& tablepath, const std::string& filename, const std::vector<std::vector<std::string>>& results, const std::string& basePath) {
    std::ofstream outfile(basePath + tablepath + filename);
    if (!outfile) {
        std::cerr << "Error opening file for writing.\n";
        return;
    }

    // Write each row of results, separating columns with '#'
    for (const auto& row : results) {
        for (size_t i = 0; i < row.size(); ++i) {
            outfile << row[i];
            if (i != row.size() - 1) {
                outfile << "#";
            }
        }
        outfile << "\n";
    }

    outfile.close();
    std::cout << "Completed analysis of regression. Results saved at " << basePath << tablepath << filename << "." << std::endl;
}

void Learner::reset_feature_metadata() {
    one_hot_values.clear();
    interaction_pairs.clear();
}
