 #include <armadillo>

#include <string>
#include <variant>
#include <vector>

#include <ilcplex/ilocplex.h>
#include <mlpack.hpp>

#include <mlpack/methods/softmax_regression/softmax_regression.hpp>
#include <mlpack/methods/bayesian_linear_regression/bayesian_linear_regression.hpp>

struct Options {
    // algorithm can be multinomial, bayesian or mixed
    // multinomial : standard multinomial logistic regression model
    // bayesian : bayesian ridge regression model
    std::string algorithm = "multinomial";
    bool use_lhs = false;
    int class_count = 5;
    double significance_threshold = 0.1;
    int max_interactions_count = 30;
    double strict_p_value = 0.01;
    double loose_p_value = 0.05;
};

class Learner {
public:
    arma::mat x_mat;
    arma::vec y_vec;

    std::vector<std::string> parameter_names;
    std::vector<std::string> parameter_ids;

    arma::mat lhs_samples_mat;

    mlpack::regression::SoftmaxRegression mlr_model;
    mlpack::regression::SoftmaxRegression mlr_model_with_interaction;

    mlpack::regression::BayesianLinearRegression gbm_model;
    mlpack::regression::BayesianLinearRegression gbm_model_with_interaction;

    std::vector<std::vector<std::string>> one_hot_values;

    std::vector<std::vector<std::string>> interaction_pairs;

    void read_parameter_file(const std::string& filename);
    void read_data_file(const std::string& filename);
    void call_learning_model(const std::string& filepath, const std::string& tablepath, Options options = Options());
    void perform_two_way_anova_filtering(double strict_p_value, double loose_p_value);
    void generate_lhs_samples(size_t n_samples);
    void apply_cplex_to_lhs(const std::string& tablepath);
    void train_multinomial_logistic_regression(int class_count);
    void train_multinomial_logistic_regression_with_interaction(int class_count, int max_interactions_count);
    void train_gaussian_bayesian_model();
    void train_gaussian_bayesian_model_with_interaction(int max_interactions_count);
    void select_significant_coefficients(double significance_threshold, const std::string& algorithm, const bool& is_interaction = false);
    void save_results(const std::string& tablepath, const std::string& filename, const std::vector<std::vector<std::string>>& results, const std::string& basePath);
    void reset_feature_metadata();

protected:
    void remove_duplicates();
    void generate_prm_file(const arma::rowvec& config,
                       const std::vector<std::string>& parameter_names,
                       const std::string& prm_filename = "/home/corbmath/MPILS/param_files/temp_config.prm");
    arma::uvec bin_labels(const arma::vec& y, int class_count);
    arma::mat one_hot_encode(const arma::mat& x_matrix, bool init = false);
    arma::mat create_interaction(const arma::mat& x_matrix, int max_interactions_count);
};