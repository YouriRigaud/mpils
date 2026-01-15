// MPILS: Multi-Phase Iterated Local Search Tuner
//
// Author: Youri Rigaud
// License: GNU GPLv3

#ifndef PARAMETER_SPACE_H
#define PARAMETER_SPACE_H

#include "parameter.h"

#include <vector>
#include <string>

class ParameterSpace {
    private:
        std::vector<Parameter> parameters_; ///< List of parameters in the space
        std::vector<std::pair<std::string, Value>> forbidden_values_; ///< List of forbidden parameter values
        std::vector<std::vector<std::pair<std::string, Value>>> forbidden_tuples_; ///< List of forbidden parameter tuples

    public:
        ParameterSpace(const std::vector<Parameter>& parameters): parameters_(parameters) {}

        /** @brief Get the list of parameters in the space */
        std::vector<Parameter>& getParameters() {
            return parameters_;
        }

        /** @brief Get the list of tuned parameters */
        std::vector<Parameter> getTunedParameters() const {
            std::vector<Parameter> tuned_params;
            for (const auto& param : parameters_) {
                if (param.isTuned()) {
                    tuned_params.push_back(param);
                }
            }
            return tuned_params;
        }

        /** @brief Get the list of selected parameters */
        std::vector<Parameter> getSelectedParameters() const {
            std::vector<Parameter> selected_params;
            for (const auto& param : parameters_) {
                if (param.isSelected()) {
                    selected_params.push_back(param);
                }
            }
            return selected_params;
        }

        /** @brief Get the list of residual parameters */
        std::vector<std::reference_wrapper<Parameter>> getResidualParameters() {
            std::vector<std::reference_wrapper<Parameter>> residual_params;
            for (auto& param : parameters_) {
                if (param.isResidual()) {
                    residual_params.emplace_back(param);
                }
            }
            return residual_params;
        }

        /** @brief Get the list of discarded parameters */
        std::vector<Parameter> getDiscardedParameters() const {
            std::vector<Parameter> discarded_params;
            for (const auto& param : parameters_) {
                if (param.isDiscarded()) {
                    discarded_params.push_back(param);
                }
            }
            return discarded_params;
        }

        /** @brief Get the list of forbidden parameter values */
        std::vector<std::pair<std::string, Value>>& getForbiddenValues() {
            return forbidden_values_;
        }

        /** @brief Get the list of forbidden parameter combinations */
        std::vector<std::vector<std::pair<std::string, Value>>>& getForbiddenTuples() {
            return forbidden_tuples_;
        }

        /** @brief Add a forbidden parameter value */
        void addForbiddenValue(const std::string& param_name, const Value& value) {
            forbidden_values_.emplace_back(param_name, value);
        }

        /** @brief Add a forbidden parameter tuple */
        void addForbiddenTuple(const std::vector<std::pair<std::string, Value>>& tuple) {
            forbidden_tuples_.push_back(tuple);
        }

};

#endif // PARAMETER_SPACE_H