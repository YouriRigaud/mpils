// MPILS: Multi-Phase Iterated Local Search Tuner
//
// Author: Youri Rigaud
// License: GNU GPLv3

#ifndef PARAMETER_SPACE_H
#define PARAMETER_SPACE_H

#include <vector>
#include <string>
#include <variant>

/**
 * @brief Class representing a value that can be of type int, double, or string.
 */
class Value {
    private:
        std::variant<int, double, std::string> value_; ///< Current value.
    
    public:
        /**
         * @brief Construct a Value object from an int.
         * @param value The value as an int.
         */
        Value(int value): value_(value) {}

        /**
         * @brief Construct a Value object from a double.
         * @param value The value as a double.
         */
        Value(double value): value_(value) {}

        /**
         * @brief Construct a Value object from a string.
         * @param value The value as a string.
         */
        Value(const std::string& value): value_(value) {}

        /**
         * @brief Construct a Value object from a C-style string.
         * @param value The value as a C-style string.
         */
        Value(const char *value): value_(std::string(value)) {}

        /**
         * @brief Get the stored value.
         * @return The value as a variant type.
         */
        std::variant<int, double, std::string> getValue() const { return value_; }

        /**
         * @brief Get the value as a string representation.
         * @return The value converted to a string.
         */
        std::string getString() const {
            if (std::holds_alternative<int>(value_)) {
                return std::to_string(std::get<int>(value_));
            } else if (std::holds_alternative<double>(value_)) {
                return std::to_string(std::get<double>(value_));
            } else if (std::holds_alternative<std::string>(value_)) {
                return std::get<std::string>(value_);
            }
            return "";
        }

        /**
         * @brief Equality operator for Value class.
         * @param other The other Value to compare with.
         * @return True if the values are equal, false otherwise.
         */
        bool operator==(const Value& other) const {
            return value_ == other.value_;
        }
};

/**
 * @brief Class representing a parameter with possible values and a default value.
 */
class Parameter{
    private:
        std::string name_;          ///< Name of the parameter.
        std::vector<Value> values_; ///< All the possible values of the parameter.
        Value default_value_;       ///< Default value of the parameter.
        bool is_tuned_ = false;    ///< Flag indicating if the parameter is being tuned.
        bool is_selected_ = false; ///< Flag indicating if the parameter is selected.
        bool is_discarded_ = false;///< Flag indicating if the parameter is discarded.
        bool is_residual_ = true; ///< Flag indicating if the parameter is residual.

    public:
        /**
         * @brief Construct a Parameter object.
         * @param name          Name of the parameter.
         * @param values        Possible values for the parameter.
         * @param default_value Default value for the parameter.
         */
        Parameter(
            const std::string& name,
            const std::vector<Value>& values,
            const Value& default_value
        ): name_(name),
           values_(values),
           default_value_(default_value)
        {}

        /**
         * @brief Get the name of the parameter.
         * @return The name of the parameter.
         */
        std::string getName() const { return name_; }

        /**
         * @brief Get the possible values of the parameter.
         * @return A vector of possible values.
         */
        std::vector<Value> getValues() const { return values_; }

        /**
         * @brief Get the default value of the parameter.
         * @return The default value.
         */
        Value getDefaultValue() const { return default_value_; }

        // Setters and getters for flags
        void setIsTuned(bool is_tuned) { is_tuned_ = is_tuned; }
        bool isTuned() const { return is_tuned_; }

        void setIsSelected(bool is_selected) { is_selected_ = is_selected; }
        bool isSelected() const { return is_selected_; }

        void setIsDiscarded(bool is_discarded) { is_discarded_ = is_discarded; }
        bool isDiscarded() const { return is_discarded_; }

        void setIsResidual(bool is_residual) { is_residual_ = is_residual; }
        bool isResidual() const { return is_residual_; }
};

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

        /** @brief Get a parameter by name */
        Parameter* getParameterByName(const std::string& name) {
            for (auto& param : parameters_) {
                if (param.getName() == name) {
                    return &param;
                }
            }
            return nullptr; // Return nullptr if not found
        }

};

#endif // PARAMETER_SPACE_H