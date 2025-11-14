/**
 * @file parameter.h
 * @brief Represents a parameter with possible values and a default value.
 *
 * The Parameter class encapsulates a parameter's name, its possible values,
 * and a default value. It supports different types of values through the
 * Value class, which can hold int, double, or string types.
 *
 * Example usage:
 * @code
 * Parameter param("max_iterations", {Value(10), Value(20), Value(30)}, Value(10));
 * std::cout << "Parameter name: " << param.getName() << std::endl;
 * @endcode
 *
 * @author Youri Rigaud
 * @copyright Copyright 2025 Youri Rigaud. All rights reserved.
 *            This software is licensed under the GNU General Public License v3.0.
 *            See the accompanying LICENSE file for full details.
 */

#ifndef PARAMETER_H
#define PARAMETER_H

#include <string>
#include <variant>
#include <vector>

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
        std::string getString() const;

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
};

#endif // PARAMETER_H