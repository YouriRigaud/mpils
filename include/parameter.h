// MPILS: Multi-Phase Iterated Local Search Tuner
//
// Author: Youri Rigaud
// License: GNU GPLv3

#ifndef PARAMETER_H
#define PARAMETER_H

#include <string>
#include <variant>
#include <vector>

class Value {
    private:
        std::variant<int, double, std::string> value_; // The value in a string format to deal with int, double, bool and string
    
    public:
        Value(int value): value_(value) {}
        Value(double value): value_(value) {}
        Value(const std::string& value): value_(value) {}
        Value(const char *value): value_(std::string(value)) {}


        std::variant<int, double, std::string> getValue() const { return value_; }
        std::string getString() const; // Get the value as a string
};

class Parameter{
    private:
        std::string name_; // Name of the parameter
        std::vector<Value> values_; // All the possible values of the parameter
        Value default_value_; // Default value of the parameter

    public:
        Parameter(
            const std::string& name,
            const std::vector<Value>& values,
            const Value& default_value
        ): name_(name),
           values_(values),
           default_value_(default_value)
        {}

        std::string getName() const { return name_; }
        std::vector<Value> getValues() const { return values_; }
        Value getDefaultValue() const { return default_value_; }
};

#endif // PARAMETER_H