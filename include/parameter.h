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
        std::variant<int, double, std::string> value; // The value in a string format to deal with int, double, bool and string
    
    public:
        Value(int val): value(val) {}
        Value(double val): value(val) {}
        Value(const std::string& val): value(val) {}
        Value(const char *val): value(std::string(val)) {}

        int getInt() const {return std::get<int>(value);}
        double getDouble() const {return std::get<double>(value);}
        std::string getString() const {return std::get<std::string>(value);}
};

class Parameter{
    protected:
        std::string name; // Name of the parameter
        std::vector<Value> values; // All the possible values of the parameter
        Value default_value; // Default value of the parameter

    public:
        Parameter(const std::string& n, const std::vector<Value>& vals, const Value& dv): name(n), values(vals), default_value(dv) {}
        std::string getName() const { return name; }
        std::vector<Value> getValues() const { return values; }
        Value getDefaultValue() const { return default_value; }
};

#endif // PARAMETER_H