// MPILS: Multi-Phase Iterated Local Search Tuner
//
// Author: Youri Rigaud
// License: GNU GPLv3

#include <string>
#include <variant>
#include <vector>

class Value {
    private:
        std::variant<int, double, bool, std::string> value; // The value in a string format to deal with int, double, bool and string
    
    public:
        Value(int val): value(val) {}
        Value(double val): value(val) {}
        Value(bool val): value(val) {}
        Value(const std::string& val): value(val) {}
        Value(const char *val): value(std::string(val)) {}

        int getInt() const {return std::get<int>(value);}
        double getDouble() const {return std::get<double>(value);}
        bool getBool() const {return std::get<bool>(value);}
        std::string getString() const {return std::get<std::string>(value);}
};

class Parameter{
    private:
        int index; // Index of the parameter
        std::string name; // Name of the parameter
        std::string type; // Type of the parameter (int, double, bool, string)
        Value default_value; // Default value of the parameter
        std::vector<Value> values; // All the possible values of the parameter

    public:
        Parameter(int i, const std::string& n, const std::string& t, const Value& dv, const std::vector<Value>& vals);
        int getIndex() const { return index; }
        std::string getName() const { return name; }
        std::string getType() const { return type; }
        Value getDefaultValue() const { return default_value; }
        std::vector<Value> getValues() const { return values; }
};