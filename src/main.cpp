// MPILS: Multi-Phase Iterated Local Search Tuner
//
// Author: Youri Rigaud
// License: GNU GPLv3

#include <iostream>
#include <string>

#include "../include/parameter.h"

int main() {
    std::cout << "Welcome to the MPILS tuner!" << std::endl;
    Value value = Value("Toto");
    std::cout << "My value: " << value.getString() << std::endl;

    Parameter param(0, std::string("param0"), std::string("string"), Value("option1"), {Value("option1"), Value("option2")});
    std::cout << "Parameter Name: " << param.getName() << ", Default Value: " << param.getDefaultValue().getString() << std::endl;
    return 0;
}
