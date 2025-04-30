#include "types.hpp"

#include <vector>
#include <string>
#include <iostream>
#include <iomanip>     // For std::setprecision, std::fixed
#include <Eigen/Dense> // For Eigen matrix operations

Vector resultToVector(const Result &result)
{
    return Eigen::Map<const Vector>(result.data(), result.size());
}

Result vectorToResult(const Vector &vec)
{
    return Result(vec.data(), vec.data() + vec.size());
}

void printResult(const std::string &experimentName, const Result &result)
{
    std::cout << experimentName << "[";
    bool first = true;
    std::cout << std::fixed << std::setprecision(4);
    for (const auto &val : result)
    {
        if (!first)
            std::cout << ", ";
        std::cout << val;
        first = false;
    }
    std::cout << "]" << std::endl;

    // Reset output format
    std::cout.unsetf(std::ios_base::floatfield);
    std::cout.precision(6); // Reset to default precision
}