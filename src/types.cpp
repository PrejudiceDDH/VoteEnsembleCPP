#include "types.hpp"

#include <vector>
#include <string>
#include <iostream>
#include <iomanip>     // For std::setprecision, std::fixed
#include <Eigen/Dense> // For Eigen matrix operations

void printResult(const std::string &experimentName, const Result &result)
{
    // Use Eigen's IO to format the output
    Eigen::IOFormat CommaInitFmt(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", ", ", "", "", "[", "]");
    std::cout << experimentName << " : " << result.transpose().format(CommaInitFmt) << std::endl;
}