#include "types.hpp"

#include <string>
#include <iostream>
#include <Eigen/Dense>

// Helper function to print a Result
void printResult(const std::string &experimentName, const Result &result)
{
    // Use Eigen's built-in IO to format the output
    Eigen::IOFormat CommaInitFmt(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", ", ", "", "", "[", "]");
    std::cout << experimentName << " : " << result.transpose().format(CommaInitFmt) << std::endl;
}