#include "LinearProgramLearner.hpp"
#include "types.hpp"

#include <vector>
#include <iostream>
#include <random>
#include <numeric>   // For std::accumulate
#include <stdexcept> // For exceptions
#include <cmath>     // For std::abs
#include <limits>    // For numeric_limits if needed
#include <Eigen/Dense>

// Implementation of core learning methods
Result LinearProgramLearner::learn(const Sample &sample)
{
    if (sample.rows() == 0 || sample.cols() != 2)
    {
        throw std::invalid_argument("LinearProgramLearner::learn: Sample must be nonempty and have exactly two columns");
    }

    double mean_xi1 = sample.col(0).mean(); // Mean of the first column
    double mean_xi2 = sample.col(1).mean(); // Mean of the second column

    Result solution(2);
    if (mean_xi1 < mean_xi2)
    {
        solution << 1.0, 0.0; // x* = [1, 0], will be automatically converted to Result, which is a vector of doubles.
    }
    else
    {
        solution << 0.0, 1.0; // x* = [0, 1]
    }

    return solution;
}

Vector LinearProgramLearner::objective(const Result &learningResult, const Sample &sample) const
{
    if (sample.rows() == 0 || sample.cols() != 2)
    {
        throw std::invalid_argument("LinearProgramLearner::objective: Sample must be nonempty and have exactly two columns");
    }

    if (learningResult.size() != 2)
    {
        throw std::invalid_argument("LinearProgramLearner::objective: Learning result must have exactly two elements");
    }

    return sample * learningResult; // Returns a vector of size num_samples.
}

bool LinearProgramLearner::isMinimization() const
{
    return true;
}

bool LinearProgramLearner::enableDeduplication() const
{
    return true;
}

bool LinearProgramLearner::isDuplicate(const Result &result1, const Result &result2) const
{
    if (result1.size() != result2.size())
    {
        throw std::invalid_argument("LinearProgramLearner::isDuplicate: Results must have the same size");
    }

    return (result1 - result2).lpNorm<1>() < tolerance; // Use the L1-distance between two vectors to check for duplicates.
}

Sample generateLPData(size_t n, const std::vector<double> &meanVector, double noiseStDev, unsigned int seed)
{
    std::cout << "\nGenerating data (N=" << n << ", meanVector=[" << meanVector[0]
              << ", " << meanVector[1] << "], noiseStDev=" << noiseStDev << ", seed="
              << seed << ")..." << std::endl;

    std::mt19937 rng(seed);
    std::normal_distribution<double> noiseDist(0.0, noiseStDev);

    Sample sample(n, 2);
    // Fill with noise first (use NullaryExpr and lambda)
    sample = Sample::NullaryExpr(n, 2, [&]()
                                 { return noiseDist(rng); });
    // Then add the mean vector
    Eigen::RowVectorXd meanRow(2);
    meanRow << meanVector[0], meanVector[1];
    sample.rowwise() += meanRow; // Automatically broadcast.
    std::cout << "Data generation completed." << std::endl;
    return sample;
}