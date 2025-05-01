#pragma once
#include "BaseLearner.hpp"
#include "types.hpp"

#include <vector>
#include <cmath>

/**
 * Consider solving the following simple linear program:
 *           min E[\xi_1 * x_1 + \xi_2 * x_2]
 *              s.t. x_1 + x_2 = 1
 *                   x_1, x_2 >= 0
 * Due to the simple nature, the solution is is trivial:
 * x* = [1, 0] if E[\xi_1] < E[\xi_2], and x* = [0, 1] otherwise.
 */
class LinearProgramLearner : public BaseLearner
{
private:
    const double tolerance = 1e-6; // Tolerance for floating point comparisons

public:
    // Constructor
    LinearProgramLearner() = default;

    // Destructor
    ~LinearProgramLearner() override = default;

    /**
     * Core learning methods
     * Solve the linear program by comparing the sample means of \xi_1 and \xi_2.
     * The sample is a matrix of size (n, 2).
     */
    Result learn(const Sample &sample) override;

    Vector objective(const Result &learningResult, const Sample &sample) const override;

    bool isMinimization() const override;

    bool enableDeduplication() const override;

    bool isDuplicate(const Result &result1, const Result &result2) const override;
};

// Function for data generation
Sample generateLPData(size_t n, const std::vector<double> &meanVector, double noiseStDev, unsigned int seed);