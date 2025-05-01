#pragma once
#include "BaseLearner.hpp"
#include "types.hpp"

#include <Eigen/Dense>
#include <utility>      // For std::pair

// Overload the BaseLearner class for linear regression
class LinearRegressionLearner : public BaseLearner
{
public:
    // Constructor
    LinearRegressionLearner() = default;

    // Destructor
    ~LinearRegressionLearner() override = default;

    // Core learning methods
    /**
     * Learn a linear regression model on the given sample.
     * Suppose the sample is a matrix of size (n, p+1), where n is the number of samples
     * and p is the number of features. The first column is the label, i.e., Y.
     */
    Result learn(const Sample &sample) override;

    // Returns a vector of size num_samples
    Vector objective(const Result &learningResult, const Sample &sample) const override;

    bool isMinimization() const override;

    bool enableDeduplication() const override;

    bool isDuplicate(const Result &result1, const Result &result2) const override;
};

// Function for data generation
std::pair<Sample, Result> generateLRData(size_t n, int p, double noiseStDev, unsigned int seed);
