#include "LinearRegressionLearner.hpp"
#include "types.hpp"

#include <stdexcept>   // For exceptions
#include <Eigen/Dense> // For Eigen matrix operations
#include <Eigen/SVD>   // For SVD decomposition
#include <iostream>    // For std::cerr (error reporting)
#include <random>      // For C++ random number generation

// Implementation of core learning methods
Result LinearRegressionLearner::learn(const Sample &sample)
{
    if (sample.rows() == 0 || sample.cols() < 2)
    {
        throw std::invalid_argument("LinearRegressionLearner::learn: Sample must be nonempty and have at least one feature and one label");
    }

    long long n = sample.rows();
    long long p = sample.cols() - 1;
    Vector Y = sample.col(0);       // Labels
    Matrix X = sample.rightCols(p); // Features

    Vector beta(p);

    if (n < p)
    { // Must be rank deficient, throw a warning and use pseudo-inverse
        std::cerr << "LinearRegressionLearner::learn: Number of samples: " << n << " is less than number of features: "
                  << p << ". Psedo-inverse will be used." << std::endl;

        // Compute SVD of X using BDCSVD.
        /**
         * Declares a BDCSVD object and factors X into U\Sigma*V^T.
         * Compute the thin version of U and V, i.e., if n < p, then U is of size (n, p)
         * and V is of size (p, p). Otherwise, U is of size (n, n) and V is of size (p, n).
         */
        Eigen::BDCSVD<Matrix> svd(X, Eigen::ComputeThinU | Eigen::ComputeThinV);

        // Solve X * beta = Y using the computed SVD
        /**
         * svd.solve(Y) computes the least-squares solution to the linear system X * beta = Y.
         */
        beta = svd.solve(Y);
    }
    else
    { // Use normal equation and Cholesky decomposition
        // This is usually faster than SVD when n is large.
        beta = (X.transpose() * X).ldlt().solve(X.transpose() * Y);
    }

    if (!beta.allFinite())
    {
        throw std::runtime_error("LinearRegressionLearner::learn: Computed beta contains non-finite values.");
    }

    return beta;
}

Vector LinearRegressionLearner::objective(const Result &learningResult, const Sample &sample) const
{
    if (sample.rows() == 0 || sample.cols() < 2)
    {
        throw std::invalid_argument("LinearRegressionLearner::objective: Sample must be nonempty and have at least one feature and one label");
    }

    if (learningResult.size() != sample.cols() - 1)
    {
        throw std::invalid_argument("LinearRegressionLearner::objective: Learning result size does not match the number of features.");
    }

    // Data extraction (note that learningResult itself is beta)
    Matrix X = sample.rightCols(sample.cols() - 1);
    Vector Y = sample.col(0);

    // Compute the predicted values
    Vector Y_pred = X * learningResult;

    // Compute the MSE
    Vector residuals = Y - Y_pred;

    return residuals.array().square(); // Compute the MSE vector
}

bool LinearRegressionLearner::isMinimization() const
{
    return true;
}

bool LinearRegressionLearner::enableDeduplication() const
{
    return false; // This is a regression problem, so deduplication check is not needed.
}

bool LinearRegressionLearner::isDuplicate(const Result &result1, const Result &result2) const
{
    // This function is not used. Use void to suppress unused variable warnings.
    (void)result1;
    (void)result2;
    return false;
}

std::pair<Sample, Result> generateLRData(size_t n, int p, double noiseStDev, unsigned int seed)
{
    std::cout << "\nGenerating data (N=" << n << ", P=" << p
              << ", noiseStDev=" << noiseStDev << ", seed="
              << seed << ")..." << std::endl;

    std::mt19937 rng(seed);
    std::normal_distribution<double> featureDist(0.0, 1.0);
    std::normal_distribution<double> noiseDist(0.0, noiseStDev);

    // Generate true beta Vector
    Result trueBeta(p);
    for (int i = 0; i < p; ++i)
    {
        trueBeta(i) = static_cast<double>(i);
    }

    // Generate feature matrix X (n x p) and fill it with random values
    Matrix X(n, p);
    X = Matrix::NullaryExpr(n, p, [&]()
                            { return featureDist(rng); });

    // Generate noise Vector
    Vector noise(n);
    noise = Vector::NullaryExpr(n, [&]()
                                { return noiseDist(rng); });

    // Generate label Y
    Vector Y = X * trueBeta + noise;

    // Combine Y and X into Sample
    Sample sample(n, p + 1);
    sample.col(0) = Y;
    sample.rightCols(p) = X;
    std::cout << "Data generation completed." << std::endl;

    return {sample, trueBeta};
}