#pragma once
#include <vector>
#include <string>
#include <Eigen/Dense>

/**
 * The dataset consists of a collection of samples, represented as a matrix.
 * Each row of the matrix corresponds to a single sample.
 */
using Sample = Eigen::MatrixXd;

/**
 * A single solution of baseLearner is expressed as an Eigen vector.
 */
using Result = Eigen::VectorXd;

// For numeric operations.
using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;

// Size limit for Eigen variables
constexpr Eigen::Index MAX_REASONABLE_SIZE = 10000000;

// Forward declarations (optional but can reduce compile times)
struct _BaseLearner;
class _SubsampleResultIO;
class _CachedEvaluator;
class _BaseVE;
class MoVE;
class ROVE;

// Utility functions for type conversions
// Function to convert Result (std::vector<double>) to Eigen VectorXd
// Vector resultToVector(const Result &result);

// Function to convert Eigen VectorXd to Result (std::vector<double>)
// Result vectorToResult(const Vector &vec);

// Helper function to print a Result
void printResult(const std::string &experimentName, const Result &result);
