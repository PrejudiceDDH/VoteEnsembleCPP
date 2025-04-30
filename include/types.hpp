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
 * A single solution of baseLearner is expressed as a vector.
 * Note that we do not use Eigen::VectorXd for Result, because it will
 * complicate the serialization in _SubsampleResultIO. Also, it will make
 * majority voting process more difficult.
 */
using Result = std::vector<double>;

// For numeric operations.
using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;

// Forward declarations (optional but can reduce compile times)
struct _BaseLearner;
class _SubsampleResultIO;
class _CachedEvaluator;
class _BaseVE;
class MoVE;
class ROVE;

// Utility functions for type conversions
// Function to convert Result (std::vector<double>) to Eigen VectorXd
Vector resultToVector(const Result &result);

// Function to convert Eigen VectorXd to Result (std::vector<double>)
Result vectorToResult(const Vector &vec);

// Helper function to print a Result
void printResult(const std::string &experimentName, const Result &result);
