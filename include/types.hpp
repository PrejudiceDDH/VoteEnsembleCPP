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

/**
 * Introduce aliases for Eigen types to improve readability.
 * Although Matrix has the same meaning as Sample and Vector has the same meaning as Result,
 * we use these aliases to distinguish between the meaning of the variables.
 */
using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;
using RowVector = Eigen::RowVectorXd;

// Size limit for Eigen variables
constexpr Eigen::Index MAX_REASONABLE_SIZE = 10000000;

// Forward declarations (optional)
struct _BaseLearner;
class _SubsampleResultIO;
class _CachedEvaluator;
class _BaseVE;
class MoVE;
class ROVE;

// Helper function to print a Result
void printResult(const std::string &experimentName, const Result &result);
