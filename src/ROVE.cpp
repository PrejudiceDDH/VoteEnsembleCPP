#include "ROVE.hpp"
#include "_BaseVE.hpp"
#include "BaseLearner.hpp"
#include "_CachedEvaluator.hpp"
#include "_SubsampleResultIO.hpp"
#include "types.hpp"

#include <vector>
#include <string>
#include <stdexcept> // For std::invalid_argument, std::runtime_error
#include <optional>  // For std::optional
#include <variant>   // For std::variant
#include <numeric>   // For std::iota
#include <algorithm> // For std::min, std::max, std::shuffle
#include <cmath>     // For std::floor, std::abs, std::pow? (No, Eigen handles math)
#include <limits>    // For std::numeric_limits
#include <iostream>  // For potential std::cerr

// Constructor
ROVE::ROVE(BaseLearner *baseLearner,
           bool dataSplit,
           int numParallelEval,
           int numParallelLearn,
           std::optional<unsigned int> randomSeed,
           const std::optional<std::string> &subsampleResultsDir,
           bool deleteSubsampleResults)
    : _BaseVE(baseLearner, numParallelLearn, randomSeed, subsampleResultsDir, deleteSubsampleResults),
      _dataSplit(dataSplit),
      _numParallelEval(std::max(1, numParallelEval))
{
    if (!_baseLearner)
        throw std::invalid_argument("ROVE constructor: baseLearner cannot be null");
}

// Helper method to compute the gap matrix
Matrix ROVE::_gapMatrix(const Matrix &evalArray)
{
    if (evalArray.rows() == 0 || evalArray.cols() == 0)
    {
        throw std::invalid_argument("ROVE::_gapMatrix: evalArray cannot be empty");
    }

    size_t B = evalArray.rows();
    size_t numCandidates = evalArray.cols();
    Matrix gapMatrix(B, numCandidates);
    if (_baseLearner->isMinimization())
    {
        // In a minimization problem, the gap = objective - min(objectives in the same row)
        Vector bestObj = evalArray.rowwise().minCoeff().eval(); // Compute the rowwise minimum
        gapMatrix = evalArray.colwise() - bestObj;
    }
    else
    {
        // In a maximization problem, the gap = max(objectives in the same row) - objective
        Vector bestObj = evalArray.rowwise().maxCoeff().eval(); // Compute the rowwise maximum
        gapMatrix = -1 * (evalArray.colwise() - bestObj);       // Note that bestObj - evalArray.colwise() is not enabled by Eigen
    }
    return gapMatrix;
}

// Static helper method to compute the probability of each candidate being epsilon-optimal
RowVector ROVE::_epsilonOptimalProb(const Matrix &gapMatrix, double epsilon)
{
    if (gapMatrix.rows() == 0 || gapMatrix.cols() == 0)
    {
        throw std::invalid_argument("ROVE::_epsilonOptimalProb: gapMatrix cannot be empty");
    }
    /**
     * (gapMatrix.array() <= epsilon) gives a boolean matrix of the same size.
     * Then, we cast is to double before taking the mean along the columns.
     * The result is a row vector of size (1, num_candidates).
     */
    return (gapMatrix.array() <= epsilon).cast<double>().colwise().mean();
}

// Helper method to choose appropriate epsilon for making the epsilon-optimal probability
double ROVE::_findEpsilon(const Matrix &gapMatrix, double autoEpsilonProb)
{
    if (gapMatrix.rows() == 0 || gapMatrix.cols() == 0)
    {
        throw std::invalid_argument("ROVE::_findEpsilon: gapMatrix cannot be empty");
    }

    if (autoEpsilonProb > 1.0)
    {
        throw std::invalid_argument("ROVE::_findEpsilon: autoEpsilonProb must be in [0, 1]");
    }

    // Get the probability of each candidate being optimal
    RowVector probArray = _epsilonOptimalProb(gapMatrix, 0.0);
    /**
     * If the maximum probability is already greater than or equal to autoEpsilonProb,
     * we do not need to allow epsilon relaxation.
     * Note that we do not need to worry about probArray being empty, since we already check gapMatrix
     */
    if (probArray.maxCoeff() >= autoEpsilonProb)
    {
        return 0.0;
    }

    double left = 0.0;
    double right = 1.0;
    probArray = _epsilonOptimalProb(gapMatrix, right);
    // Check if the maximum epsilon-optimal probability is greater than autoEpsilonProb
    while (probArray.maxCoeff() < autoEpsilonProb)
    {
        left = right;
        right *= 2.0; // Double the right bound
        probArray = _epsilonOptimalProb(gapMatrix, right);
    }

    /**
     * Perform line search to find the smallest epsilon that makes the maximum epsilon-optimal probability
     * greater than or equal to autoEpsilonProb
     */
    double tolerance = 1e-3;
    while (std::max(right - left, (right - left) / (std::abs(left) / 2.0 + std::abs(right) / 2.0 + 1e-5)) > tolerance)
    {
        double mid = (left + right) / 2.0;
        probArray = _epsilonOptimalProb(gapMatrix, mid);
        if (probArray.maxCoeff() >= autoEpsilonProb)
        {
            right = mid; // Move the right bound to mid
        }
        else
        {
            left = mid; // Move the left bound to mid
        }
    }
    return right;
}

// run method with all parameters specified
Result ROVE::run(const Sample &sample,
                 int B1, int B2,
                 std::optional<int> k1, std::optional<int> k2,
                 double epsilon, double autoEpsilonProb)
{

    long long nTotal = sample.rows();
    if (nTotal == 0)
        throw std::invalid_argument("ROVE::run: Sample size n must be greater than 0.");
    if (B1 <= 0 || B2 <= 0)
        throw std::invalid_argument("ROVE::run: Number of subsamples B1 and B2 must be positive.");

    // Determine sample sizes for two phases, depending on _dataSplit
    long long phaseOneEnd = nTotal;
    long long phaseTwoStart = 0;
    if (_dataSplit)
    {
        phaseOneEnd = nTotal / 2;
        phaseTwoStart = phaseOneEnd;
    }

    if (phaseOneEnd <= 0 || phaseTwoStart >= nTotal)
    {
        throw std::invalid_argument("ROVE::run: Insufficient sample size n = " + std::to_string(nTotal));
    }
    long long n1 = phaseOneEnd;
    long long n2 = nTotal - phaseTwoStart;

    // Determine k1 and B1 for Phase I
    int k1Val;
    int B1Val = B1;
    if (k1.has_value())
    {
        k1Val = k1.value();
        if (k1Val <= 0)
        {
            throw std::invalid_argument("ROVE::run: Provided k1 must be positive.");
        }
        if (k1Val > n1)
        { // print a warning
            std::cerr << "ROVE::run: Provided k1 is larger than sample size n1. Using n1 instead." << std::endl;
            k1Val = static_cast<int>(n1);
            B1 = 1;
        }
    }
    else
    { // Choose k1 = min(max(30, n1 / divisor), n1)}. The value of divisor depends on whether we deal with duplicates.
        int divisor = _baseLearner->enableDeduplication() ? 200 : 2;
        k1Val = static_cast<int>(std::min(static_cast<long long>(std::max(30, static_cast<int>(n1 / divisor))), n1));
    }

    // Determine k2 and B2 for Phase II
    int k2Val;
    int B2Val = B2;
    if (k2.has_value())
    {
        k2Val = k2.value();
        if (k2Val <= 0)
        {
            throw std::invalid_argument("ROVE::run: Provided k2 must be positive.");
        }
        if (k2Val > n2)
        { // print a warning
            std::cerr << "ROVE::run: Provided k2 is larger than sample size n2. Using n2 instead." << std::endl;
            k2Val = static_cast<int>(n2);
            B2 = 1;
        }
    }
    else
    { // Choose k2 = min(max(30, n2 / 200), n2)
        k2Val = static_cast<int>(std::min(static_cast<long long>(std::max(30, static_cast<int>(n2 / 200))), n2));
    }

    // 1. Phase I: Learn on subsamples and retrieve evaluation results
    Sample phaseOneSample = sample.topRows(phaseOneEnd); // Get the first phaseOneEnd rows
    std::vector<std::variant<Result, int>> learningResults = _learnOnSubsamples(phaseOneSample, k1Val, B1Val);

    // Deal with duplications
    std::vector<std::variant<Result, int>> retrievedResults; // learningResults with duplicates removed
    if (_baseLearner->enableDeduplication())
    {
        std::vector<size_t> uniqueResultIndex;
        for (size_t i = 0; i < learningResults.size(); ++i)
        {
            Result candidate1 = _loadResultIfNeeded(learningResults[i]);
            bool isDuplicate = false;
            for (size_t index : uniqueResultIndex)
            {
                Result candidate2 = _loadResultIfNeeded(learningResults[index]);
                if (_baseLearner->isDuplicate(candidate1, candidate2))
                {
                    isDuplicate = true;
                    break;
                }
            }

            if (!isDuplicate)
            {
                uniqueResultIndex.push_back(i);
                retrievedResults.push_back(learningResults[i]);
            }
        }
    }
    else
    {
        retrievedResults = std::move(learningResults);
    }

    if (retrievedResults.empty())
        throw std::runtime_error("ROVE::run: No learning results obtained during Phase I.");

    /**
     * 2. Phase II: Epsilon-optimal voting.
     * unique_ptr.get() is used to get the raw pointer from the unique_ptr
     */
    _CachedEvaluator cachedEvaluator(_baseLearner, _subsampleResultIO.get(), retrievedResults, sample, _numParallelEval);

    // Get sample indices for Phase II
    std::vector<int> phaseTwoIndices(n2);
    // Fill values, so that phaseTwoIndices = [phaseTwoStart, phaseTwoStart + 1, ..., nTotal - 1]
    std::iota(phaseTwoIndices.begin(), phaseTwoIndices.end(), static_cast<int>(phaseTwoStart));

    Matrix evalResultsPhaseTwo = cachedEvaluator._evaluateSubsamples(phaseTwoIndices, k2Val, B2Val, _rng);
    Matrix gapMatrixPhaseTwo = _gapMatrix(evalResultsPhaseTwo);

    // Determine epsilon
    if (epsilon < 0.0)
    {
        autoEpsilonProb = std::min(std::max(autoEpsilonProb, 0.0), 1.0);
        if (_dataSplit)
        {
            // When _dataSplit is enabled, we cannot determine epsilon using the Phase II data.
            std::vector<int> phaseOneIndices(n1);
            std::iota(phaseOneIndices.begin(), phaseOneIndices.end(), 0);
            Matrix evalResultsPhaseOne = cachedEvaluator._evaluateSubsamples(phaseOneIndices, k2Val, B2Val, _rng);
            Matrix gapMatrixPhaseOne = _gapMatrix(evalResultsPhaseOne);
            epsilon = _findEpsilon(gapMatrixPhaseOne, autoEpsilonProb);
        }
        else
        {
            // When _dataSplit is not enabled, we can directly use the _gapMatrix from Phase II data.
            epsilon = _findEpsilon(gapMatrixPhaseTwo, autoEpsilonProb);
        }
    }

    /**
     * Compute the epsilon-optimal probability and get the candidate with the maximum probability
     * When retrievedResults is not empty, probArray is a row vector of size (1, num_candidates)
     */
    RowVector probArray = _epsilonOptimalProb(gapMatrixPhaseTwo, epsilon);
    Eigen::Index bestCandidateIndex;
    // Use the Eigen method to get the index of the maximum element
    probArray.maxCoeff(&bestCandidateIndex);

    // 3. Finalize and clean up (similar as MoVE)
    Result finalResult = _loadResultIfNeeded(retrievedResults[bestCandidateIndex]);
    if (finalResult.size() == 0)
    {
        throw std::runtime_error("ROVE::run: The result of epsilon-optimal voting is empty.");
    }

    if (_deleteSubsampleResults &&
        _subsampleResultIO &&
        _subsampleResultIO->isExternalStorateEnabled())
    {

        std::vector<int> indicesToDelete;
        indicesToDelete.reserve(learningResults.size());
        for (const auto &resultOrIndex : learningResults)
        {
            if (std::holds_alternative<int>(resultOrIndex))
            {
                indicesToDelete.push_back(std::get<int>(resultOrIndex));
            }
        }

        if (!indicesToDelete.empty())
        {
            _subsampleResultIO->_deleteSubsampleResult(indicesToDelete);
        }
    }
    return finalResult;
}

// Overide the run method from _BaseVE (run under default parameters)
Result ROVE::run(const Sample &sample)
{
    return run(sample, 50, 200, std::nullopt, std::nullopt, -1.0, 0.5);
}