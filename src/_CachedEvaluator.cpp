#include "BaseLearner.hpp"
#include "_CachedEvaluator.hpp"
#include "_SubsampleResultIO.hpp"
#include "types.hpp"

#include <vector>
#include <unordered_map>
#include <variant>    // For std::variant
#include <stdexcept>  // For exceptions
#include <algorithm>  // For std::shuffle, std::min, std::max
#include <future>     // For std::async, std::future
#include <set>        // For std::set to find unique indices
#include <iostream>   // For std::cerr
#include <Eigen/Core> // Include Eigen Core for Map and VectorXi (if not implicitly included)

// Constructor
_CachedEvaluator::_CachedEvaluator(BaseLearner *baseLearner,
                                   _SubsampleResultIO *subsampleResultIO,
                                   const std::vector<std::variant<Result, int>> &subsampleResultList,
                                   const Sample &sample,
                                   int numParallelLearn)
    : _baseLearner(baseLearner),
      _subsampleResultIO(subsampleResultIO),
      _subsampleResultList(subsampleResultList),
      _sample(sample),
      _numParallelLearn(std::max(1, numParallelLearn))
{
    if (!_baseLearner)
        throw std::invalid_argument("_CachedEvaluator constructor: baseLearner cannot be null");
    if (!_subsampleResultIO)
        throw std::invalid_argument("_CachedEvaluator constructor: subsampleResultIO cannot be null");
    if (_subsampleResultList.empty())
        throw std::invalid_argument("_CachedEvaluator constructor: subsampleResultList cannot be empty");
    if (_sample.rows() == 0)
        throw std::invalid_argument("_CachedEvaluator constructor: sample cannot be empty");
}

// Helper function used to load a specific solution from the storage.
Result _CachedEvaluator::_loadCandidate(size_t candidateIndex) const
{
    if (candidateIndex >= _subsampleResultList.size() || candidateIndex < 0)
        throw std::out_of_range("_CachedEvaluator::_loadCandidate: candidateIndex out of range");

    /**
     * candidate is either Result or int
     * If expressed as an index, candidate is the index for the storage (not the index in the vector candidateIndex)
     */
    const auto &candidate = _subsampleResultList[candidateIndex];
    if (std::holds_alternative<Result>(candidate))
        return std::get<Result>(candidate);
    else
        return _subsampleResultIO->_loadSubsampleResult(std::get<int>(candidate));
}

// Helper function to generate B sets of subsample indices, each of size k.
std::pair<std::vector<std::vector<int>>, std::vector<int>>
_CachedEvaluator::_generateEvaluationSampleIndices(const std::vector<int> &sampleIndexList,
                                                   int k, int B, std::mt19937 &rng)
{
    size_t n = sampleIndexList.size();
    std::set<int> sampleToEvaluateSet;
    std::vector<std::vector<int>> subsampleIndices(B);

    // Fill the subsample indices and record the unique samples
    for (int b = 0; b < B; ++b)
    {
        subsampleIndices[b].reserve(k);
        // Sample k indices from sampleIndexList
        std::sample(sampleIndexList.begin(), sampleIndexList.end(), std::back_inserter(subsampleIndices[b]), k, rng);

        for (int sampleIndex : subsampleIndices[b])
        {
            sampleToEvaluateSet.insert(sampleIndex);
        }
    }

    // Convert set to vector (of indices) for convenience
    std::vector<int> sampleToEvaluate(sampleToEvaluateSet.begin(), sampleToEvaluateSet.end());
    if (sampleToEvaluate.empty())
        throw std::invalid_argument("_CachedEvaluator::_generateEvaluationSampleIndices: No samples to evaluate on.");

    return {subsampleIndices, sampleToEvaluate};
}

// Helper function to evaluate all candidates on given samples.
Matrix _CachedEvaluator::_evaluateCandidatesOnSamples(const std::vector<int> &uniqueSampleIndices)
{
    size_t numSamplesAssigned = uniqueSampleIndices.size();
    size_t numCandidates = _subsampleResultList.size();
    if (numSamplesAssigned == 0)
        throw std::invalid_argument("_CachedEvaluator::_evaluateCandidatesOnSamples: No samples to evaluate on.");
    if (numCandidates == 0)
        throw std::invalid_argument("_CachedEvaluator::_evaluateCandidatesOnSamples: No candidates to evaluate on.");

    Matrix workerResults(numSamplesAssigned, numCandidates);
    Eigen::Map<const Eigen::VectorXi> workerSampleIndicesMap(uniqueSampleIndices.data(), uniqueSampleIndices.size());
    Sample workerSampleData = _sample(workerSampleIndicesMap, Eigen::all); // Create a matrix by selecting rows from sample

    // Evaluate all candidates on the assigned samples
    for (size_t c = 0; c < numCandidates; ++c)
    {
        Result candidate = _loadCandidate(c);
        Vector evalResult = _baseLearner->objective(candidate, workerSampleData);
        // Sanity check the size of evalResult
        if (evalResult.size() != static_cast<Eigen::Index>(numSamplesAssigned))
        {
            throw std::runtime_error("BaseLearner::objective returned unexpected size. Expected " + std::to_string(workerSampleData.rows()) +
                                     ", got " + std::to_string(evalResult.size()) + ".");
        }
        workerResults.col(c) = evalResult;
    }
    return workerResults;
}

// Helper function to get cached evaluation results in parallel.
void _CachedEvaluator::_getCachedEvaluation(const std::vector<int> &sampleToEvaluate)
{
    size_t numSampleToEvaluate = sampleToEvaluate.size();
    int numWorkers = std::min(_numParallelLearn, static_cast<int>(numSampleToEvaluate));

    /**
     * One worker is responsible for evaluating all candidates on a subset of samples
     * In the inside vector of futures, each element corresponds to the evaluation of
     * one solution on a subset of samples.
     */
    std::vector<std::future<Matrix>> futures;
    futures.reserve(numWorkers);
    std::vector<std::vector<int>> allWorkerSampleIndices(numWorkers); // Store indices assigned to each worker

    /**
     * The following lambda function is used to evaluate all candidates on a subset of
     * samples assigned to the worker.
     * Returns a Matrix of size numSamplesAssigned x numCandidates.
     */
    auto taskLambda = [&](const std::vector<int> &workerSampleIndices) -> Matrix
    {
        return _evaluateCandidatesOnSamples(workerSampleIndices);
    };

    // Launch tasks in parallel
    int tasksPerWorker = numSampleToEvaluate / numWorkers;
    int remainingTasks = numSampleToEvaluate % numWorkers;
    int startIndex = 0;

    for (int i = 0; i < numWorkers; ++i)
    {
        int batchSize = tasksPerWorker + (i < remainingTasks ? 1 : 0);
        if (batchSize == 0)
            continue;
        int endIndex = startIndex + batchSize;
        allWorkerSampleIndices[i].assign(sampleToEvaluate.begin() + startIndex,
                                         sampleToEvaluate.begin() + endIndex);

        /**
         * Launch task asynchronously using std::async
         * Note that we use std::cref to pass the vector by reference
         */
        futures.push_back(std::async(std::launch::async, taskLambda, std::cref(allWorkerSampleIndices[i])));
        startIndex = endIndex;
    }

    // Collect results and populate cache
    try
    {
        for (size_t workerId = 0; workerId < futures.size(); ++workerId)
        {
            // Results from this worker, should be a Matrix of size (numSamplesAssigned, numCandidates)
            Matrix workerResults = futures[workerId].get();
            // Indices of samples evaluated by this worker
            const auto &workerSampleIndices = allWorkerSampleIndices[workerId];

            if (workerResults.rows() != static_cast<Eigen::Index>(workerSampleIndices.size()))
                throw std::runtime_error("_CachedEvaluator: Worker " + std::to_string(workerId) +
                                         " returned matrix with mismatched rows.");

            for (size_t i = 0; i < workerSampleIndices.size(); ++i)
            {
                int sampleIndex = workerSampleIndices[i];
                _cachedEvaluation[sampleIndex] = workerResults.row(i);
            }
        }
    }
    catch (const std::exception &e)
    {
        throw std::runtime_error("_CachedEvaluator::_getCachedEvaluation: Error while collecting parallel results: " +
                                 std::string(e.what()));
    }
}

// Helper function to compute the final evaluation results on subsamples.
Matrix _CachedEvaluator::_getFinalEvaluationResults(const std::vector<std::vector<int>> &subsampleIndices, int B)
{
    size_t numCandidates = _subsampleResultList.size();
    Matrix evalResultsToReturn(B, numCandidates);
    evalResultsToReturn.setZero();
    for (int b = 0; b < B; ++b)
    {
        // Sum objective values among all samples in the batch
        RowVector sumResultsForBatch = RowVector::Zero(numCandidates);
        const std::vector<int> &sampleIndicesToSum = subsampleIndices[b];
        int subsampleSize = 0;

        for (int sampleIndex : sampleIndicesToSum)
        {
            auto it = _cachedEvaluation.find(sampleIndex);
            if (it != _cachedEvaluation.end())
            {
                sumResultsForBatch += it->second; // Add the evaluation result for this sample
                ++subsampleSize;
            }
            else
            {
                throw std::runtime_error("_CachedEvaluator::_getFinalEvaluationResults: Sample index " +
                                         std::to_string(sampleIndex) + " not found in cache.");
            }
        }

        if (subsampleSize > 0)
        {
            RowVector avgResultsForBatch = sumResultsForBatch / static_cast<double>(subsampleSize); // Average
            evalResultsToReturn.row(b) = avgResultsForBatch;
        }
        else if (!sampleIndicesToSum.empty())
        {
            throw std::runtime_error("_CachedEvaluator::_getFinalEvaluationResults: Logic error: subsample " +
                                     std::to_string(b) + " has no valid samples in the cache.");
        }
    }

    return evalResultsToReturn;
}

// Main evaluation method
Matrix _CachedEvaluator::_evaluateSubsamples(const std::vector<int> &sampleIndexList, int k, int B, std::mt19937 &rng)
{
    if (B <= 0)
        throw std::invalid_argument("_CachedEvaluator::_evaluateSubsamples: Number of subsamples B must be positive.");
    size_t n = sampleIndexList.size();
    if (n < k || k <= 0)
        throw std::invalid_argument("_CachedEvaluator::_evaluateSubsamples: Sample size n must be greater than or equal to k and k must be positive.");

    /**
     * Generate B sets of subsample indices, each of size k.
     * subsampleIndices stores B sets of subsample indices
     * sampleToEvaluate stores the unique sample indices to be evaluated on
     * (_generateEvaluationSampleIndices automatically checks whether sampleIndexList is empty)
     */
    auto [subsampleIndices, sampleToEvaluate] = _generateEvaluationSampleIndices(sampleIndexList, k, B, rng);

    // Get cached evaluation results in parallel, store them in _cachedEvaluation
    _getCachedEvaluation(sampleToEvaluate);

    // Compute the final result using cache
    return _getFinalEvaluationResults(subsampleIndices, B);
}