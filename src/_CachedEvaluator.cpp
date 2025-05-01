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

// Private method to load a specific solution
// candidateIndex is the index of the solution in the vector _subsampleResultList
Result _CachedEvaluator::_loadCandidate(size_t candidateIndex) const
{
    if (candidateIndex >= _subsampleResultList.size() || candidateIndex < 0)
    {
        throw std::out_of_range("_CachedEvaluator::_loadCandidate: candidateIndex out of range");
    }

    const auto &candidate = _subsampleResultList[candidateIndex]; // candidate is either Result or int
    if (std::holds_alternative<Result>(candidate))
    {
        return std::get<Result>(candidate); // Return the Result directly
    }
    else
    {
        // Load the result from external storage using the index
        // If expressed as an index, candidate is the index for the storage (not the index in the vector candidateIndex)
        return _subsampleResultIO->_loadSubsampleResult(std::get<int>(candidate));
    }
}

// Public method to perform solution evaluation in parallel
Matrix _CachedEvaluator::_evaluateSubsamples(const std::vector<int> &sampleIndexList,
                                             int k, int B, std::mt19937 &rng)
{
    if (B <= 0)
    {
        throw std::invalid_argument("_CachedEvaluator::_evaluateSubsamples: Number of subsamples B must be positive.");
    }
    size_t n = sampleIndexList.size();
    if (n < k || k <= 0)
    {
        throw std::invalid_argument("_CachedEvaluator::_evaluateSubsamples: Sample size n must be greater than or equal to k and k must be positive.");
    }

    size_t numCandidates = _subsampleResultList.size();
    std::set<int> sampleToEvaluateSet;                      // Store unique individual samples to be evaluated on
    std::vector<std::vector<int>> subsampleIndices(B);      // B sets of subsample indices

    // Fill the subsample indices and record the unique samples
    for (int b = 0; b < B; ++b)
    {
        subsampleIndices[b].reserve(k);
        // Sample k indices from sampleIndexList
        std::sample(sampleIndexList.begin(), sampleIndexList.end(), std::back_inserter(subsampleIndices[b]), k, rng); 

        for (int sampleIndex : subsampleIndices[b])
        {
            sampleToEvaluateSet.insert(sampleIndex); // Note that we do not need to check for duplicates
        }
    }
    std::vector<int> sampleToEvaluate(sampleToEvaluateSet.begin(), sampleToEvaluateSet.end()); // Convert set to vector (of indices)
    if (sampleToEvaluate.empty())
    {
        throw std::invalid_argument("_CachedEvaluator::_evaluateSubsamples: No samples to evaluate on.");
    }

    // Parallel evaluation on unique samples
    // Setup parallel computation (similar to _learnOnSubsamples in _BaseVE)
    int numWorkers = std::min(_numParallelLearn, static_cast<int>(sampleToEvaluate.size()));
    // One worker is responsible for evaluating all candidates on a subset of samples
    // In the inside vector, each element corresponds to the evaluation of one solution on a subset of samples.
    std::vector<std::future<Matrix>> futures;
    futures.reserve(numWorkers);

    /**
     * The following lambda function is used to evaluate all candidates on a subset of samples assigned to the worker.
     * Returns a Matrix of size numSamplesAssigned x numCandidates.
     */
    auto taskLambda = [&](int workerId, const std::vector<int> &workerSampleIndices) -> Matrix
    {
        size_t numSamplesAssigned = workerSampleIndices.size();
        if (numSamplesAssigned == 0)
        {
            throw std::invalid_argument("_CachedEvaluator::_evaluateSubsamples: Worker " + std::to_string(workerId) + " has no samples to evaluate.");
        }

        Matrix workerResults(numSamplesAssigned, numCandidates);
        Eigen::Map<const Eigen::VectorXi> workerSampleIndicesMap(workerSampleIndices.data(), workerSampleIndices.size());
        Sample workerSampleData = _sample(workerSampleIndicesMap, Eigen::all); // Create a matrix by selecting rows from sample

        // Evaluate all candidates on the worker's sample
        for (size_t c = 0; c < numCandidates; ++c)
        {
            Result candidate = _loadCandidate(c);
            Vector evalResult = _baseLearner->objective(candidate, workerSampleData);
            // Sanity check the size of evalResult
            if (evalResult.size() != static_cast<Eigen::Index>(numSamplesAssigned))
            {
                throw std::runtime_error("BaseLearner::objective returned unexpected size in worker " + std::to_string(workerId) + ". Expected " + std::to_string(workerSampleData.rows()) + ", got " + std::to_string(evalResult.size()) + ".");
            }
            workerResults.col(c) = evalResult;
        }
        return workerResults;
    }; // End of taskLambda

    int tasksPerWorker = sampleToEvaluate.size() / numWorkers;
    int remainingTasks = sampleToEvaluate.size() % numWorkers;
    int startIndex = 0;

    // Launch tasks in parallel
    std::vector<std::vector<int>> allWorkerSampleIndices(numWorkers);
    for (int i = 0; i < numWorkers; ++i)
    {
        int batchSize = tasksPerWorker + (i < remainingTasks ? 1 : 0);
        if (batchSize == 0)
            continue;
        int endIndex = startIndex + batchSize;
        allWorkerSampleIndices[i].assign(sampleToEvaluate.begin() + startIndex,
                                         sampleToEvaluate.begin() + endIndex);

        // Launch task asynchronously using std::async
        futures.push_back(
            // Note that we use std::cref to pass the vector by reference
            std::async(std::launch::async, taskLambda, i, std::cref(allWorkerSampleIndices[i])));
        startIndex = endIndex;
    }

    /**
     * Collect results from all workers. Dimension of allResults is numWorkers x (numSamplesAssigned x numCandidates)
     * numSamplesAssigned can be slightly different for each worker
     */
    std::vector<Matrix> allResults;
    allResults.reserve(futures.size());

    try
    {
        for (auto &future : futures)
        {
            allResults.push_back(future.get());
        }
    }
    catch (const std::exception &e)
    {
        throw std::runtime_error("_CachedEvaluator::_evaluateSubsamples: Error while collecting parallel results: " + std::string(e.what()));
    }

    for (size_t workerId = 0; workerId < allResults.size(); ++workerId)
    {
        const auto &workerSampleIndices = allWorkerSampleIndices[workerId]; // Indices of samples evaluated by this worker
        const Matrix &workerResults = allResults[workerId];                 // Results from this worker, should be a Matrix of size (numSamplesAssigned, numCandidates)
        if (workerResults.rows() != static_cast<Eigen::Index>(workerSampleIndices.size()) ||
            workerResults.cols() != static_cast<Eigen::Index>(numCandidates))
        {
            throw std::runtime_error("_CachedEvaluator::_evaluateSubsamples: Worker " + std::to_string(workerId) +
                                     " returned unexpected result size. Expected (" + std::to_string(workerSampleIndices.size()) +
                                     ", " + std::to_string(numCandidates) + "), got (" + std::to_string(workerResults.rows()) +
                                     ", " + std::to_string(workerResults.cols()) + ").");
        }

        // For each sample evaluated by this worker
        for (size_t i = 0; i < workerSampleIndices.size(); ++i)
        {
            int sampleIndex = workerSampleIndices[i];
            _cachedEvaluation[sampleIndex] = workerResults.row(i).transpose();
        }
    } // End of the loop over workers

    // Compute the final result using cache
    Matrix evalResultsToReturn(B, numCandidates);
    evalResultsToReturn.setZero();
    for (int b = 0; b < B; ++b)
    {
        // Sum objective values among all samples in the batch
        Vector sumResultsForBatch = Vector::Zero(numCandidates);
        int subsampleSize = 0;
        const std::vector<int> &sampleIndicesToSum = subsampleIndices[b];

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
                throw std::runtime_error("_CachedEvaluator::_evaluateSubsamples: Sample index " + std::to_string(sampleIndex) + " not found in cache.");
            }
        }

        if (subsampleSize > 0)
        {
            Vector avgResultsForBatch = sumResultsForBatch / static_cast<double>(subsampleSize); // Average
            evalResultsToReturn.row(b) = avgResultsForBatch.transpose();
        }
        else if (k > 0)
        {
            throw std::runtime_error("_CachedEvaluator::_evaluateSubsamples: Logic error: subsample " + std::to_string(b) +
                                     " with k = " + std::to_string(k) + " has no valid samples in the cache.");
        }
        else
        {
            throw std::runtime_error("_CachedEvaluator::_evaluateSubsamples: Subsample size k should be positive.");
        }
    }

    return evalResultsToReturn;
}
