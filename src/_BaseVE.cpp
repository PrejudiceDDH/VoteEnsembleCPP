#include "BaseLearner.hpp"
#include "_BaseVE.hpp"
#include "_SubsampleResultIO.hpp"
#include "types.hpp"

#include <vector>
#include <string>
#include <variant>    // For std::variant
#include <stdexcept>  // For std::invalid_argument, std::runtime_error
#include <numeric>    // For std::iota
#include <algorithm>  // For std::shuffle, std::min
#include <future>     // For std::async, std::future
#include <thread>     // For std::thread::hardware_concurrency (optional)
#include <iostream>   // For std::cerr (error reporting)
#include <chrono>     // For seeding RNG with time if no seed provided
#include <Eigen/Core> // Include Eigen Core for Map and VectorXi (if not implicitly included)

// Constructor
_BaseVE::_BaseVE(BaseLearner *baseLearner,
                 int numParallelLearn,
                 std::optional<unsigned int> randomSeed,
                 const std::optional<std::string> &subsampleResultsDir,
                 bool deleteSubsampleResults)
    : _baseLearner(baseLearner),
      _numParallelLearn(std::max(1, numParallelLearn)),
      _deleteSubsampleResults(deleteSubsampleResults)
{
    if (!_baseLearner)
        throw std::invalid_argument("_BaseVE constructor: baseLearner cannot be null");

    // Initialize random seed (if not provided, use current time)
    _randomSeed = randomSeed.value_or(static_cast<unsigned int>(std::chrono::system_clock::now().time_since_epoch().count()));
    _rng.seed(_randomSeed);

    // Initialize _SubsampleResultIO
    _subsampleResultIO = std::make_unique<_SubsampleResultIO>(_baseLearner, subsampleResultsDir);
    _subsampleResultIO->_prepareSubsampleResultDir();
}

// Destructor (use default, and unique_ptr will handle cleanup)
_BaseVE::~_BaseVE() = default;

// Reset the random seed
void _BaseVE::resetRandomSeed()
{
    _rng.seed(_randomSeed);
}

// Helper function to to get a candidate solution as Result
Result _BaseVE::_loadResultIfNeeded(const std::variant<Result, int> &resultOrIndex)
{
    if (std::holds_alternative<Result>(resultOrIndex))
    {
        return std::get<Result>(resultOrIndex);
    }
    else
    {
        if (!_subsampleResultIO)
        {
            throw std::runtime_error("_BaseVE::_loadResultIfNeeded: _subsampleResultIO is not initialized.");
        }
        int index = std::get<int>(resultOrIndex);
        return _subsampleResultIO->_loadSubsampleResult(index);
    }
}

// Main learning method
std::vector<std::variant<Result, int>> _BaseVE::_learnOnSubsamples(const Sample &sample, int k, int B)
{
    if (B <= 0)
        throw std::invalid_argument("_BaseVE::_learnOnSubsamples: Number of subsamples B must be positive.");

    long long n = sample.rows();
    if (n < k)
        throw std::invalid_argument("_BaseVE::_learnOnSubsamples: Sample size n must be greater than or equal to k.");
    else if (k <= 0)
        throw std::invalid_argument("_BaseVE::_learnOnSubsamples: Subsample size k must be positive.");

    // Generate B sets of subsample indices
    std::vector<std::vector<int>> subsampleIndices(B);
    std::vector<int> nIndices(n);
    std::iota(nIndices.begin(), nIndices.end(), 0); // nIndices = {0, 1, ..., n-1}
    for (int b = 0; b < B; ++b)
    {
        subsampleIndices[b].reserve(k);
        // Sample k indices from nIndices
        std::sample(nIndices.begin(), nIndices.end(), std::back_inserter(subsampleIndices[b]), k, _rng);
    }

    /**
     * Setup parallel computation
     * Create a vector of futures to hold potentially not-yet-completed results
     * Each element of futures will be a vector of pairs (index, result).
     * result is either a Result or an index to the external storage.
     * futures has dimension: [numWorkers][numSubsamplesPerWorker]
     */
    int numWorkers = std::min(_numParallelLearn, B);
    std::vector<std::future<std::vector<std::pair<int, std::variant<Result, int>>>>> futures;
    futures.reserve(numWorkers);

    /**
     * Define a lambda function for a single worker
     * Each worker handles batches [startBatch, endBatch)
     */
    auto taskLambda = [&](int workerId, int startBatch, int endBatch)
        -> std::vector<std::pair<int, std::variant<Result, int>>>
    {
        std::vector<std::pair<int, std::variant<Result, int>>> workerResults;
        workerResults.reserve(endBatch - startBatch);

        for (int b = startBatch; b < endBatch; ++b)
        {
            /**
             * Get the subsample indices
             * Convert indices to Eigen::VectorXi
             * Then, create a matrix by selecting rows from sample and learn on it
             */
            const std::vector<int> &indices = subsampleIndices[b];
            Eigen::Map<const Eigen::VectorXi> indicesMap(indices.data(), indices.size());
            Sample subsampleData = sample(indicesMap, Eigen::all);
            Result learningResult = _baseLearner->learn(subsampleData);

            /**
             * Store result or dump it and store the index (assume _subsampleResultIO is not null)
             * Depending on whether the external storage is enabled or not
             */
            if (_subsampleResultIO->isExternalStorateEnabled())
            {
                _subsampleResultIO->_dumpSubsampleResult(learningResult, b);
                workerResults.emplace_back(b, b); // Store the index if external storage is enabled
            }
            else
            {
                workerResults.emplace_back(b, std::move(learningResult)); // Store the result otherwise
            }
        }
        return workerResults;
    }; // End of taskLambda

    int tasksPerWorker = B / numWorkers;
    int remainingTasks = B % numWorkers;
    int startIndex = 0;

    // Launch tasks in parallel
    for (int i = 0; i < numWorkers; ++i)
    {
        int batchSize = tasksPerWorker + (i < remainingTasks ? 1 : 0); // Distribute remaining tasks
        int endIndex = startIndex + batchSize;

        // Launch task asynchronously using std::async
        futures.push_back(std::async(std::launch::async, taskLambda, i, startIndex, endIndex));

        // Update startIndex for the next worker
        startIndex = endIndex;
    }

    // Collect results from all workers
    std::vector<std::pair<int, std::variant<Result, int>>> allResults;
    allResults.reserve(B);
    try
    {
        for (auto &future : futures)
        {
            std::vector<std::pair<int, std::variant<Result, int>>> workerResults = future.get();
            allResults.insert(allResults.end(),
                              std::make_move_iterator(workerResults.begin()),
                              std::make_move_iterator(workerResults.end()));
        }
    }
    catch (const std::exception &e)
    {
        throw std::runtime_error("_BaseVE::_learnOnSubsamples: Error while collecting results: " + 
                                 std::string(e.what()));
    }

    // Prepare for return, order the results by index
    std::vector<std::variant<Result, int>> allResultsToReturn(B);
    for (const auto &result : allResults)
    {
        int index = result.first;
        if (index >= 0 && index < B)
        {
            allResultsToReturn[index] = std::move(result.second);
        }
        else
        {
            std::cerr << "_BaseVE::_learnOnSubsamples: Warning: Received out-of-bounds index "
                      << index << " for subsample result." << std::endl;
        }
    }
    return allResultsToReturn;
}

// Pure virtual base method, cannot be called directly
Result _BaseVE::run(const Sample &sample)
{
    throw std::runtime_error("_BaseVE::run was called. Derived classes should implement this method.");
    return Result{};
}