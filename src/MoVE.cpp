#include "MoVE.hpp"
#include "_BaseVE.hpp"
#include "BaseLearner.hpp"
#include "_SubsampleResultIO.hpp"
#include "types.hpp"

#include <vector>
#include <optional>
#include <string>
#include <variant>
#include <iostream>  // For std::cerr
#include <stdexcept> // For std::invalid_argument, std::runtime_error
#include <algorithm> // For std::min, std::max
#include <limits>    // For numeric_limits

// Constructor
MoVE::MoVE(BaseLearner *baseLearner,
           int numParallelLearn,
           std::optional<unsigned int> randomSeed,
           const std::optional<std::string> &subsampleResultsDir,
           bool deleteSubsampleResults)
    : _BaseVE(baseLearner, numParallelLearn, randomSeed, subsampleResultsDir, deleteSubsampleResults)
{
    if (!_baseLearner || !_baseLearner->enableDeduplication())
        throw std::invalid_argument("MoVE constructor: baseLearner cannot be null and must enable deduplication.");
}

// run method with all parameters specified
Result MoVE::run(const Sample &sample, int B, std::optional<int> k)
{
    long long n = sample.rows();
    if (n == 0)
        throw std::invalid_argument("MoVE::run: Sample size n must be greater than 0.");
    if (B <= 0)
        throw std::invalid_argument("MoVE::run: Number of subsamples B must be positive.");

    int kVal;     // The value of k to be used
    int BVal = B; // The value of B to be used
    if (k.has_value())
    {
        kVal = k.value();
        if (kVal <= 0)
        {
            throw std::invalid_argument("MoVE::run: Provided k must be positive.");
        }
        if (kVal > n)
        { // print a warning, do not need to throw
            std::cerr << "MoVE::run: Provided k is larger than sample size n. Using n instead." << std::endl;
            kVal = static_cast<int>(n);
            B = 1;
        }
    }
    else
    { // choose k = min(max(30, len(sample) / 200), len(sample))
        kVal = static_cast<int>(std::min(static_cast<long long>(std::max(30, static_cast<int>(n / 200))), n));
    }

    /**
     * 1. Learn on subsamples
     * learningResults is a vector of size B, each element is either a Result or an int (index)
     */
    std::vector<std::variant<Result, int>> learningResults = _learnOnSubsamples(sample, kVal, BVal);
    if (learningResults.empty())
        throw std::runtime_error("MoVE::run: No learning results obtained.");

    /**
     * 2. Perform majority voting
     * uniqueResultIndexCounts is a vector of pairs (index, count)
     * maxIndex stores the index of the most frequent candidate (in learningResults)
     */
    std::vector<std::pair<size_t, int>> uniqueResultIndexCounts;
    size_t maxIndex = 0;
    int maxCount = 0;

    for (size_t i = 0; i < learningResults.size(); ++i)
    {
        Result candidate1 = _loadResultIfNeeded(learningResults[i]);
        if (candidate1.size() == 0)
        { // Note that Result is essentially a vector
            throw std::runtime_error("MoVE::run: Empty candidate result at index " + std::to_string(i));
        }

        size_t foundAtIndex = std::numeric_limits<size_t>::max(); // To help find the index of the candidate
        for (size_t j = 0; j < uniqueResultIndexCounts.size(); ++j)
        { // Check if candidate1 agrees with any existing candidate
            Result candidate2 = _loadResultIfNeeded(learningResults[uniqueResultIndexCounts[j].first]);
            if (_baseLearner->isDuplicate(candidate1, candidate2))
            {
                foundAtIndex = j; // Found a match
                break;
            }
        }

        int currentCount = 0; // To help check whether the current candidate becomes the most frequent
        if (foundAtIndex != std::numeric_limits<size_t>::max())
        { // Found a match, increment the count
            uniqueResultIndexCounts[foundAtIndex].second++;
            currentCount = uniqueResultIndexCounts[foundAtIndex].second;
        }
        else
        { // No match found, add a new candidate
            uniqueResultIndexCounts.emplace_back(i, 1);
            currentCount = 1;
            foundAtIndex = uniqueResultIndexCounts.size() - 1; // The index of the new candidate
        }

        // Update the most frequent candidate
        if (currentCount > maxCount)
        {
            maxCount = currentCount;
            maxIndex = uniqueResultIndexCounts[foundAtIndex].first;
        }
    } // End of the loop over learningResults

    /**
     * 3. Finalize and clean up
     */
    Result finalResult = _loadResultIfNeeded(learningResults[maxIndex]);
    if (finalResult.size() == 0)
        throw std::runtime_error("MoVE::run: The result of majority voting is empty.");

    if (_deleteSubsampleResults &&
        _subsampleResultIO &&
        _subsampleResultIO->isExternalStorateEnabled())
    {

        std::vector<int> indicesToDelete;
        indicesToDelete.reserve(learningResults.size());
        for (const auto &resultOrIndex : learningResults)
        {
            /**
             * Check if the result is an index.
             * If it is, we need to delete the corresponding result from external storage
             * Otherwise, it is a Result and is only held in memory
             */
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
Result MoVE::run(const Sample &sample)
{
    return run(sample, 200, std::nullopt);
}