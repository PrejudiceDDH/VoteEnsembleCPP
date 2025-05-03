#include "MoVE.hpp"
#include "_BaseVE.hpp"
#include "BaseLearner.hpp"
#include "_SubsampleResultIO.hpp"
#include "types.hpp"

#include <vector>
#include <string>
#include <optional>  // For std::optional
#include <variant>   // For std::variant
#include <utility>   // For std::pair
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

// Helper function to finalize the choice for B and k
std::pair<int, int> MoVE::_chooseParameters(long long n, int B_in, std::optional<int> k_in) const
{
    int kVal;        // The value of k to be used
    int BVal = B_in; // The value of B to be used
    if (k_in.has_value())
    {
        kVal = k_in.value();
        if (kVal <= 0)
        {
            throw std::invalid_argument("MoVE::_chooseParameters: Provided k must be positive.");
        }
        if (kVal > n)
        { // print a warning, do not need to throw
            std::cerr << "MoVE::_chooseParameters: Provided k is larger than sample size n. Using n instead." << std::endl;
            kVal = static_cast<int>(n);
            BVal = 1;
        }
    }
    else
    { // choose k = min(max(30, len(sample) / 200), len(sample))
        kVal = static_cast<int>(std::min(static_cast<long long>(std::max(30, static_cast<int>(n / 200))), n));
    }

    return {BVal, kVal};
}

// Helper function to implement the majority voting process in MoVE
size_t MoVE::_performMajorityVoting(const std::vector<std::variant<Result, int>> &learningResults)
{
    /**
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
            throw std::runtime_error("MoVE::_performMajorityVoting: Empty candidate result at index " + std::to_string(i));
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

    return maxIndex;
}

// run function with all parameters specified
Result MoVE::run(const Sample &sample, int B, std::optional<int> k)
{
    // Validate input and determine arameters
    long long n = sample.rows();
    if (n == 0)
        throw std::invalid_argument("MoVE::run: Sample size n must be greater than 0.");
    if (B <= 0)
        throw std::invalid_argument("MoVE::run: Number of subsamples B must be positive.");
    auto [BVal, kVal] = _chooseParameters(n, B, k);

    /**
     * Learn on subsamples and retrieve solutions as a vector
     * learningResults is a vector of size B, each element is either a Result or an int (index)
     */
    std::vector<std::variant<Result, int>> learningResults = _learnOnSubsamples(sample, kVal, BVal);
    if (learningResults.empty())
        throw std::runtime_error("MoVE::run: No learning results obtained.");

    // Perform majority voting to find the most frequently returned solution
    size_t maxIndex = _performMajorityVoting(learningResults);
    Result finalResult = _loadResultIfNeeded(learningResults[maxIndex]);
    if (finalResult.size() == 0)
        throw std::runtime_error("MoVE::run: The result of majority voting is empty.");

    // Clean up (optionally run, depending on the value of _deleteSubsampleResults)
    _cleanupSubsampleResults(learningResults);

    return finalResult;
}

// Override the run function from _BaseVE (run under default parameters)
Result MoVE::run(const Sample &sample)
{
    return run(sample, 200, std::nullopt);
}