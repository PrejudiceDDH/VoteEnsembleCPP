#pragma once
#include "types.hpp"

#include <vector>
#include <string>
#include <optional> // For std::optional
#include <random>   // For std::mt19937
#include <memory>   // For std::unique_ptr
#include <variant>  // For std::variant
#include <future>   // For std::async, std::future

// Forward declaration of classes
struct BaseLearner;
class _SubsampleResultIO; // Used in _learnOnSubsamples and _loadResultIfNeeded

/**
 * _BaseVE stands for Base VoteEnsemble, which serves as the base class for main algorithms MoVE and ROVE.
 * Note that in the current implementation, _BaseVE, MoVE, and ROVE do not
 * hold sample as a member variable. Instead, they are passed as arguments to the run function.
 */
class _BaseVE
{
protected: // Ensure those members are accessible to derived classes (MoVE and ROVE).
    // Pointer to the base learner.
    BaseLearner *_baseLearner;

    // Number of parallel learners.
    int _numParallelLearn;

    // Random number generator.
    std::mt19937 _rng;
    unsigned int _randomSeed;

    // Pointer to the _SubsampleResultIO class, which can be used by derived classes.
    std::unique_ptr<_SubsampleResultIO> _subsampleResultIO;

    /**
     * Flag to indicate whether to delete intermediate computation results.
     * This is only used in derived classes (MoVE and ROVE).
     */
    bool _deleteSubsampleResults;

    /**
     * Helper function to to get a candidate solution as Result.
     * Ensure the output is Result. If the input is the index, load the result from external storage.
     */
    Result _loadResultIfNeeded(const std::variant<Result, int> &resultOrIndex);

    // Helper function to generate B sets of subsample indices, each of size k.
    std::vector<std::vector<int>> _generateSubsampleIndices(int n, int k, int B);

    /**
     * Helper function to learn on a single subsample.
     * Receive the full sample and the indices to the subsample.
     * Return a Result or an int (index of the result).
     */
    std::variant<Result, int> _processSingleSubsample(const Sample &sample,
                                                      const std::vector<int> &indices,
                                                      int subsampleIndex);
    /**
     * Helper function to launch parallel learners to learn on B subsamples.
     * Create a vector of futures to hold potentially not-yet-completed results.
     * Each element of futures will be a vector of pairs (index, result).
     * result is either a Result or an index to the external storage.
     * futures has dimension: [numWorkers][numSubsamplesPerWorker].
     */
    std::vector<std::future<std::vector<std::pair<int, std::variant<Result, int>>>>>
    _launchLearningTasks(const Sample &sample, const std::vector<std::vector<int>> &subsampleIndices, int B);

    /**
     * Helper function to collect results from futures and order them by index.
     * Return a vector of Result or int (index of the result).
     */
    std::vector<std::variant<Result, int>> _collectResultsFromWorkers(
        std::vector<std::future<std::vector<std::pair<int, std::variant<Result, int>>>>> &futures,
        int B);

    // Helper function to clean up the subsample results if external storage is enabled.
    void _cleanupSubsampleResults(const std::vector<std::variant<Result, int>> &learningResults);

    /**
     * Main learning method, run baseLearner on B subsamples of size k by aggregating
     * the above helper functions.
     * Return a vector of Result or int (index of the result).
     */
    std::vector<std::variant<Result, int>> _learnOnSubsamples(const Sample &sample, int k, int B);

public:
    // Constructor
    _BaseVE(BaseLearner *baseLearner,
            int numParallelLearn = 1,
            std::optional<unsigned int> randomSeed = std::nullopt,
            const std::optional<std::string> &subsampleResultsDir = std::nullopt,
            bool deleteSubsampleResults = true);

    /**
     * Virtual destructor
     * Note: We cannot use default here, and can only write the implementation in cpp file.
     * The reason is that when the compiler processes _BaseVE.hpp, it may only see the forward
     * declaration of _SubsampleResultIO, instead of the full definition. So, it will cause the
     * error regarding the incomplete type.
     */
    virtual ~_BaseVE();

    // Reset the random seed (currently not used in MoVE and ROVE)
    void resetRandomSeed();

    // Run the algorithm with default parameters (to be implemented in derived classes)
    virtual Result run(const Sample &sample) = 0;
};