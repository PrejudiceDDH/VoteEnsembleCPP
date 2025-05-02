#pragma once
#include "types.hpp"

#include <vector>
#include <random>        // For std::mt19937
#include <variant>       // For std::variant
#include <unordered_map> // For the cache (mapping data index to results)
#include <string>        // For std::string (used by types.hpp)

// Forward declaration of classes
struct BaseLearner;
class _SubsampleResultIO;

class _CachedEvaluator
{
private:
    // Pointer to the base learner
    BaseLearner *_baseLearner;

    /**
     * Pointer to the subsample result IO object
     * It encapsulates all logics related to results handling
     */
    _SubsampleResultIO *_subsampleResultIO;

    /**
     * List of retrieved solutions
     * The solution* is expressed either as itself or the index to it.
     */
    const std::vector<std::variant<Result, int>> &_subsampleResultList;

    // Reference to data
    const Sample &_sample;

    // Number of parallel learners
    int _numParallelLearn;

    /**
     * Cache for storing evaluation results, expressed as a map.
     * The key is the index of the sample in _sample, and the value stores
     * the evaluation results of all candidates on that sample, represented as
     * a Vector of size num_candidates.
     * Note that the map structure is required, otherwise we cannot easily associate
     * the evaluation results with the corresponding sample indices in _sample.
     */
    std::unordered_map<int, RowVector> _cachedEvaluation;

    /**
     * Helper function used to load a specific solution from the storage.
     * candidateIndex is the index of the solution in the vector _subsampleResultList
     */
    Result _loadCandidate(size_t candidateIndex) const;

    /**
     * Helper function to generate B sets of subsample indices, each of size k.
     * Returns 1. A vector of vector, that contains B sets of subsample indices.
     *         2. A vector of indices, that contains the unique sample indices to be evaluated on.
     */
    std::pair<std::vector<std::vector<int>>, std::vector<int>>
    _generateEvaluationSampleIndices(const std::vector<int> &sampleIndexList, int k, int B, std::mt19937 &rng);

    /**
     * Helper function to evaluate all candidates on given samples.
     * This function is called by the individual worker threads to evaluate their assigned samples.
     * Returns a Matrix of size (numSamplesAssigned, numCandidates).
     */
    Matrix _evaluateCandidatesOnSamples(const std::vector<int> &uniqueSampleIndices);

    /**
     * Helper function to get cached evaluation results in parallel.
     * Setup parallel workers to evaluate all candidates on unique samples.
     * The parallel computation is similar to _learnOnSubsamples in _BaseVE
     * Store the results in _cachedEvaluation.
     */
    void _getCachedEvaluation(const std::vector<int>& sampleToEvaluate);

    /**
     * Helper function to compute the final evaluation results on subsamples.
     * The computation is based on the _cachedEvaluation map.
     * Return a matrix of size (B, num_candidates), as required by _evaluateSubsamples.
     */
    Matrix _getFinalEvaluationResults(const std::vector<std::vector<int>> &subsampleIndices, int B);

public:
    // Constructor
    _CachedEvaluator(BaseLearner *baseLearner,
                     _SubsampleResultIO *subsampleResultIO,
                     const std::vector<std::variant<Result, int>> &subsampleResultList,
                     const Sample &sample,
                     int numParallelLearn = 1);

    /**
     * Main evaluation method. The returned Matrix is a matrix of size (B, num_candidates)
     * Each row corresponds to the evaluation of all candidates on a specific subsample
     * sampleIndexList stores indices of the samples used in the evaluation
     */
    Matrix _evaluateSubsamples(const std::vector<int> &sampleIndexList, int k, int B, std::mt19937 &rng);
};