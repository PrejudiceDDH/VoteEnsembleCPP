#pragma once
#include "types.hpp"

#include <vector>
#include <random>  // For std::mt19937
#include <variant> // For std::variant
#include <map>     // For the cache (mapping data index to results)
#include <string>  // For std::string (used by types.hpp)

// Forward declaration of classes
struct BaseLearner;
class _SubsampleResultIO;

class _CachedEvaluator
{
private:
    // Private variables
    // Pointer to the base learner
    BaseLearner *_baseLearner;

    // Pointer to the subsample result IO object
    // It encapsulates all logics related to results handling
    _SubsampleResultIO *_subsampleResultIO;

    // List of retrieved solutions
    // Expressed as either the solution itself or the index of the solution
    const std::vector<std::variant<Result, int>> &_subsampleResultList;
    // Reference to data
    const Sample &_sample;

    // Number of parallel learners
    int _numParallelLearn;

    // Cache for storing evaluation results, expressed as a map
    // The key is the index of the sample in _sample, and the value stores
    // the evaluation results of all candidates on that sample
    // represented as a matrix (row vector) of size (1, num_candidates)
    std::map<int, Matrix> _cachedEvaluation;

    // Private methods
    // Helper function used to load a specific solution
    // candidateIndex is the index of the solution in the vector _subsampleResultList
    Result _loadCandidate(size_t candidateIndex) const;

public:
    // Constructor
    _CachedEvaluator(BaseLearner *baseLearner,
                     _SubsampleResultIO *subsampleResultIO,
                     const std::vector<std::variant<Result, int>> &subsampleResultList,
                     const Sample &sample,
                     int numParallelLearn = 1);

    // Main evaluation method. The returned Matrix is a matrix of size (B, num_candidates)
    // Each row corresponds to the evaluation of all candidates on a specific subsample
    Matrix _evaluateSubsamples(const std::vector<int> &sampleIndexList, // indices of the samples used in the evaluation
                                int k, int B, std::mt19937 &rng);
};