#pragma once
#include "types.hpp"
#include "_BaseVE.hpp"

#include <vector>
#include <optional>
#include <string>

// Forward declaration of classes
class _CachedEvaluator;

class ROVE : public _BaseVE
{
private:
    bool _dataSplit;
    int _numParallelEval;

    /**
     * Helper method to compute the gap matrix
     * The gap matrix has the same dimension as the evaluation matrix, returned by
     * _CachedEvaluator::_evaluateSubsamples, i.e., (B, num_candidates).
     * The (b,i)-th element of the gap matrix is the gap between the objective value
     * of the i-th candidate and the best candidate in the b-th subsample.
     */
    Matrix _gapMatrix(const Matrix &evalArray);

public:
    // Constructor
    ROVE(BaseLearner *baseLearner,
         bool dataSplit = false,
         int numParallelEval = 1,
         int numParallelLearn = 1,
         std::optional<unsigned int> randomSeed = std::nullopt,
         const std::optional<std::string> &subsampleResultsDir = std::nullopt,
         bool deleteSubsampleResults = true);

    // Destructor
    ~ROVE() override = default;

    /**
     * Helper method to compute the probability of each candidate being epsilon-optimal
     * Returns a row vector of size num_candidates.
     */
    static RowVector _epsilonOptimalProb(const Matrix &gapMatrix, double epsilon);

    /**
     * Helper method to choose appropriate epsilon for making the epsilon-optimal probability
     * appropriately large. The chosen rule is based on the paper.
     */
    static double _findEpsilon(const Matrix &gapMatrix, double autoEpsilonProb);

    // run method with all parameters specified
    virtual Result run(const Sample &sample,
                       int B1 = 50, int B2 = 200,
                       std::optional<int> k1 = std::nullopt, std::optional<int> k2 = std::nullopt,
                       double epsilon = -1.0, double autoEpsilonProb = 0.5);

    // Override the run method from _BaseVE (run under default parameters)
    Result run(const Sample &sample) override;
};