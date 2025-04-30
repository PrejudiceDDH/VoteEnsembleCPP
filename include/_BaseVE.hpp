#pragma once
#include "types.hpp"

#include <vector>
#include <random> // For std::mt19937
#include <string>
#include <optional>
#include <memory>  // For std::unique_ptr
#include <variant> // For std::variant

// Forward declaration of classes
struct BaseLearner;
class _SubsampleResultIO; // Used in _learnOnSubsamples and _loadResultIfNeeded

/**
 * Note that in the current implementation, _BaseVE, MoVE, and ROVE do not
 * hold sample as a member variable. Instead, they are passed as arguments to the run method.
 */
class _BaseVE
{
protected: // Ensure those members are accessible to derived classes (MoVE and ROVE)
    // Protected variables
    // Pointer to the base learner
    BaseLearner *_baseLearner;

    // Number of parallel learners
    int _numParallelLearn;

    // Random number generator
    std::mt19937 _rng;
    unsigned int _randomSeed;

    // Pointer to the subsample result IO object
    std::unique_ptr<_SubsampleResultIO> _subsampleResultIO;

    // Flag to indicate whether to delete subsample results
    // This is used in MoVE and ROVE. Will not be used in _BaseVE.
    bool _deleteSubsampleResults;

    // Protected methods

    // Helper function to work with isExternalStorateEnabled in _SubsampleResultIO.
    // Ensure the output is Result. If the input is the index, load the result from external storage.
    Result _loadResultIfNeeded(const std::variant<Result, int> &resultOrIndex);

    // Main learning method, run baseLearner on B subsamples of size k.
    // Return a vector of Result or int (index of the result).
    std::vector<std::variant<Result, int>> _learnOnSubsamples(const Sample &sample, int k, int B);

public:
    // Constructor
    _BaseVE(BaseLearner *baseLearner,
            int numParallelLearn = 1,
            std::optional<unsigned int> randomSeed = std::nullopt,
            const std::optional<std::string> &subsampleResultsDir = std::nullopt,
            bool deleteSubsampleResults = true);

    // Virtual destructor
    /**
     * Note: We cannot use default here, and can only write the implementation in cpp file.
     * The reason is that when the compiler processes _BaseVE.hpp, it may only see the forward
     * declaration of _SubsampleResultIO, instead of the full definition. So, it will cause the
     * error regarding the incomplete type.
     */
    virtual ~_BaseVE();

    // Public methods
    void resetRandomSeed();

    // Specific run algorithm (to be implemented in derived classes, the default case)
    virtual Result run(const Sample &sample) = 0;
};