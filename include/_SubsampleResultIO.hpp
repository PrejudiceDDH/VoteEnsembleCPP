#pragma once
#include "types.hpp"

#include <vector>
#include <filesystem> // For std::filesystem::path
#include <string>
#include <optional> // For std::optional

// Forward declaration of BaseLearner
struct BaseLearner;

class _SubsampleResultIO
{
private:
    BaseLearner *_baseLearner;

    // Optional path to a directory where results are stored
    std::optional<std::filesystem::path> _resultDir;

    // Join the directory with the index
    static std::filesystem::path _subsampleResultPath(const std::filesystem::path &dir, int index);

public:
    // Constructor
    _SubsampleResultIO(BaseLearner *baseLearner,
                       const std::optional<std::string> &subsampleResultsDir);

    // Public IO methods (they are not static because they may need access to the baseLearner, which is a private member)
    // Creates the directory.
    void _prepareSubsampleResultDir();

    // Save the learning result to a file.
    void _dumpSubsampleResult(const Result &learningResult, int index);

    // Load the learning result from a file (a single solution).
    Result _loadSubsampleResult(int index);

    // Delete the learning result files for the given indices.
    void _deleteSubsampleResult(const std::vector<int> &indexList);

    // Utility methods for C++
    // True if results should be saved to/loaded from disk.
    bool isExternalStorateEnabled() const;

    // Get the result directory path.
    const std::optional<std::filesystem::path> &getResultDir() const;
};