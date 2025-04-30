#pragma once
#include "types.hpp"
#include "MoVE.hpp"
#include "ROVE.hpp"

#include <string>
#include <optional>    // For std::optional

// Function to run MoVE with specified parameters and results printing
void runMoVE(
    const std::string &experimentName,
    BaseLearner *baseLearnerPtr,
    const Sample &sample,
    int numThreads,
    unsigned int seed,
    const std::optional<std::string>& subsampleResultsDir = std::nullopt, // Default to no external storage.
    bool deleteSubsampleResults = true,
    int B = 200,
    std::optional<int> k = std::nullopt);

// Function to run ROVE with specified parameters and results printing
void runROVE(
    const std::string &experimentName,
    BaseLearner *baseLearnerPtr,
    const Sample &sample,
    bool dataSplit,
    int numThreads,
    unsigned int seed,
    const std::optional<std::string>& subsampleResultsDir = std::nullopt, // Default to no external storage.
    bool deleteSubsampleResults = true,
    int B1 = 50,
    int B2 = 200,
    std::optional<int> k1 = std::nullopt,
    std::optional<int> k2 = std::nullopt,
    double epsilon = -1.0,
    double autoEpsilonProb = 0.5);