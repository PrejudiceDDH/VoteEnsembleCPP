#include "VoteEnsembleRunner.hpp"
#include "types.hpp"

#include <string>
#include <iostream>
#include <optional>  // For std::optional
#include <stdexcept> // For std::exception

void runMoVE(
    const std::string &experimentName,
    BaseLearner *baseLearnerPtr,
    const Sample &sample,
    int numThreads,
    unsigned int seed,
    const std::optional<std::string> &subsampleResultsDir,
    bool deleteSubsampleResults,
    int B,
    std::optional<int> k)
{
    std::cout << "\nRunning MoVE with " << experimentName << " (numThreads="
              << numThreads << ", seed=" << seed
              << ", B=" << B << ", k=" << (k ? std::to_string(*k) : "null") << ")..." << std::endl;

    if (subsampleResultsDir)
    {
        std::cout << "Subsample results will be stored in: " << *subsampleResultsDir
                  << " (delete=" << std::boolalpha << deleteSubsampleResults << ")" << std::endl;
    }
    else
    {
        std::cout << "External storage for subsample results is disabled." << std::endl;
    }

    try
    { // We write {seed} instead of seed because the declared type is std::optional
        MoVE move(baseLearnerPtr, numThreads, {seed}, subsampleResultsDir, deleteSubsampleResults);
        Result sampleBasedSolution = move.run(sample, B, k);
        printResult(experimentName + " sample-based solution: ", sampleBasedSolution);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error during MoVE execution: " << e.what() << std::endl;
    }
}

void runROVE(
    const std::string &experimentName,
    BaseLearner *baseLearnerPtr,
    const Sample &sample,
    bool dataSplit,
    int numThreads,
    unsigned int seed,
    const std::optional<std::string> &subsampleResultsDir,
    bool deleteSubsampleResults,
    int B1, int B2,
    std::optional<int> k1,
    std::optional<int> k2,
    double epsilon,
    double autoEpsilonProb)
{
    std::cout << "\nRunning ROVE with " << experimentName << " (dataSplit="
              << std::boolalpha << dataSplit << ", numThreads=" << numThreads
              << ", seed=" << seed << ", B1=" << B1 << ", B2=" << B2
              << ", k1=" << (k1 ? std::to_string(*k1) : "null")
              << ", k2=" << (k2 ? std::to_string(*k2) : "null")
              << ", epsilon=" << epsilon
              << ", autoEpsilonProb=" << autoEpsilonProb << ")..." << std::endl;

    if (subsampleResultsDir)
    {
        std::cout << "Subsample results will be stored in: " << *subsampleResultsDir
                  << " (delete=" << std::boolalpha << deleteSubsampleResults << ")" << std::endl;
    }
    else
    {
        std::cout << "External storage for subsample results is disabled." << std::endl;
    }

    try
    { // numThreads for both learning and evaluation
        ROVE rove(baseLearnerPtr, dataSplit, numThreads, numThreads, {seed}, subsampleResultsDir, deleteSubsampleResults);
        Result estimatedBeta = rove.run(sample, B1, B2, k1, k2, epsilon, autoEpsilonProb);

        printResult(experimentName + " estimated beta: ", estimatedBeta);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error during ROVE execution: " << e.what() << std::endl;
    }
}