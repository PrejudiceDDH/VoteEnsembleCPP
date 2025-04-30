#include "types.hpp"
#include "LinearRegressionLearner.hpp"
#include "LinearProgramLearner.hpp"
#include "ROVE.hpp"
#include "VoteEnsembleRunner.hpp"

#include <string>
#include <vector>
#include <iostream>    // For input/output (std::cout, std::cerr)
#include <stdexcept>   // For std::exception
#include <Eigen/Dense> // For Eigen operations during data generation
#include <algorithm>   // For std::min
#include <thread>      // For std::thread::hardware_concurrency (optional)
#include <chrono>      // For timing.

void runLRExample()
{
    std::cout << "Running VoteEnsemble on Linear Regression..." << std::endl;

    // Parameters
    const size_t n = 10000;
    const int p = 10;
    const double noiseStDev = 5.0;
    const unsigned int dataSeed = 888;
    const unsigned int algSeed = 999;
    const int numThreads = std::min(1u, std::thread::hardware_concurrency() / 2); // add u after 4 to match the type of hardware_concurrency
    const std::string subsampleResultsDir = "./LR_storage_test";

    // Generate data and print true result
    auto [sample, trueBeta] = generateLRData(n, p, noiseStDev, dataSeed);
    printResult("True beta: ", trueBeta);

    // Instantiate the base learner
    LinearRegressionLearner baseLearner;
    BaseLearner *baseLearnerPtr = &baseLearner; // Create a pointer to the base learner.

    // Run ROVE and ROVEs (default parameters with external storage disabled)
    runROVE("ROVE", baseLearnerPtr, sample, false, numThreads, algSeed);
    runROVE("ROVEs", baseLearnerPtr, sample, true, numThreads, algSeed);
    
    // Run ROVE and ROVEs (default parameters with external storage enabled)
    // runROVE("ROVE", baseLearnerPtr, sample, false, numThreads, algSeed, subsampleResultsDir + "/ROVE", false);
    // runROVE("ROVEs", baseLearnerPtr, sample, true, numThreads, algSeed, subsampleResultsDir + "/ROVEs", false);
    std::cout << "\nVoteEnsemble on Linear Regression completed." << std::endl;
}

void runLPExample()
{
    std::cout << "Running VoteEnsemble on Linear program..." << std::endl;

    // Parameters
    size_t n = 10000;
    const std::vector<double> meanVector = {0.0, 0.2};
    const double noiseStDev = 2.0;
    const unsigned int dataSeed = 888;
    const unsigned int algoSeed = 999;
    const int numThreads = std::min(1u, std::thread::hardware_concurrency() / 2); // add u after 4 to match the type of hardware_concurrency
    const std::string subsampleResultsDir = "./LP_storage_test";

    // Generate data and print true result
    Sample sample = generateLPData(n, meanVector, noiseStDev, dataSeed);
    printResult("True solution: ", {1.0, 0.0});

    // Instantiate the base learner
    LinearProgramLearner baseLearner;
    BaseLearner *baseLearnerPtr = &baseLearner;

    // Run MoVE and ROVE (default parameters with external storage disabled)
    runMoVE("MoVE", baseLearnerPtr, sample, numThreads, algoSeed);
    runROVE("ROVE", baseLearnerPtr, sample, false, numThreads, algoSeed);
    runROVE("ROVEs", baseLearnerPtr, sample, true, numThreads, algoSeed);

    // Run MoVE, ROVE, and ROVEs (default parameters with external storage enabled)
    // runMoVE("MoVE", baseLearnerPtr, sample, numThreads, algoSeed, subsampleResultsDir + "/MoVE", false);
    // runROVE("ROVE", baseLearnerPtr, sample, false, numThreads, algoSeed, subsampleResultsDir + "/ROVE", false);
    // runROVE("ROVEs", baseLearnerPtr, sample, true, numThreads, algoSeed, subsampleResultsDir + "/ROVEs", false);
    std::cout << "\nVoteEnsemble on Linear Program completed." << std::endl;
}

int main(int argc, char *argv[])
{
    auto start = std::chrono::high_resolution_clock::now();
    
    if (argc != 2)
    {
        std::cerr << "Usage: " << argv[0] << " <ExampleName>" << std::endl;
        std::cerr << "Available ExampleName: LR, LP" << std::endl;
        return 1;
    }

    std::string exampleName = argv[1];

    try
    {
        if (exampleName == "LR")
        {
            runLRExample();
        }
        else if (exampleName == "LP")
        {
            runLPExample();
        }
        else
        {
            std::cerr << "Unknown ExampleName: " << exampleName << std::endl;
            std::cerr << "Available ExampleName: LR, LP" << std::endl;
            return 1;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "\n!!! An exception occurred during execution: " << e.what() << std::endl;
        return 1;
    }
    catch (...)
    {
        std::cerr << "\n!!! An unknown exception occurred during execution." << std::endl;
        return 1;
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "VoteEnsemble on " << exampleName << " completed in " << elapsed.count() << " seconds." << std::endl;
    return 0;
}