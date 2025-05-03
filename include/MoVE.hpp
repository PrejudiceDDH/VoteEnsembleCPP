#pragma once
#include "types.hpp"
#include "_BaseVE.hpp"

#include <vector>
#include <string>
#include <optional> // For std::optional
#include <variant>  // For std::variant
#include <utility>  // For std::pair

class MoVE : public _BaseVE
{
private:
     // Helper function to finalize the choice for B and k
     std::pair<int, int> _chooseParameters(long long n, int B_in, std::optional<int> k_in) const;

     /**
      * Helper function to implement the majority voting process in MoVE
      * Returns the index of the most frequent candidate in learningResults.
      */
     size_t _performMajorityVoting(const std::vector<std::variant<Result, int>> &learningResults);

public:
     // Constructor
     MoVE(BaseLearner *baseLearner,
          int numParallelLearn = 1,
          std::optional<unsigned int> randomSeed = std::nullopt,
          const std::optional<std::string> &subsampleResultsDir = std::nullopt,
          bool deleteSubsampleResults = true);

     // Destructor
     ~MoVE() override = default;

     // run function with all parameters specified
     virtual Result run(const Sample &sample,
                        int B = 50,
                        std::optional<int> k = std::nullopt);

     // Override the run function from _BaseVE (run under default parameters)
     Result run(const Sample &sample) override;
};