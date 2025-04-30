#pragma once
#include "types.hpp"
#include "_BaseVE.hpp"

#include <vector>
#include <optional>
#include <string>

class MoVE : public _BaseVE
{
public:
     // Constructor
     MoVE(BaseLearner *baseLearner,
          int numParallelLearn = 1,
          std::optional<unsigned int> randomSeed = std::nullopt,
          const std::optional<std::string> &subsampleResultsDir = std::nullopt,
          bool deleteSubsampleResults = true);

     // Destructor
     ~MoVE() override = default;

     // Public methods
     // Specific run algorithm with all parameters specified
     virtual Result run(const Sample &sample,
                        int B = 50,
                        std::optional<int> k = std::nullopt);

     // Overide the run method from _BaseVE (run under default parameters)
     Result run(const Sample &sample) override;
};