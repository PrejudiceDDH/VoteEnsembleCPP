#pragma once
#include "types.hpp"

// We make functions pure virtual to be implemented in derived classes
struct BaseLearner
{
    virtual ~BaseLearner() = default;

    /**
     * The base learning function.
     * Receives the sample and returns a single solution.
     */
    virtual Result learn(const Sample &sample) = 0;

    /**
     * Evaluate a single solution on (possibly multiple) samples
     * Returns a vector of size num_samples
     */
    virtual Vector objective(const Result &learningResult, const Sample &sample) const = 0;
    virtual bool isMinimization() const = 0;

    // Deduplication check (Must be enabled for discrete problems, i.e., MoVE)
    virtual bool enableDeduplication() const = 0;
    virtual bool isDuplicate(const Result &result1, const Result &result2) const = 0;

    // Serialization
    virtual void dumpLearningResult(const Result &learningResult, std::ostream &out) const
    {
        // Dump a single learning result.
        if (!out)
        {
            throw std::runtime_error("Output stream is not valid for dumping learning result.");
        }

        // Eigen::VectorXd specific serialization
        Eigen::Index size = learningResult.size();
        out.write(reinterpret_cast<const char *>(&size), sizeof(size));
        if (size > 0)
        {
            // learningResult.data() returns a pointer to the raw double data buffer
            out.write(reinterpret_cast<const char *>(learningResult.data()), size * sizeof(double));
        }
        if (!out)
        {
            throw std::runtime_error("Failed to write learning result to output stream.");
        }
    }

    // Load a single learning result from external storage
    virtual Result loadLearningResult(std::istream &in) const
    {
        if (!in)
        {
            throw std::runtime_error("Input stream is not valid for loading learning result.");
        }

        // Eigen::VectorXd specific deserialization
        Eigen::Index size = 0;
        in.read(reinterpret_cast<char *>(&size), sizeof(size));
        if (!in || size < 0 || size > MAX_REASONABLE_SIZE)
        {
            throw std::runtime_error("Failed to read valid size of learning result from input stream.");
        }

        Result learningResult(size);
        if (size > 0)
        {
            // learningResult.data() returns a pointer to the raw double data buffer
            in.read(reinterpret_cast<char *>(learningResult.data()), size * sizeof(double));
            if (!in || static_cast<Eigen::Index>(in.gcount()) != size * sizeof(double))
            {
                throw std::runtime_error("Failed to read learning result data (Eigen::VectorXd) from input stream or size mismatch.");
            }
        }
        return learningResult;
    }
};