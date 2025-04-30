#include "BaseLearner.hpp"
#include "_SubsampleResultIO.hpp"
#include "types.hpp"

#include <vector>
#include <zstd.h>     // For ZSTD compression and decompression
#include <filesystem> // For path operations, create_directories, remove
#include <fstream>    // For std::ofstream, std::ifstream
#include <sstream>    // For std::stringstream (memory buffer)
#include <iostream>   // For std::cerr (error reporting)
#include <stdexcept>  // For std::runtime_error, std::invalid_argument
#include <iostream>   // For potential debug/error messages (optional)
#include <string>     // For std::to_string

// Helper functions
// Converts optional string to optional path
std::optional<std::filesystem::path> stringToPathOpt(const std::optional<std::string> &strOpt)
{
    if (strOpt)
    {
        // Note that *strOpt gives a string, and the constructor of std::filesystem::path can take a string
        try
        {
            return std::filesystem::path(*strOpt);
        }
        catch (const std::exception &e)
        {
            throw std::runtime_error("stringToPathOpt: Invalid path string provided: " + *strOpt + ": " + e.what());
        }
    }
    return std::nullopt;
}

// Constructor
_SubsampleResultIO::_SubsampleResultIO(BaseLearner *baseLearner,
                                       const std::optional<std::string> &subsampleResultsDir)
    : _baseLearner(baseLearner), _resultDir(stringToPathOpt(subsampleResultsDir))
{
    if (!_baseLearner)
    {
        throw std::invalid_argument("_SubsampleResultIO constructor: baseLearner cannot be null");
    }
}

// Private methods
// Create a path for the subsample result file (join directory with index)
std::filesystem::path _SubsampleResultIO::_subsampleResultPath(const std::filesystem::path &dir, int index)
{
    // create files look like "resultsDir/subsampleResult_0"
    return dir / ("subsampleResult_" + std::to_string(index));
}

// Public methods
void _SubsampleResultIO::_prepareSubsampleResultDir()
{
    if (_resultDir)
    {
        try
        {
            if (!std::filesystem::exists(*_resultDir))
            {
                // Note: *_resultDir dereferences the optional, giving a path
                std::filesystem::create_directories(*_resultDir);
            }
        }
        catch (const std::filesystem::filesystem_error &e)
        {
            throw std::runtime_error("_SubsampleResultIO::_prepareSubsampleResultDir: Filesystem error when creating directory: " + _resultDir->string() + ": " + e.what());
        }
        catch (const std::exception &e)
        {
            throw std::runtime_error("_SubsampleResultIO::_prepareSubsampleResultDir: Error when creating directory: " + _resultDir->string() + ": " + e.what());
        }
    }
}

void _SubsampleResultIO::_dumpSubsampleResult(const Result &learningResult, int index)
{
    if (!_resultDir)
    {
        throw std::runtime_error("_SubsampleResultIO::_dumpSubsampleResult: External storage is not enabled.");
    }
    std::filesystem::path resultPath = _subsampleResultPath(*_resultDir, index);

    // 1. Serialize Result to in-memory buffer using BaseLearner's method
    std::stringstream memoryStream(std::ios::in | std::ios::out | std::ios::binary);
    try
    {
        _baseLearner->dumpLearningResult(learningResult, memoryStream);
    }
    catch (const std::exception &e)
    {
        throw std::runtime_error("_SubsampleResultIO::_dumpSubsampleResult: Error serializing learning result for index " + std::to_string(index) + " via BaseLearner: " + e.what());
    }
    std::string serializedData = memoryStream.str(); // Get data as string/byte sequence

    // 2. Compress the serialized data using ZSTD (same as in the python version)
    size_t cBufferSize = ZSTD_compressBound(serializedData.size());
    std::vector<char> compressedData(cBufferSize);

    // Compress the data and return the size
    size_t const cSize = ZSTD_compress(compressedData.data(), cBufferSize,
                                       serializedData.data(), serializedData.size(), 1);

    if (ZSTD_isError(cSize))
    {
        throw std::runtime_error("_SubsampleResultIO::_dumpSubsampleResult: ZSTD compression error for index " + std::to_string(index) + ": " + ZSTD_getErrorName(cSize));
    }

    // 3. Write the compressed data to the file
    std::ofstream outFile(resultPath, std::ios::binary | std::ios::trunc); // binary mode, clear existing content if any
    if (!outFile)
    {
        throw std::runtime_error("_SubsampleResultIO::_dumpSubsampleResult: Failed to open file for writing: " + resultPath.string());
    }

    outFile.write(compressedData.data(), cSize);
    if (!outFile)
    {
        // Attempt close before throwing
        outFile.close();
        throw std::runtime_error("_SubsampleResultIO::_dumpSubsampleResult: Failed to write compressed data to file: " + resultPath.string());
    }

    outFile.close(); // Close the file
    if (!outFile)
    {
        throw std::runtime_error("_SubsampleResultIO::_dumpSubsampleResult: Failed to close file after writing: " + resultPath.string());
    }
}

Result _SubsampleResultIO::_loadSubsampleResult(int index)
{
    if (!_resultDir)
    {
        throw std::runtime_error("_SubsampleResultIO::_loadSubsampleResult: External storage is not enabled.");
    }
    std::filesystem::path resultPath = _subsampleResultPath(*_resultDir, index);

    if (!std::filesystem::exists(resultPath))
    {
        throw std::runtime_error("_SubsampleResultIO::_loadSubsampleResult: File not found for loading result: " + resultPath.string());
    }

    // 1. Read the compressed data from the file
    std::ifstream inFile(resultPath, std::ios::binary | std::ios::ate); // binary mode, seek to end
    if (!inFile)
    {
        throw std::runtime_error("_SubsampleResultIO::_loadSubsampleResult: Failed to open file for reading: " + resultPath.string());
    }
    std::streamsize compressedSize = inFile.tellg(); // Get the size of the file
    inFile.seekg(0, std::ios::beg);                  // Seek back to the beginning

    std::vector<char> compressedData(compressedSize);
    if (!inFile.read(compressedData.data(), compressedSize))
    {
        inFile.close();
        throw std::runtime_error("_SubsampleResultIO::_loadSubsampleResult: Failed to read compressed data from file: " + resultPath.string());
    }
    inFile.close(); // Close the file

    // 2. Decompress the data using ZSTD
    unsigned long long const bufferSize = ZSTD_getFrameContentSize(compressedData.data(), compressedSize);
    if (bufferSize == ZSTD_CONTENTSIZE_ERROR)
    {
        throw std::runtime_error("_SubsampleResultIO::_loadSubsampleResult: File is not a valid ZSTD compressed file: " + resultPath.string());
    }
    if (bufferSize == ZSTD_CONTENTSIZE_UNKNOWN)
    {
        throw std::runtime_error("_SubsampleResultIO::_loadSubsampleResult: Unknown decompressed size for file: " + resultPath.string());
    }

    // Should not happen, since Result refers to a single solution, expressed as a vector of doubles.
    // if (bufferSize > 1024 * 1024 * 1024) {
    //     throw std::runtime_error("_SubsampleResultIO::_loadSubsampleResult: Decompressed size exceeds 1GB for file: "
    //                              + resultPath.string());
    // }

    std::vector<char> decompressedData(bufferSize);
    size_t const decompressedSize = ZSTD_decompress(decompressedData.data(), bufferSize,
                                                    compressedData.data(), compressedSize);

    if (ZSTD_isError(decompressedSize))
    {
        throw std::runtime_error("_SubsampleResultIO::_loadSubsampleResult: ZSTD decompression error for index: " + std::to_string(index) + ": " + ZSTD_getErrorName(decompressedSize));
    }
    if (decompressedSize != bufferSize)
    {
        throw std::runtime_error("_SubsampleResultIO::_loadSubsampleResult: Zstd decompression size mismatch for index: " + std::to_string(index) + ": expected " + std::to_string(bufferSize) + ", got " + std::to_string(decompressedSize));
    }

    // 3. Deserialize the data back to Result using BaseLearner's method
    std::stringstream memoryStream(std::string(decompressedData.data(), decompressedSize),
                                   std::ios::in | std::ios::binary);

    Result learningResult;
    try
    {
        learningResult = _baseLearner->loadLearningResult(memoryStream);
    }
    catch (const std::exception &e)
    {
        throw std::runtime_error("_SubsampleResultIO::_loadSubsampleResult: Error deserializing learning result for index " + std::to_string(index) + " via BaseLearner: " + e.what());
    }

    return learningResult;
}

void _SubsampleResultIO::_deleteSubsampleResult(const std::vector<int> &indexList)
{
    if (!_resultDir)
    {
        return; // No need to delete if external storage is not enabled
    }
    for (int index : indexList)
    {
        std::filesystem::path resultPath = _subsampleResultPath(*_resultDir, index);
        try
        {
            if (std::filesystem::exists(resultPath))
            {
                std::filesystem::remove(resultPath);
            }
        }
        catch (const std::filesystem::filesystem_error &e)
        {
            std::cerr << "_SubsampleResultIO::_deleteSubsampleResult: Filesystem error when deleting file: "
                      << resultPath.string() << ": " << e.what() << std::endl;
        }
        catch (const std::exception &e)
        {
            std::cerr << "_SubsampleResultIO::_deleteSubsampleResult: Error when deleting file: "
                      << resultPath.string() << ": " << e.what() << std::endl;
        }
    }
}

// Utility methods
bool _SubsampleResultIO::isExternalStorateEnabled() const
{
    // As long as _resultDir is not null, we consider external storage enabled
    return _resultDir.has_value();
}

const std::optional<std::filesystem::path> &_SubsampleResultIO::getResultDir() const
{
    return _resultDir;
}