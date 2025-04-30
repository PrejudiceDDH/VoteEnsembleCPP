# VoteEnsembleCPP

A C++ implementation of Vote Ensemble methods, including MoVE (Majority Vote Ensemble) and ROVE (Ranked Order Vote Ensemble). This project provides examples using Linear Regression and a simple Linear Program as base learners.

## Dependencies

* CMake (version 3.10 or higher)
* A C++17 compliant compiler (e.g., GCC, Clang, MSVC)
* Eigen3 library
* Zstd library (for optional subsample result compression)

## Building

The project uses CMake.

1.  **Create a build directory:**
    ```bash
    mkdir build
    cd build
    ```
2.  **Run CMake:**
    ```bash
    # Adjust path to Zstd if necessary (as shown in your CMakeLists.txt)
    cmake ..
    # Example if Zstd isn't found automatically:
    # cmake .. -DZSTD_MANUAL_INCLUDE_DIR=/path/to/zstd/include -DZSTD_MANUAL_LIBRARY_DIR=/path/to/zstd/lib
    ```
3.  **Compile:**
    ```bash
    make
    # Or using cmake --build . on newer CMake versions
    # cmake --build .
    ```
    This will create an executable named `vote_ensemble_app` in the `build` directory.

## Usage

Run the compiled executable from the `build` directory, specifying the example name (`LR` or `LP`) as a command-line argument.

```bash
./vote_ensemble_app LR
```

or

```bash
./vote_ensemble_app LP
```

The program will run the corresponding example (Linear Regression or Linear Program), applying MoVE and/or ROVE, print the results, and store intermediate subsample results in test directories (`LR_storage_test`, `LP_storage_test`) if external storage is enabled.