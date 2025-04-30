# Minimum CMake version required
cmake_minimum_required(VERSION 3.10)

# Project name
project(VoteEnsembleCpp VERSION 1.0 LANGUAGES CXX)

# Set C++ standard (e.g., C++17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find required packages
# We still call find_package to potentially find the library correctly,
# but we won't rely on it for the include path.
find_package(Eigen3 REQUIRED)
find_package(Zstd REQUIRED)

# Add project's include directory
include_directories(include)

# --- Manually specify Zstd paths ---
set(ZSTD_MANUAL_INCLUDE_DIR "/usr/local/opt/zstd/include")
set(ZSTD_MANUAL_LIBRARY_DIR "/usr/local/opt/zstd/lib")
message(STATUS "Manual Zstd include path: ${ZSTD_MANUAL_INCLUDE_DIR}")
message(STATUS "Manual Zstd library path: ${ZSTD_MANUAL_LIBRARY_DIR}")

# Add manual library path for the linker
link_directories(${ZSTD_MANUAL_LIBRARY_DIR})
# --- End Manual Paths ---


# Add the executable target
add_executable(vote_ensemble_app
    src/main.cpp
    src/types.cpp
    src/_BaseVE.cpp
    src/_CachedEvaluator.cpp
    src/MoVE.cpp
    src/ROVE.cpp
    src/_SubsampleResultIO.cpp
    src/LinearRegressionLearner.cpp
    src/LinearProgramLearner.cpp
    src/VoteEnsembleRunner.cpp
)

# Add include directories *to the target*
if(NOT EXISTS "${ZSTD_MANUAL_INCLUDE_DIR}/zstd.h")
    message(FATAL_ERROR "Manually specified ZSTD include directory '${ZSTD_MANUAL_INCLUDE_DIR}' does not contain zstd.h! Please verify the path.")
endif()

target_include_directories(vote_ensemble_app PRIVATE
    ${EIGEN3_INCLUDE_DIRS}      # Add Eigen include directory
    ${ZSTD_MANUAL_INCLUDE_DIR}  # Add the manually specified Zstd include directory
)

# Link libraries to the executable
# Try linking using the explicit library name if ${ZSTD_LIBRARIES} fails
# Common name is 'zstd' -> libzstd.dylib on macOS
target_link_libraries(vote_ensemble_app PRIVATE
    Eigen3::Eigen     # Link Eigen using its imported target
    zstd              # Explicitly link libzstd (or ${ZSTD_LIBRARIES} if Zstd_FOUND sets it)
)

# Optional: Print configuration info
message(STATUS "Configuring VoteEnsembleCpp")
message(STATUS "Eigen3 found: ${Eigen3_FOUND}")
message(STATUS "Eigen3 include dirs: ${EIGEN3_INCLUDE_DIRS}")
message(STATUS "Zstd found (via find_package): ${Zstd_FOUND}")
message(STATUS "Zstd include variable (find_package): ${ZSTD_INCLUDE_DIRS}") # Likely empty
message(STATUS "Zstd library variable (find_package): ${ZSTD_LIBRARIES}") # May be empty