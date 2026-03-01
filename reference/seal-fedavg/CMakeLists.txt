cmake_minimum_required(VERSION 3.15)
project(seal_fedavg LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# If using vcpkg, SEAL should be found automatically when you pass -DCMAKE_TOOLCHAIN_FILE=...
find_package(seal REQUIRED)

add_executable(keygen keygen.cpp)
target_link_libraries(keygen PRIVATE SEAL::seal)

add_executable(hospital_encrypt hospital_encrypt.cpp)
target_link_libraries(hospital_encrypt PRIVATE SEAL::seal)

add_executable(aggregator aggregator.cpp)
target_link_libraries(aggregator PRIVATE SEAL::seal)

add_executable(decrypt decrypt.cpp)
target_link_libraries(decrypt PRIVATE SEAL::seal)
