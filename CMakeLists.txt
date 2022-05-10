cmake_minimum_required(VERSION 3.21)
project(GenRBF)

set(CMAKE_CXX_STANDARD 14)

include_directories(.)

add_executable(GenRBF
        dataset.cpp
        dataset.h
        kmeans.cpp
        kmeans.h
        main.cpp
        rbf.cpp
        rbf.h)
