cmake_minimum_required(VERSION 3.26)
project(native)

set(CMAKE_CXX_STANDARD 23)
#set(CMAKE_BUILD_TYPE Release)
#set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# Add all source files to build
file(GLOB JEVM_SRC "src/*.cpp" "src/*.h")
add_library(native SHARED ${JEVM_SRC})

# JNI dependencies
find_package(JNI REQUIRED)
include_directories(${JNI_INCLUDE_DIRS})

# Rename output
set_target_properties(native PROPERTIES PREFIX "")
set_target_properties(native PROPERTIES OUTPUT_NAME "matrix")
set_target_properties(native PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib/)

# Add c++ standard libraries to dll
set(CMAKE_SHARED_LINKER_FLAGS " -static")
target_link_libraries(native -static-libgcc -static-libstdc++)

# Create test.exe executable
add_executable(test ${JEVM_SRC})
set_target_properties(test PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR} )

set(CMAKE_EXE_LINKER_FLAGS  " -static")
target_link_libraries(test -static-libgcc -static-libstdc++)
