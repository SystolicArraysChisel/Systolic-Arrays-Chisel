# Header
cmake_minimum_required(VERSION 3.20)
project(tpu_host LANGUAGES CXX)

# Project-wide settings
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Dependencies
include(FetchContent)
FetchContent_Declare(
    json
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
)
FetchContent_MakeAvailable(json)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/releases/download/v1.15.2/googletest-1.15.2.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(googletest)

# The library
add_library(
    tpu_host_lib
    STATIC
    src/protocol.cpp
    src/config.cpp
    src/quant.cpp
    src/reference.cpp
    src/fake_tpu.cpp
    src/serial_link.cpp
    include/tpu_host/link.hpp
    include/tpu_host/tpu_driver.hpp
    src/tpu_driver.cpp
    include/tpu_host/digits.hpp
    src/digits.cpp
)
target_include_directories(tpu_host_lib PUBLIC include)
target_link_libraries(tpu_host_lib PRIVATE nlohmann_json::nlohmann_json)
target_compile_features(tpu_host_lib PUBLIC cxx_std_20)
target_compile_options(tpu_host_lib PRIVATE -Wall -Wextra -Wpedantic -Wconversion)

# The executable
add_executable(run_board src/run_board.cpp)
target_link_libraries(run_board PUBLIC tpu_host_lib)
add_executable(run_mlp src/run_mlp.cpp)
target_link_libraries(run_mlp PRIVATE tpu_host_lib)

# The user application (GUI), only if Qt 6 is installed (MSYS2: pacman -S mingw-w64-x86_64-qt6-base)
find_package(Qt6 QUIET COMPONENTS Widgets)
if(Qt6_FOUND)
    qt_standard_project_setup()
    qt_add_executable(
        tpu_app
        WIN32
        gui/main.cpp
        gui/main_window.hpp
        gui/main_window.cpp
        gui/model_page.hpp
        gui/model_page.cpp
        gui/demo_page.hpp
        gui/demo_page.cpp
        gui/programmer.hpp
        gui/programmer.cpp
        gui/widgets.hpp
        gui/widgets.cpp
    )
    target_link_libraries(tpu_app PRIVATE tpu_host_lib Qt6::Widgets)
    target_compile_definitions(
        tpu_app
        PRIVATE
            DEMO_DIR="${CMAKE_CURRENT_SOURCE_DIR}/demo"
            REPO_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../.."
    )
else()
    message(STATUS "Qt 6 not found: tpu_app is not built")
endif()

# Tests
enable_testing()
add_executable(
    tpu_host_tests
    tests/test_protocol.cpp
    tests/test_config.cpp
    tests/test_ref_vectors.cpp
    tests/test_matrix.cpp
    tests/test_quant.cpp
    tests/test_reference.cpp
    tests/test_link.cpp
    tests/test_fake_tpu.cpp
    tests/test_tpu_driver.cpp
    tests/test_serial_link.cpp
    tests/test_board.cpp
    tests/test_digits.cpp
)
target_compile_definitions(
    tpu_host_tests
    PRIVATE
        TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/data"
        REF_VECTORS="${CMAKE_CURRENT_SOURCE_DIR}/../../src/test/resources/ref_vectors.txt"
        DIGITS_MODEL="${CMAKE_CURRENT_SOURCE_DIR}/demo/digits_model.json"
        SHAPES_MODEL="${CMAKE_CURRENT_SOURCE_DIR}/demo/shapes_model.json"
)
target_link_libraries(tpu_host_tests PRIVATE tpu_host_lib GTest::gtest_main GTest::gmock)
include(GoogleTest)
gtest_discover_tests(tpu_host_tests)
