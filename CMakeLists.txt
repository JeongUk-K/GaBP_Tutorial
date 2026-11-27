cmake_minimum_required(VERSION 3.16)
project(GaBP_Tutorial LANGUAGES CXX)

option(GABP_BUILD_PYTHON "Build the gabp Python extension" OFF)
if(GABP_BUILD_PYTHON)
    find_package(Python 3.8 REQUIRED COMPONENTS Interpreter Development)
    set(GSOLVER_BUILD_PYTHON ON)
endif()

set(GSOLVER_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/solver/gsolver" CACHE PATH "gsolver source checkout")
add_subdirectory("${GSOLVER_SOURCE_DIR}" gsolver EXCLUDE_FROM_ALL)

add_library(gabp_factors STATIC
    src/factors/range_factor.cpp
    src/factors/ramge_bearing_factor.cpp
)
target_compile_features(gabp_factors PUBLIC cxx_std_17)
target_include_directories(gabp_factors
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include"
    PRIVATE "${GSOLVER_SOURCE_DIR}/gsolver/src"
)
target_link_libraries(gabp_factors PUBLIC gsolver::gsolver)

add_library(gabp_multirobot_graph STATIC src/multirobot/MultiRobotGraph.cpp)
target_compile_features(gabp_multirobot_graph PUBLIC cxx_std_17)
target_include_directories(gabp_multirobot_graph PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(gabp_multirobot_graph PUBLIC gsolver::gsolver)

if(GABP_BUILD_PYTHON)
    set_target_properties(gabp_factors gabp_multirobot_graph PROPERTIES POSITION_INDEPENDENT_CODE ON)

    # Reuse the nanobind version configured by gsolver and stage both modules for import.
    nanobind_add_module(gabp STABLE_ABI NB_STATIC python/bindings.cpp)
    target_link_libraries(gabp PRIVATE gabp_factors gabp_multirobot_graph)
    add_dependencies(gabp _gsolver)

    set(GABP_PYTHON_DIR "${CMAKE_CURRENT_BINARY_DIR}/python/$<CONFIG>")
    set_target_properties(gabp PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${GABP_PYTHON_DIR}")
    set_target_properties(_gsolver PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${GABP_PYTHON_DIR}/gsolver")
    file(GENERATE OUTPUT "${GABP_PYTHON_DIR}/gsolver/__init__.py"
        INPUT "${GSOLVER_SOURCE_DIR}/python/__init__.py")

    execute_process(
        COMMAND "${Python_EXECUTABLE}" -c
                "import os, sys, sysconfig; print(os.path.relpath(sysconfig.get_path('platlib'), sys.prefix))"
        OUTPUT_VARIABLE GABP_PYTHON_INSTALL_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(NOT GABP_PYTHON_INSTALL_DIR)
        message(FATAL_ERROR "Could not find the site-packages path for ${Python_EXECUTABLE}")
    endif()
    install(TARGETS gabp
        LIBRARY DESTINATION "${GABP_PYTHON_INSTALL_DIR}" COMPONENT gabp-python
        RUNTIME DESTINATION "${GABP_PYTHON_INSTALL_DIR}" COMPONENT gabp-python)
    install(TARGETS _gsolver
        LIBRARY DESTINATION "${GABP_PYTHON_INSTALL_DIR}/gsolver" COMPONENT gabp-python
        RUNTIME DESTINATION "${GABP_PYTHON_INSTALL_DIR}/gsolver" COMPONENT gabp-python)
    install(FILES "${GSOLVER_SOURCE_DIR}/python/__init__.py"
        DESTINATION "${GABP_PYTHON_INSTALL_DIR}/gsolver" COMPONENT gabp-python)
endif()
