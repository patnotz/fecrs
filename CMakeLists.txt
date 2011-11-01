cmake_minimum_required(VERSION 2.8)

PROJECT(fecrs)

FIND_PACKAGE(Trilinos PATHS ${TRILINOS_PATH}/lib/cmake/Trilinos ${TRILINOS_PATH})
IF(NOT Trilinos_FOUND)
  MESSAGE(FATAL_ERROR "Could not find Trilinos!")
ENDIF()
MESSAGE("   Found Trilinos!  Here are the details: ")
MESSAGE("      Trilinos_DIR = ${Trilinos_DIR}")
MESSAGE("      Trilinos_VERSION = ${Trilinos_VERSION}")
#MESSAGE("   Trilinos_PACKAGE_LIST = ${Trilinos_PACKAGE_LIST}")
#MESSAGE("   Trilinos_LIBRARIES = ${Trilinos_LIBRARIES}")
#MESSAGE("   Trilinos_INCLUDE_DIRS = ${Trilinos_INCLUDE_DIRS}")
#MESSAGE("   Trilinos_LIBRARY_DIRS = ${Trilinos_LIBRARY_DIRS}")
#MESSAGE("   Trilinos_TPL_LIST = ${Trilinos_TPL_LIST}")
#MESSAGE("   Trilinos_TPL_INCLUDE_DIRS = ${Trilinos_TPL_INCLUDE_DIRS}")
#MESSAGE("   Trilinos_TPL_LIBRARIES = ${Trilinos_TPL_LIBRARIES}")
#MESSAGE("   Trilinos_TPL_LIBRARY_DIRS = ${Trilinos_TPL_LIBRARY_DIRS}")
#MESSAGE("   Trilinos_BUILD_SHARED_LIBS = ${Trilinos_BUILD_SHARED_LIBS}")
MESSAGE("   End of Trilinos details")

IF(NOT CMAKE_CXX_COMPILER)
  MESSAGE("Setting CXX compiler.")
  SET(CMAKE_CXX_COMPILER ${Trilinos_CXX_COMPILER})
ENDIF()

# If you haven't already set the C compiler, use the same compiler
# that was used to build Trilinos on your machine.  You do _not_ 
# need to edit this line.
IF(NOT CMAKE_C_COMPILER)
  MESSAGE("Setting C compiler.")
  SET(CMAKE_C_COMPILER ${Trilinos_C_COMPILER})
ENDIF()

#add_definitions("-g -O2 -std=c++0x")
add_definitions("-g -O2")

# End of setup and error checking
#  NOTE: PROJECT command checks for compilers, so this statement
#        is moved AFTER setting CMAKE_CXX_COMPILER opton

# MPI check -- defaults to TRUE
MESSAGE("-- Checking if MPI is enabled in Trilinos:")
LIST(FIND Trilinos_TPL_LIST MPI MPI_List_ID)
IF (MPI_List_ID GREATER -1)
  MESSAGE("-- Checking if MPI is enabled in Trilinos: MPI ENABLED")
  SET(FECRS_MPI TRUE)
  ADD_DEFINITIONS(-DFECRS_MPI)
ELSE()
  MESSAGE("-- Checking if MPI is enabled in Trilinos: MPI NOT ENABLED")
  SET(FECRS_MPI FALSE)
ENDIF()

################################################################################
            
# GTest - http://code.google.com/p/googletest/
#                       
################################################################################
add_subdirectory(gtest-1.6.0)
enable_testing(true)
                                                                                             
################################################################################
# Boost
################################################################################
set(Boost_ADDITIONAL_VERSIONS 1.47 1.47.0)
find_package(Boost 1.47 REQUIRED COMPONENTS filesystem system program_options)



################################################################################
# fecrs
################################################################################
include_directories (
    src
    ${Boost_INCLUDE_DIRS}
)

link_directories (
)

SET(SOURCES 
  # e.g., src/foo.cpp
)

SET(HEADERS
  # e.g., src/foo.hpp
  src/typeinfo.hpp
)

set(FECRS_TRILINOS_LIBS ${Trilinos_LIBRARIES} ${Trilinos_TPL_LIBRARIES})

include_directories (
    ${Trilinos_INCLUDE_DIRS}
    ${Trilinos_TPL_INCLUDE_DIRS}
    src
)

link_directories (
    ${Trilinos_LIBRARY_DIRS} 
    ${Trilinos_TPL_LIBRARY_DIRS} 
)

#add_library(fecrsLib ${SOURCES} ${HEADERS})

# Now build executables from this library, tailored main()s, and Trilinos

add_executable(fecrs src/main.cpp)

SET(FECRS_LIBRARIES
    #fecrsLib
    stk_mesh_base
    stk_mesh_fem
    stk_util_diag
    stk_util_env
    stk_util_parallel
    stk_util_util
    teuchos
    epetra
    shards
    tpetra
    curl hdf5 hdf5_hl exodus netcdf z blas
)

target_link_libraries(fecrs ${FECRS_LIBRARIES} ${Boost_LIBRARIES}) 

include_directories(src ${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})
add_executable(fecrs_unit_tests src/fecrs_unit_tests.cpp)
target_link_libraries(fecrs_unit_tests ${FECRS_LIBRARY})
target_link_libraries(fecrs_unit_tests gtest gtest_main)
add_test(fecrs_UnitTests fecrs_unit_tests)


