cmake_minimum_required(VERSION 2.8)


SET(SOURCES
	main.cpp
  )

SET(HEADERS
  )

set(FECRS_TRILINOS_LIBS ${Trilinos_LIBRARIES} ${Trilinos_TPL_LIBRARIES})

include_directories (
    ${Trilinos_INCLUDE_DIRS}
    .
    ${Trilinos_TPL_INCLUDE_DIRS}
)

link_directories (
    ${Trilinos_LIBRARY_DIRS} 
    ${Trilinos_TPL_LIBRARY_DIRS} 
)

add_library(fecrsLib ${SOURCES} ${HEADERS})

find_package(Boost 1.46.0 COMPONENTS program_options REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

# Now build executables from this library, tailored main()s, and Trilinos

add_executable(fecrs main.cpp)

SET(FECRS_LIBRARIES
    fecrsLib
    stk_io
    stk_mesh_base
    stk_mesh_fem
    stk_util_diag
    stk_util_env
    stk_util_parallel
    stk_util_util
    teuchos
    epetra
    shards
    curl hdf5 hdf5_hl exodus netcdf z blas
)

target_link_libraries(fecrs ${FECRS_LIBRARIES} ${Boost_LIBRARIES}) 

if(GTEST_FOUND)
   enable_testing(true)
   add_executable(fecrs_unit_tests fecrs_unit_tests.cpp)
   target_link_libraries(fecrs_unit_tests ${FECRS_LIBRARY})
   target_link_libraries(fecrs_unit_tests ${GTEST_BOTH_LIBRARIES})
   add_test(fecrs_UnitTests fecrs_unit_tests)
endif()
