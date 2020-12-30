##
#  CMake script for the step-3 tutorial program:
##

# Set the name of the project and target:
SET(TARGET "assembly_routine_SRI")


# Declare all source files the target consists of. Here, this is only
# the one step-X.cc file, but as you expand your project you may wish
# to add other source files as well. If your project becomes much larger,
# you may want to either replace the following statement by something like
#    FILE(GLOB_RECURSE TARGET_SRC  "source/*.cc")
#    FILE(GLOB_RECURSE TARGET_INC  "include/*.h")
#    SET(TARGET_SRC ${TARGET_SRC}  ${TARGET_INC}) 
# or switch altogether to the large project CMakeLists.txt file discussed
# in the "CMake in user projects" page accessible from the "User info"
# page of the documentation.

include_directories(include)

SET(TARGET_SRC
  ${TARGET}.cc
  )

# Usually, you will not need to modify anything beyond this point...

CMAKE_MINIMUM_REQUIRED(VERSION 2.8.8)
#SET(DEAL_II_DIR /calculate/deal.II_V_8_4_1_mpi_hdf5/
#)
#SET(DEAL_II_DIR /home/dealiiuser/deal.II/
#)

FIND_PACKAGE(deal.II 9.1.1 QUIET
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
  )
IF(NOT ${deal.II_FOUND})
  MESSAGE(FATAL_ERROR "\n"
    "*** Could not locate a (sufficiently recent) version of deal.II. ***\n\n"
    "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
    "or set an environment variable \"DEAL_II_DIR\" that contains this path."
    )
ENDIF()

########################## DOXYGEN ############################
# setup target for documentation using doxygen
FIND_PACKAGE(Doxygen)
IF(NOT DOXYGEN_FOUND)
    MESSAGE("-- Doxygen was not found. Documentation will not be available.")
ENDIF()
IF(DOXYGEN_FOUND)

  FILE( MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/doc )

  ## Configure doxygen
  SET( DOXYGEN_PROJECT_NAME "Selective reduced integration (SRI) in deal.II" ) # Title of your project
  SET( DOXYGEN_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/docs ) # here you specify the output directory
  SET( DOXYGEN_QUIET YES )
  SET( DOXYGEN_ENABLE_PREPROCESSING YES )
  SET( DOXYGEN_PREDEFINED "DOXYGEN" )
  SET( DOXYGEN_EXTRACT_PRIVATE YES )
  SET( DOXYGEN_EXTRACT_PRIVATE YES )
  SET( DOXYGEN_SOURCE_BROWSER YES )
  SET( DOXYGEN_EXTRACT_ALL YES )
  SET( DOXYGEN_GENERATE_LATEX YES )

  SET( DOXYGEN_USE_MATHJAX YES )

  SET( DOXYGEN_FILE_PATTERNS *.cc *.h ) #Files to look for documentation
  SET( DOXYGEN_EXCLUDE_PATTERNS */build/* */OoO/* */docs/*) #<- Exclude doxygen from looking for documentation in these files.

  doxygen_add_docs( doc ${CMAKE_SOURCE_DIR}
                    COMMENT "Generating documentation with Doxygen")
                      
ENDIF(DOXYGEN_FOUND)
########################## END OF DOXYGEN ############################

DEAL_II_INITIALIZE_CACHED_VARIABLES()
PROJECT(${TARGET})
DEAL_II_INVOKE_AUTOPILOT() 
