# -*- CMake -*-  ---------------------------------------------------------

cmake_minimum_required(VERSION 3.1)
option(INTEL_CC "Use the Intel compiler." OFF)
option(IBM_XL "Use IBM compiler." OFF)

# Fix behavior of CMAKE_CXX_STANDARD when targeting macOS.
if (POLICY CMP0025)
   cmake_policy(SET CMP0025 NEW)
endif ()
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Detect Intel Compiler - Note this needs to be before the project statement to work correctly
if (INTEL_CC)
  find_program(CMAKE_C_COMPILER NAMES icc)
  find_program(CMAKE_CXX_COMPILER NAMES icpc)
  find_program(CMAKE_AR NAMES xiar)
  find_program(CMAKE_LINKER NAMES xild)

  if (CMAKE_C_COMPILER MATCHES CMAKE_C_COMPILER-NOTFOUND OR
      CMAKE_CXX_COMPILER MATCHES CMAKE_CXX_COMPILER-NOTFOUND OR
      CMAKE_AR MATCHES CMAKE_AR-NOTFOUND OR
      CMAKE_LINKER MATCHES CMAKE_LINKER-NOTFOUND)
    message(FATAL_ERROR "Cannot find Intel compiler.  You may need to run `. /opt/intel/bin/compilervars.sh intel64'")
  endif ()
endif (INTEL_CC)
if (IBM_XL)
  find_program(CMAKE_C_COMPILER NAMES xlc)
  find_program(CMAKE_CXX_COMPILER NAMES xlc++)
  find_program(CMAKE_AR NAMES ar)
  find_program(CMAKE_LINKER NAMES ld)
  if (CMAKE_C_COMPILER MATCHES CMAKE_C_COMPILER-NOTFOUND OR
      CMAKE_CXX_COMPILER MATCHES CMAKE_CXX_COMPILER-NOTFOUND OR
      CMAKE_AR MATCHES CMAKE_AR-NOTFOUND OR
      CMAKE_LINKER MATCHES CMAKE_LINKER-NOTFOUND)
    message(FATAL_ERROR "Cannot find IBM compiler.")
  endif ()
endif ()

project(ThreadWorker)

# project version
SET(MAJOR_VERSION 0)
SET(MINOR_VERSION 1)
SET(PATCH_VERSION 0)


SET(PACKAGE_NAME ${PROJECT_NAME})
SET(PACKAGE_TARNAME ${PROJECT_NAME})
SET(CPACK_PACKAGE_VENDOR "UNKNOWN")
SET(CPACK_PACKAGE_VERSION_MAJOR "${MAJOR_VERSION}" )
SET(CPACK_PACKAGE_VERSION_MINOR "${MINOR_VERSION}" )
SET(CPACK_PACKAGE_VERSION_PATCH "${PATCH_VERSION}" )
SET(CPACK_PACKAGE_INSTALL_DIRECTORY "${PROJECT_NAME}-${MAJOR_VERSION}.${MINOR_VERSION}.${PATCH_VERSION}")
SET(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/license.txt")
SET(CPACK_PACKAGE_INSTALL_REGISTRY_KEY "${PROJECT_NAME} ${MAJOR_VERSION}.${MINOR_VERSION}.${PATCH_VERSION}")

ENABLE_TESTING()

SET(${PROJECT_NAME}_LIBS)
SET(${PROJECT_NAME}CUDA_LIBS)
SET(${PROJECT_NAME}HIP_LIBS)
SET(${PROJECT_NAME}_SRCS)

# Add source files
SET(${PROJECT_NAME}_SRCS
	src/ThreadWorker.cpp)

OPTION(BUILD_CUDA "Build CUDA Support" ON)
OPTION(BUILD_HIP "Build HIP Support" ON)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

# Find CUDA
IF(BUILD_CUDA)
    FIND_PACKAGE(CUDA)
    MESSAGE(STATUS "CUDA found in : " ${CUDA_TOOLKIT_INCLUDE})
    IF(CUDA_FOUND)
    	INCLUDE_DIRECTORIES ( ${CUDA_TOOLKIT_INCLUDE} )
    	LIST(APPEND ${PROJECT_NAME}CUDA_LIBS ${CUDA_CUDART_LIBRARY} )
    ENDIF(CUDA_FOUND)
ENDIF()

# Search for rocm in common locations
list(APPEND CMAKE_PREFIX_PATH /opt/rocm/hip /opt/rocm)

if(BUILD_HIP)
    find_package(hip REQUIRED)
endif()

# Add shared library target
ADD_LIBRARY(${PROJECT_NAME} SHARED ${${PROJECT_NAME}_SRCS})
# change lib_target properties
SET_TARGET_PROPERTIES( ${PROJECT_NAME} PROPERTIES
	# create *nix style library versions + symbolic links
	VERSION ${MAJOR_VERSION}
	SOVERSION ${MAJOR_VERSION}
	# allow creating static and shared libs without conflicts
	CLEAN_DIRECT_OUTPUT 1
	# avoid conflicts between library and binary target names
	OUTPUT_NAME ${PROJECT_NAME} )
TARGET_LINK_LIBRARIES(${PROJECT_NAME} ${${PROJECT_NAME}_LIBS} Threads::Threads)

if(CUDA_FOUND)
	add_library(${PROJECT_NAME}CUDA SHARED ${${PROJECT_NAME}_SRCS})
    set_target_properties(${PROJECT_NAME}CUDA PROPERTIES COMPILE_FLAGS "-DHAVE_CUDA" )
	SET_TARGET_PROPERTIES( ${PROJECT_NAME}CUDA PROPERTIES
	# create *nix style library versions + symbolic links
	  VERSION ${MAJOR_VERSION}
	  SOVERSION ${MAJOR_VERSION}
	# allow creating static and shared libs without conflicts
	  CLEAN_DIRECT_OUTPUT 1
	# avoid conflicts between library and binary target names
	  OUTPUT_NAME ${PROJECT_NAME}CUDA )
	target_link_libraries(${PROJECT_NAME}CUDA ${${PROJECT_NAME}_LIBS} ${${PROJECT_NAME}CUDA_LIBS} Threads::Threads)
	INSTALL(TARGETS ${PROJECT_NAME}CUDA DESTINATION lib
			PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
			GROUP_READ GROUP_EXECUTE
			WORLD_READ WORLD_EXECUTE
			EXPORT ${PROJECT_NAME}CUDA-targets)
endif()

if(hip_FOUND)
	add_library(${PROJECT_NAME}HIP SHARED ${${PROJECT_NAME}_SRCS})
    set_target_properties(${PROJECT_NAME}HIP PROPERTIES COMPILE_FLAGS "-DHAVE_HIP" )
	set_target_properties( ${PROJECT_NAME}HIP PROPERTIES
	# create *nix style library versions + symbolic links
	  VERSION ${MAJOR_VERSION}
	  SOVERSION ${MAJOR_VERSION}
	# allow creating static and shared libs without conflicts
	  CLEAN_DIRECT_OUTPUT 1
	# avoid conflicts between library and binary target names
	  OUTPUT_NAME ${PROJECT_NAME}HIP )
	target_link_libraries(${PROJECT_NAME}HIP ${${PROJECT_NAME}_LIBS} ${${PROJECT_NAME}HIP_LIBS} Threads::Threads hip::host)
	install(TARGETS ${PROJECT_NAME}HIP DESTINATION lib
			PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
			GROUP_READ GROUP_EXECUTE
			WORLD_READ WORLD_EXECUTE
			EXPORT ${PROJECT_NAME}HIP-targets)
endif()


# install library
INSTALL(TARGETS ${PROJECT_NAME} DESTINATION lib
	PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
	GROUP_READ GROUP_EXECUTE
	WORLD_READ WORLD_EXECUTE
	EXPORT ${PROJECT_NAME}-targets)

# Add files to be installed
INSTALL(FILES src/${PROJECT_NAME}.h DESTINATION include)
#INSTALL(FILES src/${PROJECT_NAME}Version.h DESTINATION include)
INSTALL(FILES CMake/Find${PROJECT_NAME}.cmake DESTINATION .)
INSTALL(EXPORT ${PROJECT_NAME}-targets DESTINATION lib)

# Add Unit Test
ADD_EXECUTABLE(${PROJECT_NAME}Test src/${PROJECT_NAME}Test.cpp)
TARGET_LINK_LIBRARIES(${PROJECT_NAME}Test ${PROJECT_NAME})

ADD_TEST(${PROJECT_NAME}Test ${EXECUTABLE_OUTPUT_PATH}/${PROJECT_NAME}Test)


ADD_SUBDIRECTORY(doc)

# Create the ${PROJECT_NAME}Config.cmake file for installation.
CONFIGURE_FILE(${${PROJECT_NAME}_SOURCE_DIR}/CMake/${PROJECT_NAME}Config.cmake.in
               ${${PROJECT_NAME}_BINARY_DIR}/${PROJECT_NAME}Config.cmake @ONLY IMMEDIATE)


INCLUDE(InstallRequiredSystemLibraries)
SET(CPACK_PACKAGE_LIBRARIES ${PROJECT_NAME})
#set(CPACK_COMPONENTS_ALL applications libraries headers)
INCLUDE(CPack)

