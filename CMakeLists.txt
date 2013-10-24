# -*- CMake -*-  ---------------------------------------------------------

cmake_minimum_required(VERSION 2.6)
project(ThreadWorker)

# project version
SET(MAJOR_VERSION 0)
SET(MINOR_VERSION 1)
SET(PATCH_VERSION 0)

#FIND_PACKAGE(Subversion)
#IF(Subversion_FOUND)
#    Subversion_WC_INFO(${PROJECT_SOURCE_DIR} Project)
#    MESSAGE("Current revision is ${Project_WC_REVISION}")
#    #Subversion_WC_LOG(${PROJECT_SOURCE_DIR} Project)
#    #MESSAGE("Last changed log is ${Project_LAST_CHANGED_LOG}")
#
#    SET (PATCH_VERSION ${PATCH_VERSION}.${Project_WC_REVISION})
#ENDIF(Subversion_FOUND)

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
SET(${PROJECT_NAME}_SRCS)

# Add source files
SET(${PROJECT_NAME}_SRCS
	src/ThreadWorker.cpp)


#SET(Boost_ADDITIONAL_VERSIONS "1.44" "1.44.0" "1.43" "1.43.0" "1.42" "1.42.0" "1.41" ) 
SET(Boost_NO_SYSTEM_PATHS ON)
FIND_PACKAGE(Boost 1.52 REQUIRED COMPONENTS  thread system)
link_directories ( ${Boost_LIBRARY_DIRS} )
include_directories ( ${Boost_INCLUDE_DIRS} )
message(STATUS "Boost found in : " ${Boost_INCLUDE_DIRS})
if(WIN32)
	add_definitions(-DBOOST_ALL_DYN_LINK -DBOOST_BIND_ENABLE_STDCALL)
	add_definitions( "/W3 /D_CRT_SECURE_NO_WARNINGS /wd4068 /wd4996 /wd4251 /wd4275 /nologo" )
endif(WIN32)

# Find CUDA
FIND_PACKAGE(CUDA)
MESSAGE(STATUS "CUDA found in : " ${CUDA_TOOLKIT_INCLUDE})
IF(CUDA_FOUND)
	INCLUDE_DIRECTORIES ( ${CUDA_TOOLKIT_INCLUDE} )
	LIST(APPEND ${PROJECT_NAME}_LIBS ${CUDA_CUDA_LIBRARY} ${CUDA_CUDART_LIBRARY} )
ENDIF(CUDA_FOUND)

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
TARGET_LINK_LIBRARIES(${PROJECT_NAME} ${${PROJECT_NAME}_LIBS} ${Boost_THREAD_LIBRARY} ${Boost_SYSTEM_LIBRARY})

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
	target_link_libraries(${PROJECT_NAME}CUDA ${${PROJECT_NAME}_LIBS} ${Boost_THREAD_LIBRARY} ${Boost_SYSTEM_LIBRARY})
	INSTALL(TARGETS ${PROJECT_NAME}CUDA DESTINATION lib 
			PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
			GROUP_READ GROUP_EXECUTE
			WORLD_READ WORLD_EXECUTE
			EXPORT ${PROJECT_NAME}CUDA-targets)
endif(CUDA_FOUND)


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
               

#CONFIGURE_FILE(${${PROJECT_NAME}_SOURCE_DIR}/src/${PROJECT_NAME}Version.h.in
#               ${${PROJECT_NAME}_SOURCE_DIR}/src/${PROJECT_NAME}Version.h @ONLY IMMEDIATE)
               

# Save the compiler settings and library dependencies so another
# project can import them.
INCLUDE(${CMAKE_ROOT}/Modules/CMakeExportBuildSettings.cmake)
CMAKE_EXPORT_BUILD_SETTINGS(${${PROJECT_NAME}_BINARY_DIR}/${PROJECT_NAME}BuildSettings.cmake)
EXPORT_LIBRARY_DEPENDENCIES(${${PROJECT_NAME}_BINARY_DIR}/${PROJECT_NAME}LibraryDepends.cmake)


INCLUDE(InstallRequiredSystemLibraries)
SET(CPACK_PACKAGE_LIBRARIES ${PROJECT_NAME})
#set(CPACK_COMPONENTS_ALL applications libraries headers)
INCLUDE(CPack)

