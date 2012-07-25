# -*- CMake -*-  ---------------------------------------------------------


# Create documentation
CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in
               ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY IMMEDIATE)
FIND_PACKAGE(Doxygen)
IF(DOXYGEN_FOUND)
    ADD_CUSTOM_TARGET(doc ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
COMMENT "Generating API documentation with Doxygen" VERBATIM
)
ENDIF(DOXYGEN_FOUND)
