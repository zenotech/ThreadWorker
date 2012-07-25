#-----------------------------------------------------------------------------
# UseCME.cmake
#
# This module is provided as CME_USE_FILE by CMEConfig.cmake.  It can
# be INCLUDEd in a project to load the needed compiler and linker
# settings to use CME.
#

# Load the compiler settings used for CME.
IF(CME_BUILD_SETTINGS_FILE)
  INCLUDE(${CMAKE_ROOT}/Modules/CMakeImportBuildSettings.cmake)
  CMAKE_IMPORT_BUILD_SETTINGS(${CME_BUILD_SETTINGS_FILE})
ENDIF(CME_BUILD_SETTINGS_FILE)

# Add include directories needed to use CME.
INCLUDE_DIRECTORIES(${CME_INCLUDE_DIRS})

# Add link directories needed to use CME.
LINK_DIRECTORIES(${CME_LIBRARY_DIRS})
