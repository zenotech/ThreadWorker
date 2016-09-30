# - Find ThreadWorker
# Find the ThreadWorker includes and library
#
#  THREADWORKER_INCLUDES    - where to find ThreadWorker.h, etc
#  THREADWORKER_LIBRARIES   - Link these libraries when using ThreadWorker
#

if (THREADWORKER_INCLUDES AND THREADWORKER_LIBRARIES)
  # Already in cache, be silent
  set (THREADWORKER_FIND_QUIETLY TRUE)
endif (THREADWORKER_INCLUDES AND THREADWORKER_LIBRARIES)

find_path (THREADWORKER_INCLUDES ThreadWorker.h
  HINTS THREADWORKER_DIR "$ENV{ProgramFiles}/ThreadWorker" ENV THREADWORKER_DIR PATH_SUFFIXES include)

find_library (THREADWORKER_LIBRARIES  NAMES ThreadWorker HINTS THREADWORKER_DIR "$ENV{ProgramFiles}/ThreadWorker" ENV THREADWORKER_DIR PATH_SUFFIXES lib)
mark_as_advanced(THREADWORKER_LIBRARIES THREADWORKER_INCLUDES)

include (FindPackageHandleStandardArgs)
find_package_handle_standard_args (ThreadWorker DEFAULT_MSG THREADWORKER_LIBRARIES THREADWORKER_INCLUDES)
