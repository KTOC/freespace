#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "lodepng" for configuration "Debug"
set_property(TARGET lodepng APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(lodepng PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_DEBUG "CXX"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/lib/liblodepngd.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS lodepng )
list(APPEND _IMPORT_CHECK_FILES_FOR_lodepng "${_IMPORT_PREFIX}/lib/liblodepngd.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
