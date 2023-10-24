# Copyright 2023 - David Minton
# This file is part of PyOOF
# PyOOF is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License 
# as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# pyoof is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty 
# of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with pyoof. 
# If not, see: https://www.gnu.org/licenses. 

# Communicate version number and other CMake build variables to the source code
set(GLOBAL_MODULE_IN ${SRC}/globals/globals_module.f90.in)
set(GLOBAL_MODULE_OUT ${SRC}/globals/globals_module.f90)
CONFIGURE_FILE(${GLOBAL_MODULE_IN} ${GLOBAL_MODULE_OUT})

# Add the source files
SET(SRC_FILES
    ${SRC}/globals/globals_module.f90
    ${SRC}/surface/surface_module.f90
    ${SRC}/bind/bind_module.f90
    ${SRC}/driver_module.f90
)

SET(DRIVER_src ${SRC}/main.f90)

# Combine all source files 
set(PYOOF_src ${SRC_FILES})

# Turn preprocessor on for all files
SET_SOURCE_FILES_PROPERTIES(${PYOOF_src} ${DRIVER_src} PROPERTIES Fortran_PREPROCESS ON)

# Set Build Type variable
STRING(TOUPPER "${CMAKE_BUILD_TYPE}" BT)

IF(BT STREQUAL "DEBUG")
   ADD_DEFINITIONS(-DDEBUG)
ELSEIF(BT STREQUAL "PROFILE")
   ADD_DEFINITIONS(-DPROFILE)
ENDIF()

#####################################################
# Add the needed libraries 
#####################################################
# Create a library from the source files, except the driver
ADD_LIBRARY(${PYOOF_LIBRARY} ${PYOOF_src})
IF (NOT BUILD_SHARED_LIBS) 
    SET_PROPERTY(TARGET ${PYOOF_LIBRARY} PROPERTY POSITION_INDEPENDENT_CODE)
ENDIF ()


# Define the executable name
SET(PYOOF_DRIVER pyoof_exe)
ADD_EXECUTABLE(${PYOOF_DRIVER} ${DRIVER_src})
SET_PROPERTY(TARGET ${PYOOF_DRIVER} PROPERTY OUTPUT_NAME pyoof)

IF(USE_OPENMP OR USE_SIMD)
    SET_PROPERTY(TARGET ${PYOOF_LIBRARY} ${PYOOF_DRIVER} APPEND_STRING PROPERTY COMPILE_FLAGS  "${OpenMP_Fortran_FLAGS} ")
    SET_PROPERTY(TARGET ${PYOOF_LIBRARY} ${PYOOF_DRIVER} APPEND_STRING PROPERTY LINK_FLAGS  "${OpenMP_Fortran_FLAGS} ")
ENDIF()

IF (CMAKE_SYSTEM_NAME STREQUAL "Windows")
   SET_PROPERTY(TARGET ${PYOOF_LIBRARY} ${PYOOF_DRIVER} APPEND_STRING PROPERTY LINK_FLAGS  "/NODEFAULTLIB")
ENDIF()

TARGET_LINK_LIBRARIES(${PYOOF_DRIVER} PUBLIC ${PYOOF_LIBRARY})
# Check to see if the compiler allows for local-spec in do concurrent statements. Set a preprocessor variable if it does
SET(TESTFILE "${CMAKE_BINARY_DIR}/${CMAKE_FILES_DIRECTORY}")
SET(TESTFILE "${TESTFILE}/CMakeTmp/testFortranDoConcurrentLoc.f90")
FILE(WRITE "${TESTFILE}"
"
program TestDoConcurrentLoc
integer :: i
real,dimension(10) :: a
do concurrent(i = 1:10) shared(a)
   a(i) = i 
end do
end program TestDoConcurrentLoc
")
TRY_COMPILE(DOCONLOC_WORKS ${CMAKE_BINARY_DIR} ${TESTFILE} COMPILE_DEFINITIONS "${CMAKE_Fortran_FLAGS}" OUTPUT_VARIABLE OUTPUT)
IF (DOCONLOC_WORKS)
    MESSAGE(STATUS "DO CONCURRENT supports locality-spec")
    TARGET_COMPILE_DEFINITIONS(${PYOOF_DRIVER} PRIVATE -DDOCONLOC)
ELSE ()
    MESSAGE(STATUS "DO CONCURRENT does not support locality-spec")
ENDIF (DOCONLOC_WORKS)

#####################################
# Tell how to install this executable
#####################################
IF(MSVC)
    SET(CMAKE_INSTALL_PREFIX "C:\\Program Files")
    FILE(TO_CMAKE_PATH ${CMAKE_INSTALL_PREFIX} CMAKE_INSTALL_PREFIX)
ELSE()
    SET(CMAKE_INSTALL_PREFIX /usr/local)
ENDIF(MSVC)
INSTALL(TARGETS ${PYOOF_DRIVER} ${PYOOF_LIBRARY}
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    INCLUDES DESTINATION include
)

