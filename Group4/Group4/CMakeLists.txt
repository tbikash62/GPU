# CMakeList.txt : CMake project for Opencl-ex1, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

#Adding Opencl libs and include files to the proj
find_package(OpenCL)
include_directories(${OpenCL_INCLUDE_DIRS})
link_directories(${OpenCL_LIBRARY})

#adding the Boost libs to the proj
set(BOOST_INC "C:/local/boost_1_76_0_b1_rc2")
set(BOOST_LIB "C:/local/boost_1_76_0_b1_rc2/lib64-msvc-14.2/")

include_directories(${BOOST_INC})
link_directories(${BOOST_LIB})

#compile files in COre and OpenCL external libs
file(GLOB CORE_SRC "Core/*.cpp" "Core/*.c")
file(GLOB OPENCL_SRC "OpenCL/*.cpp" "OpenCL/*.c")
file(GLOB BOOST_SRC "${BOOST_LIB}/*.lib")

if(MSVC)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif()


add_executable (Group4 "OpenCLProject_Group4.cpp" ${CORE_SRC} ${OPENCL_SRC} )


target_include_directories (Group4 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} "CORE" "OPENCL" "src")
target_link_libraries (Group4 ${OpenCL_LIBRARY} imagehlp )

# TODO: Add tests and install targets if needed.
