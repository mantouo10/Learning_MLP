cmake_minimum_required(VERSION 2.8)
project(EasyML C CXX)


set( CMAKE_BUILD_TYPE "Debug" )
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -std=c++11")                          # C++ 生成共享链接库
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")							# C 生成共享链接库

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)	# 把Modules加入模块路径中,供加载

option(BUILD_EXAMPLES "building the examples" ON)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)				# 加入库路径
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/easyml)

add_subdirectory(src)													# 加入源文件路径

if (BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()
