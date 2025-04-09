message("-------------------------start-----------------")
message(${CMAKE_SOURCE_DIR})

cmake_minimum_required(VERSION 3.15.0)
project(myCodeToNewStart VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
# 在 CMake 中，prefix 并不是「文件名开头」的意思，而是依赖库的安装根目录，是类 Unix 编译/安装语境下的术语。可以把它理解为“额外的搜索目录提示”。
set(CMAKE_PREFIX_PATH "/usr/local/gtest") 
set(CMAKE_BUILD_TYPE Debug)
# 保存中间文件，可能会出现问题，因为保存中间文件时，编译会有一些不同
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -save-temps")

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -gsplit-dwarf")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -s")

message("Flag: ${CMAKE_CXX_FLAGS}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin")
# set(CMAKE_CACHEFILE_DIR "/dev/null")
# BINARY是当前构建目录，source是cmakelist所在目录， 这两个是只读变量，不能在这里修改的
# set(CMAKE_BINARY_DIR "${CMAKE_SOURCE_DIR}/build")

# find_package(GTest REQUIRED)
# find_package(TCL REQUIRED)
# find_package(Eigen3 REQUIRED)
# find_package(OpenGL REQUIRED)
# find_package(GLUT REQUIRED)
# find_package(SFML 2.5 COMPONENTS graphics window system REQUIRED)
# find_package(glfw3 REQUIRED)  # 查找 GLFW3 库
find_package(PkgConfig REQUIRED)  # 查找 pkg-config
# 用pkg-config添加GRAPHVIZ
pkg_check_modules(GRAPHVIZ REQUIRED libgvc libcgraph)  # 查找 Graphviz 的库

# file(GLOB imgui_src "imgui/*.cpp")
# add_library(imgui STATIC
#             ${imgui_src}
# ) 

# set(Eigen3_INCLUDE_DIRS ${EIGEN3_INCLUDE_DIR})
# message("eigen include path: ${EIGEN3_INCLUDE_DIR}")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include")
# include_directories("${CMAKE_CURRENT_SOURCE_DIR}/imgui")
# include_directories(${GTEST_INCLUDE_DIRS})
# include_directories(${TCL_INCLUDE_PATH})
# include_directories(${Eigen3_INCLUDE_DIRS})
include_directories(${GRAPHVIZ_INCLUDE_DIRS})

file(GLOB SOURCE "src/*.cpp")
add_executable(micrograd ${SOURCE})
target_link_libraries(micrograd ${GRAPHVIZ_LIBRARIES} pthread)


message("-------------------------end-----------------")