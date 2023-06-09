# SPDX-FileCopyrightText: 2022 UnionTech Software Technology Co., Ltd.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

#定义需要的cmake版本
cmake_minimum_required(VERSION 3.10)

# 设置工程名称
# project(libface VERSION 0.1.0)

 set(TARGET_NAME facerecognition)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS on)

set(CMAKE_C_FLAGS "-fPIC")
set(CMAKE_CXX_FLAGS "-fPIC")
# set(OpenCV_DIR "/usr/local/share/OpenCV")

find_package(PkgConfig REQUIRED)
# find_package(OpenCV REQUIRED)

pkg_check_modules(InferenceEngine REQUIRED ncnn opencv)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

include_directories(${PROJECT_BINARY_DIR})
include_directories(${PROJECT_SOURCE_DIR})

# set(PROJECT_INCLUDE
#     ${PROJECT_SOURCE_DIR}/interface/
#     ${PROJECT_SOURCE_DIR}/libfacerecognition/
# )

#文件夹包含
set(SRCS
    # libfacerecognition/utils.cpp
    libfacerecognition/facedetection.cpp
    libfacerecognition/facedetection.h
    libfacerecognition/facefeature.cpp
    libfacerecognition/facefeature.h
    libfacerecognition/facerecognition.cpp
    libfacerecognition/facerecognition.h
    libfacerecognition/liveness.cpp
    libfacerecognition/liveness.h
    interface/face.cpp
    interface/face.h
    cmdline.h
    main.cpp)

# file (GLOB_RECURSE SRCS LIST_DIRECTORIES ${PROJECT_SOURCE_DIR}/interface/*.cpp ${PROJECT_SOURCE_DIR}/libfacerecognition/*.cpp)
file (GLOB_RECURSE OUT_HEADERS ${CMAKE_CURRENT_LIST_DIR}/interface/face.h)

#-------------添加第三方库begins-----------------------#
set(INC_DIR /usr/include/)
set(LINK_DIR /usr/lib/)

include_directories(${INC_DIR})
link_directories(${LINK_DIR})

add_executable(${PROJECT_NAME} ${SRCS})

# 设置不删除生成的文件夹内容文件
set_directory_properties(PROPERTIES CLEAN_NO_CUSTOM 1)

# link_directories(${OpenCV_LIBRARY_DIRS})
# target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

target_include_directories(${PROJECT_NAME} PUBLIC
    ${PROJECT_INCLUDE}
    ${InferenceEngine_INCLUDE_DIRS}
)

target_link_libraries(${PROJECT_NAME} PRIVATE
    ${TARGET_LIBS}
    ${InferenceEngine_LIBRARIES}
)

# 设置不删除生成的文件夹内容文件
# set_directory_properties(PROPERTIES CLEAN_NO_CUSTOM 1)

#编译为库
add_library(${TARGET_NAME} SHARED ${SRCS} ${allHeaders} ${allSource})

#将库安装到指定位置
set(PREFIX /usr)
set(${TARGET_NAME} ${CMAKE_INSTALL_LIBDIR})

# set(ModelSaveSubDir models)
set(ModelInstallDir /usr/share/${TARGET_NAME})

file(GLOB MODELS_BIN ${CMAKE_SOURCE_DIR}/src/models/*.bin)
file(GLOB MODELS_PARAM ${CMAKE_SOURCE_DIR}/src/models/*.param)

set_target_properties(${TARGET_NAME} PROPERTIES VERSION 0.1.0 SOVERSION 0.1)

include(GNUInstallDirs)
configure_file(libfacerecognition.pc.in  ${PROJECT_BINARY_DIR}/libfacerecognition.pc @ONLY)

install(TARGETS ${TARGET_NAME} DESTINATION ${CMAKE_INSTALL_LIBDIR})

install(FILES ${OUT_HEADERS} DESTINATION include/libfacerecognition)

install(FILES ${PROJECT_BINARY_DIR}/libfacerecognition.pc DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig)

install(FILES ${MODELS_BIN} DESTINATION /usr/share/facerecognition/models/)
install(FILES ${MODELS_PARAM} DESTINATION /usr/share/facerecognition/models/)

# CUBE 颜色表文件 安装到usr/share下， 方便其他应用访问
# install(DIRECTORY ${PROJECT_SOURCE_DIR}/filter_cube
#     DESTINATION ${PREFIX}/share/libimagevisualresult
#     FILES_MATCHING PATTERN "*.CUBE")

# install(DIRECTORY ${PROJECT_SOURCE_DIR}/filter_cube
#     DESTINATION ${PREFIX}/share/libimagevisualresult
#     FILES_MATCHING PATTERN "*.dat")

# 加速编译优化参数
# if (CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "mips64")
#     SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wl,-O1 -Wl,--as-needed -Wl,-E -fPIE -ffunction-sections -fdata-sections -Wl,--gc-sections -ftree-vectorize -march=loongson3a -mhard-float -mno-micromips -mno-mips16 -flax-vector-conversions -mloongson-ext2 -mloongson-mmi -z noexecstack")
#     SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -Wl,-O1 -Wl,--as-needed -Wl,-E -fPIE -ffunction-sections -fdata-sections -Wl,--gc-sections -ftree-vectorize -march=loongson3a -mhard-float -mno-micromips -mno-mips16 -flax-vector-conversions -mloongson-ext2 -mloongson-mmi -z noexecstack")
#     SET(CMAKE_EXE_LINKER_FLAGS  "-pie")
# else()
#     SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wl,-O1 -Wl,--as-needed -Wl,-E -fPIE -ffunction-sections -fdata-sections -Wl,--gc-sections")
#     SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3  -Wl,-O1 -Wl,--as-needed -Wl,-E -fPIE -ffunction-sections -fdata-sections -Wl,--gc-sections")
#     SET(CMAKE_EXE_LINKER_FLAGS  "-pie")
# endif()
