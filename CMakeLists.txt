cmake_minimum_required(VERSION 3.18)
project(VoxEng VERSION 1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC)
    add_compile_options(/O2 /Ob2 /Oi /Ot /Oy /favor:ATOM /GF /DNDEBUG)
elseif(APPLE)
    message(FATAL_ERROR "VoxEng requires OpenGL 4.3 Core (Compute Shaders & SSBOs), which macOS does not support (macOS caps at OpenGL 4.1).")
else()
    add_compile_options(-O3 -march=native -ffunction-sections -fdata-sections -DNDEBUG)
    add_link_options(-Wl,--gc-sections -s)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/glm.zip")
    message(STATUS "Extracting glm.zip...")
    file(ARCHIVE_EXTRACT
        INPUT "${CMAKE_CURRENT_SOURCE_DIR}/glm.zip"
        DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/extracted_glm"
    )
    include_directories("${CMAKE_CURRENT_BINARY_DIR}/extracted_glm")
endif()

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include")
link_directories("${CMAKE_CURRENT_SOURCE_DIR}/lib")

set(SOURCES gameEngine.cpp glad.c)

add_executable(gameEngine ${SOURCES})

if(WIN32)
    target_link_libraries(gameEngine PRIVATE glfw3 opengl32 gdi32 psapi)
    if(MINGW)
        target_link_options(gameEngine PRIVATE -mwindows -static)
    endif()
else()
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(GLFW REQUIRED IMPORTED_TARGET glfw3)
    target_link_libraries(gameEngine PRIVATE PkgConfig::GLFW GL dl pthread)
endif()
