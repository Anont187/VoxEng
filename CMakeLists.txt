cmake_minimum_required(VERSION 3.12)
project(VoxEng VERSION 1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED on)

if(MSVC)
  add_compile_options(/Ox /Ob2 /Oi /Ot /Oy /favor:ATOM /GF /DNDEBUG)
else()
  add_compile_options(-Ofast -march=native -ffunction-sections -fdata-sections -DNDEBUG)
  add_link_options(-Wl,--gc-sections,--strip-all -s)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/glm.zip")
    message(STATUS "Unzip(ing) glm.zip...")
    
    file(ARCHIVE_EXTRACT 
        INPUT "${CMAKE_CURRENT_SOURCE_DIR}/glm.zip"
        DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/extracted_glm"
    )

    include_directories(${CMAKE_CURRENT_BINARY_DIR}/extracted_glm)
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/lib)

set(SOURCES 
    gameEngine.cpp
    glad.c
)

add_executable(gameEngine ${SOURCES})

if(WIN32)
  target_link_libraries(gameEngine glfw3 opengl32 gdi32 psapi)
  target_link_options(gameEngine PRIVATE -static)
elseif(APPLE)
  find_package(GLFW3 REQUIRED)
  target_link_libraries(gameEngine glfw opengl "-framework Cocoa" "-framework IOKit" "-framework CoreVideo")
else()
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(GLFW REQUIRED glfw3)
  include_directories(${GLFW_INCLUDE_DIRS})
  target_link_libraries(gameEngine ${GLFW_LIBRARIES} GL X11 pthread dl)
endif()
