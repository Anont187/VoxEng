cmake_minimum_required(3.10)
project(VoxEng VERSION 1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED on)

if(MSVC)
  add_compile_options(/02 /arch:AVX2 /DNDEBUG)
else()
  add_compile_options(-Ofast -march=native -ffunction-sections -fdata-sections -DNDEBUG)
  add_link_options(-Wl,--gc-sections,--strip-all -s)
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/lib)

set(SOURCES 
    gameEngine.cpp
    glad.c
)

if(WIN32)
  target_link_libraries(gameEngine glfw3 opengl32 gdi32 psapi)
  target_link_options(gameEngine PRIVATE -static)
elseif(APPLE)
  find_package(GLFW3 REQUIRED)
  target_link_libraries(gameEngine glfw opengl "-framework Cocoa" "-framework IOKit" "-framework CoreVideo")
else()
  find_package(GLFW3 REQUIRED)
  target_link_libraries(gameEngine glfw GL X11 lpthread lXrandr lXi dl)
endif()
