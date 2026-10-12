# VoxEng
Refined version of the previous c engine. Made in C++ and GLSL.

## HOW TO COMPILE
You need a modern C++ Compiler (e.g. GCC, CLang, or MinGW) alongside CMake.

### Here is the step by step instructions to compile this engine:

1. Clone the repo to your designated directory and run (git clone https://github.com/Anont187/VoxEng).
2. Open the terminal and go inside the directory where you cloned the repo using commands.
3. Run these commands after step 2:
```bash
# Build the configurations
cmake -B build -DCMAKE_BUILD_TYPE=Release

# Compile the build
cmake --build build --config Release
```

The compiled native Executable will be in the `/build` folder.
