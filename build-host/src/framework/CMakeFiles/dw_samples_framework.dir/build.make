# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /usr/local/driveworks-0.6/newsamples

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /usr/local/driveworks-0.6/newsamples/build-host

# Include any dependencies generated for this target.
include src/framework/CMakeFiles/dw_samples_framework.dir/depend.make

# Include the progress variables for this target.
include src/framework/CMakeFiles/dw_samples_framework.dir/progress.make

# Include the compile flags for this target's objects.
include src/framework/CMakeFiles/dw_samples_framework.dir/flags.make

src/framework/CMakeFiles/dw_samples_framework.dir/SampleFramework.cpp.o: src/framework/CMakeFiles/dw_samples_framework.dir/flags.make
src/framework/CMakeFiles/dw_samples_framework.dir/SampleFramework.cpp.o: ../src/framework/SampleFramework.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/usr/local/driveworks-0.6/newsamples/build-host/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/framework/CMakeFiles/dw_samples_framework.dir/SampleFramework.cpp.o"
	cd /usr/local/driveworks-0.6/newsamples/build-host/src/framework && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/dw_samples_framework.dir/SampleFramework.cpp.o -c /usr/local/driveworks-0.6/newsamples/src/framework/SampleFramework.cpp

src/framework/CMakeFiles/dw_samples_framework.dir/SampleFramework.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/dw_samples_framework.dir/SampleFramework.cpp.i"
	cd /usr/local/driveworks-0.6/newsamples/build-host/src/framework && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /usr/local/driveworks-0.6/newsamples/src/framework/SampleFramework.cpp > CMakeFiles/dw_samples_framework.dir/SampleFramework.cpp.i

src/framework/CMakeFiles/dw_samples_framework.dir/SampleFramework.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/dw_samples_framework.dir/SampleFramework.cpp.s"
	cd /usr/local/driveworks-0.6/newsamples/build-host/src/framework && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /usr/local/driveworks-0.6/newsamples/src/framework/SampleFramework.cpp -o CMakeFiles/dw_samples_framework.dir/SampleFramework.cpp.s

src/framework/CMakeFiles/dw_samples_framework.dir/SampleFramework.cpp.o.requires:

.PHONY : src/framework/CMakeFiles/dw_samples_framework.dir/SampleFramework.cpp.o.requires

src/framework/CMakeFiles/dw_samples_framework.dir/SampleFramework.cpp.o.provides: src/framework/CMakeFiles/dw_samples_framework.dir/SampleFramework.cpp.o.requires
	$(MAKE) -f src/framework/CMakeFiles/dw_samples_framework.dir/build.make src/framework/CMakeFiles/dw_samples_framework.dir/SampleFramework.cpp.o.provides.build
.PHONY : src/framework/CMakeFiles/dw_samples_framework.dir/SampleFramework.cpp.o.provides

src/framework/CMakeFiles/dw_samples_framework.dir/SampleFramework.cpp.o.provides.build: src/framework/CMakeFiles/dw_samples_framework.dir/SampleFramework.cpp.o


src/framework/CMakeFiles/dw_samples_framework.dir/DriveWorksSample.cpp.o: src/framework/CMakeFiles/dw_samples_framework.dir/flags.make
src/framework/CMakeFiles/dw_samples_framework.dir/DriveWorksSample.cpp.o: ../src/framework/DriveWorksSample.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/usr/local/driveworks-0.6/newsamples/build-host/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object src/framework/CMakeFiles/dw_samples_framework.dir/DriveWorksSample.cpp.o"
	cd /usr/local/driveworks-0.6/newsamples/build-host/src/framework && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/dw_samples_framework.dir/DriveWorksSample.cpp.o -c /usr/local/driveworks-0.6/newsamples/src/framework/DriveWorksSample.cpp

src/framework/CMakeFiles/dw_samples_framework.dir/DriveWorksSample.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/dw_samples_framework.dir/DriveWorksSample.cpp.i"
	cd /usr/local/driveworks-0.6/newsamples/build-host/src/framework && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /usr/local/driveworks-0.6/newsamples/src/framework/DriveWorksSample.cpp > CMakeFiles/dw_samples_framework.dir/DriveWorksSample.cpp.i

src/framework/CMakeFiles/dw_samples_framework.dir/DriveWorksSample.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/dw_samples_framework.dir/DriveWorksSample.cpp.s"
	cd /usr/local/driveworks-0.6/newsamples/build-host/src/framework && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /usr/local/driveworks-0.6/newsamples/src/framework/DriveWorksSample.cpp -o CMakeFiles/dw_samples_framework.dir/DriveWorksSample.cpp.s

src/framework/CMakeFiles/dw_samples_framework.dir/DriveWorksSample.cpp.o.requires:

.PHONY : src/framework/CMakeFiles/dw_samples_framework.dir/DriveWorksSample.cpp.o.requires

src/framework/CMakeFiles/dw_samples_framework.dir/DriveWorksSample.cpp.o.provides: src/framework/CMakeFiles/dw_samples_framework.dir/DriveWorksSample.cpp.o.requires
	$(MAKE) -f src/framework/CMakeFiles/dw_samples_framework.dir/build.make src/framework/CMakeFiles/dw_samples_framework.dir/DriveWorksSample.cpp.o.provides.build
.PHONY : src/framework/CMakeFiles/dw_samples_framework.dir/DriveWorksSample.cpp.o.provides

src/framework/CMakeFiles/dw_samples_framework.dir/DriveWorksSample.cpp.o.provides.build: src/framework/CMakeFiles/dw_samples_framework.dir/DriveWorksSample.cpp.o


src/framework/CMakeFiles/dw_samples_framework.dir/ProgramArguments.cpp.o: src/framework/CMakeFiles/dw_samples_framework.dir/flags.make
src/framework/CMakeFiles/dw_samples_framework.dir/ProgramArguments.cpp.o: ../src/framework/ProgramArguments.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/usr/local/driveworks-0.6/newsamples/build-host/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object src/framework/CMakeFiles/dw_samples_framework.dir/ProgramArguments.cpp.o"
	cd /usr/local/driveworks-0.6/newsamples/build-host/src/framework && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/dw_samples_framework.dir/ProgramArguments.cpp.o -c /usr/local/driveworks-0.6/newsamples/src/framework/ProgramArguments.cpp

src/framework/CMakeFiles/dw_samples_framework.dir/ProgramArguments.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/dw_samples_framework.dir/ProgramArguments.cpp.i"
	cd /usr/local/driveworks-0.6/newsamples/build-host/src/framework && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /usr/local/driveworks-0.6/newsamples/src/framework/ProgramArguments.cpp > CMakeFiles/dw_samples_framework.dir/ProgramArguments.cpp.i

src/framework/CMakeFiles/dw_samples_framework.dir/ProgramArguments.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/dw_samples_framework.dir/ProgramArguments.cpp.s"
	cd /usr/local/driveworks-0.6/newsamples/build-host/src/framework && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /usr/local/driveworks-0.6/newsamples/src/framework/ProgramArguments.cpp -o CMakeFiles/dw_samples_framework.dir/ProgramArguments.cpp.s

src/framework/CMakeFiles/dw_samples_framework.dir/ProgramArguments.cpp.o.requires:

.PHONY : src/framework/CMakeFiles/dw_samples_framework.dir/ProgramArguments.cpp.o.requires

src/framework/CMakeFiles/dw_samples_framework.dir/ProgramArguments.cpp.o.provides: src/framework/CMakeFiles/dw_samples_framework.dir/ProgramArguments.cpp.o.requires
	$(MAKE) -f src/framework/CMakeFiles/dw_samples_framework.dir/build.make src/framework/CMakeFiles/dw_samples_framework.dir/ProgramArguments.cpp.o.provides.build
.PHONY : src/framework/CMakeFiles/dw_samples_framework.dir/ProgramArguments.cpp.o.provides

src/framework/CMakeFiles/dw_samples_framework.dir/ProgramArguments.cpp.o.provides.build: src/framework/CMakeFiles/dw_samples_framework.dir/ProgramArguments.cpp.o


src/framework/CMakeFiles/dw_samples_framework.dir/Grid.cpp.o: src/framework/CMakeFiles/dw_samples_framework.dir/flags.make
src/framework/CMakeFiles/dw_samples_framework.dir/Grid.cpp.o: ../src/framework/Grid.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/usr/local/driveworks-0.6/newsamples/build-host/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object src/framework/CMakeFiles/dw_samples_framework.dir/Grid.cpp.o"
	cd /usr/local/driveworks-0.6/newsamples/build-host/src/framework && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/dw_samples_framework.dir/Grid.cpp.o -c /usr/local/driveworks-0.6/newsamples/src/framework/Grid.cpp

src/framework/CMakeFiles/dw_samples_framework.dir/Grid.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/dw_samples_framework.dir/Grid.cpp.i"
	cd /usr/local/driveworks-0.6/newsamples/build-host/src/framework && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /usr/local/driveworks-0.6/newsamples/src/framework/Grid.cpp > CMakeFiles/dw_samples_framework.dir/Grid.cpp.i

src/framework/CMakeFiles/dw_samples_framework.dir/Grid.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/dw_samples_framework.dir/Grid.cpp.s"
	cd /usr/local/driveworks-0.6/newsamples/build-host/src/framework && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /usr/local/driveworks-0.6/newsamples/src/framework/Grid.cpp -o CMakeFiles/dw_samples_framework.dir/Grid.cpp.s

src/framework/CMakeFiles/dw_samples_framework.dir/Grid.cpp.o.requires:

.PHONY : src/framework/CMakeFiles/dw_samples_framework.dir/Grid.cpp.o.requires

src/framework/CMakeFiles/dw_samples_framework.dir/Grid.cpp.o.provides: src/framework/CMakeFiles/dw_samples_framework.dir/Grid.cpp.o.requires
	$(MAKE) -f src/framework/CMakeFiles/dw_samples_framework.dir/build.make src/framework/CMakeFiles/dw_samples_framework.dir/Grid.cpp.o.provides.build
.PHONY : src/framework/CMakeFiles/dw_samples_framework.dir/Grid.cpp.o.provides

src/framework/CMakeFiles/dw_samples_framework.dir/Grid.cpp.o.provides.build: src/framework/CMakeFiles/dw_samples_framework.dir/Grid.cpp.o


src/framework/CMakeFiles/dw_samples_framework.dir/WindowGLFW.cpp.o: src/framework/CMakeFiles/dw_samples_framework.dir/flags.make
src/framework/CMakeFiles/dw_samples_framework.dir/WindowGLFW.cpp.o: ../src/framework/WindowGLFW.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/usr/local/driveworks-0.6/newsamples/build-host/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object src/framework/CMakeFiles/dw_samples_framework.dir/WindowGLFW.cpp.o"
	cd /usr/local/driveworks-0.6/newsamples/build-host/src/framework && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/dw_samples_framework.dir/WindowGLFW.cpp.o -c /usr/local/driveworks-0.6/newsamples/src/framework/WindowGLFW.cpp

src/framework/CMakeFiles/dw_samples_framework.dir/WindowGLFW.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/dw_samples_framework.dir/WindowGLFW.cpp.i"
	cd /usr/local/driveworks-0.6/newsamples/build-host/src/framework && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /usr/local/driveworks-0.6/newsamples/src/framework/WindowGLFW.cpp > CMakeFiles/dw_samples_framework.dir/WindowGLFW.cpp.i

src/framework/CMakeFiles/dw_samples_framework.dir/WindowGLFW.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/dw_samples_framework.dir/WindowGLFW.cpp.s"
	cd /usr/local/driveworks-0.6/newsamples/build-host/src/framework && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /usr/local/driveworks-0.6/newsamples/src/framework/WindowGLFW.cpp -o CMakeFiles/dw_samples_framework.dir/WindowGLFW.cpp.s

src/framework/CMakeFiles/dw_samples_framework.dir/WindowGLFW.cpp.o.requires:

.PHONY : src/framework/CMakeFiles/dw_samples_framework.dir/WindowGLFW.cpp.o.requires

src/framework/CMakeFiles/dw_samples_framework.dir/WindowGLFW.cpp.o.provides: src/framework/CMakeFiles/dw_samples_framework.dir/WindowGLFW.cpp.o.requires
	$(MAKE) -f src/framework/CMakeFiles/dw_samples_framework.dir/build.make src/framework/CMakeFiles/dw_samples_framework.dir/WindowGLFW.cpp.o.provides.build
.PHONY : src/framework/CMakeFiles/dw_samples_framework.dir/WindowGLFW.cpp.o.provides

src/framework/CMakeFiles/dw_samples_framework.dir/WindowGLFW.cpp.o.provides.build: src/framework/CMakeFiles/dw_samples_framework.dir/WindowGLFW.cpp.o


src/framework/CMakeFiles/dw_samples_framework.dir/MathUtils.cpp.o: src/framework/CMakeFiles/dw_samples_framework.dir/flags.make
src/framework/CMakeFiles/dw_samples_framework.dir/MathUtils.cpp.o: ../src/framework/MathUtils.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/usr/local/driveworks-0.6/newsamples/build-host/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object src/framework/CMakeFiles/dw_samples_framework.dir/MathUtils.cpp.o"
	cd /usr/local/driveworks-0.6/newsamples/build-host/src/framework && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/dw_samples_framework.dir/MathUtils.cpp.o -c /usr/local/driveworks-0.6/newsamples/src/framework/MathUtils.cpp

src/framework/CMakeFiles/dw_samples_framework.dir/MathUtils.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/dw_samples_framework.dir/MathUtils.cpp.i"
	cd /usr/local/driveworks-0.6/newsamples/build-host/src/framework && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /usr/local/driveworks-0.6/newsamples/src/framework/MathUtils.cpp > CMakeFiles/dw_samples_framework.dir/MathUtils.cpp.i

src/framework/CMakeFiles/dw_samples_framework.dir/MathUtils.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/dw_samples_framework.dir/MathUtils.cpp.s"
	cd /usr/local/driveworks-0.6/newsamples/build-host/src/framework && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /usr/local/driveworks-0.6/newsamples/src/framework/MathUtils.cpp -o CMakeFiles/dw_samples_framework.dir/MathUtils.cpp.s

src/framework/CMakeFiles/dw_samples_framework.dir/MathUtils.cpp.o.requires:

.PHONY : src/framework/CMakeFiles/dw_samples_framework.dir/MathUtils.cpp.o.requires

src/framework/CMakeFiles/dw_samples_framework.dir/MathUtils.cpp.o.provides: src/framework/CMakeFiles/dw_samples_framework.dir/MathUtils.cpp.o.requires
	$(MAKE) -f src/framework/CMakeFiles/dw_samples_framework.dir/build.make src/framework/CMakeFiles/dw_samples_framework.dir/MathUtils.cpp.o.provides.build
.PHONY : src/framework/CMakeFiles/dw_samples_framework.dir/MathUtils.cpp.o.provides

src/framework/CMakeFiles/dw_samples_framework.dir/MathUtils.cpp.o.provides.build: src/framework/CMakeFiles/dw_samples_framework.dir/MathUtils.cpp.o


src/framework/CMakeFiles/dw_samples_framework.dir/MouseView3D.cpp.o: src/framework/CMakeFiles/dw_samples_framework.dir/flags.make
src/framework/CMakeFiles/dw_samples_framework.dir/MouseView3D.cpp.o: ../src/framework/MouseView3D.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/usr/local/driveworks-0.6/newsamples/build-host/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building CXX object src/framework/CMakeFiles/dw_samples_framework.dir/MouseView3D.cpp.o"
	cd /usr/local/driveworks-0.6/newsamples/build-host/src/framework && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/dw_samples_framework.dir/MouseView3D.cpp.o -c /usr/local/driveworks-0.6/newsamples/src/framework/MouseView3D.cpp

src/framework/CMakeFiles/dw_samples_framework.dir/MouseView3D.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/dw_samples_framework.dir/MouseView3D.cpp.i"
	cd /usr/local/driveworks-0.6/newsamples/build-host/src/framework && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /usr/local/driveworks-0.6/newsamples/src/framework/MouseView3D.cpp > CMakeFiles/dw_samples_framework.dir/MouseView3D.cpp.i

src/framework/CMakeFiles/dw_samples_framework.dir/MouseView3D.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/dw_samples_framework.dir/MouseView3D.cpp.s"
	cd /usr/local/driveworks-0.6/newsamples/build-host/src/framework && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /usr/local/driveworks-0.6/newsamples/src/framework/MouseView3D.cpp -o CMakeFiles/dw_samples_framework.dir/MouseView3D.cpp.s

src/framework/CMakeFiles/dw_samples_framework.dir/MouseView3D.cpp.o.requires:

.PHONY : src/framework/CMakeFiles/dw_samples_framework.dir/MouseView3D.cpp.o.requires

src/framework/CMakeFiles/dw_samples_framework.dir/MouseView3D.cpp.o.provides: src/framework/CMakeFiles/dw_samples_framework.dir/MouseView3D.cpp.o.requires
	$(MAKE) -f src/framework/CMakeFiles/dw_samples_framework.dir/build.make src/framework/CMakeFiles/dw_samples_framework.dir/MouseView3D.cpp.o.provides.build
.PHONY : src/framework/CMakeFiles/dw_samples_framework.dir/MouseView3D.cpp.o.provides

src/framework/CMakeFiles/dw_samples_framework.dir/MouseView3D.cpp.o.provides.build: src/framework/CMakeFiles/dw_samples_framework.dir/MouseView3D.cpp.o


src/framework/CMakeFiles/dw_samples_framework.dir/Log.cpp.o: src/framework/CMakeFiles/dw_samples_framework.dir/flags.make
src/framework/CMakeFiles/dw_samples_framework.dir/Log.cpp.o: ../src/framework/Log.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/usr/local/driveworks-0.6/newsamples/build-host/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Building CXX object src/framework/CMakeFiles/dw_samples_framework.dir/Log.cpp.o"
	cd /usr/local/driveworks-0.6/newsamples/build-host/src/framework && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/dw_samples_framework.dir/Log.cpp.o -c /usr/local/driveworks-0.6/newsamples/src/framework/Log.cpp

src/framework/CMakeFiles/dw_samples_framework.dir/Log.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/dw_samples_framework.dir/Log.cpp.i"
	cd /usr/local/driveworks-0.6/newsamples/build-host/src/framework && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /usr/local/driveworks-0.6/newsamples/src/framework/Log.cpp > CMakeFiles/dw_samples_framework.dir/Log.cpp.i

src/framework/CMakeFiles/dw_samples_framework.dir/Log.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/dw_samples_framework.dir/Log.cpp.s"
	cd /usr/local/driveworks-0.6/newsamples/build-host/src/framework && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /usr/local/driveworks-0.6/newsamples/src/framework/Log.cpp -o CMakeFiles/dw_samples_framework.dir/Log.cpp.s

src/framework/CMakeFiles/dw_samples_framework.dir/Log.cpp.o.requires:

.PHONY : src/framework/CMakeFiles/dw_samples_framework.dir/Log.cpp.o.requires

src/framework/CMakeFiles/dw_samples_framework.dir/Log.cpp.o.provides: src/framework/CMakeFiles/dw_samples_framework.dir/Log.cpp.o.requires
	$(MAKE) -f src/framework/CMakeFiles/dw_samples_framework.dir/build.make src/framework/CMakeFiles/dw_samples_framework.dir/Log.cpp.o.provides.build
.PHONY : src/framework/CMakeFiles/dw_samples_framework.dir/Log.cpp.o.provides

src/framework/CMakeFiles/dw_samples_framework.dir/Log.cpp.o.provides.build: src/framework/CMakeFiles/dw_samples_framework.dir/Log.cpp.o


src/framework/CMakeFiles/dw_samples_framework.dir/ProfilerCUDA.cpp.o: src/framework/CMakeFiles/dw_samples_framework.dir/flags.make
src/framework/CMakeFiles/dw_samples_framework.dir/ProfilerCUDA.cpp.o: ../src/framework/ProfilerCUDA.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/usr/local/driveworks-0.6/newsamples/build-host/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "Building CXX object src/framework/CMakeFiles/dw_samples_framework.dir/ProfilerCUDA.cpp.o"
	cd /usr/local/driveworks-0.6/newsamples/build-host/src/framework && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/dw_samples_framework.dir/ProfilerCUDA.cpp.o -c /usr/local/driveworks-0.6/newsamples/src/framework/ProfilerCUDA.cpp

src/framework/CMakeFiles/dw_samples_framework.dir/ProfilerCUDA.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/dw_samples_framework.dir/ProfilerCUDA.cpp.i"
	cd /usr/local/driveworks-0.6/newsamples/build-host/src/framework && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /usr/local/driveworks-0.6/newsamples/src/framework/ProfilerCUDA.cpp > CMakeFiles/dw_samples_framework.dir/ProfilerCUDA.cpp.i

src/framework/CMakeFiles/dw_samples_framework.dir/ProfilerCUDA.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/dw_samples_framework.dir/ProfilerCUDA.cpp.s"
	cd /usr/local/driveworks-0.6/newsamples/build-host/src/framework && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /usr/local/driveworks-0.6/newsamples/src/framework/ProfilerCUDA.cpp -o CMakeFiles/dw_samples_framework.dir/ProfilerCUDA.cpp.s

src/framework/CMakeFiles/dw_samples_framework.dir/ProfilerCUDA.cpp.o.requires:

.PHONY : src/framework/CMakeFiles/dw_samples_framework.dir/ProfilerCUDA.cpp.o.requires

src/framework/CMakeFiles/dw_samples_framework.dir/ProfilerCUDA.cpp.o.provides: src/framework/CMakeFiles/dw_samples_framework.dir/ProfilerCUDA.cpp.o.requires
	$(MAKE) -f src/framework/CMakeFiles/dw_samples_framework.dir/build.make src/framework/CMakeFiles/dw_samples_framework.dir/ProfilerCUDA.cpp.o.provides.build
.PHONY : src/framework/CMakeFiles/dw_samples_framework.dir/ProfilerCUDA.cpp.o.provides

src/framework/CMakeFiles/dw_samples_framework.dir/ProfilerCUDA.cpp.o.provides.build: src/framework/CMakeFiles/dw_samples_framework.dir/ProfilerCUDA.cpp.o


src/framework/CMakeFiles/dw_samples_framework.dir/GenericImage.cpp.o: src/framework/CMakeFiles/dw_samples_framework.dir/flags.make
src/framework/CMakeFiles/dw_samples_framework.dir/GenericImage.cpp.o: ../src/framework/GenericImage.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/usr/local/driveworks-0.6/newsamples/build-host/CMakeFiles --progress-num=$(CMAKE_PROGRESS_10) "Building CXX object src/framework/CMakeFiles/dw_samples_framework.dir/GenericImage.cpp.o"
	cd /usr/local/driveworks-0.6/newsamples/build-host/src/framework && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/dw_samples_framework.dir/GenericImage.cpp.o -c /usr/local/driveworks-0.6/newsamples/src/framework/GenericImage.cpp

src/framework/CMakeFiles/dw_samples_framework.dir/GenericImage.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/dw_samples_framework.dir/GenericImage.cpp.i"
	cd /usr/local/driveworks-0.6/newsamples/build-host/src/framework && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /usr/local/driveworks-0.6/newsamples/src/framework/GenericImage.cpp > CMakeFiles/dw_samples_framework.dir/GenericImage.cpp.i

src/framework/CMakeFiles/dw_samples_framework.dir/GenericImage.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/dw_samples_framework.dir/GenericImage.cpp.s"
	cd /usr/local/driveworks-0.6/newsamples/build-host/src/framework && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /usr/local/driveworks-0.6/newsamples/src/framework/GenericImage.cpp -o CMakeFiles/dw_samples_framework.dir/GenericImage.cpp.s

src/framework/CMakeFiles/dw_samples_framework.dir/GenericImage.cpp.o.requires:

.PHONY : src/framework/CMakeFiles/dw_samples_framework.dir/GenericImage.cpp.o.requires

src/framework/CMakeFiles/dw_samples_framework.dir/GenericImage.cpp.o.provides: src/framework/CMakeFiles/dw_samples_framework.dir/GenericImage.cpp.o.requires
	$(MAKE) -f src/framework/CMakeFiles/dw_samples_framework.dir/build.make src/framework/CMakeFiles/dw_samples_framework.dir/GenericImage.cpp.o.provides.build
.PHONY : src/framework/CMakeFiles/dw_samples_framework.dir/GenericImage.cpp.o.provides

src/framework/CMakeFiles/dw_samples_framework.dir/GenericImage.cpp.o.provides.build: src/framework/CMakeFiles/dw_samples_framework.dir/GenericImage.cpp.o


src/framework/CMakeFiles/dw_samples_framework.dir/SimpleCamera.cpp.o: src/framework/CMakeFiles/dw_samples_framework.dir/flags.make
src/framework/CMakeFiles/dw_samples_framework.dir/SimpleCamera.cpp.o: ../src/framework/SimpleCamera.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/usr/local/driveworks-0.6/newsamples/build-host/CMakeFiles --progress-num=$(CMAKE_PROGRESS_11) "Building CXX object src/framework/CMakeFiles/dw_samples_framework.dir/SimpleCamera.cpp.o"
	cd /usr/local/driveworks-0.6/newsamples/build-host/src/framework && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/dw_samples_framework.dir/SimpleCamera.cpp.o -c /usr/local/driveworks-0.6/newsamples/src/framework/SimpleCamera.cpp

src/framework/CMakeFiles/dw_samples_framework.dir/SimpleCamera.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/dw_samples_framework.dir/SimpleCamera.cpp.i"
	cd /usr/local/driveworks-0.6/newsamples/build-host/src/framework && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /usr/local/driveworks-0.6/newsamples/src/framework/SimpleCamera.cpp > CMakeFiles/dw_samples_framework.dir/SimpleCamera.cpp.i

src/framework/CMakeFiles/dw_samples_framework.dir/SimpleCamera.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/dw_samples_framework.dir/SimpleCamera.cpp.s"
	cd /usr/local/driveworks-0.6/newsamples/build-host/src/framework && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /usr/local/driveworks-0.6/newsamples/src/framework/SimpleCamera.cpp -o CMakeFiles/dw_samples_framework.dir/SimpleCamera.cpp.s

src/framework/CMakeFiles/dw_samples_framework.dir/SimpleCamera.cpp.o.requires:

.PHONY : src/framework/CMakeFiles/dw_samples_framework.dir/SimpleCamera.cpp.o.requires

src/framework/CMakeFiles/dw_samples_framework.dir/SimpleCamera.cpp.o.provides: src/framework/CMakeFiles/dw_samples_framework.dir/SimpleCamera.cpp.o.requires
	$(MAKE) -f src/framework/CMakeFiles/dw_samples_framework.dir/build.make src/framework/CMakeFiles/dw_samples_framework.dir/SimpleCamera.cpp.o.provides.build
.PHONY : src/framework/CMakeFiles/dw_samples_framework.dir/SimpleCamera.cpp.o.provides

src/framework/CMakeFiles/dw_samples_framework.dir/SimpleCamera.cpp.o.provides.build: src/framework/CMakeFiles/dw_samples_framework.dir/SimpleCamera.cpp.o


src/framework/CMakeFiles/dw_samples_framework.dir/SimpleRenderer.cpp.o: src/framework/CMakeFiles/dw_samples_framework.dir/flags.make
src/framework/CMakeFiles/dw_samples_framework.dir/SimpleRenderer.cpp.o: ../src/framework/SimpleRenderer.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/usr/local/driveworks-0.6/newsamples/build-host/CMakeFiles --progress-num=$(CMAKE_PROGRESS_12) "Building CXX object src/framework/CMakeFiles/dw_samples_framework.dir/SimpleRenderer.cpp.o"
	cd /usr/local/driveworks-0.6/newsamples/build-host/src/framework && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/dw_samples_framework.dir/SimpleRenderer.cpp.o -c /usr/local/driveworks-0.6/newsamples/src/framework/SimpleRenderer.cpp

src/framework/CMakeFiles/dw_samples_framework.dir/SimpleRenderer.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/dw_samples_framework.dir/SimpleRenderer.cpp.i"
	cd /usr/local/driveworks-0.6/newsamples/build-host/src/framework && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /usr/local/driveworks-0.6/newsamples/src/framework/SimpleRenderer.cpp > CMakeFiles/dw_samples_framework.dir/SimpleRenderer.cpp.i

src/framework/CMakeFiles/dw_samples_framework.dir/SimpleRenderer.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/dw_samples_framework.dir/SimpleRenderer.cpp.s"
	cd /usr/local/driveworks-0.6/newsamples/build-host/src/framework && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /usr/local/driveworks-0.6/newsamples/src/framework/SimpleRenderer.cpp -o CMakeFiles/dw_samples_framework.dir/SimpleRenderer.cpp.s

src/framework/CMakeFiles/dw_samples_framework.dir/SimpleRenderer.cpp.o.requires:

.PHONY : src/framework/CMakeFiles/dw_samples_framework.dir/SimpleRenderer.cpp.o.requires

src/framework/CMakeFiles/dw_samples_framework.dir/SimpleRenderer.cpp.o.provides: src/framework/CMakeFiles/dw_samples_framework.dir/SimpleRenderer.cpp.o.requires
	$(MAKE) -f src/framework/CMakeFiles/dw_samples_framework.dir/build.make src/framework/CMakeFiles/dw_samples_framework.dir/SimpleRenderer.cpp.o.provides.build
.PHONY : src/framework/CMakeFiles/dw_samples_framework.dir/SimpleRenderer.cpp.o.provides

src/framework/CMakeFiles/dw_samples_framework.dir/SimpleRenderer.cpp.o.provides.build: src/framework/CMakeFiles/dw_samples_framework.dir/SimpleRenderer.cpp.o


src/framework/CMakeFiles/dw_samples_framework.dir/SimpleRecordingPlayer.cpp.o: src/framework/CMakeFiles/dw_samples_framework.dir/flags.make
src/framework/CMakeFiles/dw_samples_framework.dir/SimpleRecordingPlayer.cpp.o: ../src/framework/SimpleRecordingPlayer.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/usr/local/driveworks-0.6/newsamples/build-host/CMakeFiles --progress-num=$(CMAKE_PROGRESS_13) "Building CXX object src/framework/CMakeFiles/dw_samples_framework.dir/SimpleRecordingPlayer.cpp.o"
	cd /usr/local/driveworks-0.6/newsamples/build-host/src/framework && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/dw_samples_framework.dir/SimpleRecordingPlayer.cpp.o -c /usr/local/driveworks-0.6/newsamples/src/framework/SimpleRecordingPlayer.cpp

src/framework/CMakeFiles/dw_samples_framework.dir/SimpleRecordingPlayer.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/dw_samples_framework.dir/SimpleRecordingPlayer.cpp.i"
	cd /usr/local/driveworks-0.6/newsamples/build-host/src/framework && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /usr/local/driveworks-0.6/newsamples/src/framework/SimpleRecordingPlayer.cpp > CMakeFiles/dw_samples_framework.dir/SimpleRecordingPlayer.cpp.i

src/framework/CMakeFiles/dw_samples_framework.dir/SimpleRecordingPlayer.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/dw_samples_framework.dir/SimpleRecordingPlayer.cpp.s"
	cd /usr/local/driveworks-0.6/newsamples/build-host/src/framework && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /usr/local/driveworks-0.6/newsamples/src/framework/SimpleRecordingPlayer.cpp -o CMakeFiles/dw_samples_framework.dir/SimpleRecordingPlayer.cpp.s

src/framework/CMakeFiles/dw_samples_framework.dir/SimpleRecordingPlayer.cpp.o.requires:

.PHONY : src/framework/CMakeFiles/dw_samples_framework.dir/SimpleRecordingPlayer.cpp.o.requires

src/framework/CMakeFiles/dw_samples_framework.dir/SimpleRecordingPlayer.cpp.o.provides: src/framework/CMakeFiles/dw_samples_framework.dir/SimpleRecordingPlayer.cpp.o.requires
	$(MAKE) -f src/framework/CMakeFiles/dw_samples_framework.dir/build.make src/framework/CMakeFiles/dw_samples_framework.dir/SimpleRecordingPlayer.cpp.o.provides.build
.PHONY : src/framework/CMakeFiles/dw_samples_framework.dir/SimpleRecordingPlayer.cpp.o.provides

src/framework/CMakeFiles/dw_samples_framework.dir/SimpleRecordingPlayer.cpp.o.provides.build: src/framework/CMakeFiles/dw_samples_framework.dir/SimpleRecordingPlayer.cpp.o


# Object files for target dw_samples_framework
dw_samples_framework_OBJECTS = \
"CMakeFiles/dw_samples_framework.dir/SampleFramework.cpp.o" \
"CMakeFiles/dw_samples_framework.dir/DriveWorksSample.cpp.o" \
"CMakeFiles/dw_samples_framework.dir/ProgramArguments.cpp.o" \
"CMakeFiles/dw_samples_framework.dir/Grid.cpp.o" \
"CMakeFiles/dw_samples_framework.dir/WindowGLFW.cpp.o" \
"CMakeFiles/dw_samples_framework.dir/MathUtils.cpp.o" \
"CMakeFiles/dw_samples_framework.dir/MouseView3D.cpp.o" \
"CMakeFiles/dw_samples_framework.dir/Log.cpp.o" \
"CMakeFiles/dw_samples_framework.dir/ProfilerCUDA.cpp.o" \
"CMakeFiles/dw_samples_framework.dir/GenericImage.cpp.o" \
"CMakeFiles/dw_samples_framework.dir/SimpleCamera.cpp.o" \
"CMakeFiles/dw_samples_framework.dir/SimpleRenderer.cpp.o" \
"CMakeFiles/dw_samples_framework.dir/SimpleRecordingPlayer.cpp.o"

# External object files for target dw_samples_framework
dw_samples_framework_EXTERNAL_OBJECTS =

src/framework/libdw_samples_framework.a: src/framework/CMakeFiles/dw_samples_framework.dir/SampleFramework.cpp.o
src/framework/libdw_samples_framework.a: src/framework/CMakeFiles/dw_samples_framework.dir/DriveWorksSample.cpp.o
src/framework/libdw_samples_framework.a: src/framework/CMakeFiles/dw_samples_framework.dir/ProgramArguments.cpp.o
src/framework/libdw_samples_framework.a: src/framework/CMakeFiles/dw_samples_framework.dir/Grid.cpp.o
src/framework/libdw_samples_framework.a: src/framework/CMakeFiles/dw_samples_framework.dir/WindowGLFW.cpp.o
src/framework/libdw_samples_framework.a: src/framework/CMakeFiles/dw_samples_framework.dir/MathUtils.cpp.o
src/framework/libdw_samples_framework.a: src/framework/CMakeFiles/dw_samples_framework.dir/MouseView3D.cpp.o
src/framework/libdw_samples_framework.a: src/framework/CMakeFiles/dw_samples_framework.dir/Log.cpp.o
src/framework/libdw_samples_framework.a: src/framework/CMakeFiles/dw_samples_framework.dir/ProfilerCUDA.cpp.o
src/framework/libdw_samples_framework.a: src/framework/CMakeFiles/dw_samples_framework.dir/GenericImage.cpp.o
src/framework/libdw_samples_framework.a: src/framework/CMakeFiles/dw_samples_framework.dir/SimpleCamera.cpp.o
src/framework/libdw_samples_framework.a: src/framework/CMakeFiles/dw_samples_framework.dir/SimpleRenderer.cpp.o
src/framework/libdw_samples_framework.a: src/framework/CMakeFiles/dw_samples_framework.dir/SimpleRecordingPlayer.cpp.o
src/framework/libdw_samples_framework.a: src/framework/CMakeFiles/dw_samples_framework.dir/build.make
src/framework/libdw_samples_framework.a: src/framework/CMakeFiles/dw_samples_framework.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/usr/local/driveworks-0.6/newsamples/build-host/CMakeFiles --progress-num=$(CMAKE_PROGRESS_14) "Linking CXX static library libdw_samples_framework.a"
	cd /usr/local/driveworks-0.6/newsamples/build-host/src/framework && $(CMAKE_COMMAND) -P CMakeFiles/dw_samples_framework.dir/cmake_clean_target.cmake
	cd /usr/local/driveworks-0.6/newsamples/build-host/src/framework && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/dw_samples_framework.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/framework/CMakeFiles/dw_samples_framework.dir/build: src/framework/libdw_samples_framework.a

.PHONY : src/framework/CMakeFiles/dw_samples_framework.dir/build

src/framework/CMakeFiles/dw_samples_framework.dir/requires: src/framework/CMakeFiles/dw_samples_framework.dir/SampleFramework.cpp.o.requires
src/framework/CMakeFiles/dw_samples_framework.dir/requires: src/framework/CMakeFiles/dw_samples_framework.dir/DriveWorksSample.cpp.o.requires
src/framework/CMakeFiles/dw_samples_framework.dir/requires: src/framework/CMakeFiles/dw_samples_framework.dir/ProgramArguments.cpp.o.requires
src/framework/CMakeFiles/dw_samples_framework.dir/requires: src/framework/CMakeFiles/dw_samples_framework.dir/Grid.cpp.o.requires
src/framework/CMakeFiles/dw_samples_framework.dir/requires: src/framework/CMakeFiles/dw_samples_framework.dir/WindowGLFW.cpp.o.requires
src/framework/CMakeFiles/dw_samples_framework.dir/requires: src/framework/CMakeFiles/dw_samples_framework.dir/MathUtils.cpp.o.requires
src/framework/CMakeFiles/dw_samples_framework.dir/requires: src/framework/CMakeFiles/dw_samples_framework.dir/MouseView3D.cpp.o.requires
src/framework/CMakeFiles/dw_samples_framework.dir/requires: src/framework/CMakeFiles/dw_samples_framework.dir/Log.cpp.o.requires
src/framework/CMakeFiles/dw_samples_framework.dir/requires: src/framework/CMakeFiles/dw_samples_framework.dir/ProfilerCUDA.cpp.o.requires
src/framework/CMakeFiles/dw_samples_framework.dir/requires: src/framework/CMakeFiles/dw_samples_framework.dir/GenericImage.cpp.o.requires
src/framework/CMakeFiles/dw_samples_framework.dir/requires: src/framework/CMakeFiles/dw_samples_framework.dir/SimpleCamera.cpp.o.requires
src/framework/CMakeFiles/dw_samples_framework.dir/requires: src/framework/CMakeFiles/dw_samples_framework.dir/SimpleRenderer.cpp.o.requires
src/framework/CMakeFiles/dw_samples_framework.dir/requires: src/framework/CMakeFiles/dw_samples_framework.dir/SimpleRecordingPlayer.cpp.o.requires

.PHONY : src/framework/CMakeFiles/dw_samples_framework.dir/requires

src/framework/CMakeFiles/dw_samples_framework.dir/clean:
	cd /usr/local/driveworks-0.6/newsamples/build-host/src/framework && $(CMAKE_COMMAND) -P CMakeFiles/dw_samples_framework.dir/cmake_clean.cmake
.PHONY : src/framework/CMakeFiles/dw_samples_framework.dir/clean

src/framework/CMakeFiles/dw_samples_framework.dir/depend:
	cd /usr/local/driveworks-0.6/newsamples/build-host && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /usr/local/driveworks-0.6/newsamples /usr/local/driveworks-0.6/newsamples/src/framework /usr/local/driveworks-0.6/newsamples/build-host /usr/local/driveworks-0.6/newsamples/build-host/src/framework /usr/local/driveworks-0.6/newsamples/build-host/src/framework/CMakeFiles/dw_samples_framework.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/framework/CMakeFiles/dw_samples_framework.dir/depend

