# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.6

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
CMAKE_SOURCE_DIR = /home/andre/study/opencv/tutorials/face_reconition/prediction_test

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/andre/study/opencv/tutorials/face_reconition/prediction_test

# Include any dependencies generated for this target.
include CMakeFiles/prediction.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/prediction.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/prediction.dir/flags.make

CMakeFiles/prediction.dir/prediction.cpp.o: CMakeFiles/prediction.dir/flags.make
CMakeFiles/prediction.dir/prediction.cpp.o: prediction.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/andre/study/opencv/tutorials/face_reconition/prediction_test/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/prediction.dir/prediction.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/prediction.dir/prediction.cpp.o -c /home/andre/study/opencv/tutorials/face_reconition/prediction_test/prediction.cpp

CMakeFiles/prediction.dir/prediction.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/prediction.dir/prediction.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/andre/study/opencv/tutorials/face_reconition/prediction_test/prediction.cpp > CMakeFiles/prediction.dir/prediction.cpp.i

CMakeFiles/prediction.dir/prediction.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/prediction.dir/prediction.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/andre/study/opencv/tutorials/face_reconition/prediction_test/prediction.cpp -o CMakeFiles/prediction.dir/prediction.cpp.s

CMakeFiles/prediction.dir/prediction.cpp.o.requires:

.PHONY : CMakeFiles/prediction.dir/prediction.cpp.o.requires

CMakeFiles/prediction.dir/prediction.cpp.o.provides: CMakeFiles/prediction.dir/prediction.cpp.o.requires
	$(MAKE) -f CMakeFiles/prediction.dir/build.make CMakeFiles/prediction.dir/prediction.cpp.o.provides.build
.PHONY : CMakeFiles/prediction.dir/prediction.cpp.o.provides

CMakeFiles/prediction.dir/prediction.cpp.o.provides.build: CMakeFiles/prediction.dir/prediction.cpp.o


# Object files for target prediction
prediction_OBJECTS = \
"CMakeFiles/prediction.dir/prediction.cpp.o"

# External object files for target prediction
prediction_EXTERNAL_OBJECTS =

prediction: CMakeFiles/prediction.dir/prediction.cpp.o
prediction: CMakeFiles/prediction.dir/build.make
prediction: /usr/lib/libopencv_xphoto.so.3.1.0
prediction: /usr/lib/libopencv_xobjdetect.so.3.1.0
prediction: /usr/lib/libopencv_tracking.so.3.1.0
prediction: /usr/lib/libopencv_surface_matching.so.3.1.0
prediction: /usr/lib/libopencv_structured_light.so.3.1.0
prediction: /usr/lib/libopencv_stereo.so.3.1.0
prediction: /usr/lib/libopencv_saliency.so.3.1.0
prediction: /usr/lib/libopencv_rgbd.so.3.1.0
prediction: /usr/lib/libopencv_reg.so.3.1.0
prediction: /usr/lib/libopencv_plot.so.3.1.0
prediction: /usr/lib/libopencv_optflow.so.3.1.0
prediction: /usr/lib/libopencv_line_descriptor.so.3.1.0
prediction: /usr/lib/libopencv_fuzzy.so.3.1.0
prediction: /usr/lib/libopencv_dpm.so.3.1.0
prediction: /usr/lib/libopencv_dnn.so.3.1.0
prediction: /usr/lib/libopencv_datasets.so.3.1.0
prediction: /usr/lib/libopencv_ccalib.so.3.1.0
prediction: /usr/lib/libopencv_bioinspired.so.3.1.0
prediction: /usr/lib/libopencv_bgsegm.so.3.1.0
prediction: /usr/lib/libopencv_aruco.so.3.1.0
prediction: /usr/lib/libopencv_videostab.so.3.1.0
prediction: /usr/lib/libopencv_superres.so.3.1.0
prediction: /usr/lib/libopencv_stitching.so.3.1.0
prediction: /usr/lib/libopencv_photo.so.3.1.0
prediction: /usr/lib/libopencv_text.so.3.1.0
prediction: /usr/lib/libopencv_face.so.3.1.0
prediction: /usr/lib/libopencv_ximgproc.so.3.1.0
prediction: /usr/lib/libopencv_xfeatures2d.so.3.1.0
prediction: /usr/lib/libopencv_shape.so.3.1.0
prediction: /usr/lib/libopencv_video.so.3.1.0
prediction: /usr/lib/libopencv_objdetect.so.3.1.0
prediction: /usr/lib/libopencv_calib3d.so.3.1.0
prediction: /usr/lib/libopencv_features2d.so.3.1.0
prediction: /usr/lib/libopencv_ml.so.3.1.0
prediction: /usr/lib/libopencv_highgui.so.3.1.0
prediction: /usr/lib/libopencv_videoio.so.3.1.0
prediction: /usr/lib/libopencv_imgcodecs.so.3.1.0
prediction: /usr/lib/libopencv_imgproc.so.3.1.0
prediction: /usr/lib/libopencv_flann.so.3.1.0
prediction: /usr/lib/libopencv_core.so.3.1.0
prediction: CMakeFiles/prediction.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/andre/study/opencv/tutorials/face_reconition/prediction_test/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable prediction"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/prediction.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/prediction.dir/build: prediction

.PHONY : CMakeFiles/prediction.dir/build

CMakeFiles/prediction.dir/requires: CMakeFiles/prediction.dir/prediction.cpp.o.requires

.PHONY : CMakeFiles/prediction.dir/requires

CMakeFiles/prediction.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/prediction.dir/cmake_clean.cmake
.PHONY : CMakeFiles/prediction.dir/clean

CMakeFiles/prediction.dir/depend:
	cd /home/andre/study/opencv/tutorials/face_reconition/prediction_test && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/andre/study/opencv/tutorials/face_reconition/prediction_test /home/andre/study/opencv/tutorials/face_reconition/prediction_test /home/andre/study/opencv/tutorials/face_reconition/prediction_test /home/andre/study/opencv/tutorials/face_reconition/prediction_test /home/andre/study/opencv/tutorials/face_reconition/prediction_test/CMakeFiles/prediction.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/prediction.dir/depend
