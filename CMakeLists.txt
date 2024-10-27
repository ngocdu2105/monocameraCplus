cmake_minimum_required(VERSION 3.5)
project(test )
find_package(OpenCV REQUIRED PATHS /home/trinhngocdu123/build )

include_directories( ${OpenCV_INCLUDE_DIRS})
find_package(onnxruntime REQUIRED)


set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include_directories(${CMAKE_SOURCE_DIR}/)
link_directories(${CMAKE_CURRENT_BINARY_DIR}/include)

if(NOT ONNXRUNTIME_ROOTDIR)
  if(WIN32)
    set(ONNXRUNTIME_ROOTDIR "C:/Program Files/onnxruntime")
else()
    set(ONNXRUNTIME_ROOTDIR "/usr/local")
  endif()
endif()

# The ORT package has a different include directory structure to a local install via cmake.
# We added the path for the pre-built package above. Add the path for a local install to support either usage.
# TODO: If we want to support additional EPs being loadable from a local install we also need to add EP specific
# directories under /include/onnxruntime/core/providers
include_directories("${ONNXRUNTIME_ROOTDIR}/include"                           # Pre-built package
                    "${ONNXRUNTIME_ROOTDIR}/include/onnxruntime"               # Linux local install to /usr/local
                    ) # Windows local install

link_directories("${ONNXRUNTIME_ROOTDIR}/lib")
# include_directories(include)

file(GLOB SOURCE "src/*.cpp")

message(":  " ${SOURCE})

add_executable(test ${SOURCE} )



target_link_libraries(test  PUBLIC  ${OpenCV_LIBS} ${onnxruntime_LIBRARY} )
