cmake_minimum_required(VERSION 3.13 FATAL_ERROR)
project(firered_bench C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS_RELEASE "-O2")

add_definitions(-DCHIP_AX650)
add_definitions(-DENV_HAS_STD_FILESYSTEM)
add_definitions(-DENV_HAS_POSIX_FILE_STAT)

set(BSP_MSP_DIR "/data/yangrongzhao/.cache/magnetar/bsp/ax650/msp/out" CACHE PATH "AX650 BSP msp/out")
set(FIRERED_CPP "/data/yangrongzhao/Magnetar/todos/work/fireredasr-aed-ax650n/reference/hf_firered/cpp" CACHE PATH "FireRed cpp src")
set(KNF_SRC "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty_knf" CACHE PATH "kaldi-native-fbank root")

include_directories(${BSP_MSP_DIR}/include)
link_directories(${BSP_MSP_DIR}/lib)
include_directories(${FIRERED_CPP}/src)
include_directories(${KNF_SRC})
include_directories(${KNF_SRC}/kissfft)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/3rdparty)

set(KNF_SOURCES
    ${KNF_SRC}/kaldi-native-fbank/csrc/feature-fbank.cc
    ${KNF_SRC}/kaldi-native-fbank/csrc/feature-functions.cc
    ${KNF_SRC}/kaldi-native-fbank/csrc/feature-mfcc.cc
    ${KNF_SRC}/kaldi-native-fbank/csrc/feature-raw-audio-samples.cc
    ${KNF_SRC}/kaldi-native-fbank/csrc/feature-window.cc
    ${KNF_SRC}/kaldi-native-fbank/csrc/kaldi-math.cc
    ${KNF_SRC}/kaldi-native-fbank/csrc/log.cc
    ${KNF_SRC}/kaldi-native-fbank/csrc/mel-computations.cc
    ${KNF_SRC}/kaldi-native-fbank/csrc/online-feature.cc
    ${KNF_SRC}/kaldi-native-fbank/csrc/rfft.cc
    ${KNF_SRC}/kaldi-native-fbank/csrc/whisper-feature.cc
    ${KNF_SRC}/kissfft/kiss_fft.c
    ${KNF_SRC}/kissfft/kiss_fftr.c
)

add_executable(firered_bench
    main.cpp
    ${FIRERED_CPP}/src/EngineWrapper.cpp
    ${KNF_SOURCES}
)
target_link_libraries(firered_bench ax_sys ax_engine ax_interpreter pthread)

add_library(firered_sdk_lib SHARED
    sdk_lib.cpp
    ${FIRERED_CPP}/src/EngineWrapper.cpp
    ${KNF_SOURCES}
)
target_link_libraries(firered_sdk_lib ax_sys ax_engine ax_interpreter pthread)

add_executable(firered_sdk sdk_main.cpp)
target_link_libraries(firered_sdk firered_sdk_lib ax_sys ax_engine ax_interpreter pthread)

add_executable(firered_openai_server openai_server.cpp)
target_link_libraries(firered_openai_server firered_sdk_lib ax_sys ax_engine ax_interpreter pthread)

add_executable(firered_openai_client openai_client.cpp)
target_link_libraries(firered_openai_client pthread)
