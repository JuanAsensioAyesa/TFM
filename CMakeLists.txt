cmake_minimum_required(VERSION 3.15)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(ProjectSetup)
# set the project name and version
project(TFM
    VERSION   0.1
    LANGUAGES CXX CUDA
)
include(IDESetup)
include(InstallDeps)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Set up compiler and linker options.
include(AddFlags)
include(CUDASetup)

# Require C++14, and disable compiler-specific extensions (if possible).
foreach(lang CXX CUDA)
    set(CMAKE_${lang}_STANDARD          14)
    set(CMAKE_${lang}_STANDARD_REQUIRED ON)
    set(CMAKE_${lang}_EXTENSIONS        OFF)
endforeach()



list(APPEND CMAKE_MODULE_PATH "/usr/local/lib/cmake/OpenVDB/")

#list(APPEND CMAKE_MODULE_PATH "/usr/local/lib/cmake/NanoVDB/")
find_package(OpenVDB REQUIRED)
#find_package(nanoVDB REQUIRED)

add_executable(TFM_CLEAN main_clean.cc kernels.cu ./utilsSkin/utilSkin.hpp ./utilGrid/Grid.hpp )
#add_executable(TFM main.cc kernels.cu ./utilsSkin/utilSkin.hpp ./utilGrid/Grid.hpp)
add_executable(CONVERTER converter.cc  kernels.cu ./utilGrid/Grid.hpp)

target_compile_options(TFM_CLEAN PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:              
                        
                        --expt-extended-lambda
                       >)
# target_compile_options(TFM PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:              
                        
#                         --expt-extended-lambda
#                        >)
target_compile_options(CONVERTER PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:              

--expt-extended-lambda
>)


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Install rules.
install(TARGETS
    #TFM
    CONVERTER
    TFM_CLEAN

    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)
install_deps_system(lib bin)
#set_source_files_properties(nanoVdb.cc PROPERTIES LANGUAGE CUDA)

target_link_libraries(TFM_CLEAN OpenVDB::openvdb)
#target_link_libraries(TFM OpenVDB::openvdb)
target_link_libraries(CONVERTER OpenVDB::openvdb)
#target_link_libraries(prueba  NanoVDB::nanovdb)
