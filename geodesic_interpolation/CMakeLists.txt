set(CMAKE_VERBOSE_MAKEFILE ON)
cmake_minimum_required(VERSION 3.10)
project(geodesic_interpolation)

# Spécifier le standard C++
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Ajouter le chemin vers les modules CMake de GOAST
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/goast/cmake")

# Inclure le module GOAST
include(GOAST)

# Inclure les répertoires d'en-têtes
include_directories(
    ${CMAKE_SOURCE_DIR}/include            # Pour Eigen
    ${CMAKE_SOURCE_DIR}/goast/include      # Pour GOAST
    ${CMAKE_SOURCE_DIR}/Repulsor          # Pour Repulsor
    #"/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/System/Library/Frameworks/Accelerate.framework/Versions/A/Frameworks/vecLib.framework/Versions/A/Headers"         # Pour Accelerate
)

# Ajouter les options du compilateur
add_compile_options(-pthread)

# Si vous utilisez Clang, activer l'extension matrix
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-fenable-matrix)
endif()
get_cmake_property(_variableNames VARIABLES)
foreach (_variableName ${_variableNames})
    if ("${${_variableName}}" STREQUAL "NOTFOUND")
        message(STATUS "${_variableName} is NOTFOUND")
    endif()
endforeach()
set(OpenMesh_DIR "/usr/local/share/OpenMesh/cmake/OpenMeshConfig.cmake")

set(CMAKE_PREFIX_PATH "/opt/homebrew/opt/libomp" ${CMAKE_PREFIX_PATH})

 # Remplacez par votre chemin d'installation si différent
 #set(VECLIB_INCLUDE_DIR "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/System/Library/Frameworks/Accelerate.framework/Versions/A/Frameworks/vecLib.framework/Versions/A/Headers")

# Trouver les bibliothèques nécessaires
find_package(Eigen3 REQUIRED NO_MODULE)
find_package(OpenMesh REQUIRED)
find_package(OpenMP REQUIRED)
find_package(BLAS REQUIRED)

if(NOT BLAS_FOUND)
    message(FATAL_ERROR "BLAS non trouvé. Veuillez installer BLAS ou vérifier votre configuration.")
endif()
find_library(Accelerate_LIB Accelerate REQUIRED)
if(NOT Accelerate_LIB)
    message(FATAL_ERROR "Le framework Accelerate n'a pas pu être trouvé.")
endif()
set(SOURCES
    src/main.cpp
    src/MeshHandler.cpp
    src/EnergyCalculator.cpp
    # Ajoutez d'autres fichiers source ici si nécessaire
)
#find_library(ACCELERATE_LIB Accelerate REQUIRED)
# Ajouter l'exécutable
add_executable(executable ${SOURCES})

# Lier les bibliothèques
target_link_libraries(executable
    GOAST::GOAST                    # Bibliothèque GOAST
    Eigen3::Eigen
    OpenMeshCore
    OpenMeshTools
    pthread
    OpenMP::OpenMP_CXX
    ${BLAS_LIBRARIES}
    ${Accelerate_LIB}
)



# Inclure les répertoires d'en-têtes pour l'exécutable
target_include_directories(executable PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/goast/include
    ${CMAKE_SOURCE_DIR}/Repulsor

)