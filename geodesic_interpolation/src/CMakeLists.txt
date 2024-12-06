# src/CMakeLists.txt
add_executable(geodesic_interpolation main.cpp)

# Lier les bibliothèques
target_link_libraries(geodesic_interpolation Eigen3::Eigen)
target_link_libraries(geodesic_interpolation OpenMeshCore OpenMeshTools)
target_link_libraries(geodesic_interpolation ${SUITESPARSE_LIBRARIES})
# Ajouter GOAST et Repulsor si nécessaire