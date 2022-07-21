
#include <nanovdb/util/GridBuilder.h>
#include <nanovdb/util/OpenToNanoVDB.h> // converter from OpenVDB to NanoVDB (includes NanoVDB.h and GridManager.h)
#include <nanovdb/util/CudaDeviceBuffer.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_device_runtime_api.h>
#include <openvdb/tools/ValueTransformer.h>
#include <openvdb/Types.h>
#include <iostream>
#include <stdlib.h>     /* atoi */
#include <cmath>

/**
 * @brief Para pasar los datos a la funcion createSkin 
 * 
 */
struct dataSkin{
    float valueCorneum=0.0,valueSpinosum=0.0, valueBasale=0.0, valueDermis=0.0,valueHipoDermis=0.0;
};


void make_layer(openvdb::FloatGrid::Accessor& accessor, int size,int depth, openvdb::Coord& origin,float value,int increment ){
   
    // Distance value for the constant region exterior to the narrow band
    //const ValueT outside = grid.background();
    // Distance value for the constant region interior to the narrow band
    // (by convention, the signed distance is negative in the interior of
    // a level set)
    
    // Use the background value as the width in voxels of the narrow band.
    // (The narrow band is centered on the surface of the sphere, which
    // has distance 0.)
    
    // The bounding box of the narrow band is 2*dim voxels on a side.
    
    int dim = size;
    // Get a voxel accessor.
    //typename GridType::Accessor accessor = grid.getAccessor();
    
    int &i = origin[0], &j = origin[1], &k = origin[2];
    int min_size = origin[0]-size;
    int min_depth = origin[1] - depth;
    int min_size_2 = origin[2]-size;
    int k_0 = origin[2];
    int i_0 = origin[0];
    int j_0 = origin[1];
    
    for(k=k_0 ;k>min_size_2;k-=increment){
        for(i=i_0;i > min_size;i-=increment){
            for(j=j_0 ;j>min_depth;j-=increment){
                accessor.setValue(origin, value);
                //std::cout<<origin<<std::endl;
            }
            //std::cout<<i<<std::endl;
        }
    }
}


void createSkin(openvdb::FloatGrid::Accessor& accessor,int size_lado,int profundidad_total,openvdb::Coord coordenadas,dataSkin data){
    
    
    //Epidermis = 0.2mm
    //  Stratum corneum
    //  Stratum spinosum
    //  Basal membrane
    //Dermis = 1-4mm
    //Hypodermis 4-9mm  Se asigna poco
    int profundidadHipoDermis = 20;
    profundidad_total = profundidad_total - profundidadHipoDermis;
    float epidermis =0.3; //Proporcion de Epidermis
    int profundidadEpidermis = profundidad_total * epidermis;
    int profundidadBasale = profundidadEpidermis * 0.1;
    int profundidadSpinosum = profundidadEpidermis * 0.15;
    int profundidadCorneum = profundidadEpidermis - profundidadBasale - profundidadSpinosum;

    int profundidadDermis = profundidad_total - profundidadEpidermis;

    std::cout<<"Pre "<<coordenadas<<std::endl;
    coordenadas[0] = size_lado/2;
    coordenadas[2] = size_lado/2;
    make_layer(accessor,size_lado,profundidadCorneum,coordenadas,data.valueCorneum,1);
    coordenadas[0] = size_lado/2;
    coordenadas[2] = size_lado/2;
    make_layer(accessor,size_lado,profundidadSpinosum,coordenadas,data.valueSpinosum,1);
    coordenadas[0] = size_lado/2;
    coordenadas[2] = size_lado/2;
    make_layer(accessor,size_lado,profundidadBasale,coordenadas,data.valueBasale,1);
    coordenadas[0] = size_lado/2;
    coordenadas[2] = size_lado/2;
    
    make_layer(accessor,size_lado,profundidadDermis,coordenadas,data.valueDermis,1);
    
    coordenadas[0] = size_lado/2;
    coordenadas[2] = size_lado/2;
    make_layer(accessor,size_lado,profundidadHipoDermis,coordenadas,data.valueHipoDermis,1);
    std::cout<<"Post "<<coordenadas<<std::endl;
    
    //std::cout<<"Profundidad real "<<profundidadHipoDermis + profundidadBasale + profundidadSpinosum + profundidadCorneum+profundidadDermis<<std::endl;
    //makeSkin(*grid,20,openvdb::Vec3f(1.5, 2, 3));
    //makeSphere(*grid, /*radius=*/50.0, /*center=*/openvdb::Vec3f(1.5, 2, 3));
    // Associate some metadata with the grid.
    //grid.insertMeta("radius", openvdb::FloatMetadata(50.0));
    // Associate a scaling transform with the grid that sets the voxel size
    // to 0.5 units in world space.
    // grid.setTransform(
    //     openvdb::math::Transform::createLinearTransform(/*voxel size=*/0.01));
    // Identify the grid as a level set.
    //grid.setGridClass(openvdb::GRID_STAGGERED );
    // Name the grid "LevelSetSphere".
    //grid.setName("LevelSetSphere");
}

void copyNanoToOpen(const nanovdb::FloatGrid* gridNano,openvdb::FloatGrid& gridOpen,int size_lado,int profundidad_total){
    openvdb::Coord coordenadas_open;
    nanovdb::Coord coordenadas_nano;

    auto accessor_nano = gridNano->getAccessor();
    auto accessor_open = gridOpen.getAccessor();

    for(int i  =0;i>-size_lado;i--){
        for(int j = 0 ;j>-profundidad_total;j--){
            for(int k = 0 ;k>-size_lado;k--){
                coordenadas_nano = nanovdb::Coord(i,j,k);
                coordenadas_open = openvdb::Coord(i,j,k);
                
                accessor_open.setValue(coordenadas_open,accessor_nano.getValue(coordenadas_nano));
            }
        }
    }
}
void generateEndoThelial(openvdb::FloatGrid::Accessor& accessor,int size_lado,int profundidad_total,int lim_inf,int lim_sup,int modulo){
    
    float ini_endothelial = 1.0;
    openvdb::Coord coord;
    for(int i  =0;i>-size_lado;i--){
        for(int j = 0 ;j>-profundidad_total;j--){
            for(int k = 0 ;k>-size_lado;k--){
                coord = openvdb::Coord(i,j,k);
                if(coord[1]>lim_inf && coord[1]<lim_sup){
                    if(coord[0]%modulo == 0 && coord[2]%modulo == 0 ){
                        accessor.setValueOnly(coord,ini_endothelial);
                    }else{
                        accessor.setValueOnly(coord,accessor.getValue(coord));
                    }
                }else{
                    accessor.setValueOnly(coord,accessor.getValue(coord));
                }
                if(coord[1]==lim_inf || coord[1]==lim_sup){
                    accessor.setValueOnly(coord,ini_endothelial);
                }else{
                    accessor.setValueOnly(coord,accessor.getValue(coord));
                }
                    
                    
            }
        }
    }
}

void createRectangle(openvdb::FloatGrid::Accessor& accessor,openvdb::Coord esquinaIzquierda,int size,float new_value){
    //openvdb::FloatGrid::Accessor accessor = grid->getAccessor();
    for(int i = 0;i<size;i++){
        for(int j = 0;j<size;j++){
            for(int k = 0 ;k<size;k++){
                float distancia_i = std::pow(std::abs(i - size/2.0),2);
                float distancia_j = std::pow(std::abs(j - size/2.0),2);
                float distancia_k = std::pow(std::abs(k - size/2.0),2);

                float distancia = std::sqrt(distancia_i+ distancia_j+distancia_k);

                openvdb::Coord new_coords = openvdb::Coord(esquinaIzquierda[0]+i,esquinaIzquierda[1]+j,esquinaIzquierda[2]+k);
                //std::cout<<(new_value*new_value)/distancia<<std::endl;
                accessor.setValue(new_coords,new_value);
            }
        }
    }    
}

void createColumns(openvdb::FloatGrid::Accessor& accessor,int size_lado,int profundidad_total){
    //openvdb::FloatGrid::Accessor accessor = grid->getAccessor();
    float new_value = 1.0;
    for(int i = 0;i>-size_lado;i--){
        for(int j = 0;j>-profundidad_total;j--){
            for(int k = 0 ;k>-size_lado;k--){
                openvdb::Coord new_coords = openvdb::Coord(i,j,k);
                //std::cout<<(new_value*new_value)/distancia<<std::endl;
                if(-new_coords[0] % 2 ==0 ){
                    accessor.setValue(new_coords,new_value);

                }
            }
        }
    }    
}