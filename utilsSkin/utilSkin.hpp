
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

/**
 * @brief Para pasar los datos a la funcion createSkin 
 * 
 */
struct dataSkin{
    float valueCorneum=0.0,valueSpinosum=0.0, valueBasale=0.0, valueDermis=0.0,valueHipoDermis=0.0;
};

template<class GridType>
void make_layer(GridType& grid, int size,int depth, openvdb::Coord& origin,float value,int increment ){
    using ValueT = typename GridType::ValueType;
    // Distance value for the constant region exterior to the narrow band
    const ValueT outside = grid.background();
    // Distance value for the constant region interior to the narrow band
    // (by convention, the signed distance is negative in the interior of
    // a level set)
    
    // Use the background value as the width in voxels of the narrow band.
    // (The narrow band is centered on the surface of the sphere, which
    // has distance 0.)
    int padding = int(openvdb::math::RoundUp(openvdb::math::Abs(outside)));
    // The bounding box of the narrow band is 2*dim voxels on a side.
    int dim = int(size + padding);
    dim = size;
    // Get a voxel accessor.
    typename GridType::Accessor accessor = grid.getAccessor();
    
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

template<class GridType>
void createSkin(GridType& grid,int size_lado,int profundidad_total,openvdb::Coord coordenadas,dataSkin data){
    
    
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

    //std::cout<<"Pre "<<coordenadas<<std::endl;
    make_layer(grid,size_lado,profundidadCorneum,coordenadas,data.valueCorneum,1);
    coordenadas[0] = 0;
    coordenadas[2] = 0 ;
    make_layer(grid,size_lado,profundidadSpinosum,coordenadas,data.valueSpinosum,1);
    coordenadas[0] = 0;
    coordenadas[2] = 0 ;
    make_layer(grid,size_lado,profundidadBasale,coordenadas,data.valueBasale,1);
    coordenadas[0] = 0;
    coordenadas[2] = 0 ;
    make_layer(grid,size_lado,profundidadDermis,coordenadas,data.valueDermis,1);
    coordenadas[0] = 0;
    coordenadas[2] = 0 ;
    make_layer(grid,size_lado,profundidadHipoDermis,coordenadas,data.valueHipoDermis,1);

    //std::cout<<"Post "<<coordenadas<<std::endl;
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