// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0
#include <nanovdb/util/GridBuilder.h>
#include <openvdb/tools/LevelSetSphere.h> // replace with your own dependencies for generating the OpenVDB grid
#include <nanovdb/util/OpenToNanoVDB.h> // converter from OpenVDB to NanoVDB (includes NanoVDB.h and GridManager.h)
#include <nanovdb/util/CudaDeviceBuffer.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_device_runtime_api.h>
#include "kernels.h"
#include <openvdb/tools/ValueTransformer.h>
#include <openvdb/Types.h>
#include <iostream>

template<class GridType>
void make_layer(GridType& grid, int size,int depth, openvdb::Coord& origin,float value,int increment = 1){
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
/**
 * @brief Para pasar los datos a la funcion createSkin 
 * 
 */
struct dataSkin{
    float valueCorneum=0.0,valueSpinosum=0.0, valueBasale=0.0, valueDermis=0.0,valueHipoDermis=0.0;
};
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

    std::cout<<"Pre "<<coordenadas<<std::endl;
    make_layer(grid,size_lado,profundidadCorneum,coordenadas,data.valueCorneum);
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

    std::cout<<"Post "<<coordenadas<<std::endl;
    std::cout<<"Profundidad real "<<profundidadHipoDermis + profundidadBasale + profundidadSpinosum + profundidadCorneum+profundidadDermis<<std::endl;
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

int main(){
    /**
     * Creamos la piel con los datos pertinentes
     * 
     */
    openvdb::FloatGrid::Ptr grid_open =
       openvdb::FloatGrid::create(/*background value=*/2.0);
    dataSkin dataIniEndothelial;
    int size_lado = 250;
    int profundidad_total = 150;
    openvdb::Coord coordenadas;
    createSkin(*grid_open,size_lado,profundidad_total,coordenadas,dataIniEndothelial);
    
    /**
     * Transformamos a nano
     * 
     */
    auto handle_grid = nanovdb::openToNanoVDB<nanovdb::CudaDeviceBuffer>(*grid_open);
    /**
     * Subimos a la gpu y obtenemos los handle
     * 
     */
    using GridT = nanovdb::FloatGrid;
    handle_grid.deviceUpload(0,true); // Copy the NanoVDB grid to the GPU synchronously
    const GridT* nano_grid_cpu = handle_grid.grid<float>(); // get a (raw) const pointer to a NanoVDB grid of value type float on the CPU
    GridT* nano_grid_device = handle_grid.deviceGrid<float>(); // get a (raw) pointer to a NanoVDB grid of value type float on the GPU
    /**
     * Lanzamos kernel y obtenemos datos
     * 
     */
    
    int lim_inf = -profundidad_total+1;
    int lim_sup = 0 ;
    int modulo = 3;//La mitad de leafs seran endothelial
    generateEndothelial(nano_grid_device,nano_grid_cpu->tree().nodeCount(0),lim_sup,lim_inf,modulo);
    handle_grid.deviceDownload(0,true);
    /**
     * Volvemos a copiar a open
     * 
     */
    std::cout<<"Pre copy"<<std::endl;
    copyNanoToOpen(nano_grid_cpu,*grid_open,size_lado,profundidad_total);
    std::cout<<"Post copy"<<std::endl;
    /**
     * Escribimos a fichero
     * 
     */
    openvdb::io::File file("mygrids.vdb");
        // Add the grid pointer to a container.
        openvdb::GridPtrVec grids;
        //grids.push_back(grid);
        grids.push_back(grid_open);
        // Write out the contents of the container.
        file.write(grids);
        file.close();
    
    return 0;

}