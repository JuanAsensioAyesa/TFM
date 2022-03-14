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
#include "pruebaThrust.h"
#include <openvdb/tools/ValueTransformer.h>
#include <openvdb/Types.h>
#include <iostream>
float average_surrounding(const openvdb::FloatGrid::ValueOnCIter& iter,openvdb::Coord coordenadas){
    auto vec = coordenadas.asVec3i();
    //auto vec = iter.getCoord().asVec3i();
    int incrementos[] = {-1,0,1};
    int len_incrementos = 3;
    float accum = 0;
    openvdb::Coord new_coord ;
    openvdb::v9_0::math::Coord::Vec3i new_vec;
    for(int i_incremento_x = 0;i_incremento_x<len_incrementos;i_incremento_x++){
        for(int i_incremento_y = 0 ;i_incremento_y<len_incrementos;i_incremento_y++){
            for(int i_incremento_z = 0 ;i_incremento_z<len_incrementos;i_incremento_z++){
                int incremento_x = incrementos[i_incremento_x];
                int incremento_y = incrementos[i_incremento_y];
                int incremento_z = incrementos[i_incremento_z];

                
                new_vec[0] = vec[0]+incremento_x;
                new_vec[1] = vec[1]+incremento_y;
                new_vec[2] = vec[2]+incremento_z;

                new_coord = openvdb::Coord(new_vec);
                accum += iter.getTree()->getValue(new_coord);
                
            }
        }
    }
    //std::cout<<vec<<std::endl;
    accum = accum /(len_incrementos * len_incrementos * len_incrementos);
    
    return accum;
}
/**
 * @brief 
 * 
 * @tparam GridType 
 * @param grid 
 * @param size 
 * @param origin 
 */
struct Local {
    
    static inline void opCopia(const openvdb::FloatGrid::ValueOnCIter& iter,openvdb::FloatGrid::Accessor& accessor) {
        
        auto coords = iter.getCoord();
        accessor.setValue(coords,iter.getValue());
        
    }
    static inline void opAverage(const  openvdb::FloatGrid::ValueOnCIter& iter,openvdb::FloatGrid::Accessor& accessor) {
        
        auto coords = iter.getCoord();
        accessor.setValue(coords,average_surrounding(iter,coords));
       
    }
    static inline void opSet(const openvdb::FloatGrid::ValueOnCIter& iter,openvdb::FloatGrid::Accessor& accessor) {
        
        auto coords = iter.getCoord();
        accessor.setValue(coords,0.0);

    }
};
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
template<class GridType>
void createSkin(GridType& grid,int size_lado,int profundidad_total,openvdb::Coord coordenadas){
    
    
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
    make_layer(grid,size_lado,profundidadCorneum,coordenadas,0.0);
    coordenadas[0] = 0;
    coordenadas[2] = 0 ;
    make_layer(grid,size_lado,profundidadSpinosum,coordenadas,1.0,1);
    coordenadas[0] = 0;
    coordenadas[2] = 0 ;
    make_layer(grid,size_lado,profundidadBasale,coordenadas,2.0,1);
    coordenadas[0] = 0;
    coordenadas[2] = 0 ;
    make_layer(grid,size_lado,profundidadDermis,coordenadas,3.0,1);
    coordenadas[0] = 0;
    coordenadas[2] = 0 ;
    make_layer(grid,size_lado,profundidadHipoDermis,coordenadas,4.0,1);

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



template<class GridType>
void promediate(GridType& gridSource,GridType& gridDestiny,int size_lado,int profundidad_total){
    using ValueT = typename GridType::ValueType;
    const ValueT outside = gridSource.background();
    // Distance value for the constant region interior to the narrow band
    // (by convention, the signed distance is negative in the interior of
    // a level set)
    
    // Use the background value as the width in voxels of the narrow band.
    // (The narrow band is centered on the surface of the sphere, which
    // has distance 0.)
    int padding = int(openvdb::math::RoundUp(openvdb::math::Abs(outside)));
    // The bounding box of the narrow band is 2*dim voxels on a side.
    int dim = int(size_lado + padding);
    dim = size_lado;
    typename GridType::Accessor accessorSource = gridSource.getAccessor();
    typename GridType::Accessor accessorDestiny = gridDestiny.getAccessor();
    int increment = 1;
    openvdb::Coord origin(0,0,0);
    int &i = origin[0], &j = origin[1], &k = origin[2];
    int min_size = origin[0]-size_lado;
    int min_depth = origin[1] - profundidad_total;
    int min_size_2 = origin[2]-size_lado;
    int k_0 = origin[2];
    int i_0 = origin[0];
    int j_0 = origin[1];
    
    for(k=k_0 ;k>min_size_2;k-=increment){
        for(i=i_0;i > min_size;i-=increment){
            for(j=j_0 ;j>min_depth;j-=increment){
                accessorDestiny.setValue(origin, average_surrounding(gridSource.cbeginValueOn(),origin));
                //std::cout<<origin<<std::endl;
            }
            //std::cout<<i<<std::endl;
        }
    }
    
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
/// @brief This examples depends on OpenVDB, NanoVDB and CUDA.
int main()
{
    try {
        // Create an OpenVDB grid of a sphere at the origin with radius 100 and voxel size 1.
    openvdb::FloatGrid::Ptr grid =
       openvdb::FloatGrid::create(/*background value=*/2.0);
    openvdb::FloatGrid::Ptr grid2Open =
       openvdb::FloatGrid::create(/*background value=*/2.0);
        int size_lado = 250;
        int profundidad_total = 150;
        openvdb::Coord coordenadas;
        createSkin(*grid,size_lado,profundidad_total,coordenadas);
        createSkin(*grid2Open,size_lado,profundidad_total,coordenadas);
        int media_width = 20;//40 de ancho
        int media_depth = 10;//20 de profundidad
        openvdb::Coord coords_nuevas(-250/2+media_width,-profundidad_total/2+media_depth,-250/2+media_width);
        make_layer(*grid,media_width*2,media_depth*2,coords_nuevas,10.0);
        using GridT = nanovdb::FloatGrid;
        // Converts the OpenVDB to NanoVDB and returns a GridHandle that uses CUDA for memory management.
        auto handle = nanovdb::openToNanoVDB<nanovdb::CudaDeviceBuffer>(*grid);
        auto handle2 = nanovdb::openToNanoVDB<nanovdb::CudaDeviceBuffer>(*grid2Open);
        handle.deviceUpload(0,true); // Copy the NanoVDB grid to the GPU asynchronously
        handle2.deviceUpload(0, true); // Copy the NanoVDB grid to the GPU asynchronously
        const GridT* grid2 = handle.grid<float>(); // get a (raw) const pointer to a NanoVDB grid of value type float on the CPU
        GridT* deviceGrid = handle.deviceGrid<float>(); // get a (raw) pointer to a NanoVDB grid of value type float on the GPU
        GridT* deviceGrid2 = handle2.deviceGrid<float>(); // get a (raw) pointer to a NanoVDB grid of value type float on the GPU
        GridT* readGridNano= deviceGrid;
        GridT* writeGridNano = deviceGrid2;
        for(int i = 0 ;i<1001;i++){
            std::cout<<i<<std::endl;
            if(i%2==0){
                readGridNano= deviceGrid;
                writeGridNano = deviceGrid2;
            }else{
                readGridNano= deviceGrid2;
                writeGridNano = deviceGrid;
            }
            average(readGridNano, writeGridNano,grid2->tree().nodeCount(0));
            //scaleActiveVoxels(readGridNano,grid2->tree().nodeCount(0),0.5);
        }
        
        handle.deviceDownload(0, true); // Copy the NanoVDB grid to the CPU synchronously
        handle2.deviceDownload(0,true);

        //cudaStreamDestroy(stream); // Destroy the CUDA stream
        //std::cout << "Value after scaling  = " << handle.grid<float>()->tree().getValue(nanovdb::Coord(101,0,0)) << std::endl;
        
        
        // openvdb::FloatGrid::Ptr outGrid = openvdb::FloatGrid::create(2.0);
        // openvdb::FloatGrid::Ptr readGrid;
        // openvdb::FloatGrid::Ptr writeGrid;
        // for(int i = 0 ;i<10;i++){
        //     if(i%2==0){
        //         readGrid = grid;
        //         writeGrid = outGrid;
        //     }else{
        //         readGrid = outGrid;
        //         outGrid = grid;
        //     }
        //     promediate(*readGrid,*writeGrid,size_lado,profundidad_total);
        // }
        
        
        // for(int i = 0 ;i<101;i++){
        //     std::cout<<i<<std::endl;
        //     if(i%2 == 0 ){
        //         readGrid = grid;
        //         writeGrid = outGrid;
        //     }else{
        //         readGrid = outGrid;
        //         writeGrid = grid;
        //     }
        //     // openvdb::tools::transformValues(readGrid->cbeginValueOn(),*writeGrid,Local::opAverage);
        //     //openvdb::tools::transformValues(grid->cbeginValueOn(),*outGrid,Local::opSet);
        //     openvdb::tools::transformValues(readGrid->cbeginValueOn(),*writeGrid,Local::opAverage,true);
        //     //openvdb::tools::transformValues(outGrid->cbeginValueOn(),*grid,Local::opCopia);
        //     //openvdb::tools::transformValues(grid->cbeginValueOn(),*outGrid,Local::opSet);

        // }
        // //openvdb::tools::transformValues(grid->cbeginValueOn(),*outGrid,Local::opAverage);
        // int cont  = 0 ;
        // std::cout<<grid->cbeginValueOn().summary()<<std::endl;
        // //std::cout<<"Active "<<cont<<" Total:"<<size_lado*size_lado*profundidad_total<<std::endl;
        //openvdb::Coord coords_nuevas_2(0,-profundidad_total-100,0);
        //createSkin(*outGrid,size_lado,profundidad_total,coords_nuevas_2);
        std::cout<<"Pre copy"<<std::endl;
        copyNanoToOpen(grid2,*grid,size_lado,profundidad_total);
        std::cout<<"Post copy"<<std::endl;
        openvdb::io::File file("mygrids.vdb");
        // Add the grid pointer to a container.
        openvdb::GridPtrVec grids;
        //grids.push_back(grid);
        grids.push_back(grid);
        // Write out the contents of the container.
        file.write(grids);
        file.close();
    }
    catch (const std::exception& e) {
        std::cerr << "An exception occurred: \"" << e.what() << "\"" << std::endl;
    }
    return 0;
}