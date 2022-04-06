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
#include <stdlib.h>     /* atoi */
#include "pruebaThrust.h"
#include "utilsSkin/utilSkin.hpp"
#include "utilGrid/Grid.hpp"
#include <vector>
#include <openvdb/math/Stencils.h>
#include <nanovdb/util/Stencils.h>
#include <vector>
#include <fstream>


float computeMean(nanovdb::FloatGrid * grid){
    float accum = 0.0;
    int size_lado = 250;
    int profundidad_total = 150;
    auto  accessor = grid->getAccessor();
    nanovdb::Coord coord;
    int count = 0 ;
    for(int i  =0;i>-size_lado;i--){
            for(int j = 0 ;j>-profundidad_total;j--){
                for(int k = 0 ;k>-size_lado;k--){
                    coord = openvdb::Coord(i,j,k);
                    accum+=accessor.getValue(coord);
                    count++;
                }
            }
    }
    return accum/count;
}

float computeMax(nanovdb::FloatGrid * grid){
    float max = -100000.0;
    int size_lado = 250;
    int profundidad_total = 150;
    auto  accessor = grid->getAccessor();
    nanovdb::Coord coord;
    
    for(int i  =0;i>-size_lado;i--){
            for(int j = 0 ;j>-profundidad_total;j--){
                for(int k = 0 ;k>-size_lado;k--){
                    coord = openvdb::Coord(i,j,k);
                    float aux = accessor.getValue(coord);
                    if(aux > max){
                        max = aux;
                    }
                }
            }
    }
    return max;
}
void writeVector(std::vector<float>& vec,std::string fileName){
    std::ofstream myfile;
    myfile.open (fileName);
    for(auto data : vec){
        myfile<<data<<std::endl;
    }
    myfile.close();
}
int main(int argc,char * argv[]){
    using Vec3 = openvdb::Vec3s;
    using  Vec3Open = openvdb::Vec3SGrid ;
    using Vec3Nano = nanovdb::Vec3fGrid;
    Grid<> gridEndothelial(250,150,0);
    Grid<> gridTAF(250,150,0);
    Grid<> gridFibronectin(250,150,0);
    openvdb::Vec3s ini = {0.0,0.0,0.0};
    Grid<Vec3,nanovdb::Vec3f,Vec3Open,Vec3Open::Ptr,Vec3Nano> gridGradienteTAF(250,150,ini,false);
    Grid<Vec3,nanovdb::Vec3f,Vec3Open,Vec3Open::Ptr,Vec3Nano> gridGradienteFibronectin(250,150,ini,false);
    
    /**
     * Creamos la piel con los datos pertinentes
     * 
     */
    // openvdb::FloatGrid::Ptr grid_open =
    //    openvdb::FloatGrid::create(/*background value=*/2.0);
    // openvdb::FloatGrid::Ptr grid_open_2 =
    //    openvdb::FloatGrid::create(/*background value=*/2.0);
    dataSkin dataIniEndothelial;
    dataIniEndothelial.valueBasale = 0;
    dataIniEndothelial.valueCorneum = 0;
    dataIniEndothelial.valueDermis = 0;
    dataIniEndothelial.valueHipoDermis = 0;
    dataIniEndothelial.valueSpinosum = 0;
    int size_lado = 250;
    int profundidad_total = 150;
    openvdb::Coord coordenadas;
    openvdb::FloatGrid::Accessor accessor_1_endothelial = gridEndothelial.getAccessorOpen1();
    openvdb::FloatGrid::Accessor accessor_2_endothelial = gridEndothelial.getAccessorOpen2();
    
    openvdb::FloatGrid::Accessor accessor_1_TAF = gridTAF.getAccessorOpen1();
    openvdb::FloatGrid::Accessor accessor_2_TAF = gridTAF.getAccessorOpen2();

    openvdb::FloatGrid::Accessor accessor_1_Fibronectin = gridFibronectin.getAccessorOpen1();
    openvdb::FloatGrid::Accessor accessor_2_Fibronectin = gridFibronectin.getAccessorOpen2();
    
    //gridEndothelial.fillRandom();
    createSkin(accessor_1_endothelial,size_lado,profundidad_total,coordenadas,dataIniEndothelial);
    createSkin(accessor_2_endothelial,size_lado,profundidad_total,coordenadas,dataIniEndothelial);
    
    createSkin(accessor_1_TAF,size_lado,profundidad_total,coordenadas,dataIniEndothelial);
    createSkin(accessor_2_TAF,size_lado,profundidad_total,coordenadas,dataIniEndothelial);

    createSkin(accessor_1_Fibronectin,size_lado,profundidad_total,coordenadas,dataIniEndothelial);
    createSkin(accessor_2_Fibronectin,size_lado,profundidad_total,coordenadas,dataIniEndothelial);

    gridGradienteFibronectin.fillRandom();
    gridGradienteTAF.fillRandom();
    
    //gridVectorPrueba.fillRandom();
    gridEndothelial.upload();
    gridTAF.upload();
    gridFibronectin.upload();
    gridGradienteTAF.upload();
    gridGradienteFibronectin.upload();
    int n;
    std::cin>>n;
    //gridVectorPrueba.upload();
    
    int veces = 11;
    if(argc > 1){
        veces = atoi(argv[1]);
    }
    nanovdb::FloatGrid* gridRead;
    nanovdb::FloatGrid* gridWrite;
    nanovdb::FloatGrid* gridRead_CPU;
    nanovdb::FloatGrid* gridWrite_CPU;
    generateEndothelial(gridEndothelial.getPtrNano1(typePointer::DEVICE),gridEndothelial.getPtrNano1(typePointer::CPU)->tree().nodeCount(0),-39,-130,4);
    generateEndothelial(gridEndothelial.getPtrNano2(typePointer::DEVICE),gridEndothelial.getPtrNano2(typePointer::CPU)->tree().nodeCount(0),-39,-130,4);
    // gridEndothelial.writeToFile("pre.vdb");
    // auto  vector_medias = std::vector<float>();
    for(int i = 0 ;i<veces;i++){
        std::cout<<i<<std::endl;
        if(i % 2 == 0 ){
            gridRead = gridEndothelial.getPtrNano1(typePointer::DEVICE);
            gridWrite = gridEndothelial.getPtrNano2(typePointer::DEVICE);
            

        }else{
            gridRead = gridEndothelial.getPtrNano2(typePointer::DEVICE);
            gridWrite = gridEndothelial.getPtrNano1(typePointer::DEVICE);
            
        }
        
        

    }
    //pruebaGradiente(gridVectorPrueba.getPtrNanoWrite(typePointer::DEVICE),gridEndothelial.getPtrNanoRead(typePointer::DEVICE),gridEndothelial.getPtrNanoRead(typePointer::CPU)->tree().nodeCount(0));

    //gridVectorPrueba.download();
    gridEndothelial.download();
    std::cout<<"Copy pre"<<std::endl;
    gridEndothelial.copyNanoToOpen();
    std::cout<<"Copy post"<<std::endl;
    gridEndothelial.writeToFile("post.vdb"); 
   
    
    
    //gridVectorPrueba.copyNanoToOpen();
    //gridVectorPrueba.fillRandom();
    //gridVectorPrueba.writeToFile("myGrids.vdb");
    
    //writeVector(vector_medias,"vector_max.txt");

    //createSkin(*grid_open_2,size_lado,profundidad_total,coordenadas,dataIniEndothelial);
    
    // /**
    //  * Transformamos a nano
    //  * 
    //  */
    // auto handle_grid_endothelial = nanovdb::openToNanoVDB<nanovdb::CudaDeviceBuffer>(*grid_open);
    // auto handle_grid_endothelial_2 = nanovdb::openToNanoVDB<nanovdb::CudaDeviceBuffer>(*grid_open_2);
    // /**
    //  * Subimos a la gpu y obtenemos los handle
    //  * "
    //  */
    // using GridT = nanovdb::FloatGrid;
    // handle_grid_endothelial.deviceUpload(0,true); // Copy the NanoVDB grid to the GPU synchronously
    // handle_grid_endothelial_2.deviceUpload(0,true); // Copy the NanoVDB grid to the GPU synchronously
    // const GridT* nano_grid_cpu = handle_grid_endothelial.grid<float>(); // get a (raw) const pointer to a NanoVDB grid of value type float on the CPU
    // GridT* nano_grid_device = handle_grid_endothelial.deviceGrid<float>(); // get a (raw) pointer to a NanoVDB grid of value type float on the GPU
    // const GridT* nano_grid_cpu_2 = handle_grid_endothelial.grid<float>(); // get a (raw) const pointer to a NanoVDB grid of value type float on the CPU
    // GridT* nano_grid_device_2 = handle_grid_endothelial.deviceGrid<float>(); // get a (raw) pointer to a NanoVDB grid of value type float on the GPU
    // /**
    //  * Lanzamos kernel y obtenemos datos
    //  * 
    //  */
    
    // int lim_inf = -profundidad_total+1;
    // int lim_sup = 0 ;
    // int modulo = 3;//La mitad de leafs seran endothelial
    // generateEndothelial(nano_grid_device,nano_grid_cpu->tree().nodeCount(0),lim_sup,lim_inf,modulo);
    
    
    // handle_grid_endothelial.deviceDownload(0,true);

    // /**
    //  * Generamos los grid necesarios para calcular el TAF iterativamente
    //  * 
    //  */
    // openvdb::FloatGrid::Ptr grid_open_TAF_1 =
    //    openvdb::FloatGrid::create(/*background value=*/2.0);
    // openvdb::FloatGrid::Ptr grid_open_TAF_2 =
    //    openvdb::FloatGrid::create(/*background value=*/2.0);
    
    
    // //createSkin(*grid_open_TAF_1,size_lado,profundidad_total,coordenadas,dataIniEndothelial);
    // //createSkin(*grid_open_TAF_2,size_lado,profundidad_total,coordenadas,dataIniEndothelial);

    // auto handle_grid_TAF_1 = nanovdb::openToNanoVDB<nanovdb::CudaDeviceBuffer>(*grid_open_TAF_1);
    // auto handle_grid_TAF_2 = nanovdb::openToNanoVDB<nanovdb::CudaDeviceBuffer>(*grid_open_TAF_2);
    // handle_grid_TAF_1.deviceUpload(0,true);
    // handle_grid_TAF_2.deviceUpload(0,true);
    // handle_grid_endothelial.deviceUpload(0,true);
    
    // GridT* grid_cpu_TAF_1 = handle_grid_TAF_1.grid<float>(); // get a (raw) const pointer to a NanoVDB grid of value type float on the CPU
    // GridT* grid_device_TAF_1 = handle_grid_TAF_1.deviceGrid<float>(); // get a (raw) pointer to a NanoVDB grid of value type float on the GPU

    // GridT* grid_cpu_TAF_2 = handle_grid_TAF_2.grid<float>(); // get a (raw) const pointer to a NanoVDB grid of value type float on the CPU
    // GridT* grid_device_TAF_2 = handle_grid_TAF_2.deviceGrid<float>(); // get a (raw) pointer to a NanoVDB grid of value type float on the GPU

    // GridT* readGridTAF = grid_device_TAF_1;
    // GridT* writeGridTAF = grid_device_TAF_2;
    // int veces = 11;
    // if(argc > 1){
    //     veces = atoi(argv[1]);
    // }
    // for(int i = 0 ;i<veces;i++){
    //     std::cout<<i<<std::endl;
    //     if(i%2 == 0 ){
    //         readGridTAF = grid_device_TAF_1;
    //         writeGridTAF = grid_device_TAF_2;
    //     }else{
    //         readGridTAF = grid_device_TAF_2;
    //         writeGridTAF = grid_device_TAF_1;
    //     }
    //     equationTAF(nano_grid_device,readGridTAF,writeGridTAF,grid_cpu_TAF_1->tree().nodeCount(0));
    // }
    // handle_grid_TAF_1.deviceDownload(0,true);
    // handle_grid_TAF_2.deviceDownload(0,true);
    // handle_grid_endothelial.deviceDownload(0,true);
    

    // /**
    //  * Volvemos a copiar a open
    //  * 
    //  */
    // std::cout<<"Pre copy"<<std::endl;
    // copyNanoToOpen(grid_cpu_TAF_2,*grid_open_TAF_1,size_lado,profundidad_total);
    // copyNanoToOpen(nano_grid_cpu,*grid_open,size_lado,profundidad_total);
    // std::cout<<"Post copy"<<std::endl;

    





    
    
    return 0;

}