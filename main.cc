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

    openvdb::Coord esquina_izquierda = openvdb::Coord();
    esquina_izquierda[0] = -size_lado / 2;
    esquina_izquierda[1] = -profundidad_total/2;
    esquina_izquierda[2] = -size_lado /2;
    gridGradienteFibronectin.fillRandom();
    gridGradienteTAF.fillRandom();
    int tamanio_tumor = 20;//Tamanio en voxels
   
    openvdb::FloatGrid::Accessor accessorFibronectin1 = gridFibronectin.getAccessorOpen1();
    openvdb::FloatGrid::Accessor accessorFibronectin2 = gridFibronectin.getAccessorOpen2();
    openvdb::FloatGrid::Accessor accessorTAF1 = gridTAF.getAccessorOpen1();
    openvdb::FloatGrid::Accessor accessorTAF2 = gridTAF.getAccessorOpen2();
    createRectangle(accessorTAF1,esquina_izquierda,tamanio_tumor,4.0);
    createRectangle(accessorTAF2,esquina_izquierda,tamanio_tumor,4.0);
    createRectangle(accessorFibronectin1,esquina_izquierda,tamanio_tumor,4.0);
    createRectangle(accessorFibronectin2,esquina_izquierda,tamanio_tumor,4.0);


    
    
    //gridVectorPrueba.fillRandom();
    gridEndothelial.upload();
    gridTAF.upload();
    gridFibronectin.upload();
    gridGradienteTAF.upload();
    gridGradienteFibronectin.upload();
    // int n;
    // std::cin>>n;
    //gridVectorPrueba.upload();
    
    int veces = 11;
    if(argc > 1){
        veces = atoi(argv[1]);
    }
    nanovdb::FloatGrid* gridRead;
    nanovdb::FloatGrid* gridWrite;
    nanovdb::FloatGrid* gridRead_CPU;
    nanovdb::FloatGrid* gridWrite_CPU;
    uint64_t nodeCount = gridGradienteFibronectin.getPtrNano1(typePointer::CPU)->tree().nodeCount(0);
    std::cout<<"NodeCount "<<nodeCount<<std::endl;
    generateEndothelial(gridEndothelial.getPtrNano1(typePointer::DEVICE),nodeCount,-39,-130,4);
    generateEndothelial(gridEndothelial.getPtrNano2(typePointer::DEVICE),nodeCount,-39,-130,4);
    generateGradientFibronectin(gridFibronectin.getPtrNano1(typePointer::DEVICE),gridEndothelial.getPtrNano1(typePointer::DEVICE),gridGradienteFibronectin.getPtrNano1(typePointer::DEVICE),nodeCount);
    generateGradientTAF(gridTAF.getPtrNano1(typePointer::DEVICE),gridEndothelial.getPtrNano1(typePointer::DEVICE),gridGradienteTAF.getPtrNano1(typePointer::DEVICE),nodeCount);
    // generateGradientFibronectin(gridFibronectin.getPtrNano1(typePointer::DEVICE),gridEndothelial.getPtrNano1(typePointer::DEVICE),gridGradienteFibronectin.getPtrNano2(typePointer::DEVICE),nodeCount);
    // generateGradientTAF(gridTAF.getPtrNano1(typePointer::DEVICE),gridEndothelial.getPtrNano1(typePointer::DEVICE),gridGradienteTAF.getPtrNano2(typePointer::DEVICE),nodeCount);
    //pruebaGradiente(gridGradienteTAF.getPtrNano1(typePointer::DEVICE),gridEndothelial.getPtrNano1(typePointer::DEVICE),nodeCount);
    

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
        
        

    // }
    //pruebaGradiente(gridVectorPrueba.getPtrNanoWrite(typePointer::DEVICE),gridEndothelial.getPtrNanoRead(typePointer::DEVICE),gridEndothelial.getPtrNanoRead(typePointer::CPU)->tree().nodeCount(0));

    //gridVectorPrueba.download();
    gridTAF.download();
    gridFibronectin.download();
    gridGradienteFibronectin.download();
    gridGradienteTAF.download();
    gridGradienteFibronectin.copyNanoToOpen();
    gridGradienteTAF.copyNanoToOpen();
    gridEndothelial.download();
    //std::cout<<"Copy pre"<<std::endl;
    gridEndothelial.copyNanoToOpen();
    //std::cout<<"Copy post"<<std::endl;
    gridEndothelial.writeToFile("../Grids/Endothelial.vdb"); 
    gridTAF.writeToFile("../Grids/TAF.vdb");
    gridFibronectin.writeToFile("../Grids/Fibronectin.vdb");
    gridGradienteTAF.writeToFile("../Grids/TAFGradient.vdb");
    gridGradienteFibronectin.writeToFile("../Grids/FibronectinGradient.vdb");

    return 0;

}