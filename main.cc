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
    Grid<> gridEndothelial(250,150,0.1);
    Grid<> gridTAF(250,150,0.1);
    Grid<> gridFibronectin(250,150,0.1);
    Grid<> gridMDE(250,150,0.1);
    Grid<> gridDivergenciaEndothelial(250,150,0.1);
    openvdb::Vec3s ini = {0.0,0.0,0.0};
    Grid<Vec3,nanovdb::Vec3f,Vec3Open,Vec3Open::Ptr,Vec3Nano> gridGradienteTAF(250,150,ini,false);
    Grid<Vec3,nanovdb::Vec3f,Vec3Open,Vec3Open::Ptr,Vec3Nano> gridGradienteFibronectin(250,150,ini,false);
    Grid<Vec3,nanovdb::Vec3f,Vec3Open,Vec3Open::Ptr,Vec3Nano> gridGradienteEndothelial(250,150,ini,false);

    /**
     * Creamos la piel con los datos pertinentes
     * 
     */
    // openvdb::FloatGrid::Ptr grid_open =
    //    openvdb::FloatGrid::create(/*background value=*/2.0);
    // openvdb::FloatGrid::Ptr grid_open_2 =
    //    openvdb::FloatGrid::create(/*background value=*/2.0);
    dataSkin dataIniEndothelial;
    dataIniEndothelial.valueBasale = 0.1;
    dataIniEndothelial.valueCorneum = 0.1;
    dataIniEndothelial.valueDermis = 0.1;
    dataIniEndothelial.valueHipoDermis = 0.1;
    dataIniEndothelial.valueSpinosum = 0.1;
    int size_lado = 250;
    int profundidad_total = 150;
    openvdb::Coord coordenadas;
    openvdb::FloatGrid::Accessor accessor_1_endothelial = gridEndothelial.getAccessorOpen1();
    openvdb::FloatGrid::Accessor accessor_2_endothelial = gridEndothelial.getAccessorOpen2();
    
    openvdb::FloatGrid::Accessor accessor_1_TAF = gridTAF.getAccessorOpen1();
    openvdb::FloatGrid::Accessor accessor_2_TAF = gridTAF.getAccessorOpen2();

    openvdb::FloatGrid::Accessor accessor_1_Fibronectin = gridFibronectin.getAccessorOpen1();
    openvdb::FloatGrid::Accessor accessor_2_Fibronectin = gridFibronectin.getAccessorOpen2();

    openvdb::FloatGrid::Accessor accessor_1_MDE = gridMDE.getAccessorOpen1();
    openvdb::FloatGrid::Accessor accessor_2_MDE = gridMDE.getAccessorOpen2();
    
    openvdb::FloatGrid::Accessor accessor_1_Divergencia= gridDivergenciaEndothelial.getAccessorOpen1();

    //gridEndothelial.fillRandom();
    createSkin(accessor_1_endothelial,size_lado,profundidad_total,coordenadas,dataIniEndothelial);
    createSkin(accessor_2_endothelial,size_lado,profundidad_total,coordenadas,dataIniEndothelial);
    
    createSkin(accessor_1_TAF,size_lado,profundidad_total,coordenadas,dataIniEndothelial);
    createSkin(accessor_2_TAF,size_lado,profundidad_total,coordenadas,dataIniEndothelial);

    createSkin(accessor_1_Fibronectin,size_lado,profundidad_total,coordenadas,dataIniEndothelial);
    createSkin(accessor_2_Fibronectin,size_lado,profundidad_total,coordenadas,dataIniEndothelial);

    createSkin(accessor_1_MDE,size_lado,profundidad_total,coordenadas,dataIniEndothelial);
    createSkin(accessor_2_MDE,size_lado,profundidad_total,coordenadas,dataIniEndothelial);

    createSkin(accessor_1_Divergencia,size_lado,profundidad_total,coordenadas,dataIniEndothelial);

    openvdb::Coord esquina_izquierda = openvdb::Coord();
    esquina_izquierda[0] = -size_lado / 2;
    esquina_izquierda[1] = -profundidad_total/2;
    esquina_izquierda[2] = -size_lado /2;
    gridGradienteFibronectin.fillRandom();
    gridGradienteTAF.fillRandom();
    gridGradienteEndothelial.fillRandom();
    int tamanio_tumor = 20;//Tamanio en voxels
   
    openvdb::FloatGrid::Accessor accessorFibronectin1 = gridFibronectin.getAccessorOpen1();
    openvdb::FloatGrid::Accessor accessorFibronectin2 = gridFibronectin.getAccessorOpen2();
    openvdb::FloatGrid::Accessor accessorTAF1 = gridTAF.getAccessorOpen1();
    openvdb::FloatGrid::Accessor accessorTAF2 = gridTAF.getAccessorOpen2();
    createRectangle(accessorTAF1,esquina_izquierda,tamanio_tumor,1);
    createRectangle(accessorTAF2,esquina_izquierda,tamanio_tumor,1);
    createRectangle(accessorFibronectin1,esquina_izquierda,tamanio_tumor,1);
    createRectangle(accessorFibronectin2,esquina_izquierda,tamanio_tumor,1);
    // esquina_izquierda[0]-=30;
    // createRectangle(accessor_1_endothelial,esquina_izquierda,tamanio_tumor,1);
    // createRectangle(accessor_2_endothelial,esquina_izquierda,tamanio_tumor,1);
    //createRectangle(accessor_1_MDE,esquina_izquierda,tamanio_tumor,10);
    //createRectangle(accessor_2_MDE,esquina_izquierda,tamanio_tumor,10);
    //gridMDE.fillRandom();
    
    
    
    //gridVectorPrueba.fillRandom();
    gridEndothelial.upload();
    gridTAF.upload();
    gridFibronectin.upload();
    gridGradienteTAF.upload();
    gridGradienteFibronectin.upload();
    gridMDE.upload();
    gridGradienteEndothelial.upload();
    gridDivergenciaEndothelial.upload();
    // int n;
    // std::cin>>n;
    //gridVectorPrueba.upload();
    
    int veces = 11;
    if(argc > 1){
        veces = atoi(argv[1]);
    }
    nanovdb::FloatGrid* gridRead;
    nanovdb::FloatGrid* gridWrite;
    nanovdb::FloatGrid* gridReadTAF;
    nanovdb::FloatGrid* gridWriteTAF;
    nanovdb::FloatGrid* gridReadMDE;
    nanovdb::FloatGrid* gridWriteMDE;
    nanovdb::FloatGrid* gridReadFibronectin;
    nanovdb::FloatGrid* gridWriteFibtronectin;
    nanovdb::FloatGrid* gridRead_CPU;
    nanovdb::FloatGrid* gridWrite_CPU;
    uint64_t nodeCount = gridGradienteFibronectin.getPtrNano1(typePointer::CPU)->tree().nodeCount(0);
    std::cout<<"NodeCount "<<nodeCount<<std::endl;
    generateEndothelial(gridEndothelial.getPtrNano1(typePointer::DEVICE),nodeCount,-39,-130,10);
    generateEndothelial(gridEndothelial.getPtrNano2(typePointer::DEVICE),nodeCount,-39,-130,10);
  
    //generateGradientFibronectin(gridFibronectin.getPtrNano1(typePointer::DEVICE),gridEndothelial.getPtrNano1(typePointer::DEVICE),gridGradienteFibronectin.getPtrNano2(typePointer::DEVICE),nodeCount);
    //generateGradientTAF(gridTAF.getPtrNano1(typePointer::DEVICE),gridEndothelial.getPtrNano1(typePointer::DEVICE),gridGradienteTAF.getPtrNano2(typePointer::DEVICE),nodeCount);
    

    // gridEndothelial.writeToFile("pre.vdb");
    auto  vector_medias = std::vector<float>();
    auto  vector_max = std::vector<float>();
    for(int i = 0 ;i<veces;i++){
        std::cout<<i<<std::endl;
        if(i % 2 == 0 ){
            gridRead = gridEndothelial.getPtrNano1(typePointer::DEVICE);
            gridWrite = gridEndothelial.getPtrNano2(typePointer::DEVICE);

            gridReadTAF = gridTAF.getPtrNano1(typePointer::DEVICE);
            gridWriteTAF = gridTAF.getPtrNano2(typePointer::DEVICE);

            gridReadFibronectin = gridFibronectin.getPtrNano1(typePointer::DEVICE);
            gridWriteFibtronectin = gridFibronectin.getPtrNano2(typePointer::DEVICE);

            gridReadMDE = gridMDE.getPtrNano1(typePointer::DEVICE);
            gridWriteMDE = gridMDE.getPtrNano2(typePointer::DEVICE);
            

        }else{
            gridRead = gridEndothelial.getPtrNano2(typePointer::DEVICE);
            gridWrite = gridEndothelial.getPtrNano1(typePointer::DEVICE);

            gridReadTAF = gridTAF.getPtrNano2(typePointer::DEVICE);
            gridWriteTAF = gridTAF.getPtrNano1(typePointer::DEVICE);

            gridReadFibronectin = gridFibronectin.getPtrNano2(typePointer::DEVICE);
            gridWriteFibtronectin = gridFibronectin.getPtrNano1(typePointer::DEVICE);
            
            gridReadMDE = gridMDE.getPtrNano2(typePointer::DEVICE);
            gridWriteMDE = gridMDE.getPtrNano1(typePointer::DEVICE);
        }
        equationMDE(gridRead,gridReadMDE,gridWriteMDE,nodeCount);
        equationFibronectin(gridRead,gridReadFibronectin,gridReadMDE,gridWriteFibtronectin,nodeCount);
        equationTAF(gridRead,gridReadTAF,gridWriteTAF,nodeCount);
        generateGradientFibronectin(gridReadFibronectin,gridRead,gridGradienteFibronectin.getPtrNano1(typePointer::DEVICE),nodeCount);
        generateGradientTAF(gridReadTAF,gridRead,gridGradienteTAF.getPtrNano1(typePointer::DEVICE),nodeCount);
        pruebaGradiente(gridGradienteEndothelial.getPtrNano1(typePointer::DEVICE),gridRead,nodeCount);
        equationEndothelial(gridRead,gridWrite,gridReadTAF,gridReadFibronectin,gridGradienteTAF.getPtrNano1(typePointer::DEVICE),gridGradienteFibronectin.getPtrNano1(typePointer::DEVICE),nodeCount);
        
        
        
        //gridFibronectin.download();
        
        //vector_medias.push_back(computeMean(gridFibronectin.getPtrNano1(typePointer::CPU)));
        //gridFibronectin.upload();
        // gridEndothelial.download();
        // vector_max.push_back(computeMax(gridEndothelial.getPtrNano1(typePointer::CPU)));
        // gridEndothelial.upload();

    }
    
    divergence(gridGradienteEndothelial.getPtrNano1(typePointer::DEVICE),gridDivergenciaEndothelial.getPtrNano1(typePointer::DEVICE),nodeCount);

    // }
    //pruebaGradiente(gridVectorPrueba.getPtrNanoWrite(typePointer::DEVICE),gridEndothelial.getPtrNanoRead(typePointer::DEVICE),gridEndothelial.getPtrNanoRead(typePointer::CPU)->tree().nodeCount(0));
    writeVector(vector_medias,"mediasTAF.txt");
    //writeVector(vector_max,"maxEndothelial.txt");
    //gridVectorPrueba.download();
    gridTAF.download();
    gridFibronectin.download();
    gridGradienteFibronectin.download();
    gridGradienteTAF.download();
    gridGradienteEndothelial.download();
    gridEndothelial.download();
    gridMDE.download();
    gridDivergenciaEndothelial.download();
    gridGradienteEndothelial.copyNanoToOpen();
    gridGradienteFibronectin.copyNanoToOpen();
    gridGradienteTAF.copyNanoToOpen();
    gridMDE.copyNanoToOpen();
    gridDivergenciaEndothelial.copyNanoToOpen();
    //std::cout<<"Copy pre"<<std::endl;
    gridEndothelial.copyNanoToOpen();
    gridFibronectin.copyNanoToOpen();
    //std::cout<<"Copy post"<<std::endl;
    gridEndothelial.writeToFile("../Grids/Endothelial.vdb"); 
    gridTAF.writeToFile("../Grids/TAF.vdb");
    gridFibronectin.writeToFile("../Grids/Fibronectin.vdb");
    gridGradienteTAF.writeToFile("../Grids/TAFGradient.vdb");
    gridGradienteFibronectin.writeToFile("../Grids/FibronectinGradient.vdb");
    gridMDE.writeToFile("../Grids/MDE.vdb");
    gridGradienteEndothelial.writeToFile("../Grids/EndothelialGradient.vdb");
    gridDivergenciaEndothelial.writeToFile("../Grids/EndothelialDivergence.vdb");

    return 0;

}