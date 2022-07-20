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
#include <cctype>
#include <random>
#include <openvdb/tools/PoissonSolver.h>
#include <cmath>

float computeMeam(nanovdb::FloatGrid * grid){
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
float computeMaxAbs(nanovdb::FloatGrid * grid){
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
                    
                    if(std::abs(aux) > std::abs(max)){
                        max = aux;
                    }
                }
            }
    }
    return max;
}


float computeMin(nanovdb::FloatGrid * grid){
    float min = 100000.0;
    int size_lado = 250;
    int profundidad_total = 150;
    auto  accessor = grid->getAccessor();
    nanovdb::Coord coord;
    
    for(int i  =0;i>-size_lado;i--){
            for(int j = 0 ;j>-profundidad_total;j--){
                for(int k = 0 ;k>-size_lado;k--){
                    coord = openvdb::Coord(i,j,k);
                    float aux = accessor.getValue(coord);
                    if(aux < min){
                        min = aux;
                    }
                }
            }
    }
    return min;
}

void writeVector(std::vector<float>& vec,std::string fileName){
    std::ofstream myfile;
    myfile.open (fileName);
    for(auto data : vec){
        myfile<<data<<std::endl;
    }
    myfile.close();
}
float valor_ini =  0;
int main(int argc,char * argv[]){
    std::cout<<" CORREGIDOS READ/WRITE "<<std::endl;
    std::cout<<" DISCRETE CAMBIA CONTINUE, INCIAL CONTINUE TODO A 0"<<std::endl;
    /*
        ESTA EN EL PRODUCT NEGADA LA N
    */
    using Vec3 = openvdb::Vec3s;
    using  Vec3Open = openvdb::Vec3SGrid ;
    using Vec3Nano = nanovdb::Vec3fGrid;
    //valor_ini = 0.1;
    Grid<> gridEndothelial(250,150,valor_ini);//Este es para calcular la derivada y ver hacia donde va la migracion
    valor_ini = 0.0 ;
    Grid<> gridTAF(250,150,valor_ini);
    valor_ini = 0;
    Grid<> gridFibronectin(250,150,valor_ini);
    Grid<> gridMDE(250,150,valor_ini);
    Grid<> gridDivergenciaTAF(250,150,valor_ini);
    Grid<> gridDivergenciaFibronectin(250,150,valor_ini);
    Grid<> gridTAFEndothelial(250,150,valor_ini);//Tendra el producto de c * n para calcular su gradiente
    Grid<> gridBplus(250,150,valor_ini,false);
    Grid<> gridBMinus(250,150,valor_ini,false);
    Grid<> gridTumorCells(250,150,valor_ini);
    Grid<> gridOxygen(250,150,valor_ini,false);
    Grid<> gridPressure(250,150,valor_ini,false);
    Grid<> gridPressureLaplacian(250,150,valor_ini,false);
    Grid<> gridDeadCells(250,150,valor_ini,false);
    
    openvdb::Vec3s ini = {0.0,0.0,0.0};
    Grid<Vec3,nanovdb::Vec3f,Vec3Open,Vec3Open::Ptr,Vec3Nano> gridGradienteTAF(250,150,ini,false);
    Grid<Vec3,nanovdb::Vec3f,Vec3Open,Vec3Open::Ptr,Vec3Nano> gridGradienteFibronectin(250,150,ini,false);
    Grid<Vec3,nanovdb::Vec3f,Vec3Open,Vec3Open::Ptr,Vec3Nano> gridGradienteEndothelial(250,150,ini,false);
    Grid<Vec3,nanovdb::Vec3f,Vec3Open,Vec3Open::Ptr,Vec3Nano> gridTummorFlux(250,150,ini,false);
    //Grid<bool,bool,openvdb::BoolGrid,openvdb::BoolGrid::Ptr,nanovdb::BoolGrid> gridTipEndothelial(250,150,false,false);
    Grid<> gridTipEndothelial(250,150,0.0);
    Grid<> gridEndothelialDiscrete(250,150,0.0);//este almacenara la vasculatura como tal, de forma discreta
    /**
     * Creamos la piel con los datos pertinentes
     * 
     */
    // openvdb::FloatGrid::Ptr grid_open =
    //    openvdb::FloatGrid::create(/*background value=*/2.0);
    // openvdb::FloatGrid::Ptr grid_open_2 =
    //    openvdb::FloatGrid::create(/*background value=*/2.0);
    dataSkin dataIniEndothelial;
    dataIniEndothelial.valueBasale = valor_ini;
    dataIniEndothelial.valueCorneum = valor_ini;
    dataIniEndothelial.valueDermis = valor_ini;
    dataIniEndothelial.valueHipoDermis = valor_ini;
    dataIniEndothelial.valueSpinosum = valor_ini;
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
    
    openvdb::FloatGrid::Accessor accessor_1_Divergencia= gridDivergenciaTAF.getAccessorOpen1();
    openvdb::FloatGrid::Accessor accessor_1_Tip = gridTipEndothelial.getAccessorOpen1();
    openvdb::FloatGrid::Accessor accessor_2_Tip = gridTipEndothelial.getAccessorOpen2();

    openvdb::FloatGrid::Accessor accessor_1_Endothelial_discrete = gridEndothelialDiscrete.getAccessorOpen1();
    openvdb::FloatGrid::Accessor accessor_2_Endothelial_discrete = gridEndothelialDiscrete.getAccessorOpen2();
     

    //gridEndothelial.fillRandom();
    createSkin(accessor_1_Endothelial_discrete,size_lado,profundidad_total,coordenadas,dataIniEndothelial);
    createSkin(accessor_2_Endothelial_discrete,size_lado,profundidad_total,coordenadas,dataIniEndothelial);
    
    createSkin(accessor_1_Fibronectin,size_lado,profundidad_total,coordenadas,dataIniEndothelial);
    createSkin(accessor_2_Fibronectin,size_lado,profundidad_total,coordenadas,dataIniEndothelial);

    createSkin(accessor_1_MDE,size_lado,profundidad_total,coordenadas,dataIniEndothelial);
    createSkin(accessor_2_MDE,size_lado,profundidad_total,coordenadas,dataIniEndothelial);

    createSkin(accessor_1_Divergencia,size_lado,profundidad_total,coordenadas,dataIniEndothelial);
    createSkin(accessor_1_Tip,size_lado,profundidad_total,coordenadas,dataIniEndothelial);
    createSkin(accessor_2_Tip,size_lado,profundidad_total,coordenadas,dataIniEndothelial);

    //valor_ini = 0.8;
    dataIniEndothelial.valueBasale = valor_ini;
    dataIniEndothelial.valueCorneum = valor_ini;
    dataIniEndothelial.valueDermis = valor_ini;
    dataIniEndothelial.valueHipoDermis = valor_ini;
    dataIniEndothelial.valueSpinosum = valor_ini;   
    createSkin(accessor_1_TAF,size_lado,profundidad_total,coordenadas,dataIniEndothelial);
    createSkin(accessor_2_TAF,size_lado,profundidad_total,coordenadas,dataIniEndothelial);
    //gridTAF.fillRandom();

    //valor_ini = 0.0000001;
    valor_ini = 0.1;
    dataIniEndothelial.valueBasale = valor_ini;
    dataIniEndothelial.valueCorneum = valor_ini;
    dataIniEndothelial.valueDermis = valor_ini;
    dataIniEndothelial.valueHipoDermis = valor_ini;
    dataIniEndothelial.valueSpinosum = valor_ini;

    createSkin(accessor_1_endothelial,size_lado,profundidad_total,coordenadas,dataIniEndothelial);
    createSkin(accessor_2_endothelial,size_lado,profundidad_total,coordenadas,dataIniEndothelial);
    
    openvdb::Coord esquina_izquierda = openvdb::Coord();
    esquina_izquierda[0] = -size_lado / 2;
    esquina_izquierda[1] = -profundidad_total/2-20;
    esquina_izquierda[2] = -size_lado /2;
    gridGradienteFibronectin.fillRandom();
    gridGradienteTAF.fillRandom();
    gridGradienteEndothelial.fillRandom();
    gridDivergenciaFibronectin.fillRandom();
    gridTAFEndothelial.fillRandom();

    gridDivergenciaTAF.fillValue(0.0);
    gridBplus.fillValue(0.0);
    gridBMinus.fillValue(0.0);
    gridTumorCells.fillValue(0);
    gridOxygen.fillValue(0);
    gridPressure.fillValue(0);
    gridTummorFlux.fillValue(ini);
    gridDeadCells.fillValue(0.0);
    gridPressureLaplacian.fillValue(0.0);


    gridEndothelialDiscrete.fillValue(0.0);
    int tamanio_tumor = 30;//Tamanio en voxels
    //tamanio_tumor = 1;
    //esquina_izquierda[1]-=20;
    openvdb::FloatGrid::Accessor accessorFibronectin1 = gridFibronectin.getAccessorOpen1();
    openvdb::FloatGrid::Accessor accessorFibronectin2 = gridFibronectin.getAccessorOpen2();
    openvdb::FloatGrid::Accessor accessorTAF1 = gridTAF.getAccessorOpen1();
    openvdb::FloatGrid::Accessor accessorTAF2 = gridTAF.getAccessorOpen2();
    openvdb::FloatGrid::Accessor accessorTummor1 = gridTumorCells.getAccessorOpen1();
    openvdb::FloatGrid::Accessor accessorTummor2 = gridTumorCells.getAccessorOpen2();
    openvdb::FloatGrid::Accessor accessorPressure1 = gridPressure.getAccessorOpen1();
    //openvdb::FloatGrid::Accessor accessorPressure2 = gridPressure.getAccessorOpen2();
    
    tamanio_tumor = 1;
    createRectangle(accessorTummor1,esquina_izquierda,tamanio_tumor,1.0);
    createRectangle(accessorTummor2,esquina_izquierda,tamanio_tumor,1.0);
    //gridTumorCells.interpolate();
    tamanio_tumor = 1;
    //createRectangle(accessorPressure1,esquina_izquierda,tamanio_tumor,10.0);
    //createRectangle(accessorPressure2,esquina_izquierda,tamanio_tumor,1.0);
    
    //createRectangle(accessor_1_endothelial,esquina_izquierda,tamanio_tumor,1.0);
    //createRectangle(accessor_2_endothelial,esquina_izquierda,tamanio_tumor,1.0);
    //createRectangle(accessorFibronectin1,esquina_izquierda,tamanio_tumor,10);
    //createRectangle(accessorFibronectin2,esquina_izquierda,tamanio_tumor,10);
    esquina_izquierda[0]-=10;
    esquina_izquierda[1]+=30;
    //createRectangle(accessor_1_Endothelial_discrete,esquina_izquierda,tamanio_tumor,1.0);
    //createRectangle(accessor_2_Endothelial_discrete,esquina_izquierda,tamanio_tumor,1.0);

    //createRectangle(accessor_1_endothelial,esquina_izquierda,tamanio_tumor,1);
    //createRectangle(accessor_2_endothelial,esquina_izquierda,tamanio_tumor,1);
    //createRectangle(accessor_1_endothelial,esquina_izquierda,1,1);
    //createRectangle(accessor_2_endothelial,esquina_izquierda,tamanio_tumor,1);
    
    //esquina_izquierda[2] += 10;
    tamanio_tumor = 1;
    createRectangle(accessor_1_Tip,esquina_izquierda,tamanio_tumor,1);
    createRectangle(accessor_2_Tip,esquina_izquierda,tamanio_tumor,1);
    //createRectangle(accessor_1_endothelial,esquina_izquierda,tamanio_tumor,1);
    //createRectangle(accessor_2_endothelial,esquina_izquierda,tamanio_tumor,1);


    

    // esquina_izquierda[0]+=50;
    // esquina_izquierda[1]+=2;
    // tamanio_tumor = 40;
    // // createRectangle(accessor_1_Tip,esquina_izquierda,tamanio_tumor,1);
    // // createRectangle(accessor_2_Tip,esquina_izquierda,tamanio_tumor,1);
    // createRectangle(accessor_1_endothelial,esquina_izquierda,tamanio_tumor,1);
    // createRectangle(accessor_2_endothelial,esquina_izquierda,tamanio_tumor,1);


    
    // createRectangle(accessor_1_Endothelial_discrete,esquina_izquierda,tamanio_tumor,1);
    // createRectangle(accessor_2_Endothelial_discrete,esquina_izquierda,tamanio_tumor,1);
    // esquina_izquierda[0]-=25;
    // esquina_izquierda[1]+=20;
    // createRectangle(accessor_1_Tip,esquina_izquierda,tamanio_tumor,1);
    // createRectangle(accessor_2_Tip,esquina_izquierda,tamanio_tumor,1);
    // //createRectangle(accessor_1_endothelial,esquina_izquierda,tamanio_tumor,1);
    // //createRectangle(accessor_2_endothelial,esquina_izquierda,tamanio_tumor,1);


    
    // createRectangle(accessor_1_Endothelial_discrete,esquina_izquierda,tamanio_tumor,1);
    // createRectangle(accessor_2_Endothelial_discrete,esquina_izquierda,tamanio_tumor,1);

    // esquina_izquierda[1]-=40;
    // createRectangle(accessor_1_Tip,esquina_izquierda,tamanio_tumor,1);
    // createRectangle(accessor_2_Tip,esquina_izquierda,tamanio_tumor,1);
    //createRectangle(accessor_1_endothelial,esquina_izquierda,tamanio_tumor,1);
    //createRectangle(accessor_2_endothelial,esquina_izquierda,tamanio_tumor,1);


    tamanio_tumor = 10;
    createRectangle(accessor_1_Endothelial_discrete,esquina_izquierda,tamanio_tumor,1);
    createRectangle(accessor_2_Endothelial_discrete,esquina_izquierda,tamanio_tumor,1);
    esquina_izquierda[0]+= 30;
    createRectangle(accessor_1_Endothelial_discrete,esquina_izquierda,tamanio_tumor,1);
    createRectangle(accessor_2_Endothelial_discrete,esquina_izquierda,tamanio_tumor,1);
    //createColumns(accessor_1_Endothelial_discrete,size_lado,profundidad_total);
    //generateEndoThelial(accessor_1_Endothelial_discrete,size_lado,profundidad_total,-130,-39,5);
    //createRectangle(accessor_2_endothelial,esquina_izquierda,tamanio_tumor,1);
    //createRectangle(accessor_1_MDE,esquina_izquierda,tamanio_tumor,10);
    //createRectangle(accessor_2_MDE,esquina_izquierda,tamanio_tumor,10);
    //gridMDE.fillRandom();

    int veces = 11;
    if(argc > 1){
        veces = atoi(argv[1]);
    }
    // for(int i = 0 ;i<veces;i++){
    //     std::cout<<"Previo a solve"<<std::endl;
    //     auto state = openvdb::math::pcg::terminationDefaults<float>();
    //     auto tree = gridTAF.getPtrOpen1()->tree();
    //     auto newTree2 = openvdb::tools::poisson::solve<openvdb::v9_0::FloatTree>(tree,state);
    //     gridTAF.getPtrOpen1()->setTree(newTree2->copy());

    //     tamanio_tumor = 10;
    //     accessorTAF1 = gridTAF.getAccessorOpen1();
    //     //createRectangle(accessorTAF1,esquina_izquierda,tamanio_tumor,0.0);

    //     state = openvdb::math::pcg::terminationDefaults<float>();
    //     auto tree2 = gridTAF.getPtrOpen1()->tree();
    //     newTree2 = openvdb::tools::poisson::solve<openvdb::v9_0::FloatTree>(tree2,state);
    //     gridTAF.getPtrOpen2()->setTree(newTree2->copy());
    //     std::cout<<"Post solve"<<std::endl;
    // }
    
    
    
    //gridEndothelial.fillRandom();
    //gridVectorPrueba.fillRandom();
    gridEndothelial.upload();
    // gridTAF.upload();
    // gridFibronectin.upload();
    // gridGradienteTAF.upload();
    // gridGradienteFibronectin.upload();
    // gridMDE.upload();
    gridGradienteEndothelial.upload();
    gridDivergenciaTAF.upload();
    // //gridDivergenciaFibronectin.upload();
    // gridTAFEndothelial.upload();
    // gridTipEndothelial.upload();
    gridEndothelialDiscrete.upload();

    gridBplus.upload();
    gridBMinus.upload();
    gridTumorCells.upload();
    gridOxygen.upload();
    gridPressure.upload();
    gridDeadCells.upload();
    gridTummorFlux.upload();
    gridPressureLaplacian.upload();
    // int n;
    // std::cin>>n;
    //gridVectorPrueba.upload();
    
    
    nanovdb::FloatGrid* gridRead;
    nanovdb::FloatGrid* gridWrite;
    nanovdb::FloatGrid* gridReadTAF;
    nanovdb::FloatGrid* gridWriteTAF;
    nanovdb::FloatGrid* gridReadTAF_CPU;
    nanovdb::FloatGrid* gridWriteTAF_CPU;
    nanovdb::FloatGrid* gridReadMDE;
    nanovdb::FloatGrid* gridWriteMDE;
    nanovdb::FloatGrid* gridReadFibronectin;
    nanovdb::FloatGrid* gridWriteFibtronectin;
    nanovdb::FloatGrid* gridRead_CPU;
    nanovdb::FloatGrid* gridWrite_CPU;
    nanovdb::FloatGrid* gridTip_Read;
    nanovdb::FloatGrid* gridTip_Write;
    nanovdb::FloatGrid* endothelialContinueRead;
    nanovdb::FloatGrid* endothelialContinueWrite;
    nanovdb::FloatGrid* tummorCellsRead;
    nanovdb::FloatGrid* tummorCellsWrite;
    
    //int n = 1000;

    // auto var = gridWrite_CPU->tree().getFirstNode<0>() + (n >> 9);
    // var->CoordToOffset
    uint64_t nodeCount = gridEndothelial.getPtrNano1(typePointer::CPU)->tree().nodeCount(0);
    std::cout<<"NodeCount "<<nodeCount<<std::endl;
    

    generateEndothelial(gridEndothelialDiscrete.getPtrNano1(typePointer::DEVICE),nodeCount,-39,-130,5);
    generateEndothelial(gridEndothelialDiscrete.getPtrNano2(typePointer::DEVICE),nodeCount,-39,-130,5);
    
    // generateGradientFibronectin(gridFibronectin.getPtrNano1(typePointer::DEVICE),gridEndothelial.getPtrNano1(typePointer::DEVICE),gridGradienteFibronectin.getPtrNano2(typePointer::DEVICE),nodeCount);
    // generateGradientTAF(gridTAF.getPtrNano1(typePointer::DEVICE),gridEndothelial.getPtrNano1(typePointer::DEVICE),gridGradienteTAF.getPtrNano2(typePointer::DEVICE),nodeCount);
    
    // for(int i = 0 ;i<500;i++){
    //     //std::cout<<i<<std::endl;
    //     if(i%2 == 0 ){
    //         gridReadTAF = gridTAF.getPtrNano1(typePointer::DEVICE);
    //         gridWriteTAF = gridTAF.getPtrNano2(typePointer::DEVICE);
    //     }else{
    //         gridReadTAF = gridTAF.getPtrNano2(typePointer::DEVICE);
    //         gridWriteTAF = gridTAF.getPtrNano1(typePointer::DEVICE);
    //     }
    //     laplacian(gridReadTAF,gridWriteTAF,nodeCount);
    // }
    std::cout<<"Laplaciano calculado"<<std::endl;

    // gridEndothelial.writeToFile("pre.vdb");
    auto  vector_medias_Endothelial = std::vector<float>();
    auto  vector_max = std::vector<float>();
    auto vector_medias_TAF = std::vector<float>();
    auto vector_medias_Fibronectin = std::vector<float>();
    auto vector_medias_MDE = std::vector<float>();
    auto vector_medias_diver_TAF = std::vector<float>();
    auto vector_medias_diver_Fibronectin = std::vector<float>();

    using u32    = uint_least32_t; 
    using engine = std::mt19937;
    std::random_device os_seed;
    u32 seed = os_seed();

    engine generator( seed );
    std::uniform_int_distribution< u32 > distribute( 1, nodeCount);
    
    openvdb::v9_0::FloatTree::Ptr newTree;
    int laplacianos = 0;
    for(int i = 0 ;i<veces;i++){
        std::cout<<i<<std::endl;
        bool condition = i%30 == 0;
        
        //condition = false;
        float prevMax=1.0;
        if(condition){
            laplacianos++;
            float prevMax=1;
            //if(i>0){
                gridEndothelialDiscrete.download();
                gridOxygen.download();
                gridOxygen.copyNanoToOpen();
                gridEndothelialDiscrete.copyNanoToOpen();
            //}
            prevMax = computeMax(gridOxygen.getPtrNano1(typePointer::CPU));

            auto state = openvdb::math::pcg::terminationDefaults<float>();prevMax = computeMax(gridReadTAF_CPU);
            
            if(i%2==0){
                auto tree = gridEndothelialDiscrete.getPtrOpen1()->tree();
                newTree = openvdb::tools::poisson::solve<openvdb::v9_0::FloatTree>(tree,state);
                gridOxygen.getPtrOpen1()->setTree(newTree->copy());

            }else{
                auto tree = gridEndothelialDiscrete.getPtrOpen2()->tree();
                newTree = openvdb::tools::poisson::solve<openvdb::v9_0::FloatTree>(tree,state);
                gridOxygen.getPtrOpen2()->setTree(newTree->copy());

            }
            
            gridOxygen.upload();
            gridEndothelialDiscrete.upload();
            
            
            std::cout<<"Poisson out"<<std::endl;
        }
        if(i % 2 == 0 ){
            gridRead = gridEndothelialDiscrete.getPtrNano1(typePointer::DEVICE);
            gridWrite = gridEndothelialDiscrete.getPtrNano2(typePointer::DEVICE);

            gridReadTAF = gridTAF.getPtrNano1(typePointer::DEVICE);
            gridWriteTAF = gridTAF.getPtrNano2(typePointer::DEVICE);

            gridReadTAF_CPU = gridTAF.getPtrNano1(typePointer::CPU);
            gridWriteTAF_CPU = gridTAF.getPtrNano2(typePointer::CPU);

            gridReadFibronectin = gridFibronectin.getPtrNano1(typePointer::DEVICE);
            gridWriteFibtronectin = gridFibronectin.getPtrNano2(typePointer::DEVICE);

            gridReadMDE = gridMDE.getPtrNano1(typePointer::DEVICE);
            gridWriteMDE = gridMDE.getPtrNano2(typePointer::DEVICE);

            gridTip_Read = gridTipEndothelial.getPtrNano1(typePointer::DEVICE);
            gridTip_Write = gridTipEndothelial.getPtrNano2(typePointer::DEVICE);

            endothelialContinueRead = gridEndothelial.getPtrNano1(typePointer::DEVICE);
            endothelialContinueWrite = gridEndothelial.getPtrNano2(typePointer::DEVICE);

            tummorCellsRead = gridTumorCells.getPtrNano1(typePointer::DEVICE);
            tummorCellsWrite = gridTumorCells.getPtrNano2(typePointer::DEVICE);
            

        }else{
            gridRead = gridEndothelialDiscrete.getPtrNano2(typePointer::DEVICE);
            gridWrite = gridEndothelialDiscrete.getPtrNano1(typePointer::DEVICE);

            gridReadTAF = gridTAF.getPtrNano2(typePointer::DEVICE);
            gridWriteTAF = gridTAF.getPtrNano1(typePointer::DEVICE);

            gridReadTAF_CPU = gridTAF.getPtrNano2(typePointer::CPU);
            gridWriteTAF_CPU = gridTAF.getPtrNano1(typePointer::CPU);

            gridReadFibronectin = gridFibronectin.getPtrNano2(typePointer::DEVICE);
            gridWriteFibtronectin = gridFibronectin.getPtrNano1(typePointer::DEVICE);
            
            gridReadMDE = gridMDE.getPtrNano2(typePointer::DEVICE);
            gridWriteMDE = gridMDE.getPtrNano1(typePointer::DEVICE);

            gridTip_Read = gridTipEndothelial.getPtrNano2(typePointer::DEVICE);
            gridTip_Write = gridTipEndothelial.getPtrNano1(typePointer::DEVICE);

            endothelialContinueRead = gridEndothelial.getPtrNano2(typePointer::DEVICE);
            endothelialContinueWrite = gridEndothelial.getPtrNano1(typePointer::DEVICE);

            tummorCellsRead = gridTumorCells.getPtrNano2(typePointer::DEVICE);
            tummorCellsWrite = gridTumorCells.getPtrNano1(typePointer::DEVICE);

        }
        
        if(condition){
            //float maxAbs = computeMaxAbs(gridReadTAF_CPU);
            float max = computeMax(gridOxygen.getPtrNano1(typePointer::CPU));
            float min = computeMin(gridOxygen.getPtrNano1(typePointer::CPU));
            float addition = max - min;
            float newMax = max + addition;
            // if(maxAbs < 0 ){
            //     absolute(gridReadTAF,nodeCount);
            //     maxAbs = -maxAbs;
            // }
            addMax(gridOxygen.getPtrNano1(typePointer::DEVICE),addition,nodeCount);
            normalize(gridOxygen.getPtrNano1(typePointer::DEVICE),newMax,prevMax,nodeCount);
        }
        
        // if(i == 0 ){
        //     equationEndothelial(endothelialContinueRead,endothelialContinueRead,gridReadTAF,gridReadFibronectin,gridGradienteTAF.getPtrNano1(typePointer::DEVICE),gridGradienteFibronectin.getPtrNano1(typePointer::DEVICE),gridTip_Read,nodeCount);
        //     cleanEndothelial(endothelialContinueRead,nodeCount);
        // }
        auto gridNi = gridRead;
        // //std::cout<<"First endothelial"<<std::endl;
        // equationMDE(gridNi,gridReadMDE,gridWriteMDE,nodeCount);
        // equationFibronectin(gridNi,gridReadFibronectin,gridWriteMDE,gridWriteFibtronectin,nodeCount);
        // equationTAF(gridNi,gridReadTAF,gridWriteTAF,nodeCount);
        // product(gridReadTAF,endothelialContinueRead,gridTAFEndothelial.getPtrNano1(typePointer::DEVICE),nodeCount);

        // generateGradientFibronectin(gridReadFibronectin,gridRead,gridGradienteFibronectin.getPtrNano1(typePointer::DEVICE),nodeCount);
        // generateGradientTAF(gridReadTAF,gridTAFEndothelial.getPtrNano1(typePointer::DEVICE),gridGradienteTAF.getPtrNano1(typePointer::DEVICE),nodeCount);
        // equationEndothelial(endothelialContinueRead,endothelialContinueWrite,gridWriteTAF,gridWriteFibtronectin ,gridGradienteTAF.getPtrNano1(typePointer::DEVICE),gridGradienteFibronectin.getPtrNano1(typePointer::DEVICE),gridTip_Read,nodeCount);
        
        // equationEndothelialDiscrete(gridRead,gridWrite,endothelialContinueRead,endothelialContinueWrite,gridWriteTAF,gridTip_Read,gridTip_Write,distribute(generator),nodeCount);
        // seed = os_seed();

        // engine generator( seed );
        // branching(gridTip_Write,gridReadTAF,seed,nodeCount);
        
        // cleanEndothelial(endothelialContinueWrite,nodeCount);
        // // //regenerateEndothelial(endothelialContinueWrite,gridRead,nodeCount);
        // // ////generateEndothelial(gridEndothelial.getPtrNano1(typePointer::DEVICE),nodeCount,-39,-130,5);
        // // //generateEndothelial(gridEndothelial.getPtrNano2(typePointer::DEVICE),nodeCount,-39,-130,5);
        
        // divergence(gridGradienteTAF.getPtrNano1(typePointer::DEVICE),gridDivergenciaTAF.getPtrNano1(typePointer::DEVICE),nodeCount);
        // //divergence(gridGradienteFibronectin.getPtrNano1(typePointer::DEVICE),gridDivergenciaFibronectin.getPtrNano1(typePointer::DEVICE),nodeCount);
        
        equationBplusSimple(tummorCellsRead,gridBplus.getPtrNano1(typePointer::DEVICE),gridOxygen.getPtrNano1(typePointer::DEVICE),nodeCount);
        // // equationBminusSimple(gridTumorCells.getPtrNano1(typePointer::DEVICE),gridBMinus.getPtrNano1(typePointer::DEVICE),gridOxygen.getPtrNano1(typePointer::DEVICE),nodeCount);
        equationPressure(tummorCellsRead,gridPressure.getPtrNano1(typePointer::DEVICE),nodeCount);
        // //laplacian(gridPressure.getPtrNano1(typePointer::DEVICE),gridPressureLaplacian.getPtrNano1(typePointer::DEVICE),nodeCount);
        // // for(int j = 0 ;j<10;j++){
        // //      laplacian(gridPressureLaplacian.getPtrNano1(typePointer::DEVICE),gridPressureLaplacian.getPtrNano1(typePointer::DEVICE),nodeCount);

        // // }
        equationFluxSimple(gridPressure.getPtrNano1(typePointer::DEVICE),tummorCellsRead,gridTummorFlux.getPtrNano1(typePointer::DEVICE),nodeCount);
        // pruebaGradiente(gridGradienteEndothelial.getPtrNano1(typePointer::DEVICE),gridPressure.getPtrNano1(typePointer::DEVICE),nodeCount);
        // //divergence(gridGradienteEndothelial.getPtrNano1(typePointer::DEVICE),gridDivergenciaTAF.getPtrNano1(typePointer::DEVICE),nodeCount);
        // divergence(gridTummorFlux.getPtrNano1(typePointer::DEVICE),gridDivergenciaTAF.getPtrNano1(typePointer::DEVICE),nodeCount);

        equationTumorSimple(gridTummorFlux.getPtrNano1(typePointer::DEVICE),gridBplus.getPtrNano1(typePointer::DEVICE),gridBMinus.getPtrNano1(typePointer::DEVICE),tummorCellsRead,tummorCellsWrite,nodeCount);
        for(int j = 0 ;j <1 ;j++){
            average(tummorCellsWrite,gridBplus.getPtrNano1(typePointer::DEVICE),nodeCount);
            copy(gridBplus.getPtrNano1(typePointer::DEVICE),tummorCellsWrite,nodeCount);
        }
        
        //gridTumorCells.interpolate();
        // if(i == veces-1){
        //     discretize(gridTumorCells.getPtrNano1(typePointer::DEVICE),nodeCount);

        // }
        //gridFibronectin.download();
        // gridTAF.download();
        // gridEndothelial.download();
        // vector_medias_Endothelial.push_back(computeMax(gridEndothelial.getPtrNano1(typePointer::CPU)));
        // gridEndothelial.upload();
        // gridTumorCells.download();
        // vector_max.push_back(computeMax(gridTumorCells.getPtrNano1(typePointer::CPU)));
        // gridTumorCells.upload();
        // gridMDE.download();
        // vector_medias_MDE.push_back(computeMax(gridMDE.getPtrNano1(typePointer::CPU)));
        // gridMDE.upload();

        // gridFibronectin.download();
        // vector_medias_Fibronectin.push_back(computeMax(gridFibronectin.getPtrNano1(typePointer::CPU)));
        // gridFibronectin.upload();

        // gridTAF.download();
        // vector_medias_TAF.push_back(computeMax(gridTAF.getPtrNano1(typePointer::CPU)));
        // gridTAF.upload();

        // gridDivergenciaFibronectin.download();
        // vector_medias_diver_Fibronectin.push_back(computeMax(gridDivergenciaFibronectin.getPtrNano1(typePointer::CPU)));
        // gridDivergenciaFibronectin.upload();

        // gridDivergenciaTAF.download();
        // vector_medias_diver_TAF.push_back(computeMax(gridDivergenciaTAF.getPtrNano1(typePointer::CPU)));
        // gridDivergenciaTAF.upload();

        //vector_medias.push_back(computeMax(gridTAF.getPtrNano1(typePointer::CPU)));
        //gridTAF.upload();
        //vector_medias.push_back(computeMax(gridFibronectin.getPtrNano1(typePointer::CPU)));
        //gridFibronectin.upload();
        // gridEndothelial.download();
        // vector_max.push_back(computeMax(gridEndothelial.getPtrNano1(typePointer::CPU)));
        // gridEndothelial.upload();

    }
    
    

    // }
    //pruebaGradiente(gridVectorPrueba.getPtrNanoWrite(typePointer::DEVICE),gridEndothelial.getPtrNanoRead(typePointer::DEVICE),gridEndothelial.getPtrNanoRead(typePointer::CPU)->tree().nodeCount(0));
    writeVector(vector_medias_Endothelial,"mediasEndothelial.txt");
    writeVector(vector_medias_MDE,"mediasMDE.txt");
    writeVector(vector_medias_Fibronectin,"mediasFibronectin.txt");
    writeVector(vector_medias_TAF,"mediasTAF.txt");
    writeVector(vector_medias_diver_TAF,"mediasDiverTAF.txt");
    writeVector(vector_medias_diver_Fibronectin,"mediasDiverFibronectin.txt");
    
    //gridVectorPrueba.download();
    std::cout<<"Hola"<<std::endl;
    gridTAF.download();
    gridFibronectin.download();
    gridGradienteFibronectin.download();
    gridGradienteTAF.download();
    gridGradienteEndothelial.download();
    gridEndothelial.download();
    gridMDE.download();
    gridDivergenciaTAF.download();
    //gridDivergenciaFibronectin.download();
    gridTAFEndothelial.download();
    gridTipEndothelial.download();
    gridEndothelialDiscrete.download();
    gridBplus.download();
    gridBMinus.download();
    gridTumorCells.download();
    gridOxygen.download();
    gridPressure.download();
    gridDeadCells.download();
    gridTummorFlux.download();
    gridPressureLaplacian.download();
    // gridEndothelialDiscrete.copyNanoToOpen();
    // gridTipEndothelial.copyNanoToOpen();
    // gridTAF.copyNanoToOpen();
    // std::cout<<"Hola2"<<std::endl;
    gridGradienteEndothelial.copyNanoToOpen();
    // gridGradienteFibronectin.copyNanoToOpen();
    // gridGradienteTAF.copyNanoToOpen();
    // gridMDE.copyNanoToOpen();
    gridDivergenciaTAF.copyNanoToOpen();
    // //gridDivergenciaFibronectin.copyNanoToOpen();
    // //std::cout<<"Copy pre"<<std::endl;
    gridEndothelial.copyNanoToOpen();
    // gridFibronectin.copyNanoToOpen();
    // gridTAFEndothelial.copyNanoToOpen();

    gridBplus.copyNanoToOpen();
    gridBMinus.copyNanoToOpen();
    gridTumorCells.copyNanoToOpen();
    gridOxygen.copyNanoToOpen();
    gridPressure.copyNanoToOpen();
    gridDeadCells.copyNanoToOpen();
    gridTummorFlux.copyNanoToOpen();
    gridPressureLaplacian.copyNanoToOpen();

    std::cout<<"Hola3"<<std::endl;
    vector_max.push_back(computeMax(gridTumorCells.getPtrNano1(typePointer::CPU)));
    writeVector(vector_max,"maxTummor.txt");
    //std::cout<<"Copy post"<<std::endl;
    gridEndothelial.writeToFile("../grids/Endothelial.vdb"); 
    gridTAF.writeToFile("../grids/TAF.vdb");
    gridFibronectin.writeToFile("../grids/Fibronectin.vdb");
    gridGradienteTAF.writeToFile("../grids/TAFGradient.vdb");
    gridGradienteFibronectin.writeToFile("../grids/FibronectinGradient.vdb");
    gridMDE.writeToFile("../grids/MDE.vdb");
    gridGradienteEndothelial.writeToFile("../grids/EndothelialGradient.vdb");
    gridDivergenciaTAF.writeToFile("../grids/TAFDivergence.vdb");
    gridDivergenciaFibronectin.writeToFile("../grids/FibronectinDivergence.vdb");
    gridTAFEndothelial.writeToFile("../grids/TAFEndothelial.vdb");
    gridTipEndothelial.writeToFile("../grids/TipEndothelial.vdb");
    gridEndothelialDiscrete.writeToFile("../grids/EndothelialDiscrete.vdb");
    gridBplus.writeToFile("../grids/gridBplus.vdb");
    gridBMinus.writeToFile("../grids/gridBminus.vdb");
    gridTumorCells.writeToFile("../grids/gridTummorCells.vdb");
    gridOxygen.writeToFile("../grids/gridOxygen.vdb");
    gridPressure.writeToFile("../grids/gridPressure.vdb");
    gridDeadCells.writeToFile("../grids/gridDeadCells.vdb");
    gridTummorFlux.writeToFile("../grids/gridTumorFlux.vdb");
    gridPressureLaplacian.writeToFile("../grids/PressureLaplacian.vdb");



    // std::cout<<"Max "<<computeMax(gridTAF.getPtrNano1(typePointer::CPU))<<std::endl;
    // std::cout<<"Min "<<computeMin(gridTAF.getPtrNano1(typePointer::CPU))<<std::endl;

    return 0;

}