#include <nanovdb/util/GridBuilder.h>
#include <openvdb/tools/LevelSetSphere.h> // replace with your own dependencies for generating the OpenVDB grid
#include <nanovdb/util/OpenToNanoVDB.h> // converter from OpenVDB to NanoVDB (includes NanoVDB.h and GridManager.h)
#include <nanovdb/util/CudaDeviceBuffer.h>
#include "kernels.h"
#include <openvdb/tools/ValueTransformer.h>
#include <openvdb/Types.h>
#include <iostream>
#include <stdlib.h>     /* atoi */
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
#include <map>
#include <string>

template<class gridClass,typename gridType>
void initializeAll(std::map<std::string,gridClass*>& map,gridType val){
    for(auto it = map.begin();it != map.end();it++){
        it->second->fillValue(val);
    }
}
template<class gridClass>
void fillRandomAll(std::map<std::string,gridClass*>& map){
    for(auto it = map.begin();it != map.end();it++){
        it->second->fillRandom();
    }
}
template<class gridClass>
void uploadAll(std::map<std::string,gridClass*>& map,std::vector<std::string> gridNames){
    for(auto it = gridNames.begin();it != gridNames.end();it++){
        map.at(*it)->upload();
    }
}
template<class gridClass>
void downloadAll(std::map<std::string,gridClass*>& map,std::vector<std::string> gridNames){
    for(auto it = gridNames.begin();it != gridNames.end();it++){
        map.at(*it)->download();
    }
}
template<class gridClass>
void copyAll(std::map<std::string,gridClass*>& map){
    for(auto it = map.begin();it != map.end();it++){
        it->second->copyNanoToOpen();
    }
}
template<class gridClass>
void writeAll(std::map<std::string,gridClass*>& map){
    for(auto it = map.begin();it != map.end();it++){
        it->second->writeToFile("../grids/new/"+it->first+".vdb");
    }
}
template<class gridClass,class accessor>
void getAllAccessor(std::map<std::string,gridClass*>& map,std::map<std::string,accessor>& accessorMap,int accessorNumber = 1){
    
    if(accessorNumber == 1){
        for(auto it = map.begin();it != map.end();it++){
            accessorMap[it->first] = it->second->getAccessorOpen1();
        }
    }else{
        for(auto it = map.begin();it != map.end();it++){
            accessorMap[it->first] = it->second->getAccessorOpen2();
        }
    }
}
template<class gridClass,class accessor>
void getAllNanoAccessor(std::map<std::string,gridClass*>& map,std::map<std::string,accessor*>& accessorMap,typePointer type,int accessorNumber = 1){
    if(accessorNumber == 1){
        for(auto it = map.begin();it != map.end();it++){
            accessorMap[it->first] = it->second->getPtrNano1(type);
        }
    }else{
        for(auto it = map.begin();it != map.end();it++){
            if(it->second->getCreateBoth()){
                accessorMap[it->first] = it->second->getPtrNano2(type);

            }else{
                accessorMap[it->first] = it->second->getPtrNano1(type);

            }
        }
    }
}
int main(int argc ,char * argv[]){
    using Vec3 = openvdb::Vec3s;
    using  Vec3Open = openvdb::Vec3SGrid ;
    using Vec3Nano = nanovdb::Vec3fGrid;
    int n_veces = 10;
    int size_lado = 250;
    int profundidad_total = 150;
    if(argc > 1){
        n_veces = atoi(argv[1]);
    }
    std::map<std::string,Grid<>*> gridsFloat ;
    std::map<std::string,Grid<Vec3,nanovdb::Vec3f,Vec3Open,Vec3Open::Ptr,Vec3Nano>*>gridsVec;
    std::map<std::string,nanovdb::FloatGrid*> nanoFloatMap1;
    std::map<std::string,nanovdb::FloatGrid*> nanoFloatMap2;
    std::map<std::string,nanovdb::Vec3fGrid*> nanoVecMap;
    std::vector<std::string> floatNames = {"Endothelial","TAF","Fibronectin","MDE","TAFEndothelial","Bplus","BMinus","TummorCells","Oxygen",
    "Pressure","PressureLaplacian","DeadCells","TipEndothelial","EndothelialDiscrete"};
    std::vector<std::string> endothelialGrids = {"Endothelial","TAF","Fibronectin","MDE","TAFEndothelial","TipEndothelial"};
    std::vector<std::string> tummorGrids = {"Bplus","BMinus","TummorCells","Oxygen","Pressure","PressureLaplacian","DeadCells"};
    std::vector<std::string> vecNames  = {"GradienteTAF","GradienteFibronectin","GradienteEndothelial","TummorFlux"};
    bool createBoth;
    for(auto it = floatNames.begin();it!=floatNames.end();it++){
        std::string name = *it;
        createBoth = name == "Endothelial" || name == "TAF" || name == "Fibronectin" || name == "MDE" || name == "TAFEndothelial"
        ||name == "TipEndothelial" || name == "EndothelialDiscrete";
        gridsFloat[name] = new Grid<>(size_lado,profundidad_total,0.0,createBoth);
    }
    for(auto it = vecNames.begin();it != vecNames.end();it++){
        std::string name = *it;
        openvdb::Vec3s ini = {0.0,0.0,0.0};
        gridsVec[name] = new Grid<Vec3,nanovdb::Vec3f,Vec3Open,Vec3Open::Ptr,Vec3Nano>(size_lado,profundidad_total,ini,false);
    }
    initializeAll<Grid<>,float>(gridsFloat,0.0);
    fillRandomAll<Grid<Vec3,nanovdb::Vec3f,Vec3Open,Vec3Open::Ptr,Vec3Nano>>(gridsVec);
    // uploadAll<Grid<>>(gridsFloat,floatNames);
    

    getAllNanoAccessor<Grid<Vec3,nanovdb::Vec3f,Vec3Open,Vec3Open::Ptr,Vec3Nano>,nanovdb::Vec3fGrid>(gridsVec,nanoVecMap,typePointer::DEVICE,1);

    std::map<std::string,nanovdb::FloatGrid*>* gridFloatRead;
    std::map<std::string,nanovdb::FloatGrid*>* gridFloatWrite;
    gridsFloat["EndothelialDiscrete"]->upload();//Este siempre tendrá que estar en GPU
    uint64_t nodeCount = gridsFloat["EndothelialDiscrete"]->getPtrNano1(typePointer::CPU)->tree().nodeCount(0);
    for(int i = 0 ;i<n_veces;i++){
        std::cout<<i<<std::endl;
        downloadAll<Grid<>>(gridsFloat,tummorGrids);
        uploadAll<Grid<>>(gridsFloat,endothelialGrids);
        uploadAll<Grid<Vec3,nanovdb::Vec3f,Vec3Open,Vec3Open::Ptr,Vec3Nano>>(gridsVec,vecNames);
        getAllNanoAccessor<Grid<>,nanovdb::FloatGrid>(gridsFloat,nanoFloatMap1,typePointer::DEVICE,1);
        getAllNanoAccessor<Grid<>,nanovdb::FloatGrid>(gridsFloat,nanoFloatMap2,typePointer::DEVICE,2);
        if(i%2 == 0 ){
            gridFloatRead = &nanoFloatMap1;
            gridFloatWrite = &nanoFloatMap2;
        }else{
            gridFloatRead = &nanoFloatMap2;
            gridFloatWrite = &nanoFloatMap1;
        }
        laplacian(gridFloatRead->at("MDE"),gridFloatRead->at("MDE"),nodeCount);
        //equationMDE(gridFloatRead->at("EndothelialDiscrete"),gridFloatRead->at("MDE"),gridFloatWrite->at("MDE"),nodeCount);
        // equationFibronectin(gridFloatRead->at("EndothelialDiscrete"),gridFloatRead->at("Fibronectin"),gridFloatRead->at("MDA"),gridFloatWrite->at("Fibronectin"),nodeCount);
        // equationTAF(gridFloatRead->at("EndothelialDiscrete"),gridFloatRead->at("TAF"),gridFloatWrite->at("TAF"),nodeCount);
        // product(gridFloatRead->at("TAF"),gridFloatRead->at("Endothelial"),gridFloatRead->at("TAFEndothelial"),nodeCount);
        // generateGradientFibronectin(gridFloatRead->at("Fibronectin"),gridFloatRead->at("EndothelialDiscrete"),nanoVecMap.at("GradienteFibronectin"),nodeCount);
        // generateGradientTAF(gridFloatRead->at("TAF"),gridFloatRead->at("TAFEndothelial"),nanoVecMap.at("GradienteTAF"),nodeCount);
        // equationEndothelial(gridFloatRead->at("EndothelialDiscrete"),gridFloatWrite->at("EndothelialDiscrete"),gridFloatRead->at("TAF"),gridFloatRead->at("Fibronectin"),nanoVecMap.at("GradienteTAF"),nanoVecMap.at("GradienteFibronectin"),gridFloatRead->at("TipEndothelial"),nodeCount);
        //downloadAll<Grid<>>(gridsFloat,endothelialGrids);
        downloadAll<Grid<Vec3,nanovdb::Vec3f,Vec3Open,Vec3Open::Ptr,Vec3Nano>>(gridsVec,vecNames);
        downloadAll<Grid<>>(gridsFloat,endothelialGrids);
        int dummy;
        std::cin>>dummy;
        uploadAll<Grid<>>(gridsFloat,tummorGrids);
        
    }


    downloadAll<Grid<>>(gridsFloat,floatNames);
    downloadAll<Grid<Vec3,nanovdb::Vec3f,Vec3Open,Vec3Open::Ptr,Vec3Nano>>(gridsVec,vecNames);
    copyAll<Grid<Vec3,nanovdb::Vec3f,Vec3Open,Vec3Open::Ptr,Vec3Nano>>(gridsVec);
    copyAll<Grid<>>(gridsFloat);
    writeAll<Grid<>>(gridsFloat);
    copyAll<Grid<Vec3,nanovdb::Vec3f,Vec3Open,Vec3Open::Ptr,Vec3Nano>>(gridsVec);
    writeAll<Grid<Vec3,nanovdb::Vec3f,Vec3Open,Vec3Open::Ptr,Vec3Nano>>(gridsVec);

    return 0 ;
}