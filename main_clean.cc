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
void uploadAll(std::map<std::string,gridClass*>& map){
    for(auto it = map.begin();it != map.end();it++){
        it->second->upload();
    }
}
template<class gridClass>
void downloadAll(std::map<std::string,gridClass*>& map){
    for(auto it = map.begin();it != map.end();it++){
        it->second->download();
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
    std::vector<std::string> floatNames = {"Endothelial","TAF","Fibronectin","MDE","TAFEndothelial","Bplus","BMinus","TummorCells","Oxygen",
    "Pressure","PressureLaplacian","DeadCells","TipEndothelial","EndothelialDiscrete"};
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

    uploadAll<Grid<>>(gridsFloat);
    uploadAll<Grid<Vec3,nanovdb::Vec3f,Vec3Open,Vec3Open::Ptr,Vec3Nano>>(gridsVec);

    downloadAll<Grid<>>(gridsFloat);

    downloadAll<Grid<Vec3,nanovdb::Vec3f,Vec3Open,Vec3Open::Ptr,Vec3Nano>>(gridsVec);


    copyAll<Grid<Vec3,nanovdb::Vec3f,Vec3Open,Vec3Open::Ptr,Vec3Nano>>(gridsVec);
    copyAll<Grid<>>(gridsFloat);
    writeAll<Grid<>>(gridsFloat);
    copyAll<Grid<Vec3,nanovdb::Vec3f,Vec3Open,Vec3Open::Ptr,Vec3Nano>>(gridsVec);
    writeAll<Grid<Vec3,nanovdb::Vec3f,Vec3Open,Vec3Open::Ptr,Vec3Nano>>(gridsVec);

    return 0 ;
}