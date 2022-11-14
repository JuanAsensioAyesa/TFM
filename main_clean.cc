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
float resolvePoisson(std::map<std::string,Grid<>*>& grids,int i,std::string gridName,std::string gridDestinyName, openvdb::v9_0::FloatTree::Ptr& newTree){

    grids[gridName]->download();
    grids[gridName]->copyNanoToOpen();
    grids[gridDestinyName]->download();
    float prevMax = computeMax(grids[gridName]->getPtrNano1(typePointer::CPU));
    auto state = openvdb::math::pcg::terminationDefaults<float>();
    if(i%2==0){
        auto tree = grids[gridName]->getPtrOpen1()->tree();
        newTree = openvdb::tools::poisson::solve<openvdb::v9_0::FloatTree>(tree,state);
        grids[gridDestinyName]->getPtrOpen1()->setTree(newTree->copy());
    }else{
        auto tree = grids[gridName]->getPtrOpen1()->tree();
        newTree = openvdb::tools::poisson::solve<openvdb::v9_0::FloatTree>(tree,state);
        grids[gridDestinyName]->getPtrOpen2()->setTree(newTree->copy());
    }
    grids[gridName]->upload();
    grids[gridDestinyName]->upload();

    return prevMax;
}
void normalizePoisson(std::map<std::string,Grid<>*>& grids,std::string gridName,float prevMax,uint64_t nodeCount){

    float max = computeMax(grids[gridName]->getPtrNano1(typePointer::CPU));
    float min = computeMin(grids[gridName]->getPtrNano1(typePointer::CPU));
    float addition = max-min;
    float newMax = max + addition;

    addMax(grids[gridName]->getPtrNano1(typePointer::DEVICE),addition,nodeCount);
    normalize(grids[gridName]->getPtrNano1(typePointer::DEVICE),newMax,prevMax,nodeCount);
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
    std::vector<std::string> floatNames = {"Endothelial","TAF","Fibronectin","MDE","TAFEndothelial","Bplus","Bminus","TummorCells","Oxygen",
    "Pressure","Diffusion","DeadCells","TipEndothelial","EndothelialDiscrete"};
    std::vector<std::string> endothelialGrids = {"Endothelial","TAF","Fibronectin","MDE","TAFEndothelial","TipEndothelial"};
    std::vector<std::string> tummorGrids = {"Bplus","BMinus","TummorCells","Oxygen","Pressure","PressureLaplacian","DeadCells"};
    std::vector<std::string> vecNames  = {"vecGrid"};
    
    bool createBoth;
    for(auto it = floatNames.begin();it!=floatNames.end();it++){
        std::string name = *it;
        createBoth = name == "Endothelial" || name == "TAF" || name == "Fibronectin" || name == "MDE" || name == "TAFEndothelial"
        ||name == "TipEndothelial" || name == "EndothelialDiscrete" || name == "TummorCells";
        if(name == "Endothelial"){
            gridsFloat[name] = new Grid<>(size_lado,profundidad_total,0.0,createBoth);

        }else{
            gridsFloat[name] = new Grid<>(size_lado,profundidad_total,0.0,createBoth);

        }
    }
    for(auto it = vecNames.begin();it != vecNames.end();it++){
        std::string name = *it;
        openvdb::Vec3s ini = {0.0,0.0,0.0};
        gridsVec[name] = new Grid<Vec3,nanovdb::Vec3f,Vec3Open,Vec3Open::Ptr,Vec3Nano>(size_lado,profundidad_total,ini,false);
    }
    initializeAll<Grid<>,float>(gridsFloat,0.0);
    float max_diffussion = 0.1;
    float max_paper = 83.0;
    dataSkin diffussionCoef;
    diffussionCoef.valueCorneum = 83.0 / max_paper * max_diffussion;
    diffussionCoef.valueSpinosum = 8.3/ max_paper * max_diffussion;
    diffussionCoef.valueBasale = 41.5 / max_paper * max_diffussion;
    diffussionCoef.valueDermis = 30 / max_paper * max_diffussion;
    diffussionCoef.valueHipoDermis = 0.01666 / max_paper * max_diffussion;

    diffussionCoef.valueCorneum = 30.0 / max_paper * max_diffussion;
    diffussionCoef.valueSpinosum = 83./ max_paper * max_diffussion;
    diffussionCoef.valueBasale = 8.3 / max_paper * max_diffussion;
    diffussionCoef.valueDermis = 41.5 / max_paper * max_diffussion;
    diffussionCoef.valueHipoDermis = 20. / max_paper * max_diffussion;
    auto accessorDif = gridsFloat.at("Diffusion")->getAccessorOpen1();
    openvdb::Coord coord_dif;
    createSkin(accessorDif,size_lado,profundidad_total,coord_dif,diffussionCoef);
    std::cout<<"Ey"<<std::endl;

    fillRandomAll<Grid<Vec3,nanovdb::Vec3f,Vec3Open,Vec3Open::Ptr,Vec3Nano>>(gridsVec);
    std::cout<<"Ey"<<std::endl;
    std::map<std::string,nanovdb::FloatGrid*>* gridFloatRead;
    std::map<std::string,nanovdb::FloatGrid*>* gridFloatWrite;
    
    gridsFloat["EndothelialDiscrete"]->upload();//Este siempre tendrá que estar en GPU
    std::cout<<"Ey"<<std::endl;

    openvdb::v9_0::FloatTree::Ptr newTreeOxygen;
    openvdb::v9_0::FloatTree::Ptr newTreeTAF;
    bool condition;
    float prevMax;
    float prevMaxDiscrete;
    float prevMaxTummor;
    openvdb::Coord esquina_izquierda;
    esquina_izquierda[0] = 0 ;
    esquina_izquierda[1] =-34 ;
    esquina_izquierda[2] =0 ;
    int tamanio_tumor = 1;
    auto accessor_tummor_1 = gridsFloat["TummorCells"]->getAccessorOpen1();
    auto accessor_tummor_2 = gridsFloat["TummorCells"]->getAccessorOpen2();
    auto accessor_Dead_1 = gridsFloat["DeadCells"]->getAccessorOpen1();
    //auto accessor_Dead_2 = gridsFloat["DeadCells"]->getAccessorOpen2();
    gridsFloat["Endothelial"]->fillValue(0.1);
    std::cout<<"Ey"<<std::endl;
    createRectangle(accessor_tummor_1,esquina_izquierda,tamanio_tumor,1.0);
    createRectangle(accessor_tummor_2,esquina_izquierda,tamanio_tumor,1.0);
    auto accesor_Endothelial_1 = gridsFloat["Endothelial"]->getAccessorOpen1();
    auto accesor_Endothelial_2 = gridsFloat["Endothelial"]->getAccessorOpen2();
    //createRectangle(accesor_Endothelial_1,esquina_izquierda,20,1.0);
    //createRectangle(accesor_Endothelial_2,esquina_izquierda,20,1.0);
    //createRectangle(accessor_Dead_1,esquina_izquierda,tamanio_tumor,0.1);
    //createRectangle(accessor_tummor_2,esquina_izquierda,tamanio_tumor,1.0);
    
    uploadAll<Grid<>>(gridsFloat,floatNames);
    uploadAll<Grid<Vec3,nanovdb::Vec3f,Vec3Open,Vec3Open::Ptr,Vec3Nano>>(gridsVec,vecNames);

    getAllNanoAccessor<Grid<Vec3,nanovdb::Vec3f,Vec3Open,Vec3Open::Ptr,Vec3Nano>,nanovdb::Vec3fGrid>(gridsVec,nanoVecMap,typePointer::DEVICE,1);
    getAllNanoAccessor<Grid<>,nanovdb::FloatGrid>(gridsFloat,nanoFloatMap1,typePointer::DEVICE,1);
    getAllNanoAccessor<Grid<>,nanovdb::FloatGrid>(gridsFloat,nanoFloatMap2,typePointer::DEVICE,2);
    uint64_t nodeCount = gridsFloat["EndothelialDiscrete"]->getPtrNano1(typePointer::CPU)->tree().nodeCount(0);
    generateEndothelial(gridsFloat["EndothelialDiscrete"]->getPtrNano1(typePointer::DEVICE),nodeCount,-35,-130,5);
    generateEndothelial(gridsFloat["EndothelialDiscrete"]->getPtrNano2(typePointer::DEVICE),nodeCount,-35,-130,5);
    using u32    = uint_least32_t; 
    using engine = std::mt19937;
    std::random_device os_seed;
    u32 seed = os_seed();

    engine generator( seed );
    std::uniform_int_distribution< u32 > distribute( 1, nodeCount);

    for(int i = 0 ;i<n_veces;i++){
        std::cout<<i<<std::endl;
        condition = i % 100 == 0 ;//|| i == n_veces - 2;
        
        if(condition  ){
            if(i > 0){
                //std::cout<<"Jelous"<<std::endl;
                degradeOxygen(gridFloatRead->at("EndothelialDiscrete"),gridFloatWrite->at("TummorCells"),1.0,nodeCount);
                degradeOxygen(gridFloatWrite->at("EndothelialDiscrete"),gridFloatWrite->at("TummorCells"),1.0,nodeCount);

            }
            prevMaxDiscrete = resolvePoisson(gridsFloat,i,"EndothelialDiscrete","Oxygen",newTreeOxygen);
            prevMaxTummor = resolvePoisson(gridsFloat,i,"Bplus","TAF",newTreeTAF);
            getAllNanoAccessor<Grid<>,nanovdb::FloatGrid>(gridsFloat,nanoFloatMap1,typePointer::DEVICE,1);
            getAllNanoAccessor<Grid<>,nanovdb::FloatGrid>(gridsFloat,nanoFloatMap2,typePointer::DEVICE,2);
            gridsFloat["Bplus"]->fillValue(0.0);
            //Hacer aqui bien de laplacianos
            //average(nanoFloatMap1.at("EndothelialDiscrete"),nanoFloatMap1.at("Oxygen"),nodeCount);

            
        }
        
        if(i%2 == 0 ){
            gridFloatRead = &nanoFloatMap1;
            gridFloatWrite = &nanoFloatMap2;
        }else{
            gridFloatRead = &nanoFloatMap2;
            gridFloatWrite = &nanoFloatMap1;
        }

        if(condition){
            normalizePoisson(gridsFloat,"Oxygen",prevMaxDiscrete,nodeCount);
            normalizePoisson(gridsFloat,"TAF",prevMaxTummor,nodeCount);
            for(int j = 0 ;j < 5 ;j++){
                average(nanoFloatMap1.at("Oxygen"),nanoFloatMap1.at("Bplus"),nodeCount);
                copy(nanoFloatMap1.at("Bplus"),nanoFloatMap1.at("Oxygen"),nodeCount);
                //generateEndothelial(nanoFloatMap1.at("Oxygen"),nodeCount,-39,-130,5);
            }
            generateEndothelial(nanoFloatMap1.at("Oxygen"),nodeCount,-35,-130,5);
            degradeOxygen(gridFloatRead->at("Oxygen"),gridFloatWrite->at("TummorCells"),0.4,nodeCount);
            newTips(gridFloatRead->at("TipEndothelial"),gridFloatRead->at("EndothelialDiscrete"),gridFloatRead->at("TAF"),seed,nodeCount);
            degradeDiscrete(gridFloatRead->at("EndothelialDiscrete"),gridFloatRead->at("TummorCells"),nodeCount);
        }

        equationMDE(gridFloatRead->at("EndothelialDiscrete"),gridFloatRead->at("MDE"),gridFloatWrite->at("MDE"),nodeCount);
        equationFibronectin(gridFloatRead->at("EndothelialDiscrete"),gridFloatRead->at("Fibronectin"),gridFloatRead->at("MDE"),gridFloatWrite->at("Fibronectin"),nodeCount);
        //equationTAF(gridFloatRead->at("EndothelialDiscrete"),gridFloatRead->at("TAF"),gridFloatWrite->at("TAF"),nodeCount);
        product(gridFloatRead->at("TAF"),gridFloatRead->at("Endothelial"),gridFloatRead->at("TAFEndothelial"),nodeCount);
        
        //laplacian(gridFloatRead->at("Endothelial"),gridFloatWrite->at("Endothelial"),nodeCount);
        //std::cout<<"Read "<<gridFloatRead->at("Endothelial")<<" Write "<<gridFloatWrite->at("Endothelial")<<std::endl;
        factorEndothelial(gridFloatRead->at("Endothelial"),gridFloatWrite->at("Endothelial"),nodeCount);
        generateGradientTAF(gridFloatRead->at("TAF"),gridFloatRead->at("TAFEndothelial"),nanoVecMap.at("vecGrid"),nodeCount);
        factorTAF(gridFloatRead->at("Endothelial"),gridFloatWrite->at("Endothelial"),gridFloatRead->at("TAF"),nanoVecMap.at("vecGrid"),nodeCount);
        generateGradientFibronectin(gridFloatRead->at("Fibronectin"),gridFloatRead->at("EndothelialDiscrete"),nanoVecMap.at("vecGrid"),nodeCount);
        factorFibronectin(gridFloatRead->at("Endothelial"),gridFloatWrite->at("Endothelial"),gridFloatRead->at("Fibronectin"),nanoVecMap.at("vecGrid"),nodeCount);
        equationEndothelialDiscrete(gridFloatRead->at("EndothelialDiscrete"),gridFloatWrite->at("EndothelialDiscrete"),gridFloatRead->at("Endothelial"),gridFloatWrite->at("Endothelial"),gridFloatRead->at("TAF"),gridFloatRead->at("TipEndothelial"),gridFloatWrite->at("TipEndothelial"),seed,nodeCount);
        branching(gridFloatWrite->at("TipEndothelial"),gridFloatRead->at("TAF"),seed,nodeCount);

        equationBplusSimple(gridFloatRead->at("TummorCells"),gridFloatRead->at("Bplus"),gridFloatRead->at("Oxygen"),nodeCount);
        equationBminusSimple(gridFloatRead->at("TummorCells"),gridFloatRead->at("Bminus"),gridFloatRead->at("Oxygen"),nodeCount);
        equationPressure(gridFloatRead->at("TummorCells"),gridFloatRead->at("Pressure"),nodeCount);
        equationFluxSimple(gridFloatRead->at("Pressure"),gridFloatRead->at("TummorCells"),nanoVecMap.at("vecGrid"),gridFloatRead->at("Diffusion"),nodeCount);
        equationTumorSimple(nanoVecMap.at("vecGrid"),gridFloatRead->at("Bplus"),gridFloatRead->at("Bminus"),gridFloatRead->at("TummorCells"),gridFloatWrite->at("TummorCells"),nodeCount);
        for(int j = 0 ;j<1;j++){
            average(gridFloatWrite->at("TummorCells"),gridFloatRead->at("Bplus"),nodeCount);
            copy(gridFloatRead->at("Bplus"),gridFloatWrite->at("TummorCells"),nodeCount);
        }
        
        // //newTips(gridFloatRead->at("TipEndothelial"),gridFloatRead->at("TAF"),gridFloatRead->at("EndothelialDiscrete"),seed,nodeCount);

        addDeadCells(gridFloatRead->at("Bminus"),gridFloatRead->at("DeadCells"),nodeCount);
        closeTips(gridFloatRead->at("TipEndothelial"),gridFloatRead->at("TummorCells"),nodeCount);
        //degradeOxygen(gridFloatRead->at("Oxygen"),gridFloatWrite->at("TummorCells"),0.035,nodeCount);

        // for(int j = 0;j<5;j++){
        //     average(gridFloatWrite->at("Oxygen"),gridFloatRead->at("Bplus"),nodeCount);
        //     copy(gridFloatRead->at("Bplus"),gridFloatWrite->at("Oxygen"),nodeCount);
        // }
        
        
        
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