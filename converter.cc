#include <iostream>
#include "utilGrid/Grid.hpp"
#include <string>
#include "albedoKernel.h"
using namespace std;

int main(int argc , char* argv[]){
    using Vec3 = openvdb::Vec3s;
    using  Vec3Open = openvdb::Vec3SGrid ;
    using Vec3Nano = nanovdb::Vec3fGrid;
    cout<<"Hola"<<std::endl;
    string filenameTummor = "../grids/new/TummorCells360.vdb";
    string filenameO2 = "../grids/new/Oxygen360.vdb";
    Grid<> gridOxygen(filenameO2);
    Grid<> gridTummor(filenameTummor);
    openvdb::Vec3s ini = {0.0,0.0,0.0};
    Grid<Vec3,nanovdb::Vec3f,Vec3Open,Vec3Open::Ptr,Vec3Nano> albedoOxygen(250,150,ini,false);
    Grid<> gridAux(250,150,0.0,false);;
    
    gridAux.fillRandom();
    albedoOxygen.fillRandom();
    albedoOxygen.upload();
    gridAux.upload();
    gridOxygen.upload();
    gridTummor.upload();
    
    uint64_t nodeCount = gridOxygen.getPtrNano1(typePointer::CPU)->tree().nodeCount(0);
    for(int j = 0 ;j < 1 ;j++){
                average(gridTummor.getPtrNano1(typePointer::DEVICE),gridAux.getPtrNano1(typePointer::DEVICE),nodeCount);
                copy(gridAux.getPtrNano1(typePointer::DEVICE),gridTummor.getPtrNano1(typePointer::DEVICE),nodeCount);
                //generateEndothelial(nanoFloatMap1.at("Oxygen"),nodeCount,-39,-130,5);
    }
    //albedoHemogoblin(gridOxygen.getPtrNano1(typePointer::DEVICE),albedoOxygen.getPtrNano1(typePointer::DEVICE),nodeCount);
    albedoTotal(gridOxygen.getPtrNano1(typePointer::DEVICE),gridTummor.getPtrNano1(typePointer::DEVICE),albedoOxygen.getPtrNano1(typePointer::DEVICE),nodeCount);

    gridTummor.download();
    gridOxygen.download();
    albedoOxygen.download();
    albedoOxygen.copyNanoToOpen();
    gridOxygen.copyNanoToOpen();
    
    // gridOxygen.writeToFile("../grids/readedEndothelial.vdb");
    albedoOxygen.writeToFile("../grids/albedoOxygen.vdb");
    

    return 0;
}