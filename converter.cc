#include <iostream>
#include "utilGrid/Grid.hpp"
#include <string>
#include "kernels.h"
using namespace std;

int main(int argc , char* argv[]){
    using Vec3 = openvdb::Vec3s;
    using  Vec3Open = openvdb::Vec3SGrid ;
    using Vec3Nano = nanovdb::Vec3fGrid;
    cout<<"Hola"<<std::endl;
    string filename = "../grids/new/TummorCells.vdb";
    Grid<> gridOxygen(filename);
    openvdb::Vec3s ini = {0.0,0.0,0.0};
    Grid<Vec3,nanovdb::Vec3f,Vec3Open,Vec3Open::Ptr,Vec3Nano> albedoOxygen(250,150,ini,false);

    
    albedoOxygen.fillRandom();
    albedoOxygen.upload();
    
    gridOxygen.upload();
    
    uint64_t nodeCount = gridOxygen.getPtrNano1(typePointer::CPU)->tree().nodeCount(0);
    
    albedoHemogoblin(gridOxygen.getPtrNano1(typePointer::DEVICE),albedoOxygen.getPtrNano1(typePointer::DEVICE),nodeCount);
    
    gridOxygen.download();
    albedoOxygen.download();
    albedoOxygen.copyNanoToOpen();
    gridOxygen.copyNanoToOpen();
    
    // gridOxygen.writeToFile("../grids/readedEndothelial.vdb");
    albedoOxygen.writeToFile("../grids/albedoOxygen.vdb");
    

    return 0;
}