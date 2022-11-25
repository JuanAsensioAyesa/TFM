#include <iostream>
#include "utilGrid/Grid.hpp"
#include <string>
#include "albedoKernel.h"
#include <stdlib.h>     /* atoi */

using namespace std;

int main(int argc , char* argv[]){
    using Vec3 = openvdb::Vec3s;
    using  Vec3Open = openvdb::Vec3SGrid ;
    using Vec3Nano = nanovdb::Vec3fGrid;
    if(argc < 4){
        std::cout<<"Converter gridTummor gridOxygen i"<<std::endl;
        return -1;
    }
    
    string filenameTummor = std::string(argv[1]);
    string filenameO2 = std::string(argv[2]);
    int iFile = atoi(argv[3]);
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
    //std::cout<<"Ey"<<std::endl;
    uint64_t nodeCount = gridOxygen.getPtrNano1(typePointer::CPU)->tree().nodeCount(0);
    //std::cout<<"Ey"<<std::endl;
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
    albedoOxygen.writeToFile("/home/juanasensio/Desktop/TFM/codigo/grids/albedoOxygen_"+std::to_string(iFile)+".vdb");
    

    return 0;
}