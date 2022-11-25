#include <iostream>
#include "utilGrid/Grid.hpp"
#include <string>
#include "albedoKernel.h"
#include <stdlib.h>     /* atoi */

using namespace std;

int main(int argc , char* argv[]){

    std::string filenameTummor ;
    std::string filenameDeadCells;
    std::string filenameOut;
    Grid<> gridOut(250,150,0.0,false);
    gridOut.fillRandom();
    
    for(int i = 0 ;i<=400;i = i + 40){
        std::cout<<i<<std::endl;
        
        filenameTummor = "/home/juanasensio/Desktop/TFM/codigo/grids/new/TummorCells"+std::to_string(i)+".vdb";
        filenameDeadCells = "/home/juanasensio/Desktop/TFM/codigo/grids/new/DeadCells"+std::to_string(i)+".vdb";

        Grid<> gridTummor(filenameTummor);
        Grid<> gridDeadCells(filenameDeadCells);
        gridTummor.upload();
        gridDeadCells.upload();
        gridOut.upload();
        uint64_t nodeCount = gridOut.getPtrNano1(typePointer::CPU)->tree().nodeCount(0);
        copy(gridTummor.getPtrNano1(typePointer::DEVICE),gridOut.getPtrNano1(typePointer::DEVICE),nodeCount);
        product(gridDeadCells.getPtrNano1(typePointer::DEVICE),-1.0,nodeCount);
        copy(gridDeadCells.getPtrNano1(typePointer::DEVICE),gridOut.getPtrNano1(typePointer::DEVICE),nodeCount);
        add(gridTummor.getPtrNano1(typePointer::DEVICE),gridOut.getPtrNano1(typePointer::DEVICE),nodeCount);

        gridTummor.download();
        gridDeadCells.download();
        
        gridOut.download();
        gridOut.copyNanoToOpen();
        gridOut.writeToFile("/home/juanasensio/Desktop/TFM/codigo/grids/new/gridOut"+std::to_string(i)+".vdb");
    }

}