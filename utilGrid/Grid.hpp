#ifndef GRID_HPP
#define GRID_HPP
#include <openvdb/tools/ValueTransformer.h>
#include <openvdb/Types.h>
#include <iostream>
#include <stdlib.h>     /* atoi */
//#include "../utilsSkin/utilSkin.h"
#include <nanovdb/util/GridBuilder.h>
#include <openvdb/tools/LevelSetSphere.h> // replace with your own dependencies for generating the OpenVDB handle
#include <nanovdb/util/OpenToNanoVDB.h> // converter from OpenVDB to NanoVDB (includes NanoVDB.h and GridManager.h)
#include <nanovdb/util/CudaDeviceBuffer.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_device_runtime_api.h>
#include <memory>

template<typename OpenGridType,class GridTypeOpen,class GridTypeNano>
class Grid{
    private:
        GridTypeOpen gridOpen_1;
        GridTypeOpen gridOpen_2;
        
        std::shared_ptr<GridTypeOpen> gridOpen_1_ptr;
        std::shared_ptr<GridTypeOpen>  gridOpen_2_ptr;

        nanovdb::GridHandle<nanovdb::CudaDeviceBuffer> handleNano_1;
        nanovdb::GridHandle<nanovdb::CudaDeviceBuffer> handleNano_2;

        
        
        
        GridTypeNano* gridNano_1_cpu;
        GridTypeNano* gridNano_2_cpu;
        GridTypeNano* gridNano_1_device;
        GridTypeNano* gridNano_2_device;


        int profundidad_total;
        int size_lado;

    public:
        
        Grid(int size_lado,int profundidad_total,OpenGridType value){
            this->profundidad_total = profundidad_total;
            this->size_lado = size_lado;

            gridOpen_1_ptr = gridOpen_1.create(value);
            
            

            handleNano_1 = nanovdb::openToNanoVDB<nanovdb::CudaDeviceBuffer>(gridOpen_1);
            handleNano_2 = nanovdb::openToNanoVDB<nanovdb::CudaDeviceBuffer>(gridOpen_2);

            gridNano_1_cpu = handleNano_1.grid<OpenGridType>();
            gridNano_2_cpu = handleNano_2.grid<OpenGridType>();

            gridNano_1_device = handleNano_1.deviceGrid<OpenGridType>();
            gridNano_2_device = handleNano_2.deviceGrid<OpenGridType>();

        }

        std::shared_ptr<GridTypeOpen> getPtrOpen1(){
            return gridOpen_1_ptr;
        };

        std::shared_ptr<GridTypeOpen> getPtrOpen2(){
            return gridOpen_2_ptr;
        }
        
        enum typePointer{CPU,DEVICE};
        GridTypeNano* getPtrNano1(typePointer type){
            switch(type){
                case CPU:
                    return gridNano_1_cpu;
                case DEVICE:
                    return gridNano_1_device;
            };
            
        }

        GridTypeNano* getPtrNano2(typePointer type){
            switch(type){
                case CPU:
                    return gridNano_2_cpu;
                case DEVICE:
                    return gridNano_2_device;
            };
            
        }


        
};

#endif