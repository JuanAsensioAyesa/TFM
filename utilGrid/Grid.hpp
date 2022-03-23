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
#include <string>

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
        bool uploaded;

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
            uploaded = false;

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

        void upload(){
            if(!uploaded){
                handleNano_1.deviceUpload(false);
                handleNano_2.deviceUpload(false);
                uploaded = true;

            }
        }

        void download(){
            if(uploaded){
                handleNano_1.deviceDownload(true);
                handleNano_2.deviceDownload(true);
                uploaded = false;
            }
        }

        void writeToFile(string file){
            openvdb::io::File file(file);
            openvdb::GridPtrVec grid;
            grids.push_back(gridOpen_1_ptr);
            
            file.write(grids);
            file.close();
        }


        void copyNanoToOpen(){
            openvdb::Coord coordenadas_open;
            nanovdb::Coord coordenadas_nano;

            auto accessor_nano_2 = gridNano_2_cpu->getAccessor();
            auto accessor_open_2 = gridOpen_1_ptr->getAccessor();
            for(int i  =0;i>-size_lado;i--){
                for(int j = 0 ;j>-profundidad_total;j--){
                    for(int k = 0 ;k>-size_lado;k--){
                        coordenadas_nano = nanovdb::Coord(i,j,k);
                        coordenadas_open = openvdb::Coord(i,j,k);
                        
                        accessor_open_2.setValue(coordenadas_open,accessor_nano_2.getValue(coordenadas_nano));
                    }
                }
            }
        }
        
};

#endif