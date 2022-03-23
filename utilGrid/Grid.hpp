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
#include <type_traits>
#include <random>

enum typePointer{CPU,DEVICE};

template<typename OpenGridType=float,typename NanoGridType=float,class GridTypeOpen=openvdb::FloatGrid,class GridTypeNano=nanovdb::FloatGrid>
class Grid{
    private:
        GridTypeOpen gridOpen_read;
        GridTypeOpen gridOpen_write;
        
        std::shared_ptr<GridTypeOpen> gridOpen_read_ptr;
        std::shared_ptr<GridTypeOpen>  gridOpen_write_ptr;

        nanovdb::GridHandle<nanovdb::CudaDeviceBuffer> handleNano_read;
        nanovdb::GridHandle<nanovdb::CudaDeviceBuffer> handleNano_write;

        
        
        
        GridTypeNano* gridNano_read_cpu;
        GridTypeNano* gridNano_write_cpu;
        GridTypeNano* gridNano_read_device;
        GridTypeNano* gridNano_write_device;


        int profundidad_total;
        int size_lado;
        bool uploaded;

    public:
        
        Grid(int size_lado,int profundidad_total,OpenGridType value){
            this->profundidad_total = profundidad_total;
            this->size_lado = size_lado;

            gridOpen_write_ptr = gridOpen_write.create(value);
            gridOpen_read_ptr = gridOpen_read.create(value);
            
            

            

            
            uploaded = false;

        }

        std::shared_ptr<GridTypeOpen> getPtrOpenRead(){
            return gridOpen_read_ptr;
        };

        std::shared_ptr<GridTypeOpen> getPtrOpenWrite(){
            return gridOpen_write_ptr;
        }
        
        typename GridTypeOpen::Accessor getAccessorOpenRead(){
            return gridOpen_read.getAccessor();
        }

        typename GridTypeOpen::Accessor getAccessorOpenWrite(){
            return gridOpen_write.getAccessor();
        }
        
        GridTypeNano* getPtrNanoRead(typePointer type){
            switch(type){
                case CPU:
                    return gridNano_read_cpu;
                case DEVICE:
                    return gridNano_read_device;
            };
            
        }

        GridTypeNano* getPtrNanoWrite(typePointer type){
            switch(type){
                case CPU:
                    return gridNano_write_cpu;
                case DEVICE:
                    return gridNano_write_device;
            };
            
        }

        void upload(){
            if(!uploaded){
                std::cout<<"Upload"<<std::endl;
                handleNano_read = nanovdb::openToNanoVDB<nanovdb::CudaDeviceBuffer>(gridOpen_read);
                handleNano_write = nanovdb::openToNanoVDB<nanovdb::CudaDeviceBuffer>(gridOpen_write);
                handleNano_read.deviceUpload(0,true);
                handleNano_write.deviceUpload(0,true);
                uploaded = true;
                gridNano_read_cpu = handleNano_read.grid<NanoGridType>();
                gridNano_write_cpu = handleNano_write.grid<NanoGridType>();

                gridNano_read_device = handleNano_read.deviceGrid<NanoGridType>();
                gridNano_write_device = handleNano_write.deviceGrid<NanoGridType>();

            }
        }

        void download(){
            if(uploaded){
                handleNano_read.deviceDownload(0,true);
                handleNano_write.deviceDownload(0,true);
                uploaded = false;
            }
        }

        void writeToFile(std::string filename){
            openvdb::io::File file(filename);
            openvdb::GridPtrVec grids;
            grids.push_back(gridOpen_write_ptr);
            
            file.write(grids);
            file.close();
        }


        void copyNanoToOpen(){
            openvdb::Coord coordenadas_open;
            nanovdb::Coord coordenadas_nano;

            auto accessor_nano_write = gridNano_write_cpu->getAccessor();
            auto accessor_open_write = gridOpen_write_ptr->getAccessor();
            for(int i  =0;i>-size_lado;i--){
                for(int j = 0 ;j>-profundidad_total;j--){
                    for(int k = 0 ;k>-size_lado;k--){
                        coordenadas_nano = nanovdb::Coord(i,j,k);
                        coordenadas_open = openvdb::Coord(i,j,k);
                        if constexpr(std::is_same<OpenGridType,float>::value){
                            accessor_open_write.setValue(coordenadas_open,accessor_nano_write.getValue(coordenadas_nano));
                        }else if constexpr(std::is_same<OpenGridType,openvdb::Vec3s>::value) {
                            nanovdb::Vec3f vec = accessor_nano_write.getValue(coordenadas_nano);
                            openvdb::Vec3s vec_open = accessor_open_write.getValue(coordenadas_open);
                            for(int c =0 ;c<3;c++){
                                vec_open[c] = vec[c];
                            }
                            accessor_open_write.setValue(coordenadas_open,vec_open);
                        }
                        
                    }
                }
            }
        }

        void fillRandom(){
            std::random_device rd;

            //
            // Engines 
            //
            std::mt19937 e2(rd());
            std::uniform_real_distribution<> dist(0, 10);
            auto accessor_open = gridOpen_read.getAccessor();
            auto accessor_open_write = gridOpen_write.getAccessor();
            for(int i  =0;i>-size_lado;i--){
                for(int j = 0 ;j>-profundidad_total;j--){
                    for(int k = 0 ;k>-size_lado;k--){
                        openvdb::Coord coordenadas_open = openvdb::Coord(i,j,k);
                        if constexpr(std::is_same<OpenGridType,float>::value){
                            //accessor_open.setValue(coordenadas_open,dist(e2));
                            //accessor_open_write.setValue(coordenadas_open,dist(e2));

                            accessor_open.setValue(coordenadas_open,i*i);
                            accessor_open_write.setValue(coordenadas_open,i*i);
                        }else if constexpr(std::is_same<OpenGridType,openvdb::Vec3s>::value){
                            openvdb::Vec3s vec;
                            vec[0] = dist(e2);
                            vec[1]  = dist(e2);
                            vec[2] = dist(e2);
                            accessor_open.setValue(coordenadas_open,vec);
                            accessor_open_write.setValue(coordenadas_open,vec);
                        }
                    }
                    
                }
            }
        }
        
};

#endif