#ifndef GRID_HPP
#define GRID_HPP
#include <openvdb/openvdb.h>
#include <openvdb/tools/GridTransformer.h>
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
#include <algorithm>


enum typePointer{CPU,DEVICE};

template<typename OpenGridType=float,typename NanoGridType=float,class GridTypeOpen=openvdb::FloatGrid,class GridOpenPtr = openvdb::FloatGrid::Ptr,class GridTypeNano=nanovdb::FloatGrid>
class Grid{
    private:

        
        GridOpenPtr gridOpen_1_ptr;
        GridOpenPtr  gridOpen_2_ptr;

        nanovdb::GridHandle<nanovdb::CudaDeviceBuffer> handleNano_1;
        nanovdb::GridHandle<nanovdb::CudaDeviceBuffer> handleNano_2;

        
        
        
        GridTypeNano* gridNano_1_cpu;
        GridTypeNano* gridNano_2_cpu;
        GridTypeNano* gridNano_1_device;
        GridTypeNano* gridNano_2_device;


        int profundidad_total;
        int size_lado;
        bool uploaded;
        bool first_upload;
        bool createBoth;

    public:
        
        Grid(int size_lado,int profundidad_total,OpenGridType value,bool createBoth = true,bool sphere = false){
            this->profundidad_total = profundidad_total;
            this->size_lado = size_lado;
            this->createBoth = createBoth;
            if(!sphere){
                gridOpen_1_ptr = GridTypeOpen::create(value);
                // gridOpen_1_ptr->setTransform(
                //     openvdb::math::Transform::createLinearTransform(/*voxel size=*/0.05));
                if(createBoth){
                    gridOpen_2_ptr = GridTypeOpen::create(value);
                    // gridOpen_2_ptr->setTransform(
                    // openvdb::math::Transform::createLinearTransform(/*voxel size=*/0.05));
                
                }
            }else{
                gridOpen_1_ptr = GridTypeOpen::create(value);
                if(createBoth){
                    gridOpen_2_ptr = GridTypeOpen::create(value);
                }
            }
            
            
            
            
            // std::cout<<"ptr 1 "<<gridOpen_1_ptr<<std::endl;
            // std::cout<<"ptr 2 "<<gridOpen_2_ptr<<std::endl;
            
            

            

            first_upload = true;
            uploaded = false;

        }

        std::shared_ptr<GridTypeOpen> getPtrOpen1(){
            return gridOpen_1_ptr;
        };

        std::shared_ptr<GridTypeOpen> getPtrOpen2(){
            return gridOpen_2_ptr;
        }
        
        typename GridTypeOpen::Accessor getAccessorOpen1(){
            return gridOpen_1_ptr->getAccessor();
        }

        typename GridTypeOpen::Accessor getAccessorOpen2(){
            return gridOpen_2_ptr->getAccessor();
        }
        
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
                //std::cout<<"Upload"<<std::endl;
                if(first_upload){
                    handleNano_1 = nanovdb::openToNanoVDB<nanovdb::CudaDeviceBuffer>(*gridOpen_1_ptr);
                    if(createBoth){
                        handleNano_2 = nanovdb::openToNanoVDB<nanovdb::CudaDeviceBuffer>(*gridOpen_2_ptr);
                    }
                    //first_upload = false;
                }
                
                handleNano_1.deviceUpload(0,true);
                uploaded = true;
                gridNano_1_cpu = handleNano_1.grid<NanoGridType>();
                

                gridNano_1_device = handleNano_1.deviceGrid<NanoGridType>();
                if(createBoth){
                    handleNano_2.deviceUpload(0,true);
                    gridNano_2_cpu = handleNano_2.grid<NanoGridType>();
                    gridNano_2_device = handleNano_2.deviceGrid<NanoGridType>();
                }
                

            }
        }

        void download(){
            if(uploaded){
                handleNano_1.deviceDownload(0,true);
                if(createBoth){
                    handleNano_2.deviceDownload(0,true);
                }
                
                uploaded = false;
            }
        }

        void writeToFile(std::string filename){
            openvdb::io::File file(filename);
            openvdb::GridPtrVec grids;
            grids.push_back(gridOpen_1_ptr);
            if(createBoth){
                grids.push_back(gridOpen_2_ptr);
            }
            
            
            
            file.write(grids);
            file.close();
        }


        void copyNanoToOpen(){
            openvdb::Coord coordenadas_open;
            nanovdb::Coord coordenadas_nano;

            typename GridTypeNano::AccessorType  accessor_nano_2 = gridNano_1_cpu->getAccessor();
            typename GridTypeOpen::Accessor accessor_open_2 = gridOpen_1_ptr->getAccessor();
            typename GridTypeNano::AccessorType accessor_nano_1 = gridNano_1_cpu->getAccessor();
            typename GridTypeOpen::Accessor accessor_open_1 = gridOpen_1_ptr->getAccessor();
            
            if(createBoth){
                accessor_nano_2 = gridNano_2_cpu->getAccessor();
                accessor_open_2 = gridOpen_2_ptr->getAccessor();
            }
            
            for(int i  =0;i>-size_lado;i--){
                for(int j = 0 ;j>-profundidad_total;j--){
                    for(int k = 0 ;k>-size_lado;k--){
                        int i_new = i + size_lado / 2.0;
                        int k_new = k + size_lado /2.0;
                        coordenadas_nano = nanovdb::Coord(i_new,j,k_new);
                        coordenadas_open = openvdb::Coord(i_new,j,k_new);
                        if constexpr(std::is_same<OpenGridType,float>::value){
                            if(createBoth){
                                
                                accessor_open_2.setValue(coordenadas_open,accessor_nano_2.getValue(coordenadas_nano));
                                
                            }
                            
                            accessor_open_1.setValue(coordenadas_open,accessor_nano_1.getValue(coordenadas_nano));
                        }else if constexpr(std::is_same<OpenGridType,openvdb::Vec3s>::value) {
                            nanovdb::Vec3f vec;
                            openvdb::Vec3s vec_open ;
                            if(createBoth){
                                vec  = accessor_nano_2.getValue(coordenadas_nano);
                                vec_open = accessor_open_2.getValue(coordenadas_open);
                            }
                            nanovdb::Vec3f vec_1 = accessor_nano_1.getValue(coordenadas_nano);
                            openvdb::Vec3s vec_open_1 = accessor_open_1.getValue(coordenadas_open);
                            for(int c =0 ;c<3;c++){
                                if(createBoth){
                                    vec_open[c] = vec[c];
                                }
                                
                                vec_open_1[c] = vec_1[c];
                            }
                            if(createBoth){
                                accessor_open_2.setValue(coordenadas_open,vec_open);
                            }
                            
                            accessor_open_1.setValue(coordenadas_open,vec_open_1);
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
            std::uniform_real_distribution<> dist(0, 1);
            auto accessor_open = gridOpen_1_ptr->getAccessor();
            auto accessor_open_2 = gridOpen_1_ptr->getAccessor();
            if(createBoth){
                accessor_open_2 =  gridOpen_2_ptr->getAccessor();
            }
             
            for(int i  =0;i>-size_lado;i--){
                for(int j = 0 ;j>-profundidad_total;j--){
                    for(int k = 0 ;k>-size_lado;k--){
                        int i_new = i + size_lado / 2.0;
                        int k_new = k + size_lado /2.0;
                        openvdb::Coord coordenadas_open = openvdb::Coord(i_new,j,k_new);
                        if constexpr(std::is_same<OpenGridType,float>::value){
                            // accessor_open.setValue(coordenadas_open,dist(e2));
                            // if(createBoth){
                            //     accessor_open_2.setValue(coordenadas_open,dist(e2));
                            // }
                            accessor_open.setValue(coordenadas_open,(float(size_lado) - std::abs(i))/size_lado);
                            if(createBoth){
                                accessor_open_2.setValue(coordenadas_open,(float(size_lado) - std::abs(i))/size_lado);
                            }
                            // accessor_open.setValue(coordenadas_open,0.0);
                            // if(createBoth){
                            //     accessor_open_2.setValue(coordenadas_open,0.0);
                            // }
                            // // accessor_open.setValue(coordenadas_open,0.1);
                            // if(createBoth){
                            //     accessor_open_2.setValue(coordenadas_open,0.1);
                            // }
                            // accessor_open.setValue(coordenadas_open,(float(std::abs(j)))/profundidad_total);
                            // if(createBoth){
                            //     accessor_open_2.setValue(coordenadas_open,(float(std::abs(j)))/profundidad_total);
                            // }
                            //accessor_open.setValue(coordenadas_open,i*i*i);
                            //accessor_open_2.setValue(coordenadas_open,i*i*i);
                        }else if constexpr(std::is_same<OpenGridType,openvdb::Vec3s>::value){
                            openvdb::Vec3s vec;
                            //Para probar bien los gradientes
                            vec[0] = -(i+2*size_lado/3) * (i +size_lado/3) * (i -size_lado/2);
                            vec[1]  = -(j+2*profundidad_total/3) * (j +profundidad_total/3) *(j - profundidad_total/2);
                            vec[2] =  -(k+2*size_lado/3) * (k +size_lado/3) * (k - size_lado/2);

                            float max_value_abs = 200;
                            
                            for(int iVec = 0 ;iVec <3;iVec++){
                                // vec[iVec] = std::min({vec[iVec],max_value_abs});
                                // vec[iVec] = std::max({vec[iVec],-max_value_abs});
                                vec[iVec] = 0.0;
                                vec[iVec] = 0.0;
                            }
                            


                            //vec[0]*=vec[0]*0.001;
                            // vec[1]*=vec[1];
                            // vec[2]*=vec[2];
                            accessor_open.setValue(coordenadas_open,vec);
                            if(createBoth){
                                accessor_open_2.setValue(coordenadas_open,vec);
                            }
                            
                        }else if constexpr(std::is_same<OpenGridType,bool>::value){
                            accessor_open.setValue(coordenadas_open,true);
                            if(createBoth){
                                accessor_open_2.setValue(coordenadas_open,false);
                            }
                        }
                    }
                    
                }
            }
        }
        
        void fillValue(OpenGridType value){
            auto accessor_open = gridOpen_1_ptr->getAccessor();
            auto accessor_open_2 = gridOpen_1_ptr->getAccessor();
            if(createBoth){
                accessor_open_2 =  gridOpen_2_ptr->getAccessor();
            }
             
            for(int i  =0;i>-size_lado;i--){
                for(int j = 0 ;j>-profundidad_total;j--){
                    for(int k = 0 ;k>-size_lado;k--){
                        int i_new = i + size_lado / 2.0;
                        int k_new = k + size_lado /2.0;
                        openvdb::Coord coordenadas_open = openvdb::Coord(i_new,j,k_new);
                        accessor_open.setValue(coordenadas_open,value);
                        if(createBoth){
                            accessor_open_2.setValue(coordenadas_open,value);
                        }
                    }
                }
            }

        }
        void uploadOne(int which){
            switch(which){
                case 1:
                    break;
                case 2:
                    break;
            };
        }

        void interpolate(){ 
            if constexpr(std::is_same<OpenGridType,float>::value){
                download();
                //copyNanoToOpen();
                openvdb::FloatGrid::Ptr
                sourceGrid = getPtrOpen1(),
                targetGrid = getPtrOpen2();
                const openvdb::math::Transform
                &sourceXform = sourceGrid->transform(),
                &targetXform = targetGrid->transform();
                // Compute a source grid to target grid transform.
                // (For this example, we assume that both grids' transforms are linear,
                // so that they can be represented as 4 x 4 matrices.)
                openvdb::Mat4R xform =
                    sourceXform.baseMap()->getAffineMap()->getMat4() *
                    targetXform.baseMap()->getAffineMap()->getMat4().inverse();
                // Create the transformer.
                openvdb::tools::GridTransformer transformer(xform);
                // Resample using nearest-neighbor interpolation.
                transformer.transformGrid<openvdb::tools::QuadraticSampler, openvdb::FloatGrid>(
                *sourceGrid, *targetGrid);
                auto tree = targetGrid->tree();
                gridOpen_2_ptr->setTree(tree.copy());
                upload();
            }
        }
};

#endif