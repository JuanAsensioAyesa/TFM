// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0
#include <nanovdb/util/GridBuilder.h>
#include <nanovdb/NanoVDB.h> // this defined the core tree data structure of NanoVDB accessable on both the host and device
#include <stdio.h> // for printf
#include <thrust/iterator/counting_iterator.h>
#include <thrust/for_each.h>

#include <nanovdb/util/Primitives.h>
#include <nanovdb/util/CudaDeviceBuffer.h>
#include "pruebaThrust.h"

// This is called by the host only
void cpu_kernel(const nanovdb::NanoGrid<float>* cpuGrid)
{
    printf("NanoVDB cpu; %4.2f\n", cpuGrid->tree().getValue(nanovdb::Coord(99, 0, 0)));
}

// This is called by the device only
__global__ void gpu_kernel(const nanovdb::NanoGrid<float>* deviceGrid)
{
    printf("NanoVDB gpu: %4.2f\n", deviceGrid->tree().getValue(nanovdb::Coord(99, 0, 0)));
}

// This is called by the client code on the host
void launch_kernels(const nanovdb::NanoGrid<float>* deviceGrid,
                               const nanovdb::NanoGrid<float>* cpuGrid,
                               cudaStream_t                    stream)
{
    gpu_kernel<<<1, 1, 0, stream>>>(deviceGrid); // Launch the device kernel asynchronously

    cpu_kernel(cpuGrid); // Launch the host "kernel" (synchronously)
}

void scaleActiveVoxels(nanovdb::FloatGrid *grid_d, uint64_t leafCount, float scale)
{
    auto kernel = [grid_d, scale] __device__ (const uint64_t n) {
        auto *leaf_d = grid_d->tree().getFirstNode<0>() + (n >> 9);// this only works if grid->isSequential<0>() == true
        const int i = n & 511;
        const float v = scale * leaf_d->getValue(i);
        if (leaf_d->isActive(i)) {
            leaf_d->setValueOnly(i, v);// only possible execution divergence
        }
    };

    thrust::counting_iterator<uint64_t, thrust::device_system_tag> iter(0);
    thrust::for_each(iter, iter + 512*leafCount, kernel);
}

void setZero(nanovdb::FloatGrid *grid_d,uint64_t leafCount){
    
    auto kernel = [grid_d] __device__ (const uint64_t n) {
        auto *leaf_d = grid_d->tree().getFirstNode<0>() + (n >> 9);// this only works if grid->isSequential<0>() == true
        const int i = n & 511;
        float scale = 2;
        const float v = scale * leaf_d->getValue(i);
        if (leaf_d->isActive(i)) {
            leaf_d->setValueOnly(i, 0);// only possible execution divergence
        }
    };

    thrust::counting_iterator<uint64_t, thrust::device_system_tag> iter(0);
    thrust::for_each(iter, iter + 512*leafCount, kernel);
}

inline float averageSurrounding(nanovdb::Coord coordenadas,nanovdb::FloatGrid *grid_source,nanovdb::FloatGrid *grid_destiny){
    float incrementos[] = {-1,0,1};
    int len_incrementos = 3;
    float accum = 0.0;
    nanovdb::Coord new_coord;
    float new_vec[3];
    auto vec = coordenadas.asVec3s();
    for(int i_incremento_x = 0;i_incremento_x<len_incrementos;i_incremento_x++){
        for(int i_incremento_y = 0 ;i_incremento_y<len_incrementos;i_incremento_y++){
            for(int i_incremento_z = 0 ;i_incremento_z<len_incrementos;i_incremento_z++){
                int incremento_x = incrementos[i_incremento_x];
                int incremento_y = incrementos[i_incremento_y];
                int incremento_z = incrementos[i_incremento_z];

                
                new_vec[0] = vec[0]+incremento_x;
                new_vec[1] = vec[1]+incremento_y;
                new_vec[2] = vec[2]+incremento_z;

                new_coord = nanovdb::Coord(new_vec[0],new_vec[1],new_vec[2]);
                accum += grid_source->tree().getValue(new_coord);
                
            }
        }
    }
    //std::cout<<vec<<std::endl;
    accum = accum /(len_incrementos * len_incrementos * len_incrementos);

    
    return accum;
}

void average(nanovdb::FloatGrid *grid_source,nanovdb::FloatGrid *grid_destiny,uint64_t leafCount){
    auto kernel = [grid_source,grid_destiny] __device__ (const uint64_t n) {
        auto *leaf_d = grid_source->tree().getFirstNode<0>() + (n >> 9);// this only works if grid->isSequential<0>() == true
        const int i = n & 511;//Es el tamanio por defecto de leafNodes
        auto *leaf_d_destiny = grid_destiny->tree().getFirstNode<0>() + (n >> 9);
        if (leaf_d->isActive(i)) {
            auto coord = leaf_d->offsetToGlobalCoord(i);
            float incrementos[] = {-1,0,1};
            int len_incrementos = 3;
            float accum = 0.0;
            nanovdb::Coord new_coord = nanovdb::Coord();
            //float new_vec[3];
            auto vec = coord.asVec3s();
            
            //printf("%d %f %f %f\n",i,vec[0],coord[1],vec[2]);
            auto acc = grid_source->tree().getAccessor();
            auto acc_destiny = grid_destiny->tree().getAccessor();
            int incrementadas = 0 ;
            for(int i_incremento_x = 0;i_incremento_x<len_incrementos;i_incremento_x++){
                for(int i_incremento_y = 0 ;i_incremento_y<len_incrementos;i_incremento_y++){
                    for(int i_incremento_z = 0 ;i_incremento_z<len_incrementos;i_incremento_z++){
                        int incremento_x = incrementos[i_incremento_x];
                        int incremento_y = incrementos[i_incremento_y];
                        int incremento_z = incrementos[i_incremento_z];

                        
                        // new_vec[0] = vec[0]+incremento_x;
                        // new_vec[1] = vec[1]+incremento_y;
                        // new_vec[2] = vec[2]+incremento_z;

                        if(acc.isActive(coord.offsetBy(incremento_x,incremento_y,incremento_z))){
                            float aux = acc.getValue(coord.offsetBy(incremento_x,incremento_y,incremento_z));
                            accum += aux;
                            incrementadas++;
                        }
                        
                        
                    }
                }
            }
            //std::cout<<vec<<std::endl;
            accum = accum /incrementadas;
            //printf("%f \n",accum);
            //printf("%f\n",accum);
            //acc.setValue(coord,accum);
            leaf_d_destiny->setValueOnly(i, accum);// only possible execution divergence
        }
    };
    thrust::counting_iterator<uint64_t, thrust::device_system_tag> iter(0);
    thrust::for_each(iter, iter + 512*leafCount, kernel);
}



