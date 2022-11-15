#include <nanovdb/util/GridBuilder.h>
#include <nanovdb/NanoVDB.h> // this defined the core tree data structure of NanoVDB accessable on both the host and device
#include <stdio.h> // for //printf
#include <thrust/iterator/counting_iterator.h>
#include <thrust/for_each.h>

#include <nanovdb/util/Primitives.h>
#include <nanovdb/util/CudaDeviceBuffer.h>
#include "pruebaThrust.h"
#include <nanovdb/util/Stencils.h>
#include <thrust/random/uniform_real_distribution.h>
#include <thrust/random/linear_congruential_engine.h>
#include <thrust/device_vector.h>
#include <thrust/random.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>

__device__ float average(nanovdb::Coord coord,nanovdb::FloatGrid * input_grid,uint64_t n){
    auto accessor_endothelial = input_grid->tree().getAccessor();
    auto* leaf = input_grid->tree().getFirstNode<0>() + (n >> 9);
    int desplazamientos[] = {-6,-5.-4,-3,-2,-1,0,1,2,3,4,5,6};
    //int desplazamientos[] = {-1,0,1};
    float len_desp = 13.0;
    float n_i = 0;
    bool esVecino = false;
    float total = 0.0;
    float accum = 0.0;
    //Se calcula n_i , que determina si se es vecino de una endothelial
    //if(accessor_endothelial.getValue(coord)<threshold_vecino){
    ////printf("%d %d %d\n",coord[0],coord[1],coord[2]);
    

    
    for(int dimension = 0 ;dimension <3;dimension++){

            for(int desplazamiento = 0;desplazamiento<len_desp;desplazamiento++){
                nanovdb::Coord new_coord = coord;
                new_coord[dimension] += desplazamientos[desplazamiento];
                if(accessor_endothelial.isActive(new_coord)){
                    float value_i = accessor_endothelial.getValue(new_coord);
                    total = total + 1.0;
                    accum = accum + value_i;
                }
                
                ////printf("Position Self %d, new_coord %d %d %d value_i %f\n",positionSelf,new_coord[0],new_coord[1],new_coord[2],value_i);
                ////printf("Value i %f\n",value_i);
                
            }
    
        }
    
    if(total == 0.0){
        total = 1.0;
    }
    // if(accum > 0.0 ){
    //     //printf("accum: %f\n",accum);
    // }
    //return 1;
    return accum / total;
}

void average(nanovdb::FloatGrid* grid,nanovdb::FloatGrid* destiny,u_int64_t leafCount){
    auto kernel = [grid,destiny] __device__ (const uint64_t n) {
        auto *leaf = grid->tree().getFirstNode<0>() + (n >> 9);
        auto *leaf_destiny = destiny->tree().getFirstNode<0>() + (n >> 9);
        const int i = n & 511;
        
        auto coord = leaf->offsetToGlobalCoord(i);
        float new_value = average(coord,grid,n);
        
        leaf_destiny->setValue(coord,new_value);
    };
    thrust::counting_iterator<uint64_t, thrust::device_system_tag> iter(0);
    thrust::for_each(iter, iter + 512*leafCount, kernel);
}

void albedoTotal(nanovdb::FloatGrid* oxygen,nanovdb::FloatGrid*melanine,nanovdb::Vec3fGrid* albedo,u_int64_t leafCount){
    auto kernel = [oxygen,melanine,albedo] __device__ (const uint64_t n) {
        auto *leaf_Oxygen= oxygen->tree().getFirstNode<0>() + (n >> 9);
        auto *leaf_Melanine = melanine->tree().getFirstNode<0>() + (n >> 9);
        auto *leaf_albedo = albedo->tree().getFirstNode<0>() + (n >> 9 );
        const int i = n & 511;
        auto coord = leaf_Oxygen->offsetToGlobalCoord(i);
        float o2Concentration = leaf_Oxygen->getValue(i);
        float melanineConcentration = leaf_Melanine->getValue(i);
        float total = o2Concentration + melanineConcentration;
        if(total > 1.0){
            o2Concentration = o2Concentration  / total;
            melanineConcentration = melanineConcentration / total;
        }
        
        nanovdb::Vec3f base_color = {1,203.0/256.0,190.0/256.0};
        nanovdb::Vec3f melanine_color = {80.0/256.0,41.0/256.0,21.0/256.0}; //{230.0/256.0,191.0/256.0,170.0/256.0};
        nanovdb::Vec3f hemoglobin_color = {180.0/256.0,10/256.0,10.0/256.0};
        nanovdb::Vec3f new_albedo;
        for(int i = 0 ;i<3;i++){
            new_albedo[i] = base_color[i] * (1- o2Concentration - melanineConcentration) + melanine_color[i] * melanineConcentration + hemoglobin_color[i] * o2Concentration;
        }
        // for(int i = 0; i<3;i++){
        //     new_albedo[i] = 1.0 - new_albedo[i];
        // }
        leaf_albedo->setValueOnly(coord,new_albedo);

    };
    thrust::counting_iterator<uint64_t, thrust::device_system_tag> iter(0);
    thrust::for_each(iter, iter + 512*leafCount, kernel);
}
void copy(nanovdb::FloatGrid* source ,nanovdb::FloatGrid* destiny,u_int64_t leafCount){
    auto kernel = [source,destiny] __device__ (const uint64_t n) {
        auto* leaf_source = source->tree().getFirstNode<0>() + (n >> 9);
        auto* leaf_destiny = destiny->tree().getFirstNode<0>() + (n >> 9);
        const int i = n & 511;
        
        auto coord = leaf_source->offsetToGlobalCoord(i);
        leaf_destiny->setValue(coord,leaf_source->getValue(coord));
    };
    thrust::counting_iterator<uint64_t, thrust::device_system_tag> iter(0);
    thrust::for_each(iter, iter + 512*leafCount, kernel);
}