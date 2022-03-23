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
#include <nanovdb/util/Stencils.h>
/**
 * @brief Genera la estrucura inicial de las celulas endoteliales, cada cilindro tendra el tamanio de un leaf node (8x8x8)
 * 
 */
void generateEndothelial(nanovdb::FloatGrid *grid_d, uint64_t leafCount, int lim_sup,int lim_inf,int modulo)
{
    auto kernel = [grid_d, lim_sup,lim_inf,modulo] __device__ (const uint64_t n) {
        auto *leaf_d = grid_d->tree().getFirstNode<0>() + (n >> 9);// this only works if grid->isSequential<0>() == true
        const int i = n & 511;
        
        auto coord  = leaf_d->origin();
        //printf("%d %d %d\n",coord[0],coord[1],coord[2]);
        if(coord[1]>lim_inf && coord[1]<lim_sup){
            if(coord[0]%modulo == 0 && coord[2]%modulo == 0 ){
                leaf_d->setValueOnly(i,1.0);
            }
        }
        if(coord[1]==lim_inf || coord[1]==lim_sup){
            leaf_d->setValueOnly(i,1.0);
        }
        
    };

    thrust::counting_iterator<uint64_t, thrust::device_system_tag> iter(0);
    thrust::for_each(iter, iter + 512*leafCount, kernel);
}

/**
 * @brief Implementa la ecuacion en diferencias relacionada con el TAF 
 *  (Ecuacion 8)
 * 
 */
void equationTAF(nanovdb::FloatGrid* input_grid_endothelial,nanovdb::FloatGrid* input_grid_TAF,nanovdb::FloatGrid* output_grid_TAF,uint64_t leafCount){
    auto kernel = [input_grid_endothelial,input_grid_TAF,output_grid_TAF] __device__ (const uint64_t n){
        auto *leaf_d = output_grid_TAF->tree().getFirstNode<0>() + (n >> 9);// this only works if grid->isSequential<0>() == true
        const int i = n & 511;

        //Coordenadas del voxel globales
        auto coord = leaf_d->offsetToGlobalCoord(i);
        auto accessor_endothelial = input_grid_endothelial->tree().getAccessor();
        auto accessor_TAF_in = input_grid_TAF->tree().getAccessor();
        auto accessor_TAF_out = output_grid_TAF->tree().getAccessor();
        int incrementos_vecinos[] = {-1,0,1};
        int len_incrementos = 3;
        
        float n_i = 0;
        bool esVecino = false;
        //Se calcula n_i , que determina si se es vecino de una endothelial
        for(int i_incremento_x = 0;i_incremento_x<len_incrementos && !esVecino;i_incremento_x++){
            for(int i_incremento_y = 0 ;i_incremento_y<len_incrementos && !esVecino;i_incremento_y++){
                for(int i_incremento_z = 0 ;i_incremento_z<len_incrementos && !esVecino;i_incremento_z++){
                    int incremento_x = incrementos_vecinos[i_incremento_x];
                    int incremento_y = incrementos_vecinos[i_incremento_y];
                    int incremento_z = incrementos_vecinos[i_incremento_z];


                    if(accessor_endothelial.isActive(coord.offsetBy(incremento_x,incremento_y,incremento_z))){
                        n_i = accessor_endothelial.getValue(coord.offsetBy(incremento_x,incremento_y,incremento_z));
                        
                    }
                    esVecino = n_i != 0.0;//Esto igual esta feo
                    
                }
            }
        }

        float n_c = 0.025;
        //printf("%f\n",n_i);

        float c = accessor_TAF_in.getValue(coord);

        float new_c = -n_c * n_i * c;

        leaf_d->setValueOnly(i,new_c);

        

    };
    thrust::counting_iterator<uint64_t, thrust::device_system_tag> iter(0);
    thrust::for_each(iter, iter + 512*leafCount, kernel);
}

void pruebaGradiente(nanovdb::Vec3fGrid  *grid_d,nanovdb::CurvatureStencil<nanovdb::FloatGrid>* stencilNano ,uint64_t leafCount)
{
    auto kernel = [grid_d,stencilNano] __device__ (const uint64_t n) {
        auto *leaf_d = grid_d->tree().getFirstNode<0>() + (n >> 9);// this only works if grid->isSequential<0>() == true
        const int i = n & 511;
        
        auto coord = leaf_d->offsetToGlobalCoord(i);
        const nanovdb::Coord coord_nano = coord;
        
        //printf("%d %d %d\n",coord[0],coord[1],coord[2]);
        
        stencilNano->moveTo(coord_nano);
        leaf_d->setValueOnly(coord,stencilNano->gradient());
        
        
    };

    thrust::counting_iterator<uint64_t, thrust::device_system_tag> iter(0);
    thrust::for_each(iter, iter + 512*leafCount, kernel);
}
