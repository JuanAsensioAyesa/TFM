// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: MPL-2.0
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


const float threshold_vecino = 0;
const float time_factor = 6; //Timestep de 6 minutos pasado a segundos
__device__ const float ini_endothelial = 1.0;

/**
 * @brief Genera la estrucura inicial de las celulas endoteliales, cada cilindro tendra el tamanio de un leaf node (8x8x8)
 * 
 */
void generateEndothelial(nanovdb::FloatGrid *grid_d, uint64_t leafCount, int lim_sup,int lim_inf,int modulo)
{
    auto kernel = [grid_d, lim_sup,lim_inf,modulo] __device__ (const uint64_t n) {
        auto *leaf_d = grid_d->tree().getFirstNode<0>() + (n >> 9);// this only works if grid->isSequential<0>() == true
        const int i = n & 511;
        
        auto coord_indi = leaf_d->offsetToGlobalCoord(i);
        auto coord = leaf_d->origin();
        coord = coord_indi;
        ////printf("%d %d %d\n",coord[0],coord[1],coord[2]);
        if(coord[1]>lim_inf && coord[1]<lim_sup){
            if(coord[0]%modulo == 0 && coord[2]%modulo == 0 ){
                leaf_d->setValueOnly(i,ini_endothelial);
            }else{
                leaf_d->setValueOnly(i,leaf_d->getValue(i));
            }
        }else{
            leaf_d->setValueOnly(i,leaf_d->getValue(i));
        }
        if(coord_indi[1]==lim_inf || coord_indi[1]==lim_sup){
            leaf_d->setValueOnly(i,ini_endothelial);
        }else{
            leaf_d->setValueOnly(i,leaf_d->getValue(i));
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
        auto *leaf_s = input_grid_TAF->tree().getFirstNode<0>() + (n >> 9);
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
        //if(accessor_endothelial.getValue(coord)<threshold_vecino){
            // for(int i_incremento_x = 0;i_incremento_x<len_incrementos && !esVecino;i_incremento_x++){
            //     for(int i_incremento_y = 0 ;i_incremento_y<len_incrementos && !esVecino;i_incremento_y++){
            //         for(int i_incremento_z = 0 ;i_incremento_z<len_incrementos && !esVecino;i_incremento_z++){
            //             int incremento_x = incrementos_vecinos[i_incremento_x];
            //             int incremento_y = incrementos_vecinos[i_incremento_y];
            //             int incremento_z = incrementos_vecinos[i_incremento_z];
    
    
            //             if(accessor_endothelial.isActive(coord.offsetBy(incremento_x,incremento_y,incremento_z))){
            //                 n_i = accessor_endothelial.getValue(coord.offsetBy(incremento_x,incremento_y,incremento_z));
                            
            //             }
            //             esVecino = n_i > threshold_vecino;//Esto igual esta feo
                        
            //         }
            //     }
            // }
        //}

        float n_c = 0.025;
        ////printf("%f\n",n_i);
        esVecino = accessor_endothelial.getValue(coord)==1.0;
        if(esVecino){
            ////printf("VECINO\n");
            n_i = 1.0;
        }else{
            n_i  = 0.0;
        }
        float c = leaf_s->getValue(i);
        
        
        float derivative = -n_c * n_i* c;
        // if(c > 0){
        //     //printf("%f %f %f\n",c,n_i,derivative);
        // }
        float old_c = leaf_s->getValue(i);
        auto new_value = old_c + derivative * time_factor;
        if(new_value < 0 ){
            new_value = 0 ;
        }
        if(new_value >1 ){
            new_value = 1;
        }
        
        leaf_d->setValueOnly(i,new_value);
        

        

    };
    thrust::counting_iterator<uint64_t, thrust::device_system_tag> iter(0);
    thrust::for_each(iter, iter + 512*leafCount, kernel);
}

void equationFibronectin(nanovdb::FloatGrid* input_grid_endothelial,nanovdb::FloatGrid* input_grid_Fibronectin,nanovdb::FloatGrid* input_grid_MDE,nanovdb::FloatGrid* output_grid_Fibronectin,uint64_t leafCount){
    auto kernel = [input_grid_endothelial,input_grid_Fibronectin,input_grid_MDE,output_grid_Fibronectin] __device__ (const uint64_t n) {
        auto *leaf_d = output_grid_Fibronectin->tree().getFirstNode<0>() + (n >> 9);// this only works if grid->isSequential<0>() == true
        auto *leaf_s = input_grid_Fibronectin->tree().getFirstNode<0>() + (n >> 9);
        auto *leaf_mde = input_grid_MDE->tree().getFirstNode<0>() + (n >> 9);
        const int i = n & 511;

        //Coordenadas del voxel globales
        auto coord = leaf_d->offsetToGlobalCoord(i);
        auto accessor_endothelial = input_grid_endothelial->tree().getAccessor();
        auto accessor_Fibronectin_in = input_grid_Fibronectin->tree().getAccessor();
        auto accessor_Fibronectin_out = output_grid_Fibronectin->tree().getAccessor();
        int incrementos_vecinos[] = {-1,0,1};
        int len_incrementos = 3;
        
        float n_i = 0;
        bool esVecino = false;
        //Se calcula n_i , que determina si se es vecino de una endothelial
        //if(accessor_endothelial.getValue(coord)<threshold_vecino){
            // for(int i_incremento_x = 0;i_incremento_x<len_incrementos && !esVecino;i_incremento_x++){
            //     for(int i_incremento_y = 0 ;i_incremento_y<len_incrementos && !esVecino;i_incremento_y++){
            //         for(int i_incremento_z = 0 ;i_incremento_z<len_incrementos && !esVecino;i_incremento_z++){
            //             int incremento_x = incrementos_vecinos[i_incremento_x];
            //             int incremento_y = incrementos_vecinos[i_incremento_y];
            //             int incremento_z = incrementos_vecinos[i_incremento_z];
    
    
            //             if(accessor_endothelial.isActive(coord.offsetBy(incremento_x,incremento_y,incremento_z))){
            //                 n_i = accessor_endothelial.getValue(coord.offsetBy(incremento_x,incremento_y,incremento_z));
                            
            //             }
            //             esVecino = n_i > threshold_vecino;//Esto igual esta feo
                        
            //         }
            //     }
            // }
        //}
        
        
        float production_rate = 0.0125;
        float degradation_rate = 0.1;
        esVecino = accessor_endothelial.getValue(coord) == 1.0;
        if(esVecino){
            n_i = 1.0;
        }else{
            n_i = 0.0;
        }
        
        float old_f = leaf_s->getValue(i);
        float old_mde = leaf_mde->getValue(i);

        float derivative = production_rate * n_i - degradation_rate * old_f * old_mde;
        // if(derivative > 0 ){
        //     //printf("%f\n",old_f + derivative * time_factor);
        // }
        auto new_value = old_f + derivative*time_factor;
        if(new_value < 0 ){
            new_value =  0;
        }
        if(new_value > 1){
            new_value = 1;
        }
        leaf_d->setValueOnly(i,new_value);
        //leaf_d->setValueOnly(i,n_i);

    };
    thrust::counting_iterator<uint64_t, thrust::device_system_tag> iter(0);
    thrust::for_each(iter, iter + 512*leafCount, kernel);
}

void equationMDE(nanovdb::FloatGrid* input_grid_endothelial,nanovdb::FloatGrid* input_grid_MDE,nanovdb::FloatGrid* output_grid_MDE,uint64_t leafCount){
    auto kernel = [input_grid_endothelial,input_grid_MDE,output_grid_MDE] __device__ (const uint64_t n) {
        auto *leaf_d = output_grid_MDE->tree().getFirstNode<0>() + (n >> 9);// this only works if grid->isSequential<0>() == true
        auto *leaf_s = input_grid_MDE->tree().getFirstNode<0>() + (n >> 9);
        const int i = n & 511;

        //Coordenadas del voxel globales
        auto coord = leaf_d->offsetToGlobalCoord(i);
        auto accessor_endothelial = input_grid_endothelial->tree().getAccessor();
        auto accessor_Fibronectin_in = input_grid_MDE->tree().getAccessor();
        auto accessor_Fibronectin_out = output_grid_MDE->tree().getAccessor();
        int incrementos_vecinos[] = {-1,0,1};
        int len_incrementos = 3;
        
        float n_i = 0;
        bool esVecino = false;
        // //Se calcula n_i , que determina si se es vecino de una endothelial
        //if(accessor_endothelial.getValue(coord)<threshold_vecino){
            // for(int i_incremento_x = 0;i_incremento_x<len_incrementos && !esVecino;i_incremento_x++){
            //     for(int i_incremento_y = 0 ;i_incremento_y<len_incrementos && !esVecino;i_incremento_y++){
            //         for(int i_incremento_z = 0 ;i_incremento_z<len_incrementos && !esVecino;i_incremento_z++){
            //             int incremento_x = incrementos_vecinos[i_incremento_x];
            //             int incremento_y = incrementos_vecinos[i_incremento_y];
            //             int incremento_z = incrementos_vecinos[i_incremento_z];
    
    
            //             if(accessor_endothelial.isActive(coord.offsetBy(incremento_x,incremento_y,incremento_z))){
            //                 n_i = accessor_endothelial.getValue(coord.offsetBy(incremento_x,incremento_y,incremento_z));
                            
            //             }
            //             esVecino = n_i > threshold_vecino;//Esto igual esta feo
                        
            //         }
            //     }
            // }
        //}
        //esVecino = accessor_endothelial.getValue(coord)==1.0;
        float production_rate = 0.0000015;
        
        float diffussion_coefficient = 0.0025;
        float degradation_rate = 0.75;
        ////printf("%f %f %f\n",production_rate,diffussion_coefficient,degradation_rate);
        nanovdb::CurvatureStencil<nanovdb::FloatGrid> stencilNano(*input_grid_MDE);
        stencilNano.moveTo(coord);
        float laplacian = stencilNano.laplacian() ;
        esVecino = accessor_endothelial.getValue(coord)==1.0;
        if(esVecino){
            n_i = 1.0;
        }else{
            n_i = 0.0;
        }
        float old_mde = leaf_s->getValue(i);
        float derivative = n_i * production_rate + diffussion_coefficient * laplacian * old_mde - degradation_rate * old_mde;
        derivative *= 1e-9;
        // float factor_1 = n_i * production_rate;
        // float factor_2 = diffussion_coefficient * laplacian * old_mde;
        // float factor_3 = degradation_rate * old_mde;
        // if(laplacian!=0){
        //     //printf("%f\n",laplacian);
        // }
        
        // if(factor_1 != 0|| factor_2 != 0 || factor_3!=0){
        //     //printf("%f %f %f\n",n_i * production_rate,diffussion_coefficient*laplacian*old_mde,degradation_rate*old_mde);

        // }
        // if(derivative > 0 && factor_1 > 0.000002){
        //     //printf("%f %f %f\n",factor_1,factor_2,factor_3);
        // }
        //float derivative = diffussion_coefficient * laplacian;
        auto new_value = old_mde + derivative * time_factor;
        if(new_value < 0 ){
            new_value = 0;
        }
        if(new_value >1 ){
            new_value = 1;
        }
        leaf_d->setValueOnly(i,new_value);
        //leaf_d->setValueOnly(i,n_i);

    };
    thrust::counting_iterator<uint64_t, thrust::device_system_tag> iter(0);
    thrust::for_each(iter, iter + 512*leafCount, kernel);
}

void pruebaGradiente(nanovdb::Vec3fGrid  *grid_d,nanovdb::FloatGrid* gridSource ,uint64_t leafCount)
{
    auto kernel = [grid_d,gridSource] __device__ (const uint64_t n) {
        auto *leaf_d = grid_d->tree().getFirstNode<0>() + (n >> 9);// this only works if grid->isSequential<0>() == true
        auto *leaf_s = gridSource->tree().getFirstNode<0>() + (n >> 9);
        const int i = n & 511;
        
        auto coord = leaf_d->offsetToGlobalCoord(i);
        const nanovdb::Coord coord_nano = coord;
        ////printf("%d %d %d\n",coord[0],coord[1],coord[2]);
        nanovdb::CurvatureStencil<nanovdb::FloatGrid> stencilNano(*gridSource);
        
        auto accessor = gridSource->tree().getAccessor();
        
        stencilNano.moveTo(coord_nano);
        auto gradiente = stencilNano.gradient();
        // s
        // auto suma = gradiente[0]+gradiente[1]+gradiente[2];
        // if(suma!=0){
        //     nanovdb::Coord coord_aux = coord;
        //     coord_aux[0] = 0 ;
        //     coord_aux[1] = 0 ;
        //     coord_aux[2] = 0 ;
        //     ////printf("%f %f\n",suma,leaf_s->getValue(coord));
        //     //printf("%f \n",accessor.getValue(coord));
        // }
        leaf_d->setValueOnly(coord,gradiente);
        
        
    };

    thrust::counting_iterator<uint64_t, thrust::device_system_tag> iter(0);
    thrust::for_each(iter, iter + 512*leafCount, kernel);
}


__device__ float chemotacticSensivity(float c){
    float chemotacticMigration = 0.38;
    float chemotacticConstant = 0.6;
    return chemotacticMigration /(1 + chemotacticConstant*c);
}
__device__ int getPosition(nanovdb::Coord coord_self,nanovdb::FloatGrid * endothelialTip){
    auto accessor = endothelialTip->tree().getAccessor();
    nanovdb::Coord coord_i = coord_self;

    coord_i[2] = coord_i[2] - 1; //Comprobamos frente
    if(accessor.getValue(coord_i) != 0 ){
        return 6;//coord_self esta a la dch de la celula endothelial
    }
    coord_i[2] = coord_self[2];

    coord_i[2] = coord_i[2] + 1; //Comprobamos frente
    if(accessor.getValue(coord_i) != 0 ){
        return 5;//coord_self esta a la dch de la celula endothelial
    }
    coord_i[2] = coord_self[2];

    coord_i[1] = coord_i[1] +1; //Comprobamos arriba;
    if(accessor.getValue(coord_i) != 0 ){
        return 3;//coord_self esta debajo de la celula endothelial
    }
    coord_i[1] = coord_self[1];

    coord_i[0] = coord_i[0] +1; //Comprobamos derecha
    if(accessor.getValue(coord_i) != 0 ){
        return 4;//coord_self esta a la izq de la celula endothelial
    }
    coord_i[0] = coord_self[0];

    coord_i[1] = coord_i[1] - 1; //Comprobamos abajo
    if(accessor.getValue(coord_i) != 0 ){
        return 1;//coord_self esta arriba de la celula endotelial
    }
    coord_i[1] = coord_self[1];

    coord_i[0] = coord_i[0] - 1; //Comprobamos izquierda
    if(accessor.getValue(coord_i) != 0 ){
        return 2;//coord_self esta a la dch de la celula endothelial
    }
    coord_i[0] = coord_self[0];

    
    return 0;


}
__device__ bool  isNextToEndothelial(nanovdb::Coord coord,nanovdb::FloatGrid * input_grid_endothelial){
    auto accessor_endothelial = input_grid_endothelial->tree().getAccessor();
    int incrementos_vecinos[] = {-1,0,1};
    int len_incrementos = 3;
    float n_i = 0;
    bool esVecino = false;
    //Se calcula n_i , que determina si se es vecino de una endothelial
    //if(accessor_endothelial.getValue(coord)<threshold_vecino){
    ////printf("%d %d %d\n",coord[0],coord[1],coord[2]);
    for(int i_incremento_x = 0;i_incremento_x<len_incrementos && !esVecino;i_incremento_x++){
        for(int i_incremento_y = 0 ;i_incremento_y<len_incrementos && !esVecino;i_incremento_y++){
            for(int i_incremento_z = 0 ;i_incremento_z<len_incrementos && !esVecino;i_incremento_z++){
                int incremento_x = incrementos_vecinos[i_incremento_x];
                int incremento_y = incrementos_vecinos[i_incremento_y];
                int incremento_z = incrementos_vecinos[i_incremento_z];


                if(accessor_endothelial.isActive(coord.offsetBy(incremento_x,incremento_y,incremento_z))){
                    n_i = accessor_endothelial.getValue(coord.offsetBy(incremento_x,incremento_y,incremento_z));
                    esVecino = n_i > threshold_vecino;//Esto igual esta feo
                }else{
                    esVecino = false;
                }
                
                
            }
        }
    }
    return esVecino;
}
__device__ float average(nanovdb::Coord coord,nanovdb::FloatGrid * input_grid,uint64_t n){
    auto accessor_endothelial = input_grid->tree().getAccessor();
    auto* leaf = input_grid->tree().getFirstNode<0>() + (n >> 9);
    //int desplazamientos[] = {-4,-3,-2,-1,0,1,2,3,4};
    int desplazamientos[] = {-1,0,1};
    int len_desp = 3;
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
__device__ bool isNextToEndothelialDiscrete(nanovdb::Coord coord,nanovdb::FloatGrid * input_grid_endothelial){
    auto accessor_endothelial = input_grid_endothelial->tree().getAccessor();
    int desplazamientos[] = {-1,1};
    int len_desp = 2;
    
    //int desplazamiento_max[3];
    
    
    //nanovdb::Coord coord_max = coord;
    //float max_derivative = computeEndothelial(coord,stencilEndothelial,stencilTAF,stencilFibronectin);

    //Se calcula el maximo
    bool esVecino = false;
    for(int dimension = 0 ;dimension <3&&!esVecino;dimension++){

        for(int desplazamiento = 0;desplazamiento<len_desp&&!esVecino;desplazamiento++){
            nanovdb::Coord new_coord = coord;
            new_coord[dimension] += desplazamientos[desplazamiento];
            float value_i = accessor_endothelial.getValue(new_coord);
            esVecino = value_i > threshold_vecino;
        }
    }
    return esVecino;
}
/*
    Ecuacion 6
*/
void equationEndothelial(nanovdb::FloatGrid * grid_s,nanovdb::FloatGrid * grid_d,nanovdb::FloatGrid* gridTAF,nanovdb::FloatGrid* gridFibronectin,nanovdb::Vec3fGrid* gradientTAF,nanovdb::Vec3fGrid* gradientFibronectin,nanovdb::FloatGrid* gridTip,uint64_t leafCount){
    auto kernel = [grid_s,grid_d,gridTAF,gridFibronectin,gradientTAF,gradientFibronectin,gridTip] __device__ (const uint64_t n) {
        auto *leaf_d = grid_d->tree().getFirstNode<0>() + (n >> 9);// this only works if grid->isSequential<0>() == true
        auto *leaf_s = grid_s->tree().getFirstNode<0>() + (n >> 9);// this only works if grid->isSequential<0>() == true
        auto *leaf_taf = gridTAF->tree().getFirstNode<0>() + (n >> 9);
        const int i = n & 511;
        
        auto coord = leaf_d->offsetToGlobalCoord(i);
        const nanovdb::Coord coord_nano = coord;
        /*
            Primera parte: Difusion aleatoria
        */
        nanovdb::CurvatureStencil<nanovdb::FloatGrid> stencilNano(*grid_s);
        stencilNano.moveTo(coord_nano);
        float old_n = leaf_s->getValue(coord_nano);
        float laplacian = stencilNano.laplacian();
        ////printf("%f\n",laplacian);
        float factorEndothelial = laplacian * 0.0003  ;
        /*
            Segunda parte, chimiotaxis TAF
        */
        nanovdb::CurvatureStencil<nanovdb::Vec3fGrid> stencilTAF(*gradientTAF);
        
        stencilTAF.moveTo(coord_nano);
        auto gradientTAF = stencilTAF.gradient();
        float taf_value = leaf_taf->getValue(coord_nano);
        for(int index = 0 ;index <3;index++){
             gradientTAF[index] *= chemotacticSensivity(taf_value);
             gradientTAF[index] *= old_n;
        }
        float factorTAF = gradientTAF[0][0] + gradientTAF[1][1] + gradientTAF[2][2];
        
        //factorTAF *=10;
        ////printf("%f  %f  %f\n",gradientTAF[0][0],gradientTAF[1][1],gradientTAF[2][2]);
        
        //float factorTAF  = stencilNano.gaussianCurvature() ;
        /*
            Tercera parte, Fibronectin
        */
        nanovdb::CurvatureStencil<nanovdb::Vec3fGrid> stencilFibronectin(*gradientFibronectin);
        stencilFibronectin.moveTo(coord_nano);
        auto gradientFibronectin = stencilFibronectin.gradient();
        float factorFibronectin = gradientFibronectin[0][0] + gradientFibronectin[1][1] + gradientFibronectin[2][2];
        factorFibronectin = factorFibronectin * 0.28;
        
        ////printf("%f %f %f\n",factorEndothelial,factorTAF,factorFibronectin);

        //float derivative = factorEndothelial  + factorTAF ;//+ factorFibronectin;
        float derivative = factorEndothelial - factorTAF - factorFibronectin;
        
        // if(derivative > 0 ){
        //     //printf("%f\n",derivative);
        // }
        // if(derivative > 100){
        //     //printf("%f %f %f\n",factorEndothelial,factorTAF,factorFibronectin);
        // }
        //float derivative = -factorTAF;
        int positionSelf = getPosition(coord_nano,gridTip);
        auto new_value = old_n + derivative * time_factor;
        if(isNextToEndothelialDiscrete(coord_nano,gridTip)){
            if(derivative != 0){
                ////printf("endo:%f taf:%f fibro:%f new:%f positionSelf:%d coord: %d %d %d\n",factorEndothelial,factorTAF,factorFibronectin,new_value,positionSelf,coord_nano[0],coord_nano[1],coord_nano[2]);
            }
        }
        
        if(new_value < 0 ){
            new_value = 0 ;
        }
        if(new_value > 1){
            new_value = 1;
        }
        
        leaf_d->setValueOnly(coord_nano,new_value);//6 minutos //* 60 segundos
        //leaf_d->setValueOnly(coord_nano,derivative);//6 minutos //* 60 segundos

    };
    thrust::counting_iterator<uint64_t, thrust::device_system_tag> iter(0);
    thrust::for_each(iter, iter + 512*leafCount, kernel);
}
void factorEndothelial(nanovdb::FloatGrid * grid_s,nanovdb::FloatGrid * grid_d,uint64_t leafCount){
    auto kernel = [grid_s,grid_d] __device__ (const uint64_t n) {
        auto *leaf_d = grid_d->tree().getFirstNode<0>() + (n >> 9);// this only works if grid->isSequential<0>() == true
        auto *leaf_s = grid_s->tree().getFirstNode<0>() + (n >> 9);// this only works if grid->isSequential<0>() == true
        
        const int i = n & 511;
              
        auto coord = leaf_d->offsetToGlobalCoord(i);
        auto coord_s = leaf_s->offsetToGlobalCoord(i);
        
        nanovdb::CurvatureStencil<nanovdb::FloatGrid> stencilNano(*grid_s);
        stencilNano.moveTo(coord_s);
        float old_n = leaf_s->getValue(coord);
        float laplacian = stencilNano.laplacian();
        ////printf("%f\n",laplacian);
        float factorEndothelial = laplacian * 0.0003 * 1e-4;
        float new_value = old_n+factorEndothelial * time_factor;
        if(new_value > 1){
            new_value = 1;
        }
        if(new_value  <0 ){
            new_value = 0;
        }

        leaf_d->setValueOnly(coord,new_value);
    };
    thrust::counting_iterator<uint64_t, thrust::device_system_tag> iter(0);
    thrust::for_each(iter, iter + 512*leafCount, kernel);
}
void factorTAF(nanovdb::FloatGrid * grid_s,nanovdb::FloatGrid * grid_d,nanovdb::FloatGrid* gridTAF,nanovdb::Vec3fGrid* gradientTAF,uint64_t leafCount){
    auto kernel = [grid_s,grid_d,gridTAF,gradientTAF] __device__ (const uint64_t n) {
        auto *leaf_d = grid_d->tree().getFirstNode<0>() + (n >> 9);// this only works if grid->isSequential<0>() == true
        auto *leaf_s = grid_s->tree().getFirstNode<0>() + (n >> 9);// this only works if grid->isSequential<0>() == true
        auto *leaf_taf = gridTAF->tree().getFirstNode<0>() + (n >> 9);
        const int i = n & 511;
        
        auto coord = leaf_d->offsetToGlobalCoord(i);

        nanovdb::CurvatureStencil<nanovdb::Vec3fGrid> stencilTAF(*gradientTAF);
        float old_n = leaf_s->getValue(coord);
        float current_n = leaf_d->getValue(coord);
        stencilTAF.moveTo(coord);
        auto gradientTAF = stencilTAF.gradient();
        float taf_value = leaf_taf->getValue(coord);
        for(int index = 0 ;index <3;index++){
             gradientTAF[index] *= chemotacticSensivity(taf_value);
             gradientTAF[index] *= old_n;
        }
        float factorTAF = gradientTAF[0][0] + gradientTAF[1][1] + gradientTAF[2][2];

        float new_n = current_n - factorTAF * time_factor;
        if(new_n > 1){
            new_n = 1;
        }
        if(new_n < 0 ){
            new_n = 0;
        }
        leaf_d->setValueOnly(coord,new_n);
        

    };
    thrust::counting_iterator<uint64_t, thrust::device_system_tag> iter(0);
    thrust::for_each(iter, iter + 512*leafCount, kernel);
}
void factorFibronectin(nanovdb::FloatGrid * grid_s,nanovdb::FloatGrid * grid_d,nanovdb::FloatGrid* gridFibronectin,nanovdb::Vec3fGrid* gradientFibronectin,uint64_t leafCount){
    auto kernel = [grid_s,grid_d,gridFibronectin,gradientFibronectin] __device__ (const uint64_t n) {
        auto *leaf_d = grid_d->tree().getFirstNode<0>() + (n >> 9);// this only works if grid->isSequential<0>() == true
        auto *leaf_s = grid_s->tree().getFirstNode<0>() + (n >> 9);// this only works if grid->isSequential<0>() == true
        auto *leaf_Fibronectin = gridFibronectin->tree().getFirstNode<0>() + (n >> 9);
        const int i = n & 511;
        
        auto coord = leaf_d->offsetToGlobalCoord(i);
        float current_n = leaf_d->getValue(coord);
        nanovdb::CurvatureStencil<nanovdb::Vec3fGrid> stencilFibronectin(*gradientFibronectin);
        stencilFibronectin.moveTo(coord);
        auto gradientFibronectin = stencilFibronectin.gradient();
        float factorFibronectin = gradientFibronectin[0][0] + gradientFibronectin[1][1] + gradientFibronectin[2][2];
        factorFibronectin = factorFibronectin * 0.28;
        float new_value = current_n-factorFibronectin*time_factor;
        if(new_value < 0 ){
            new_value = 0;
        }
        if(new_value > 1){
            new_value = 1;
        }
        leaf_d->setValueOnly(coord,new_value);

    };
    thrust::counting_iterator<uint64_t, thrust::device_system_tag> iter(0);
    thrust::for_each(iter, iter + 512*leafCount, kernel);
}
__device__ float computeEndothelial(nanovdb::Coord coord_nano,nanovdb::CurvatureStencil<nanovdb::FloatGrid>& stencilEndothelial,nanovdb::CurvatureStencil<nanovdb::Vec3fGrid> &stencilTAF,nanovdb::CurvatureStencil<nanovdb::Vec3fGrid>& stencilFibronectin){
    /*
    Primera parte: Difusion aleatoria
    */
    
    stencilEndothelial.moveTo(coord_nano);
    //float old_n = leaf_s->getValue(coord_nano);
    float laplacian = stencilEndothelial.laplacian();
    ////printf("%f\n",laplacian);
    float factorEndothelial = laplacian * 0.0003 ;

    /*
    Segunda parte, chimiotaxis TAF
    */
    
    
    stencilTAF.moveTo(coord_nano);
    auto gradientTAF = stencilTAF.gradient();
    float factorTAF = gradientTAF[0][0] + gradientTAF[1][1] + gradientTAF[2][2];
    /*
    Tercera parte, Fibronectin
    */
    
    stencilFibronectin.moveTo(coord_nano);
    auto gradientFibronectin = stencilFibronectin.gradient();
    float factorFibronectin = gradientFibronectin[0][0] + gradientFibronectin[1][1] + gradientFibronectin[2][2];
    factorFibronectin = factorFibronectin * 0.28;
    
    ////printf("%f %f\n",factorTAF,factorFibronectin);

    float derivative = factorEndothelial  - factorTAF - factorFibronectin;
    //derivative = -factorTAF-factorFibronectin;
    derivative = -factorTAF;
    return derivative;
}


__device__ bool isMax(nanovdb::Coord coord_self,nanovdb::FloatGrid * endothelialTip,nanovdb::FloatGrid * endothelial,nanovdb::FloatGrid* endothelialDiscrete){
    auto accessor_endothelial = endothelial->tree().getAccessor();
    auto accessor_discrete = endothelialDiscrete->tree().getAccessor();
    int positionSelf = getPosition(coord_self,endothelialTip);
    ////printf("%d\n",positionSelf);
    nanovdb::Coord coord_endothelial = coord_self;
    switch(positionSelf){
        case 1:
            coord_endothelial[1] -=1;
            break;
        case 2:
            coord_endothelial[0] -= 1;
            break;
        case 3:
            coord_endothelial[1] += 1;
            break;
        case 4:
            coord_endothelial[0] += 1;
            break;
        case 5:
            coord_endothelial[2] += 1;
            break;
        case 6:
            coord_endothelial[2] -= 1;
            break;
        default:
            break;

    };
    int desplazamientos[] = {-1,1};
    int len_desp = 2;
    
    nanovdb::Coord coord_max;
    float value_max = -1;
    if(positionSelf != 0 ){
        for(int dimension = 0 ;dimension <3;dimension++){

            for(int desplazamiento = 0;desplazamiento<len_desp;desplazamiento++){
                nanovdb::Coord new_coord = coord_endothelial;
                new_coord[dimension] += desplazamientos[desplazamiento];
                float value_i = accessor_endothelial.getValue(new_coord);
                ////printf("Position Self %d, new_coord %d %d %d value_i %f\n",positionSelf,new_coord[0],new_coord[1],new_coord[2],value_i);
                ////printf("Value i %f\n",value_i);
                if(accessor_discrete.getValue(new_coord) == 0.0 && value_i > value_max){
                    value_max = value_i;
                    coord_max = new_coord;
                }
                if(positionSelf == 1){
                    ////printf("desplazamiento %d dimension %d discrete %f value_i %f value_max %f new coord %d %d %d\n",
                    //desplazamiento,dimension,accessor_discrete.getValue(new_coord),value_i,value_max,new_coord[0],new_coord[1],new_coord[2]);
                }
            }
     
        }
    }
    float endothelialSelf = accessor_endothelial.getValue(coord_self);
    ////printf("PositionSelf %d, coord_self %d %d %d Coord max %d %d %d value Self %f\n",positionSelf,coord_self[0],coord_self[1],coord_self[2],coord_max[0],coord_max[1],coord_max[2],endothelialSelf);
    ////printf("Value_max %f\n",value_max);
    //return positionSelf == 4;
    return positionSelf != 0 && coord_max == coord_self;


}

__device__ void moveRandom(nanovdb::Coord coord_self,nanovdb::FloatGrid* gridEndothelial,nanovdb::FloatGrid* gridEndothelialDiscrete,nanovdb::FloatGrid* gridTip,float randomValue,int n ){
    auto accessor_endothelial = gridEndothelial->tree().getAccessor();
    auto accessor_discrete = gridEndothelialDiscrete->tree().getAccessor();
    auto accessor_tip = gridTip->tree().getAccessor();
    auto *leaf_tip = gridTip->tree().getFirstNode<0>()+(n>>9);
    auto *leaf_discrete = gridEndothelialDiscrete->tree().getFirstNode<0>()+(n>>9);
    int positionSelf = getPosition(coord_self,gridTip);
    ////printf("%d\n",positionSelf);
    nanovdb::Coord coord_endothelial = coord_self;
    switch(positionSelf){
        case 1:
            coord_endothelial[1] -=1;
            break;
        case 2:
            coord_endothelial[0] -= 1;
            break;
        case 3:
            coord_endothelial[1] += 1;
            break;
        case 4:
            coord_endothelial[0] += 1;
            break;
        case 5:
            coord_endothelial[2] += 1;
            break;
        case 6:
            coord_endothelial[2] -= 1;
            break;
        default:
            break;

    };
    if(positionSelf == 0 ){
        return;
    }


    int desplazamientos[] = {-1,1};
    const int len_desp = 2;
    
    nanovdb::Coord coord_max;
    float value_accum = 0.0;
    const int length = 3 * len_desp;
    float values[length];
    nanovdb::Coord coords[length];
    int i_value = 0;
    if(positionSelf != 0 ){
        for(int dimension = 0 ;dimension <3;dimension++){

            for(int desplazamiento = 0;desplazamiento<len_desp;desplazamiento++){
                nanovdb::Coord new_coord = coord_endothelial;
                new_coord[dimension] += desplazamientos[desplazamiento];
                float value_i = accessor_endothelial.getValue(new_coord);
                ////printf("Position Self %d, new_coord %d %d %d value_i %f\n",positionSelf,new_coord[0],new_coord[1],new_coord[2],value_i);
                value_accum += value_i;
                values[i_value] = value_i;
                coords[i_value] = new_coord;
                i_value++;
            }
        }
    }
    // for(int i = 0 ;i<length;i++){
    //     values[i] = 1-values[i] / value_accum;
    // } 
    thrust::sort(thrust::device, values, values + length);
    bool decided = false;
    for(int i = 0 ;i<length && !decided;i++){
        if(coords[i]==coord_self && randomValue>values[i]){
            leaf_tip->setValue(coords[i],2);
            leaf_discrete->setValue(coords[i],1);
            decided = true;
        }
    }


}

void equationEndothelialDiscrete(nanovdb::FloatGrid * grid_source_discrete,nanovdb::FloatGrid * grid_destiny_discrete,nanovdb::FloatGrid* gridDerivativeEndothelial,nanovdb::FloatGrid* gridDerivativeEndothelialWrite,nanovdb::FloatGrid* gridTAF,nanovdb::FloatGrid * gridTipRead,nanovdb::FloatGrid* gridTipWrite,int seed,uint64_t leafCount){
    
    auto kernel = [grid_source_discrete,grid_destiny_discrete,gridDerivativeEndothelial,gridDerivativeEndothelialWrite,gridTAF,gridTipRead,gridTipWrite,seed] __device__ (const uint64_t n) {
        auto *leaf_d = grid_destiny_discrete->tree().getFirstNode<0>() + (n >> 9);// this only works if grid->isSequential<0>() == true
        auto *leaf_s = grid_source_discrete->tree().getFirstNode<0>() + (n >> 9);// this only works if grid->isSequential<0>() == true
        auto *leaf_tip_write = gridTipWrite->tree().getFirstNode<0>()+(n>>9);
        auto *leaf_tip_read = gridTipRead->tree().getFirstNode<0>()+(n>>9);
        auto *leaf_endothelial_write = gridDerivativeEndothelialWrite->tree().getFirstNode<0>()+(n>>9);
        auto *leaf_endothelial_read = gridDerivativeEndothelial->tree().getFirstNode<0>()+(n>>9);
        //auto *leaf_TAF = gridTAF->tree().getFirstNode<0>() + (n >> 9);
        const int i = n & 511;

        

        auto coord = leaf_tip_write->offsetToGlobalCoord(i);
        auto coord_d = leaf_d->offsetToGlobalCoord(i);
        //float taf_value = leaf_TAF->getValue(coord);

        // if(coord[0] == 0&&coord[1]==0 && coord[2]==0){
        //     //printf("RAndom %f\n",random);

        // }        

        //leaf_d->setValue(coord,random);

        // nanovdb::Coord coord_dummy;
        // coord_dummy[0] = 0 ;
        // coord_dummy[1] =0 ;
        // coord_dummy[2] = 0 ;
        //leaf_d->setValueOnly(coord_dummy,100);
        // if(leaf_tip_read->getValue(i)>0){
        //     //printf("TIP\n");
        // }
       // static int first = true;
       if(isNextToEndothelialDiscrete(coord_d,gridTipRead)){
        //if(leaf_tip->getValue(i)>0){
            //if((coord_d[1]-1)%2 == 0){
            //if(coord_d[1]%2 == 0 ){
            int positionSelf = getPosition(coord_d,gridTipRead);
            //moveRandom(coord,gridDerivativeEndothelial,grid_destiny_discrete,gridTipWrite,random,n);
            //leaf_d->setValue(coord_d,1.0);
            auto coord_left = coord_d;
            coord_left[0]-=1;
            float tip_left = leaf_tip_read->getValue(coord_left);
            ////printf("Next, endothelial: %f ,position:%d , tip left:%f\n",leaf_endothelial_read->getValue(coord_d),positionSelf,tip_left);

            if(isMax(coord_d,gridTipRead,gridDerivativeEndothelial,grid_source_discrete)){
                if(leaf_s->getValue(coord)>0.0){
                    //printf("WTF\n");
                }

                //printf("Is max %d \n",positionSelf);
                leaf_d->setValue(coord_d,1.0);
                leaf_tip_write->setValue(coord_d,2.0);
                //leaf_endothelial_write->setValue(coord_d,1.0);
                
            
            }
            
            // leaf_d->setValue(coord_d,1.0);
            // //leaf_s->setValue(coord_d,1.0);
            // if(leaf_s->getValue(coord_d)==0){
            //     leaf_tip_write->setValue(coord_d,2.0);
            // }
            //leaf_tip->setValue(coord_d,0);
            //coord_d[0]+=1;
            //leaf_tip->setValue(coord_d,1);

            //FALTA EL BRANCHING
           
        }else{
            float value = leaf_s->getValue(i);
            leaf_d->setValue(coord_d,value);
        }
        // }else if(false&&isNextToEndothelialDiscrete(coord_d,grid_source_discrete)){
        //     //first = false;
        //     if(taf_value >= 0.8 && random >= 1.0-vector_probabilidades[3]){
        //         ////printf("NEW TIP\n");
        //         leaf_tip_write->setValueOnly(coord,1.0);
        //         leaf_d->setValueOnly(coord,1.0);
        //     }else if(taf_value >=0.7 && random >= 1-vector_probabilidades[2] ){
        //         ////printf("NEW TIP\n");
        //         leaf_tip_write->setValueOnly(coord,1.0);
        //         leaf_d->setValueOnly(coord,1.0);
        //     }else if(taf_value >= 0.5&& random >= 1-vector_probabilidades[1]){
        //         ////printf("NEW TIP\n");
        //         leaf_tip_write->setValueOnly(coord,1.0);
        //         leaf_d->setValueOnly(coord,1.0);
        //     }else if(taf_value >=0.3&& random >= 1-vector_probabilidades[0]){
        //         ////printf("NEW TIP\n");
        //         leaf_tip_write->setValueOnly(coord,1.0);
        //         leaf_d->setValueOnly(coord,1.0);
        //     }else{
        //         //NO hay branch
        //         //leaf_tip->setValueOnly(coord,0.0);
        //     }
        //     //leaf_d->setValueOnly(coord,1.0);
        // }else{
        //     leaf_tip_write->setValue(coord_d,0);
        // }
        
       
        //leaf_d->setValueOnly(i,0.0);
        
        
    
    
    
    
    };
    thrust::counting_iterator<uint64_t, thrust::device_system_tag> iter(0);
    thrust::for_each(iter, iter + 512*leafCount, kernel);
}
void newTips(nanovdb::FloatGrid* gridEndothelialTip,nanovdb::FloatGrid* gridEndothelialDiscrete,nanovdb::FloatGrid* gridTAF,int seed,int leafCount){
    auto kernel = [gridEndothelialTip,gridTAF,gridEndothelialDiscrete,seed] __device__ (const uint64_t n) {
        auto* leaf_tip = gridEndothelialTip->tree().getFirstNode<0>() + (n >> 9);
        auto* leaf_taf = gridTAF->tree().getFirstNode<0>() + (n >> 9);
        auto* leaf_discrete = gridEndothelialDiscrete->tree().getFirstNode<0>() + (n >> 9);
        const int i = n & 511;

        thrust::minstd_rand rng;
        thrust::default_random_engine randEng;
        thrust::uniform_real_distribution<float> uniDist;
        int discard = seed+n;
        randEng.discard(discard);
        float random = uniDist(randEng);

        auto coord_d = leaf_tip->offsetToGlobalCoord(i);
        auto coord = coord_d;
        float value = leaf_tip->getValue(i);
        float taf_value = leaf_taf->getValue(i);
        float new_value = 1;
        float vector_probabilidades[] = {0.02,0.03,0.04,0.08};

        if(isNextToEndothelialDiscrete(coord_d,gridEndothelialDiscrete)){
            if(taf_value >= 0.8 && random >= 1.0-vector_probabilidades[3]){
                //printf("NEW TIP\n");
                 leaf_tip->setValueOnly(coord,1.0);
                 leaf_discrete->setValueOnly(coord,1.0);
                 //leaf_d->setValueOnly(coord,1.0);
            }else if(taf_value >=0.7 && random >= 1-vector_probabilidades[2] ){
                //printf("NEW TIP\n");
                 leaf_tip->setValueOnly(coord,1.0);
                 leaf_discrete->setValueOnly(coord,1.0);
                 //leaf_d->setValueOnly(coord,1.0);
            }
        }
    };
    thrust::counting_iterator<uint64_t, thrust::device_system_tag> iter(0);
    thrust::for_each(iter, iter + 512*leafCount, kernel);
}

void branching(nanovdb::FloatGrid* gridEndothelialTip,nanovdb::FloatGrid* gridTAF,int seed,int leafCount){
    auto kernel = [gridEndothelialTip,gridTAF,seed] __device__ (const uint64_t n) {
        auto* leaf_tip = gridEndothelialTip->tree().getFirstNode<0>() + (n >> 9);
        auto* leaf_taf = gridTAF->tree().getFirstNode<0>() + (n >> 9);
        const int i = n & 511;

        thrust::minstd_rand rng;
        thrust::default_random_engine randEng;
        thrust::uniform_real_distribution<float> uniDist;
        int discard = seed+n;
        randEng.discard(discard);
        float random = uniDist(randEng);

        auto coord_d = leaf_tip->offsetToGlobalCoord(i);
        auto coord = coord_d;
        float value = leaf_tip->getValue(i);
        float taf_value = leaf_taf->getValue(i);
        float new_value = 1;
        float vector_probabilidades[] = {0.04,0.06,0.08,0.2};
        if(value == 1.0){
            if(taf_value >= 0.8 && random >= 1.0-vector_probabilidades[3]){
                //printf("NEW TIP\n");
                 leaf_tip->setValueOnly(coord,1.0);
                 //leaf_d->setValueOnly(coord,1.0);
            }else if(taf_value >=0.7 && random >= 1-vector_probabilidades[2] ){
                //printf("NEW TIP\n");
                 leaf_tip->setValueOnly(coord,1.0);
                 //leaf_d->setValueOnly(coord,1.0);
            }else if(taf_value >= 0.5&& random >= 1-vector_probabilidades[1]){
                //printf("NEW TIP\n");
                 leaf_tip->setValueOnly(coord,1.0);
                 //leaf_d->setValueOnly(coord,1.0);
            }else if(taf_value >=0.3&& random >= 1-vector_probabilidades[0]){
                //printf("NEW TIP\n");
                 leaf_tip->setValueOnly(coord,1.0);
                 //leaf_d->setValueOnly(coord,1.0);
            }else{
                //NO hay branch
                leaf_tip->setValue(coord_d,0.0);
                new_value = 0;
                //leaf_tip->setValueOnly(coord,0.0);
            }
        }else if(value == 2){
            new_value = value -1;
        }else{
            new_value = value;
        }
        

        // if(true || i%2==0){
        //     new_value = value-1;
        // }
        //float new_value = value-1;
        if(new_value < 0 ){
            new_value =0;
        }
        if(new_value != 0.0 && new_value != 1.0){
            //printf("%f\n",new_value);
        }
        //new_value = random;
        leaf_tip->setValue(coord_d,new_value);

        
        

    };
    thrust::counting_iterator<uint64_t, thrust::device_system_tag> iter(0);
    thrust::for_each(iter, iter + 512*leafCount, kernel);
}


/*
    Genera el gradiente escalado del TAF, para poder calcular la divergencia
*/
void generateGradientTAF(nanovdb::FloatGrid * gridTAF,nanovdb::FloatGrid * gridTAFEndothelial,nanovdb::Vec3fGrid* gradientTAF,uint64_t leafCount){
    auto kernel = [gridTAF,gridTAFEndothelial,gradientTAF] __device__ (const uint64_t n) {
        auto *leaf_s = gridTAF->tree().getFirstNode<0>() + (n >> 9);// this only works if grid->isSequential<0>() == true
        auto *leaf_TAFEndothelial = gridTAFEndothelial->tree().getFirstNode<0>() + (n >> 9);
        auto *leaf_Gradient = gradientTAF->tree().getFirstNode<0>() + (n >> 9);
        auto accessor_aux = gradientTAF->getAccessor();
        const int i = n & 511;
        auto coord = leaf_s->offsetToGlobalCoord(i);
        const nanovdb::Coord coord_nano = coord;
        nanovdb::CurvatureStencil<nanovdb::FloatGrid> stencilNano(*gridTAFEndothelial);
        ////printf("%d %d %d\n",coord_nano[0],coord_nano[1],coord_nano[2]);
        stencilNano.moveTo(coord_nano);
        auto gradient = stencilNano.gradient();
        float sensivity = chemotacticSensivity(leaf_s->getValue(i));
        
        gradient = gradient *sensivity * 1e-4;
        // if(coord[0]== 0 || coord[1]==0||coord[2]==0){
        //     gradient[0] = 0;
        //     gradient[1] = 0 ;
        //     gradient[2] = 0;
        // }
        // if(gradient[0]!=0 || gradient[1]!=0|| gradient[2]!=0){
        //     //printf("<%f %f %f>  sensivity:%f\n",gradient[0],gradient[1],gradient[2],sensivity);
        // }
        
        
        //gradient[0] = 2.0;
        //gradient[1] = 1.0 ;
        //gradient[2] = 3.0;
        leaf_Gradient->setValueOnly(i,gradient);
        //auto aux = accessor_aux.getValue(coord);
        ////printf("%f %f %f\n",aux[0],aux[1],aux[2]);

    };
    thrust::counting_iterator<uint64_t, thrust::device_system_tag> iter(0);
    thrust::for_each(iter, iter + 512*leafCount, kernel);
}

/*
    Genera el gradiente escalado de la Fibronectina, para poder calcular la divergencia
*/
void generateGradientFibronectin(nanovdb::FloatGrid * gridFibronectin,nanovdb::FloatGrid * gridEndothelial,nanovdb::Vec3fGrid* gradientFibronectin,uint64_t leafCount){
    auto kernel = [gridFibronectin,gridEndothelial,gradientFibronectin] __device__ (const uint64_t n) {
        auto *leaf_s = gridFibronectin->tree().getFirstNode<0>() + (n >> 9);// this only works if grid->isSequential<0>() == true
        auto *leaf_Endothelial = gridEndothelial->tree().getFirstNode<0>() + (n >> 9);
        auto *leaf_Gradient = gradientFibronectin->tree().getFirstNode<0>() + (n >> 9);
        const int i = n & 511;
        auto coord = leaf_s->offsetToGlobalCoord(i);
        const nanovdb::Coord coord_nano = coord;
        nanovdb::CurvatureStencil<nanovdb::FloatGrid> stencilNano(*gridFibronectin);
        stencilNano.moveTo(coord_nano);
        auto gradient = stencilNano.gradient();
        
        float endothelialValue = leaf_Endothelial->getValue(i);
        gradient = gradient  * endothelialValue * 1e-7;
        
        leaf_Gradient->setValueOnly(coord,gradient);

    };
    thrust::counting_iterator<uint64_t, thrust::device_system_tag> iter(0);
    thrust::for_each(iter, iter + 512*leafCount, kernel);
}

void divergence(nanovdb::Vec3fGrid *grid_s,nanovdb::FloatGrid *grid_d,uint64_t leafCount){
    auto kernel = [grid_s,grid_d] __device__ (const uint64_t n) {
        auto *leaf_s = grid_s->tree().getFirstNode<0>() + (n >> 9);// this only works if grid->isSequential<0>() == true
        auto *leaf_d = grid_d->tree().getFirstNode<0>() + (n >> 9);// this only works if grid->isSequential<0>() == true

        const int i = n & 511;
        
        auto coord = leaf_s->offsetToGlobalCoord(i);

        nanovdb::CurvatureStencil<nanovdb::Vec3fGrid> stencil(*grid_s);
        stencil.moveTo(coord);
        auto gradient = stencil.gradient();
        auto divergence = gradient[0][0]+gradient[1][1] + gradient[2][2];
        // if(gradient[0][0]!= 0 && divergence != 0 ){
        //     //printf("%f\n",divergence);
        // }
        leaf_d->setValueOnly(coord,divergence);

    };
    thrust::counting_iterator<uint64_t, thrust::device_system_tag> iter(0);
    thrust::for_each(iter, iter + 512*leafCount, kernel);
}
void laplacian(nanovdb::FloatGrid * grid_s,nanovdb::FloatGrid * grid_d, uint64_t leafCount){
    auto kernel = [grid_s,grid_d] __device__ (const uint64_t n) {
        auto *leaf_s = grid_s->tree().getFirstNode<0>() + (n >> 9);// this only works if grid->isSequential<0>() == true
        auto *leaf_d = grid_d->tree().getFirstNode<0>() + (n >> 9);// this only works if grid->isSequential<0>() == true

        const int i = n & 511;
        
        auto coord = leaf_s->offsetToGlobalCoord(i);

        nanovdb::CurvatureStencil<nanovdb::FloatGrid> stencil(*grid_s);
        const nanovdb::Coord coord_nano = coord;
        stencil.moveTo(coord_nano);
        auto old_value = leaf_s->getValue(i);
        auto laplacian = stencil.laplacian();
        // if(laplacian < 0.0){
        //     laplacian=0.0;
        // }
        // if(laplacian!= 0){
        //     //printf("%f\n",laplacian);
        // }
        auto new_value = old_value + laplacian* 1e-7;

        // if(new_value < 0 ){
        //     new_value = 0;
        // }
        // if(new_value > 1){
        //     new_value = 1;
        // }        
        leaf_d->setValueOnly(i,new_value);
    };
    thrust::counting_iterator<uint64_t, thrust::device_system_tag> iter(0);
    thrust::for_each(iter, iter + 512*leafCount, kernel);
}

void product(nanovdb::FloatGrid * gridTAF,nanovdb::FloatGrid * gridEndothelial,nanovdb::FloatGrid *grid_d, uint64_t leafCount){
    auto kernel = [gridTAF,gridEndothelial,grid_d] __device__ (const uint64_t n) {
        auto *leaf_d = grid_d->tree().getFirstNode<0>() + (n >> 9);// this only works if grid->isSequential<0>() == true
        auto *leaf_TAF = gridTAF->tree().getFirstNode<0>() + (n >> 9);// this only works if grid->isSequential<0>() == true
        auto *leaf_Endothelial = gridEndothelial->tree().getFirstNode<0>() + (n >> 9);// this only works if grid->isSequential<0>() == true

        const int i = n & 511;
        
        //auto coord = leaf_d->offsetToGlobalCoord(i);
        auto endothelial = leaf_Endothelial->getValue(i);
        auto taf = leaf_TAF->getValue(i);
        
        auto new_value = leaf_TAF->getValue(i)*leaf_Endothelial->getValue(i);
        // if(endothelial != 0){
        //     //printf("end:%f taf:%f newVal:%f\n",endothelial,taf,new_value);
        // }
        new_value = leaf_TAF->getValue(i);
        //new_value = leaf_Endothelial->getValue(i);
        leaf_d->setValueOnly(i,new_value);

    };
    thrust::counting_iterator<uint64_t, thrust::device_system_tag> iter(0);
    thrust::for_each(iter, iter + 512*leafCount, kernel);
}

void cleanEndothelial(nanovdb::FloatGrid * gridEndothelial,uint64_t leafCount){
    auto kernel = [gridEndothelial] __device__ (const uint64_t n) {
        auto *leaf_Endothelial = gridEndothelial->tree().getFirstNode<0>() + (n >> 9);// this only works if grid->isSequential<0>() == true

        const int i = n & 511;

        auto coord = leaf_Endothelial->offsetToGlobalCoord(i);
        // if(coord[1]<-149){
        //     //printf("%d\n",coord[1]);
        // }
        if(coord[0] == -250 || coord[1] == -150 || coord[2] == -250){
            //leaf_Endothelial->setValueOnly(i,0.00001);
            leaf_Endothelial->setValueOnly(i,0.1);
            ////printf("Uese\n");
        }
        if(coord[0] == 0 || coord[1] == 0 || coord[2] == 0){
            //leaf_Endothelial->setValueOnly(i,0.00001);
            leaf_Endothelial->setValueOnly(i,0.1);
            ////printf("Uese\n");
        }
        // if(coord[0] <= -230 || coord[1] <= -130 || coord[2] <= -230){
        //     //leaf_Endothelial->setValueOnly(i,0.00001);
        //     leaf_Endothelial->setValueOnly(i,0);
        //     ////printf("Uese\n");
        // }
        // if(coord[0] >= -20 || coord[1] >= -20 || coord[2] >= -20){
        //     //leaf_Endothelial->setValueOnly(i,0.00001);
        //     leaf_Endothelial->setValueOnly(i,0);
        //     ////printf("Uese\n");
        // }
    };
    thrust::counting_iterator<uint64_t, thrust::device_system_tag> iter(0);
    thrust::for_each(iter, iter + 512*leafCount, kernel);
}

void normalize(nanovdb::FloatGrid * gridTAF,float maxValue,float prevMax, uint64_t leafCount){
    auto kernel = [gridTAF,maxValue,prevMax] __device__ (const uint64_t n) {
       // auto *leaf_d = grid_d->tree().getFirstNode<0>() + (n >> 9);// this only works if grid->isSequential<0>() == true
        auto *leaf_TAF = gridTAF->tree().getFirstNode<0>() + (n >> 9);// this only works if grid->isSequential<0>() == true
        //auto *leaf_Endothelial = gridEndothelial->tree().getFirstNode<0>() + (n >> 9);// this only works if grid->isSequential<0>() == true

        const int i = n & 511;
        
        //auto coord = leaf_d->offsetToGlobalCoord(i);

        
        float new_value = leaf_TAF->getValue(i);
        if(maxValue > 0 ){
            new_value = (1.0 - new_value / maxValue)*prevMax;
        }
        leaf_TAF->setValueOnly(i,new_value);

    };
    thrust::counting_iterator<uint64_t, thrust::device_system_tag> iter(0);
    thrust::for_each(iter, iter + 512*leafCount, kernel);
}
void normalize(nanovdb::FloatGrid * grid,float maxValue, uint64_t leafCount){
    auto kernel = [grid,maxValue] __device__ (const uint64_t n) {
        auto* leaf = grid->tree().getFirstNode<0>() + (n >> 9);// this only works if grid->isSequential<0>() == true
        const int i = n & 511;
        float new_value = leaf->getValue(i);
        new_value = new_value / maxValue;
        leaf->setValueOnly(i,new_value);
    };
    thrust::counting_iterator<uint64_t, thrust::device_system_tag> iter(0);
    thrust::for_each(iter, iter + 512*leafCount, kernel);
}
void addMax(nanovdb::FloatGrid * gridTAF, float maxValue,uint64_t leafCount){
    auto kernel = [gridTAF,maxValue] __device__ (const uint64_t n) {
       // auto *leaf_d = grid_d->tree().getFirstNode<0>() + (n >> 9);// this only works if grid->isSequential<0>() == true
        auto *leaf_TAF = gridTAF->tree().getFirstNode<0>() + (n >> 9);// this only works if grid->isSequential<0>() == true
        //auto *leaf_Endothelial = gridEndothelial->tree().getFirstNode<0>() + (n >> 9);// this only works if grid->isSequential<0>() == true

        const int i = n & 511;
        
        //auto coord = leaf_d->offsetToGlobalCoord(i);

        
        float new_value = leaf_TAF->getValue(i) + maxValue;
        
        leaf_TAF->setValueOnly(i,new_value);

    };
    thrust::counting_iterator<uint64_t, thrust::device_system_tag> iter(0);
    thrust::for_each(iter, iter + 512*leafCount, kernel);
}
void degradeOxygen(nanovdb::FloatGrid * gridOxygen, nanovdb::FloatGrid * gridTummor,float factor,uint64_t leafCount){
    auto kernel = [gridOxygen,gridTummor,factor] __device__ (const uint64_t n) {
       // auto *leaf_d = grid_d->tree().getFirstNode<0>() + (n >> 9);// this only works if grid->isSequential<0>() == true
        auto *leaf_Oxygen = gridOxygen->tree().getFirstNode<0>() + (n >> 9);// this only works if grid->isSequential<0>() == true
        auto *leaf_Tummor = gridTummor->tree().getFirstNode<0>() + (n >> 9);// this only works if grid->isSequential<0>() == true
        //auto *leaf_Endothelial = gridEndothelial->tree().getFirstNode<0>() + (n >> 9);// this only works if grid->isSequential<0>() == true

        const int i = n & 511;
        
        //auto coord = leaf_d->offsetToGlobalCoord(i);

        
        float new_value = leaf_Oxygen->getValue(i) - leaf_Tummor->getValue(i)* factor;
        if(new_value < 0 ){
            new_value = 0 ;
        }
        leaf_Oxygen->setValueOnly(i,new_value);

    };
    thrust::counting_iterator<uint64_t, thrust::device_system_tag> iter(0);
    thrust::for_each(iter, iter + 512*leafCount, kernel);
}
void absolute(nanovdb::FloatGrid * gridTAF, uint64_t leafCount){
    auto kernel = [gridTAF] __device__ (const uint64_t n) {
       // auto *leaf_d = grid_d->tree().getFirstNode<0>() + (n >> 9);// this only works if grid->isSequential<0>() == true
        auto *leaf_TAF = gridTAF->tree().getFirstNode<0>() + (n >> 9);// this only works if grid->isSequential<0>() == true
        //auto *leaf_Endothelial = gridEndothelial->tree().getFirstNode<0>() + (n >> 9);// this only works if grid->isSequential<0>() == true

        const int i = n & 511;
        
        //auto coord = leaf_d->offsetToGlobalCoord(i);

        
        float new_value = leaf_TAF->getValue(i) ;
        new_value = -1.0 * new_value;
        
        leaf_TAF->setValueOnly(i,new_value);

    };
    thrust::counting_iterator<uint64_t, thrust::device_system_tag> iter(0);
    thrust::for_each(iter, iter + 512*leafCount, kernel);
}

void regenerateEndothelial(nanovdb::FloatGrid* gridEndothelialContinue,nanovdb::FloatGrid* gridEndothelialDiscrete,u_int64_t leafCount){
    auto kernel = [gridEndothelialContinue,gridEndothelialDiscrete] __device__ (const uint64_t n) {
        auto *leaf_Endo = gridEndothelialContinue->tree().getFirstNode<0>() + (n >> 9);
        auto *leaf_Discrete = gridEndothelialDiscrete->tree().getFirstNode<0>() + (n >> 9);
        const int i = n & 511;
        
        auto coord = leaf_Endo->offsetToGlobalCoord(i);
        if(leaf_Discrete->getValue(i)>0.0){
            leaf_Endo->setValue(coord,1.0);
        }
    };
    thrust::counting_iterator<uint64_t, thrust::device_system_tag> iter(0);
    thrust::for_each(iter, iter + 512*leafCount, kernel);
}

void equationBplusSimple(nanovdb::FloatGrid* gridTumor,nanovdb::FloatGrid* gridBplus,nanovdb::FloatGrid* gridOxygen,u_int64_t leafCount){
    auto kernel = [gridTumor,gridBplus,gridOxygen] __device__ (const uint64_t n) {
        auto *leaf_Tumor = gridTumor->tree().getFirstNode<0>() + (n >> 9);
        auto *leaf_Bplus = gridBplus->tree().getFirstNode<0>() + (n >> 9);
        auto *leaf_Oxygen = gridOxygen->tree().getFirstNode<0>() + (n >> 9);
        
        const int i = n & 511;
        
        auto coord = leaf_Tumor->offsetToGlobalCoord(i);
        float oxygen = leaf_Oxygen->getValue(i);
        float oxygenThreshold = 0.3;
        
        
        if(oxygen>oxygenThreshold){
            float c_max = 2.0;
            float TtcProliferation= 10.0 * 5.0;
            float tumor_cells = leaf_Tumor->getValue(i);
            float new_value = 0 ;
            new_value = 1.0/TtcProliferation * tumor_cells * (1.0-tumor_cells/c_max);
            // if(new_value > 0.0){
            //     //printf("%f\n",new_value);
            // }
            leaf_Bplus->setValue(coord,new_value);
            //leaf_Bplus->setValue(coord,1.0*tumor_cells);
        }else{
            leaf_Bplus->setValue(coord,0.0);
        }

    };
    thrust::counting_iterator<uint64_t, thrust::device_system_tag> iter(0);
    thrust::for_each(iter, iter + 512*leafCount, kernel);
}
void equationBminusSimple(nanovdb::FloatGrid* gridTumor,nanovdb::FloatGrid* gridBminus,nanovdb::FloatGrid* gridOxygen,u_int64_t leafCount){
    auto kernel = [gridTumor,gridBminus,gridOxygen] __device__ (const uint64_t n) {
        auto *leaf_Tumor = gridTumor->tree().getFirstNode<0>() + (n >> 9);
        auto *leaf_Bminus = gridBminus->tree().getFirstNode<0>() + (n >> 9);
        auto *leaf_Oxygen = gridOxygen->tree().getFirstNode<0>() + (n >> 9);
        
        const int i = n & 511;
        
        auto coord = leaf_Tumor->offsetToGlobalCoord(i);
        float oxygen = leaf_Oxygen->getValue(i);
        float oxygenThreshold = 0.01;
        
        
        if(oxygen<oxygenThreshold){
            float TtcDeath = 100;
            float tumor_cells = leaf_Tumor->getValue(i);
            float new_value = 0.0;

            new_value = -1.0/TtcDeath * tumor_cells;
            leaf_Bminus->setValue(coord,new_value);
        }else{
            leaf_Bminus->setValue(coord,0.0);
        }

    };
    thrust::counting_iterator<uint64_t, thrust::device_system_tag> iter(0);
    thrust::for_each(iter, iter + 512*leafCount, kernel);
}
void addDeadCells(nanovdb::FloatGrid* gridBminus,nanovdb::FloatGrid* gridDeadCells,u_int64_t leafCount){
    auto kernel = [gridDeadCells,gridBminus] __device__ (const uint64_t n) {
        auto *leaf_DeadCells = gridDeadCells->tree().getFirstNode<0>() + (n >> 9);
        auto *leaf_Bminus = gridBminus->tree().getFirstNode<0>() + (n >> 9);
        //auto *leaf_Oxygen = gridOxygen->tree().getFirstNode<0>() + (n >> 9);
        
        const int i = n & 511;
        auto coord = leaf_DeadCells->offsetToGlobalCoord(i);

        float currentDead = leaf_DeadCells->getValue(i);
        float new_data = currentDead - leaf_Bminus->getValue(i);
        if(new_data>1){
            new_data = 1;
        }
        leaf_DeadCells->setValue(coord,new_data);

    };
    thrust::counting_iterator<uint64_t, thrust::device_system_tag> iter(0);
    thrust::for_each(iter, iter + 512*leafCount, kernel);
}
void equationPressure(nanovdb::FloatGrid* gridTumor,nanovdb::FloatGrid* gridPressure,u_int64_t leafCount){
    auto kernel = [gridTumor,gridPressure] __device__ (const uint64_t n) {
        auto *leaf_Tumor = gridTumor->tree().getFirstNode<0>() + (n >> 9);
        auto *leaf_Pressure = gridPressure->tree().getFirstNode<0>() + (n >> 9);

        const int i = n & 511;
        
        auto coord = leaf_Tumor->offsetToGlobalCoord(i);
       

        float cbNorm = 1.0;
        float cbMax = 2.0;

        float tumor_cells = leaf_Tumor->getValue(i);

        if(tumor_cells >= cbMax){
            //Do nothing ????
            //int a = 0 ;
            float value = leaf_Pressure->getValue(coord);
            float new_value = 0.0;

            new_value = (tumor_cells - cbNorm)/(cbMax-cbNorm);
            // if(value != 0 ){
            //     //printf("%f\n",value);
            // }
            leaf_Pressure->setValue(coord,1.0);
        }else if(tumor_cells>=cbNorm){
            float new_value = 0.0;

            new_value = (tumor_cells - cbNorm)/(cbMax-cbNorm);
            // if(new_value < 0 ){
            //     new_value = 0.0;
            // }
            leaf_Pressure->setValue(coord,new_value);
        }else{
            leaf_Pressure->setValue(coord,0.0);
        }

    };
    thrust::counting_iterator<uint64_t, thrust::device_system_tag> iter(0);
    thrust::for_each(iter, iter + 512*leafCount, kernel);
}

void equationTumorSimple(nanovdb::Vec3fGrid* gridFlux,nanovdb::FloatGrid* gridBplus,nanovdb::FloatGrid* gridBminus,nanovdb::FloatGrid* gridTumorRead,nanovdb::FloatGrid* gridTumorWrite,u_int64_t leafCount){
    auto kernel = [gridFlux,gridBplus,gridBminus,gridTumorRead,gridTumorWrite] __device__ (const uint64_t n) {
        auto *leaf_Bplus = gridBplus->tree().getFirstNode<0>() + (n >> 9);
        auto *leaf_Bminus = gridBminus->tree().getFirstNode<0>() + (n >> 9);
        auto *leaf_Flux = gridFlux->tree().getFirstNode<0>() + (n >> 9);
        auto leaf_tumor_read = gridTumorRead->tree().getFirstNode<0>() + (n>>9);
        auto leaf_tumor_write = gridTumorWrite->tree().getFirstNode<0>() + (n>>9);

        const int i = n & 511;
        
        auto coord = leaf_Flux->offsetToGlobalCoord(i);
       
        
        nanovdb::CurvatureStencil<nanovdb::Vec3fGrid> stencilNano(*gridFlux);
        
        
        
        stencilNano.moveTo(coord);
        auto gradient = stencilNano.gradient();
        auto divergence = gradient[0][0]+gradient[1][1] + gradient[2][2];
        auto value_flux = leaf_Flux->getValue(coord);
        // if(value_flux[0]!=0.0||value_flux[1]!=0.0||value_flux[2]){
        //     //printf("gradient matrix %f %f %f ; %f %f %f ; %f %f %f\n",gradient[0][0],gradient[0][1],gradient[0][2],gradient[1][0],gradient[1][1],
        //                                                             gradient[1][2],gradient[2][0],gradient[2][1],gradient[2][2]);
        // }
        float old_m = leaf_tumor_read->getValue(i);
        float b_plus = leaf_Bplus->getValue(i);
        float b_minus = leaf_Bminus->getValue(i);
        float factor_divergence = -divergence * old_m * 0.0001;
        if(factor_divergence<0){
            factor_divergence = 0 ;
        }
        // if(factor_divergence!=0){
        //     //printf("%f %f\n",factor_divergence,b_plus);
        // }
        float derivative = factor_divergence + b_plus + b_minus;
        // if(derivative < 0 ) {
        //     derivative  = 0 ;
        // }
        float new_value = old_m + derivative * time_factor;
        if(new_value > 2.0){
            new_value =2.0 ;
        }
        if(new_value < 0.0){
            new_value  = 0.0;
        }
        // if(b_plus != 0.0){
        //     //printf("diver:%f plus:%f minus:%f new:%f\n",divergence,b_plus,b_minus,new_value);

        // }
        // if(new_value != 0.0){
        //     //printf("new value tumor:%f\n",new_value);
        // }
        leaf_tumor_write->setValue(coord,new_value);
    };
    thrust::counting_iterator<uint64_t, thrust::device_system_tag> iter(0);
    thrust::for_each(iter, iter + 512*leafCount, kernel);
}

void equationFluxSimple(nanovdb::FloatGrid* gridPressure,nanovdb::FloatGrid* gridTumor,nanovdb::Vec3fGrid* gridFlux,nanovdb::FloatGrid* diffusionGrid,u_int64_t leafCount){
    auto kernel = [gridPressure,gridTumor,gridFlux,diffusionGrid] __device__ (const uint64_t n) {
        auto *leaf_Pressure = gridPressure->tree().getFirstNode<0>() + (n >> 9);
        auto *leaf_Tumor = gridTumor->tree().getFirstNode<0>() + (n>>9);
        auto *leaf_Flux = gridFlux->tree().getFirstNode<0>() + (n>>9);
        auto *leaf_diffusion= diffusionGrid->tree().getFirstNode<0>() + (n>>9);
        const int i = n & 511;
        
        auto coord = leaf_Tumor->offsetToGlobalCoord(i);
       
        nanovdb::CurvatureStencil<nanovdb::FloatGrid> stencilNano(*gridPressure);
        stencilNano.moveTo(coord);
        auto gradiente = stencilNano.gradient();
        
        // if(leaf_Pressure->getValue(coord)>0.0){
        //     //printf("%f %f %f\n",gradiente[0],gradiente[1],gradiente[2]);
        // }
        float diffussion_coefficient = leaf_diffusion->getValue(i);
        float tumor_cells = leaf_Tumor->getValue(i);
        for(int i = 0;i<3;i++){
            gradiente[i] *= -diffussion_coefficient;//* tumor_cells;
        }
        if(gradiente[0]!=0 || gradiente[1] != 0 || gradiente[2]!=0){
            //printf("%f %f %f \n",gradiente[0],gradiente[1],gradiente[2]);
        }
        leaf_Flux->setValue(coord,gradiente);

    };
    thrust::counting_iterator<uint64_t, thrust::device_system_tag> iter(0);
    thrust::for_each(iter, iter + 512*leafCount, kernel);
}

void discretize(nanovdb::FloatGrid* grid,u_int64_t leafCount){
    auto kernel = [grid] __device__ (const uint64_t n) {
        auto *leaf = grid->tree().getFirstNode<0>() + (n >> 9);
        const int i = n & 511;
        
        auto coord = leaf->offsetToGlobalCoord(i);
        float value = leaf->getValue(i);
        if(value > 0.0){
            value  = 1.0;
        }
        leaf->setValue(coord,value);
    };
    thrust::counting_iterator<uint64_t, thrust::device_system_tag> iter(0);
    thrust::for_each(iter, iter + 512*leafCount, kernel);
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

void albedoHemogoblin(nanovdb::FloatGrid* oxygen,nanovdb::Vec3fGrid* albedo,u_int64_t leafCount){
    auto kernel = [oxygen,albedo] __device__ (const uint64_t n) {
        nanovdb::Vec3f base_color = {1,203.0/256.0,190.0/256.0};
        nanovdb::Vec3f hemoglobin_color = {230.0/256.0,191.0/256.0,170.0/256.0};
        auto* leaf_Oxygen = oxygen->tree().getFirstNode<0>() + (n >> 9);
        auto* leaf_Albedo = albedo->tree().getFirstNode<0>() + (n >> 9);
        const int i = n & 511;
        
        auto coord = leaf_Oxygen->offsetToGlobalCoord(i);
        nanovdb::Vec3f new_color = {0.0,0.0,0.0};
        for(int i =0 ;i<3;i++){
            new_color[i] =  base_color[i] - hemoglobin_color[i]* leaf_Oxygen->getValue(coord);
            if(new_color[i] - 1.0 > 0.0){
                new_color[i] = 1.0;
            }
            if(new_color[i] < 0.0){
                new_color[i] = 0.0;
            }
        }
        // int coord_abs = coord[0];
        // if(coord_abs < 0 ){
        //     coord_abs = -coord_abs;
        // }
        // if(coord_abs < 30){
        //     coord_abs = 0 ;

        // }

        // new_color[0] = (float)coord_abs / 250.0;
        leaf_Albedo->setValue(coord,new_color);
    };
    thrust::counting_iterator<uint64_t, thrust::device_system_tag> iter(0);
    thrust::for_each(iter, iter + 512*leafCount, kernel);
}
void closeTips(nanovdb::FloatGrid* gridEndothelialTip, nanovdb::FloatGrid* gridTummorCells,uint64_t leafCount){
    auto kernel = [gridEndothelialTip,gridTummorCells] __device__ (const uint64_t n) {
        auto *leaf_endothelial= gridEndothelialTip->tree().getFirstNode<0>() + (n >> 9);
        auto *leaf_tummor = gridTummorCells->tree().getFirstNode<0>() + (n >> 9);
        const int i = n & 511;
        auto coord = leaf_endothelial->offsetToGlobalCoord(i);
        float tummorDensity = leaf_tummor->getValue(coord);
        if(tummorDensity > 0.9){
            leaf_endothelial->setValueOnly(coord,0);
        }else{
            leaf_endothelial->setValueOnly(coord,leaf_endothelial->getValue(coord));
        }

    };
    thrust::counting_iterator<uint64_t, thrust::device_system_tag> iter(0);
    thrust::for_each(iter, iter + 512*leafCount, kernel);
}

void degradeDiscrete(nanovdb::FloatGrid* gridEndothelialDiscrete, nanovdb::FloatGrid* gridTummorCells,uint64_t leafCount){
    auto kernel = [gridEndothelialDiscrete,gridTummorCells] __device__ (const uint64_t n) {
        auto *leaf_endothelial= gridEndothelialDiscrete->tree().getFirstNode<0>() + (n >> 9);
        auto *leaf_tummor = gridTummorCells->tree().getFirstNode<0>() + (n >> 9);
        const int i = n & 511;
        auto coord = leaf_tummor->offsetToGlobalCoord(i);
        float tummorDensity = leaf_tummor->getValue(coord);
        if(tummorDensity > 0.9){
            leaf_endothelial->setValueOnly(coord,0);
        }else{
            leaf_endothelial->setValueOnly(coord,leaf_endothelial->getValue(coord));
        }
    };
    thrust::counting_iterator<uint64_t, thrust::device_system_tag> iter(0);
    thrust::for_each(iter, iter + 512*leafCount, kernel);
}
// void baseSkinColor(nanovdb::FloatGrid* melaninConcentration,nanovdb::Vec3fGrid* skinColor,u_int64_t leafCount){
//     auto kernel = [melaninConcentration,skinColor] __device__ (const uint64_t n) {
//         auto* leaf_Melanin = melaninConcentration->tree().getFirstNode<0>() + (n >> 9);
//         auto* leaf_Skin = skinColor->tree().getFirstNode<0>() + (n >> 9);
//         const int i = n & 511;
        
//         auto coord = leaf_Oxygen->offsetToGlobalCoord(i);
//     };
//     thrust::counting_iterator<uint64_t, thrust::device_system_tag> iter(0);
//     thrust::for_each(iter, iter + 512*leafCount, kernel);
// }

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
        o2Concentration = o2Concentration *0 ;/// total;
        melanineConcentration = melanineConcentration ;/// total;
        nanovdb::Vec3f base_color = {1,203.0/256.0,190.0/256.0};
        nanovdb::Vec3f melanine_color = {230.0/256.0,191.0/256.0,170.0/256.0};
        nanovdb::Vec3f hemoglobin_color = {230.0/256.0,10/256.0,10.0/256.0};
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