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

const float threshold_vecino = 0.0;
const float time_factor = 6*60; //Timestep de 6 minutos pasado a segundos
__device__ const float ini_endothelial = 0.6;

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
        //coord = coord_indi;
        //printf("%d %d %d\n",coord[0],coord[1],coord[2]);
        if(coord[1]>lim_inf && coord[1]<lim_sup){
            if(coord[0]%modulo == 0 && coord[2]%modulo == 0 ){
                leaf_d->setValueOnly(i,ini_endothelial);
            }
        }
        if(coord_indi[1]==lim_inf || coord_indi[1]==lim_sup){
            leaf_d->setValueOnly(i,ini_endothelial);
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
            for(int i_incremento_x = 0;i_incremento_x<len_incrementos && !esVecino;i_incremento_x++){
                for(int i_incremento_y = 0 ;i_incremento_y<len_incrementos && !esVecino;i_incremento_y++){
                    for(int i_incremento_z = 0 ;i_incremento_z<len_incrementos && !esVecino;i_incremento_z++){
                        int incremento_x = incrementos_vecinos[i_incremento_x];
                        int incremento_y = incrementos_vecinos[i_incremento_y];
                        int incremento_z = incrementos_vecinos[i_incremento_z];
    
    
                        if(accessor_endothelial.isActive(coord.offsetBy(incremento_x,incremento_y,incremento_z))){
                            n_i = accessor_endothelial.getValue(coord.offsetBy(incremento_x,incremento_y,incremento_z));
                            
                        }
                        esVecino = n_i > threshold_vecino;//Esto igual esta feo
                        
                    }
                }
            }
        //}

        float n_c = 0.025;
        //printf("%f\n",n_i);
        if(esVecino){
            n_i = 1.0;
        }else{
            n_i  = 0.0;
        }
        float c = leaf_s->getValue(i);
        
        
        float derivative = -n_c * n_i* c;
        // if(c > 0){
        //     printf("%f %f %f\n",c,n_i,derivative);
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
            for(int i_incremento_x = 0;i_incremento_x<len_incrementos && !esVecino;i_incremento_x++){
                for(int i_incremento_y = 0 ;i_incremento_y<len_incrementos && !esVecino;i_incremento_y++){
                    for(int i_incremento_z = 0 ;i_incremento_z<len_incrementos && !esVecino;i_incremento_z++){
                        int incremento_x = incrementos_vecinos[i_incremento_x];
                        int incremento_y = incrementos_vecinos[i_incremento_y];
                        int incremento_z = incrementos_vecinos[i_incremento_z];
    
    
                        if(accessor_endothelial.isActive(coord.offsetBy(incremento_x,incremento_y,incremento_z))){
                            n_i = accessor_endothelial.getValue(coord.offsetBy(incremento_x,incremento_y,incremento_z));
                            
                        }
                        esVecino = n_i > threshold_vecino;//Esto igual esta feo
                        
                    }
                }
            }
        //}
        
        
        float production_rate = 0.0125;
        float degradation_rate = 0.1;
        if(esVecino){
            n_i = 1.0;
        }else{
            n_i = 0.0;
        }
        
        float old_f = leaf_s->getValue(i);
        float old_mde = leaf_mde->getValue(i);

        float derivative = production_rate * n_i - degradation_rate * old_f * old_mde;
        // if(derivative > 0 ){
        //     printf("%f\n",old_f + derivative * time_factor);
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
        //Se calcula n_i , que determina si se es vecino de una endothelial
        if(accessor_endothelial.getValue(coord)<threshold_vecino){
            for(int i_incremento_x = 0;i_incremento_x<len_incrementos && !esVecino;i_incremento_x++){
                for(int i_incremento_y = 0 ;i_incremento_y<len_incrementos && !esVecino;i_incremento_y++){
                    for(int i_incremento_z = 0 ;i_incremento_z<len_incrementos && !esVecino;i_incremento_z++){
                        int incremento_x = incrementos_vecinos[i_incremento_x];
                        int incremento_y = incrementos_vecinos[i_incremento_y];
                        int incremento_z = incrementos_vecinos[i_incremento_z];
    
    
                        if(accessor_endothelial.isActive(coord.offsetBy(incremento_x,incremento_y,incremento_z))){
                            n_i = accessor_endothelial.getValue(coord.offsetBy(incremento_x,incremento_y,incremento_z));
                            
                        }
                        esVecino = n_i > threshold_vecino;//Esto igual esta feo
                        
                    }
                }
            }
        }
        float production_rate = 0.0000015;
        
        float diffussion_coefficient = 0.0025;
        float degradation_rate = 0.75;
        //printf("%f %f %f\n",production_rate,diffussion_coefficient,degradation_rate);
        nanovdb::CurvatureStencil<nanovdb::FloatGrid> stencilNano(*input_grid_MDE);
        stencilNano.moveTo(coord);
        float laplacian = stencilNano.laplacian();
        if(esVecino){
            n_i = 1.0;
        }else{
            n_i = 0.0;
        }
        float old_mde = leaf_s->getValue(i);
        float derivative = n_i * production_rate + diffussion_coefficient * laplacian * old_mde - degradation_rate * old_mde;
        // float factor_1 = n_i * production_rate;
        // float factor_2 = diffussion_coefficient * laplacian * old_mde;
        // float factor_3 = degradation_rate * old_mde;
        // if(laplacian!=0){
        //     printf("%f\n",laplacian);
        // }
        
        // if(factor_1 != 0|| factor_2 != 0 || factor_3!=0){
        //     printf("%f %f %f\n",n_i * production_rate,diffussion_coefficient*laplacian*old_mde,degradation_rate*old_mde);

        // }
        // if(derivative > 0 && factor_1 > 0.000002){
        //     printf("%f %f %f\n",factor_1,factor_2,factor_3);
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
        //printf("%d %d %d\n",coord[0],coord[1],coord[2]);
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
        //     //printf("%f %f\n",suma,leaf_s->getValue(coord));
        //     printf("%f \n",accessor.getValue(coord));
        // }
        leaf_d->setValueOnly(coord,gradiente);
        
        
    };

    thrust::counting_iterator<uint64_t, thrust::device_system_tag> iter(0);
    thrust::for_each(iter, iter + 512*leafCount, kernel);
}



/*
    Ecuacion 6
*/
void equationEndothelial(nanovdb::FloatGrid * grid_s,nanovdb::FloatGrid * grid_d,nanovdb::FloatGrid* gridTAF,nanovdb::FloatGrid* gridFibronectin,nanovdb::Vec3fGrid* gradientTAF,nanovdb::Vec3fGrid* gradientFibronectin,uint64_t leafCount){
    auto kernel = [grid_s,grid_d,gridTAF,gridFibronectin,gradientTAF,gradientFibronectin] __device__ (const uint64_t n) {
        auto *leaf_d = grid_d->tree().getFirstNode<0>() + (n >> 9);// this only works if grid->isSequential<0>() == true
        auto *leaf_s = grid_s->tree().getFirstNode<0>() + (n >> 9);// this only works if grid->isSequential<0>() == true
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
        //printf("%f\n",laplacian);
        float factorEndothelial = laplacian * 0.0003 ;
        /*
            Segunda parte, chimiotaxis TAF
        */
        nanovdb::CurvatureStencil<nanovdb::Vec3fGrid> stencilTAF(*gradientTAF);
        
        stencilTAF.moveTo(coord_nano);
        auto gradientTAF = stencilTAF.gradient();
        float factorTAF = gradientTAF[0][0] + gradientTAF[1][1] + gradientTAF[2][2];
        
        //printf("%f  %f  %f\n",gradientTAF[0][0],gradientTAF[1][1],gradientTAF[2][2]);
        
        //float factorTAF  = stencilNano.gaussianCurvature() ;
        /*
            Tercera parte, Fibronectin
        */
        nanovdb::CurvatureStencil<nanovdb::Vec3fGrid> stencilFibronectin(*gradientFibronectin);
        stencilFibronectin.moveTo(coord_nano);
        auto gradientFibronectin = stencilFibronectin.gradient();
        float factorFibronectin = gradientFibronectin[0][0] + gradientFibronectin[1][1] + gradientFibronectin[2][2];
        factorFibronectin = factorFibronectin * 0.28;
        
        //printf("%f %f\n",factorTAF,factorFibronectin);

        float derivative = factorEndothelial  - factorTAF - factorFibronectin;
        // if(derivative > 100){
        //     printf("%f %f %f\n",factorEndothelial,factorTAF,factorFibronectin);
        // }
        //float derivative = -factorTAF;
        auto new_value = old_n + derivative * time_factor;
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

__device__ float chemotacticSensivity(float c){
    float chemotacticMigration = 0.38;
    float chemotacticConstant = 0.6;
    return chemotacticMigration /(1 + chemotacticConstant*c);
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
        //printf("%d %d %d\n",coord_nano[0],coord_nano[1],coord_nano[2]);
        stencilNano.moveTo(coord_nano);
        auto gradient = stencilNano.gradient();
        float sensivity = chemotacticSensivity(leaf_s->getValue(i));
        
        gradient = gradient *sensivity;
        // if(coord[0]== 0 || coord[1]==0||coord[2]==0){
        //     gradient[0] = 0;
        //     gradient[1] = 0 ;
        //     gradient[2] = 0;
        // }
        // if(gradient[0]!=0){
        //     printf("%f %f\n",sensivity,endothelialValue);
        // }
        
        
        //gradient[0] = 2.0;
        //gradient[1] = 1.0 ;
        //gradient[2] = 3.0;
        leaf_Gradient->setValueOnly(i,gradient);
        //auto aux = accessor_aux.getValue(coord);
        //printf("%f %f %f\n",aux[0],aux[1],aux[2]);

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
        gradient = gradient  * endothelialValue;
        
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
        // if(laplacian!= 0){
        //     printf("%f\n",laplacian);
        // }
        auto new_value = old_value + laplacian*0.06;

        // if(new_value < 0 ){
        //     new_value = 0;
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

        auto new_value = leaf_TAF->getValue(i)*leaf_Endothelial->getValue(i);

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
        //     printf("%d\n",coord[1]);
        // }
        if(coord[0] == -250 || coord[1] == -150 || coord[2] == -250){
            leaf_Endothelial->setValueOnly(i,0);
            //printf("Uese\n");
        }
        if(coord[0] == 0 || coord[1] == 0 || coord[2] == 0){
            leaf_Endothelial->setValueOnly(i,0);
            //printf("Uese\n");
        }
    };
    thrust::counting_iterator<uint64_t, thrust::device_system_tag> iter(0);
    thrust::for_each(iter, iter + 512*leafCount, kernel);
}