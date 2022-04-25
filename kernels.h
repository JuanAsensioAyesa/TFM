#ifndef KERNELS_H
#define KERNELS_H
#include <nanovdb/NanoVDB.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <nanovdb/util/Stencils.h>

void generateEndothelial(nanovdb::FloatGrid *grid_d, uint64_t leafCount, int lim_sup,int lim_inf,int modulo);
void equationTAF(nanovdb::FloatGrid* input_grid_endothelial,nanovdb::FloatGrid* input_grid_TAF,nanovdb::FloatGrid* output_grid_TAF,uint64_t leafCount);
void equationFibronectin(nanovdb::FloatGrid* input_grid_endothelial,nanovdb::FloatGrid* input_grid_Fibronectin,nanovdb::FloatGrid* input_grid_MDE,nanovdb::FloatGrid* output_grid_Fibronectin,uint64_t leafCount);
void equationMDE(nanovdb::FloatGrid* input_grid_endothelial,nanovdb::FloatGrid* input_grid_MDE,nanovdb::FloatGrid* output_grid_MDE,uint64_t leafCount);
void pruebaGradiente(nanovdb::Vec3fGrid  *grid_d,nanovdb::FloatGrid* gridSource ,uint64_t leafCount);
void equationEndothelial(nanovdb::FloatGrid * grid_s,nanovdb::FloatGrid * grid_d,nanovdb::FloatGrid* gridTAF,nanovdb::FloatGrid* gridFibronectin,nanovdb::Vec3fGrid* gradientTAF,nanovdb::Vec3fGrid* gradientFibronectin,uint64_t leafCount);
void generateGradientTAF(nanovdb::FloatGrid * gridTAF,nanovdb::FloatGrid * gridTAFEndothelial,nanovdb::Vec3fGrid* gradientTAF,uint64_t leafCount);
void generateGradientFibronectin(nanovdb::FloatGrid * gridFibronectin,nanovdb::FloatGrid * gridEndothelial,nanovdb::Vec3fGrid* gradientFibronectin,uint64_t leafCount);
void divergence(nanovdb::Vec3fGrid *grid_s,nanovdb::FloatGrid *grid_d,uint64_t leafCount);
void laplacian(nanovdb::FloatGrid * grid_s,nanovdb::FloatGrid * grid_d, uint64_t leafCount);
void product(nanovdb::FloatGrid * gridTAF,nanovdb::FloatGrid * gridEndothelial,nanovdb::FloatGrid *grid_d, uint64_t leafCount);
#endif