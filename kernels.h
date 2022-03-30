#ifndef KERNELS_H
#define KERNELS_H
#include <nanovdb/NanoVDB.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <nanovdb/util/Stencils.h>

void generateEndothelial(nanovdb::FloatGrid *grid_d, uint64_t leafCount, int lim_sup,int lim_inf,int modulo);
void equationTAF(nanovdb::FloatGrid* input_grid_endothelial,nanovdb::FloatGrid* input_grid_TAF,nanovdb::FloatGrid* output_grid_TAF,uint64_t leafCount);
void pruebaGradiente(nanovdb::Vec3fGrid  *grid_d,nanovdb::FloatGrid* gridSource ,uint64_t leafCount);
void pruebaLaplaciano(nanovdb::FloatGrid * grid_s,nanovdb::FloatGrid * grid_d,uint64_t leafCount);
#endif