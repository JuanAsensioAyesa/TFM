#ifndef KERNELS_H
#define KERNELS_H
#include <nanovdb/NanoVDB.h>
#include <cuda.h>
#include <cuda_runtime.h>


void generateEndothelial(nanovdb::FloatGrid *grid_d, uint64_t leafCount, int lim_sup,int lim_inf,int modulo);


#endif