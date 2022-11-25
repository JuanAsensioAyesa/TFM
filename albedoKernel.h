#ifndef KERNELS_H
#define KERNELS_H
#include <nanovdb/NanoVDB.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <nanovdb/util/Stencils.h>
#include<curand_kernel.h>

void albedoTotal(nanovdb::FloatGrid* oxygen,nanovdb::FloatGrid*melanine,nanovdb::Vec3fGrid* albedo,u_int64_t leafCount);
void average(nanovdb::FloatGrid* grid,nanovdb::FloatGrid* destiny,u_int64_t leafCount);
void copy(nanovdb::FloatGrid* source ,nanovdb::FloatGrid* destiny,u_int64_t leafCount);
void product(nanovdb::FloatGrid* source ,float factor,u_int64_t leafCount);
void add(nanovdb::FloatGrid* source ,nanovdb::FloatGrid* destiny,u_int64_t leafCount);
#endif