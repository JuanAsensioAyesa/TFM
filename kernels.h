#ifndef KERNELS_H
#define KERNELS_H
#include <nanovdb/NanoVDB.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <nanovdb/util/Stencils.h>
#include<curand_kernel.h>

void generateEndothelial(nanovdb::FloatGrid *grid_d, uint64_t leafCount, int lim_sup,int lim_inf,int modulo);
void equationTAF(nanovdb::FloatGrid* input_grid_endothelial,nanovdb::FloatGrid* input_grid_TAF,nanovdb::FloatGrid* output_grid_TAF,uint64_t leafCount);
void equationFibronectin(nanovdb::FloatGrid* input_grid_endothelial,nanovdb::FloatGrid* input_grid_Fibronectin,nanovdb::FloatGrid* input_grid_MDE,nanovdb::FloatGrid* output_grid_Fibronectin,uint64_t leafCount);
void equationMDE(nanovdb::FloatGrid* input_grid_endothelial,nanovdb::FloatGrid* input_grid_MDE,nanovdb::FloatGrid* output_grid_MDE,uint64_t leafCount);
void pruebaGradiente(nanovdb::Vec3fGrid  *grid_d,nanovdb::FloatGrid* gridSource ,uint64_t leafCount);
void equationEndothelial(nanovdb::FloatGrid * grid_s,nanovdb::FloatGrid * grid_d,nanovdb::FloatGrid* gridTAF,nanovdb::FloatGrid* gridFibronectin,nanovdb::Vec3fGrid* gradientTAF,nanovdb::Vec3fGrid* gradientFibronectin,nanovdb::FloatGrid* gridTip,uint64_t leafCount);
void factorEndothelial(nanovdb::FloatGrid * grid_s,nanovdb::FloatGrid * grid_d,uint64_t leafCount);
void factorTAF(nanovdb::FloatGrid * grid_s,nanovdb::FloatGrid * grid_d,nanovdb::FloatGrid* gridTAF,nanovdb::Vec3fGrid* gradientTAF,uint64_t leafCount);
void factorFibronectin(nanovdb::FloatGrid * grid_s,nanovdb::FloatGrid * grid_d,nanovdb::FloatGrid* gridFibronectin,nanovdb::Vec3fGrid* gradientFibronectin,uint64_t leafCount);
void generateGradientTAF(nanovdb::FloatGrid * gridTAF,nanovdb::FloatGrid * gridTAFEndothelial,nanovdb::Vec3fGrid* gradientTAF,uint64_t leafCount);
void generateGradientFibronectin(nanovdb::FloatGrid * gridFibronectin,nanovdb::FloatGrid * gridEndothelial,nanovdb::Vec3fGrid* gradientFibronectin,uint64_t leafCount);
void divergence(nanovdb::Vec3fGrid *grid_s,nanovdb::FloatGrid *grid_d,uint64_t leafCount);
void laplacian(nanovdb::FloatGrid * grid_s,nanovdb::FloatGrid * grid_d, uint64_t leafCount);
void product(nanovdb::FloatGrid * gridTAF,nanovdb::FloatGrid * gridEndothelial,nanovdb::FloatGrid *grid_d, uint64_t leafCount);
void cleanEndothelial(nanovdb::FloatGrid * gridEndothelial,uint64_t leafCount);
void equationEndothelialDiscrete(nanovdb::FloatGrid * grid_source_discrete,nanovdb::FloatGrid * grid_destiny_discrete,nanovdb::FloatGrid* gridDerivativeEndothelial,nanovdb::FloatGrid* gridDerivativeEndothelialWrite,nanovdb::FloatGrid* gridTAF,nanovdb::FloatGrid * gridTipRead,nanovdb::FloatGrid* gridTipWrite,int seed,uint64_t leafCount);
void branching(nanovdb::FloatGrid* gridEndothelialTip,nanovdb::FloatGrid* gridTAF,int seed,int leafCount);
void normalize(nanovdb::FloatGrid * gridTAF,float maxValue, float prevMax,uint64_t leafCount);
void normalize(nanovdb::FloatGrid * grid,float maxValue, uint64_t leafCount);
void addMax(nanovdb::FloatGrid * gridTAF, float maxValue,uint64_t leafCount);
void absolute(nanovdb::FloatGrid * gridTAF, uint64_t leafCount);
void regenerateEndothelial(nanovdb::FloatGrid* gridEndothelialContinue,nanovdb::FloatGrid* gridEndothelialDiscrete,u_int64_t leafCount);
void equationBplusSimple(nanovdb::FloatGrid* gridTumor,nanovdb::FloatGrid* gridBplus,nanovdb::FloatGrid* gridOxygen,u_int64_t leafCount);
void equationBminusSimple(nanovdb::FloatGrid* gridTumor,nanovdb::FloatGrid* gridBminus,nanovdb::FloatGrid* gridOxygen,u_int64_t leafCount);
void equationPressure(nanovdb::FloatGrid* gridTumor,nanovdb::FloatGrid* gridPressure,u_int64_t leafCount);
void equationFluxSimple(nanovdb::FloatGrid* gridPressure,nanovdb::FloatGrid* gridTumor,nanovdb::Vec3fGrid* gridFlux,nanovdb::FloatGrid* diffusionGrid,u_int64_t leafCount);
void equationTumorSimple(nanovdb::Vec3fGrid* gridFlux,nanovdb::FloatGrid* gridBplus,nanovdb::FloatGrid* gridBminus,nanovdb::FloatGrid* gridTumorRead,nanovdb::FloatGrid* gridTumorWrite,u_int64_t leafCount);
void discretize(nanovdb::FloatGrid* grid,u_int64_t leafCount);
void average(nanovdb::FloatGrid* grid,nanovdb::FloatGrid* destiny,u_int64_t leafCount);
void copy(nanovdb::FloatGrid* source ,nanovdb::FloatGrid* destiny,u_int64_t leafCount);
void albedoHemogoblin(nanovdb::FloatGrid* oxygen,nanovdb::Vec3fGrid* albedo,u_int64_t leafCount);
#endif