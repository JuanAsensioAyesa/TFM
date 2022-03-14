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
        //const float v = scale * leaf_d->getValue(i);
        if (leaf_d->isActive(i)) {
            leaf_d->setValueOnly(i, 0);// only possible execution divergence
        }
    };

    thrust::counting_iterator<uint64_t, thrust::device_system_tag> iter(0);
    thrust::for_each(iter, iter + 512*leafCount, kernel);
}




