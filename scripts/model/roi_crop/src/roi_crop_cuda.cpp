// #include <THC/THC.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <stdbool.h>
#include <stdio.h>
#include "roi_crop_cuda_kernel.h"

#define real float

// this symbol will be resolved automatically from PyTorch libs
// extern THCState *state;

// Bilinear sampling is done in BHWD (coalescing is not obvious in BDHW)
// we assume BHWD format in inputImages
// we assume BHW(YX) format on grids

int BilinearSamplerBHWD_updateOutput_cuda(at::Tensor inputImages, at::Tensor grids, at::Tensor output){
//  THCState *state = getCutorchState(L);
//  THCudaTensor *inputImages = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");
//  THCudaTensor *grids = (THCudaTensor *)luaT_checkudata(L, 3, "torch.CudaTensor");
//  THCudaTensor *output = (THCudaTensor *)luaT_checkudata(L, 4, "torch.CudaTensor");

  int success = 0;
  success = BilinearSamplerBHWD_updateOutput_cuda_kernel(output.size(1),
                                               output.size(3),
                                               output.size(2),
                                               output.size(0),
                                               inputImages.size(1),
                                               inputImages.size(2),
                                               inputImages.size(3),
                                               inputImages.size(0),
                                               inputImages.data_ptr<float>(),
                                               inputImages.stride(0), 
                                               inputImages.stride(1),
                                               inputImages.stride(2),
                                               inputImages.stride(3),
                                               grids.data_ptr<float>(),
                                               grids.stride(0), 
                                               grids.stride(3),
                                               grids.stride(1),
                                               grids.stride(2),
                                               output.data_ptr<float>(),
                                               output.stride(0),
                                               output.stride(1),
                                               output.stride(2),
                                               output.stride(3),
                                               at::cuda::getCurrentCUDAStream());

  //check for errors
  if (!success) {
    TORCH_CHECK(false, "aborting");
  }
  return 1;
}

int BilinearSamplerBHWD_updateGradInput_cuda(at::Tensor inputImages, at::Tensor grids, at::Tensor gradInputImages,
                                        at::Tensor gradGrids, at::Tensor gradOutput)
{
//  THCState *state = getCutorchState(L);
//  THCudaTensor *inputImages = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");
//  THCudaTensor *grids = (THCudaTensor *)luaT_checkudata(L, 3, "torch.CudaTensor");
//  THCudaTensor *gradInputImages = (THCudaTensor *)luaT_checkudata(L, 4, "torch.CudaTensor");
//  THCudaTensor *gradGrids = (THCudaTensor *)luaT_checkudata(L, 5, "torch.CudaTensor");
//  THCudaTensor *gradOutput = (THCudaTensor *)luaT_checkudata(L, 6, "torch.CudaTensor");

  int success = 0;
  success = BilinearSamplerBHWD_updateGradInput_cuda_kernel(gradOutput.size(1),
                                                  gradOutput.size(3),
                                                  gradOutput.size(2),
                                                  gradOutput.size(0),
                                                  inputImages.size(1),
                                                  inputImages.size(2),
                                                  inputImages.size(3),
                                                  inputImages.size(0),
                                                  inputImages.data_ptr<float>(),
                                                  inputImages.stride(0), 
                                                  inputImages.stride(1),
                                                  inputImages.stride(2),
                                                  inputImages.stride(3),
                                                  grids.data_ptr<float>(),
                                                  grids.stride(0), 
                                                  grids.stride(3),
                                                  grids.stride(1),
                                                  grids.stride(2),
                                                  gradInputImages.data_ptr<float>(),
                                                  gradInputImages.stride(0), 
                                                  gradInputImages.stride(1),
                                                  gradInputImages.stride(2),
                                                  gradInputImages.stride(3),
                                                  gradGrids.data_ptr<float>(),
                                                  gradGrids.stride(0),
                                                  gradGrids.stride(3),
                                                  gradGrids.stride(1),
                                                  gradGrids.stride(2),
                                                  gradOutput.data_ptr<float>(),
                                                  gradOutput.stride(0), 
                                                  gradOutput.stride(1),
                                                  gradOutput.stride(2),
                                                  gradOutput.stride(3),
                                                  at::cuda::getCurrentCUDAStream());

  //check for errors
  if (!success) {
    TORCH_CHECK(false, "aborting");
  }
  return 1;
}
