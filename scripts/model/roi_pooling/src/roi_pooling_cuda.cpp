// #include <THC/THC.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <math.h>
#include "roi_pooling_kernel.h"

// extern THCState *state;

int roi_pooling_forward_cuda(int pooled_height, int pooled_width, float spatial_scale,
                        at::Tensor features, at::Tensor rois, at::Tensor output, at::Tensor argmax)
{
    // Grab the input tensor
    float * data_flat = features.data_ptr<float>();
    float * rois_flat = rois.data_ptr<float>();

    float * output_flat = output.data_ptr<float>();
    int * argmax_flat = argmax.data_ptr<int>();

    // Number of ROIs
    int num_rois = rois.size(0);
    int size_rois = rois.size(1);
    if (size_rois != 5)
    {
        return 0;
    }

    // batch size
    // int batch_size = THCudaTensor_size(state, features, 0);
    // if (batch_size != 1)
    // {
    //     return 0;
    // }
    // data height
    int data_height = features.size(2);
    // data width
    int data_width = features.size(3);
    // Number of channels
    int num_channels = features.size(1);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    ROIPoolForwardLaucher(
        data_flat, spatial_scale, num_rois, data_height,
        data_width, num_channels, pooled_height,
        pooled_width, rois_flat,
        output_flat, argmax_flat, stream);

    return 1;
}

int roi_pooling_backward_cuda(int pooled_height, int pooled_width, float spatial_scale,
                        at::Tensor top_grad, at::Tensor rois, at::Tensor bottom_grad, at::Tensor argmax)
{
    // Grab the input tensor
    float * top_grad_flat = top_grad.data_ptr<float>();
    float * rois_flat = rois.data_ptr<float>();

    float * bottom_grad_flat = bottom_grad.data_ptr<float>();
    int * argmax_flat = argmax.data_ptr<int>();

    // Number of ROIs
    int num_rois = rois.size(0);
    int size_rois = rois.size(1);
    if (size_rois != 5)
    {
        return 0;
    }

    // batch size
    int batch_size = bottom_grad.size(0);
    // if (batch_size != 1)
    // {
    //     return 0;
    // }
    // data height
    int data_height = bottom_grad.size(2);
    // data width
    int data_width = bottom_grad.size(3);
    // Number of channels
    int num_channels = bottom_grad.size(1);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    ROIPoolBackwardLaucher(
        top_grad_flat, spatial_scale, batch_size, num_rois, data_height,
        data_width, num_channels, pooled_height,
        pooled_width, rois_flat,
        bottom_grad_flat, argmax_flat, stream);

    return 1;
}
