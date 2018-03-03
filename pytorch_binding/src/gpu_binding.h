int gpu_ctc(THCudaTensor *probs,
                        THCudaTensor *grads,
                        THIntTensor *labels_ptr,
                        THIntTensor *label_sizes_ptr,
                        THIntTensor *sizes,
                        int blank_label,
                        THFloatTensor *costs);