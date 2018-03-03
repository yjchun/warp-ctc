#include <iostream>
#include <vector>

#include <numeric>

#include "ctc.h"

#ifdef WARPCTC_ENABLE_GPU
    #include "THC.h"
    #include "THCTensor.h"
    #include "detail/reduce.h"
    extern THCState* state;
#else
    #include "TH.h"
#endif

extern "C" int cpu_ctc(THFloatTensor *probs,
                        THFloatTensor *grads,
                        THIntTensor *labels,
                        THIntTensor *label_sizes,
                        THIntTensor *sizes,
                        int blank_label,
                        THFloatTensor *costs) {

    float *probs_ptr = probs->storage->data + probs->storageOffset;
    float *grads_ptr;
    if (grads->storage) {
            grads_ptr = grads->storage->data + grads->storageOffset;
    } else {
            grads_ptr = NULL; // this will trigger the score forward code path
    }

    int *sizes_ptr = sizes->storage->data + sizes->storageOffset;
    int *labels_ptr = labels->storage->data + labels->storageOffset;
    int *label_sizes_ptr = label_sizes->storage->data + label_sizes->storageOffset;
    float *costs_ptr = costs->storage->data + costs->storageOffset;

    ctcOptions options;
    memset(&options, 0, sizeof(options));
    options.loc = CTC_CPU;
    options.num_threads = 0; // will use default number of threads
    options.blank_label = blank_label;

#if defined(CTC_DISABLE_OMP) || defined(APPLE)
    // have to use at least one
    options.num_threads = std::max(options.num_threads, (unsigned int) 1);
#endif

    size_t cpu_size_bytes;
    get_workspace_size(label_sizes_ptr, sizes_ptr,
                       (int) probs->size[2], (int)probs->size[1],
                       options, &cpu_size_bytes);

    float* cpu_workspace = (float*) new unsigned char[cpu_size_bytes];

    compute_ctc_loss(probs_ptr, grads_ptr,
                     labels_ptr, label_sizes_ptr,
                     sizes_ptr, probs->size[2],
                     probs->size[1], costs_ptr,
                     cpu_workspace, options);

    delete cpu_workspace;
    return 1;
}
#ifdef WARPCTC_ENABLE_GPU
   extern "C" int gpu_ctc(THCudaTensor *probs,
                           THCudaTensor *grads,
                           THIntTensor *labels,
                           THIntTensor *label_sizes,
                           THIntTensor *sizes,
                           int blank_label,
                           THFloatTensor *costs
                           ) {

       float *probs_ptr = probs->storage->data + probs->storageOffset;
       float *grads_ptr;
       if (grads->storage) {
               grads_ptr = grads->storage->data + grads->storageOffset;
       } else {
               grads_ptr = NULL; // this will trigger the score forward code path
       }
       int *sizes_ptr = sizes->storage->data + sizes->storageOffset;
       int *labels_ptr = labels->storage->data + labels->storageOffset;
       int *label_sizes_ptr = label_sizes->storage->data + label_sizes->storageOffset;
       float *costs_ptr = costs->storage->data + costs->storageOffset;

       ctcOptions options;
       memset(&options, 0, sizeof(options));
       options.loc = CTC_GPU;
       options.stream = THCState_getCurrentStream(state);
       options.blank_label = blank_label;

       size_t gpu_size_bytes;
       get_workspace_size(label_sizes_ptr, sizes_ptr,
                          (int) probs->size[2], (int)probs->size[1],
                          options, &gpu_size_bytes);

       float* gpu_workspace;
       THCudaMalloc(state, (void **) &gpu_workspace, gpu_size_bytes);

       compute_ctc_loss(probs_ptr, grads_ptr,
                        labels_ptr, label_sizes_ptr,
                        sizes_ptr, probs->size[2],
                        probs->size[1], costs_ptr,
                        gpu_workspace, options);

       THCudaFree(state, (void *) gpu_workspace);
       return 1;
   }
#endif
