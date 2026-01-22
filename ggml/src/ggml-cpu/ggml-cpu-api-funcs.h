
#ifndef CPU_NAME
#error "CPU_NAME macro not defined, must be defined before including " __FILE__
#endif

#ifdef CPU_PREFIX
#undef CPU_PREFIX
#endif

#define CAT3(a,b,c) a##b##c
#define CAT2(a,b) CAT3(a, _, b)
#define CAT(a,b) CAT2(a,b)

#define CPU_PREFIX(name) CAT(CPU_NAME, name)


// included from ggml-cpu-dispatch.cpp
// declare CPU API functions as defined in ggml-cpu.h but with CPU name prefix

GGML_BACKEND_API void    CPU_PREFIX(ggml_numa_init)(enum ggml_numa_strategy numa); // call once for better performance on NUMA systems
GGML_BACKEND_API bool    CPU_PREFIX(ggml_is_numa)(void); // true if init detected that system has >1 NUMA node

GGML_BACKEND_API struct ggml_tensor * CPU_PREFIX(ggml_new_i32)(struct ggml_context * ctx, int32_t value);
GGML_BACKEND_API struct ggml_tensor * CPU_PREFIX(ggml_new_f32)(struct ggml_context * ctx, float value);

GGML_BACKEND_API struct ggml_tensor * CPU_PREFIX(ggml_set_i32) (struct ggml_tensor * tensor, int32_t value);
GGML_BACKEND_API struct ggml_tensor * CPU_PREFIX(ggml_set_f32) (struct ggml_tensor * tensor, float value);

GGML_BACKEND_API int32_t CPU_PREFIX(ggml_get_i32_1d)(const struct ggml_tensor * tensor, int i);
GGML_BACKEND_API void    CPU_PREFIX(ggml_set_i32_1d)(const struct ggml_tensor * tensor, int i, int32_t value);

GGML_BACKEND_API int32_t CPU_PREFIX(ggml_get_i32_nd)(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3);
GGML_BACKEND_API void    CPU_PREFIX(ggml_set_i32_nd)(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3, int32_t value);

GGML_BACKEND_API float   CPU_PREFIX(ggml_get_f32_1d)(const struct ggml_tensor * tensor, int i);
GGML_BACKEND_API void    CPU_PREFIX(ggml_set_f32_1d)(const struct ggml_tensor * tensor, int i, float value);

GGML_BACKEND_API float   CPU_PREFIX(ggml_get_f32_nd)(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3);
GGML_BACKEND_API void    CPU_PREFIX(ggml_set_f32_nd)(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3, float value);

GGML_BACKEND_API struct ggml_threadpool *      CPU_PREFIX(ggml_threadpool_new)           (struct ggml_threadpool_params  * params);
GGML_BACKEND_API void                          CPU_PREFIX(ggml_threadpool_free)          (struct ggml_threadpool * threadpool);
// GGML_BACKEND_API int                           CPU_PREFIX(ggml_threadpool_get_n_threads) (struct ggml_threadpool * threadpool);
GGML_BACKEND_API void                          CPU_PREFIX(ggml_threadpool_pause)         (struct ggml_threadpool * threadpool);
GGML_BACKEND_API void                          CPU_PREFIX(ggml_threadpool_resume)        (struct ggml_threadpool * threadpool);

GGML_BACKEND_API struct ggml_cplan CPU_PREFIX(ggml_graph_plan)(
              const struct ggml_cgraph * cgraph,
                                   int   n_threads, /* = GGML_DEFAULT_N_THREADS */
                struct ggml_threadpool * threadpool /* = NULL */ );
GGML_BACKEND_API enum ggml_status  CPU_PREFIX(ggml_graph_compute)(struct ggml_cgraph * cgraph, struct ggml_cplan * cplan);

GGML_BACKEND_API enum ggml_status  CPU_PREFIX(ggml_graph_compute_with_ctx)(struct ggml_context * ctx, struct ggml_cgraph * cgraph, int n_threads);



GGML_BACKEND_API const struct ggml_type_traits_cpu * CPU_PREFIX(ggml_get_type_traits_cpu)(enum ggml_type type);

GGML_BACKEND_API void CPU_PREFIX(ggml_cpu_init)(void);

//
// CPU backend
//

GGML_BACKEND_API ggml_backend_t CPU_PREFIX(ggml_backend_cpu_init)(void);

GGML_BACKEND_API bool CPU_PREFIX(ggml_backend_is_cpu)                (ggml_backend_t backend);
GGML_BACKEND_API void CPU_PREFIX(ggml_backend_cpu_set_n_threads)     (ggml_backend_t backend_cpu, int n_threads);
GGML_BACKEND_API void CPU_PREFIX(ggml_backend_cpu_set_threadpool)    (ggml_backend_t backend_cpu, ggml_threadpool_t threadpool);
GGML_BACKEND_API void CPU_PREFIX(ggml_backend_cpu_set_abort_callback)(ggml_backend_t backend_cpu, ggml_abort_callback abort_callback, void * abort_callback_data);

GGML_BACKEND_API ggml_backend_reg_t CPU_PREFIX(ggml_backend_cpu_reg)(void);

GGML_BACKEND_API void CPU_PREFIX(ggml_cpu_fp32_to_i32)(const float *, int32_t *, int64_t);
GGML_BACKEND_API void CPU_PREFIX(ggml_cpu_fp32_to_fp32)(const float *, float *, int64_t);
GGML_BACKEND_API void CPU_PREFIX(ggml_cpu_fp32_to_fp16)(const float *, ggml_fp16_t *, int64_t);
GGML_BACKEND_API void CPU_PREFIX(ggml_cpu_fp16_to_fp32)(const ggml_fp16_t *, float *, int64_t);
GGML_BACKEND_API void CPU_PREFIX(ggml_cpu_fp32_to_bf16)(const float *, ggml_bf16_t *, int64_t);
GGML_BACKEND_API void CPU_PREFIX(ggml_cpu_bf16_to_fp32)(const ggml_bf16_t *, float *, int64_t);


#undef CPU_PREFIX
#undef CAT
#undef CAT2
#undef CAT3
#undef CPU_NAME
