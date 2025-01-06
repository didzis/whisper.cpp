#ifdef __x86_64__

// disable things
#ifdef GGML_BACKEND_DL
#undef GGML_BACKEND_DL
#endif
#define ggml_backend_cpu_x86_score __xyz_hidden_ggml_backend_cpu_x86_score
// a quick hack
#include "cpu-feats-x86.cpp"

#include <cstdio> // for fprintf()

#define PRINTF(...) fprintf(stderr, __VA_ARGS__)


#ifdef  __cplusplus
extern "C" {
#endif

// some definitions and declarations from ggml-cpu.h

    // the compute plan that needs to be prepared for ggml_graph_compute()
    // since https://github.com/ggerganov/ggml/issues/287
    struct ggml_cplan {
        size_t    work_size; // size of work buffer, calculated by `ggml_graph_plan()`
        uint8_t * work_data; // work buffer, to be allocated by caller before calling to `ggml_graph_compute()`

        int n_threads;
        struct ggml_threadpool * threadpool;

        // abort ggml_graph_compute when true
        ggml_abort_callback abort_callback;
        void *              abort_callback_data;
    };

    // numa strategies
    enum ggml_numa_strategy {
        GGML_NUMA_STRATEGY_DISABLED   = 0,
        GGML_NUMA_STRATEGY_DISTRIBUTE = 1,
        GGML_NUMA_STRATEGY_ISOLATE    = 2,
        GGML_NUMA_STRATEGY_NUMACTL    = 3,
        GGML_NUMA_STRATEGY_MIRROR     = 4,
        GGML_NUMA_STRATEGY_COUNT
    };

    // x86
    GGML_BACKEND_API int ggml_cpu_has_sse3       (void);
    GGML_BACKEND_API int ggml_cpu_has_ssse3      (void);
    GGML_BACKEND_API int ggml_cpu_has_avx        (void);
    GGML_BACKEND_API int ggml_cpu_has_avx_vnni   (void);
    GGML_BACKEND_API int ggml_cpu_has_avx2       (void);
    GGML_BACKEND_API int ggml_cpu_has_f16c       (void);
    GGML_BACKEND_API int ggml_cpu_has_fma        (void);
    GGML_BACKEND_API int ggml_cpu_has_avx512     (void);
    GGML_BACKEND_API int ggml_cpu_has_avx512_vbmi(void);
    GGML_BACKEND_API int ggml_cpu_has_avx512_vnni(void);
    GGML_BACKEND_API int ggml_cpu_has_avx512_bf16(void);
    GGML_BACKEND_API int ggml_cpu_has_amx_int8   (void);
    // ARM
    GGML_BACKEND_API int ggml_cpu_has_neon       (void);
    GGML_BACKEND_API int ggml_cpu_has_arm_fma    (void);
    GGML_BACKEND_API int ggml_cpu_has_fp16_va    (void);
    GGML_BACKEND_API int ggml_cpu_has_dotprod    (void);
    GGML_BACKEND_API int ggml_cpu_has_matmul_int8(void);
    GGML_BACKEND_API int ggml_cpu_has_sve        (void);
    GGML_BACKEND_API int ggml_cpu_get_sve_cnt    (void);  // sve vector length in bytes
    // other
    GGML_BACKEND_API int ggml_cpu_has_riscv_v    (void);
    GGML_BACKEND_API int ggml_cpu_has_vsx        (void);
    GGML_BACKEND_API int ggml_cpu_has_wasm_simd  (void);
    GGML_BACKEND_API int ggml_cpu_has_llamafile  (void);

    GGML_BACKEND_API void ggml_cpu_init(void);
    GGML_BACKEND_API ggml_backend_t ggml_backend_cpu_init(void);
    GGML_BACKEND_API ggml_backend_reg_t ggml_backend_cpu_reg(void);




// API declarations for CPU variant: sandybridge

GGML_BACKEND_API void    sandybridge_ggml_numa_init(enum ggml_numa_strategy numa); // call once for better performance on NUMA systems
GGML_BACKEND_API bool    sandybridge_ggml_is_numa(void); // true if init detected that system has >1 NUMA node

GGML_BACKEND_API struct ggml_tensor * sandybridge_ggml_new_i32(struct ggml_context * ctx, int32_t value);
GGML_BACKEND_API struct ggml_tensor * sandybridge_ggml_new_f32(struct ggml_context * ctx, float value);

GGML_BACKEND_API struct ggml_tensor * sandybridge_ggml_set_i32 (struct ggml_tensor * tensor, int32_t value);
GGML_BACKEND_API struct ggml_tensor * sandybridge_ggml_set_f32 (struct ggml_tensor * tensor, float value);

GGML_BACKEND_API int32_t sandybridge_ggml_get_i32_1d(const struct ggml_tensor * tensor, int i);
GGML_BACKEND_API void    sandybridge_ggml_set_i32_1d(const struct ggml_tensor * tensor, int i, int32_t value);

GGML_BACKEND_API int32_t sandybridge_ggml_get_i32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3);
GGML_BACKEND_API void    sandybridge_ggml_set_i32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3, int32_t value);

GGML_BACKEND_API float   sandybridge_ggml_get_f32_1d(const struct ggml_tensor * tensor, int i);
GGML_BACKEND_API void    sandybridge_ggml_set_f32_1d(const struct ggml_tensor * tensor, int i, float value);

GGML_BACKEND_API float   sandybridge_ggml_get_f32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3);
GGML_BACKEND_API void    sandybridge_ggml_set_f32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3, float value);

GGML_BACKEND_API struct ggml_threadpool *      sandybridge_ggml_threadpool_new           (struct ggml_threadpool_params  * params);
GGML_BACKEND_API void                          sandybridge_ggml_threadpool_free          (struct ggml_threadpool * threadpool);
// GGML_BACKEND_API int                           sandybridge_ggml_threadpool_get_n_threads (struct ggml_threadpool * threadpool);
GGML_BACKEND_API void                          sandybridge_ggml_threadpool_pause         (struct ggml_threadpool * threadpool);
GGML_BACKEND_API void                          sandybridge_ggml_threadpool_resume        (struct ggml_threadpool * threadpool);

GGML_BACKEND_API struct ggml_cplan sandybridge_ggml_graph_plan(
              const struct ggml_cgraph * cgraph,
                                   int   n_threads, /* = GGML_DEFAULT_N_THREADS */
                struct ggml_threadpool * threadpool /* = NULL */ );
GGML_BACKEND_API enum ggml_status  sandybridge_ggml_graph_compute(struct ggml_cgraph * cgraph, struct ggml_cplan * cplan);

GGML_BACKEND_API enum ggml_status  sandybridge_ggml_graph_compute_with_ctx(struct ggml_context * ctx, struct ggml_cgraph * cgraph, int n_threads);



GGML_BACKEND_API const struct ggml_type_traits_cpu * sandybridge_ggml_get_type_traits_cpu(enum ggml_type type);

GGML_BACKEND_API void sandybridge_ggml_cpu_init(void);

//
// CPU backend
//

GGML_BACKEND_API ggml_backend_t sandybridge_ggml_backend_cpu_init(void);

GGML_BACKEND_API bool sandybridge_ggml_backend_is_cpu                (ggml_backend_t backend);
GGML_BACKEND_API void sandybridge_ggml_backend_cpu_set_n_threads     (ggml_backend_t backend_cpu, int n_threads);
GGML_BACKEND_API void sandybridge_ggml_backend_cpu_set_threadpool    (ggml_backend_t backend_cpu, ggml_threadpool_t threadpool);
GGML_BACKEND_API void sandybridge_ggml_backend_cpu_set_abort_callback(ggml_backend_t backend_cpu, ggml_abort_callback abort_callback, void * abort_callback_data);

GGML_BACKEND_API ggml_backend_reg_t sandybridge_ggml_backend_cpu_reg(void);




// API declarations for CPU variant: haswell

GGML_BACKEND_API void    haswell_ggml_numa_init(enum ggml_numa_strategy numa); // call once for better performance on NUMA systems
GGML_BACKEND_API bool    haswell_ggml_is_numa(void); // true if init detected that system has >1 NUMA node

GGML_BACKEND_API struct ggml_tensor * haswell_ggml_new_i32(struct ggml_context * ctx, int32_t value);
GGML_BACKEND_API struct ggml_tensor * haswell_ggml_new_f32(struct ggml_context * ctx, float value);

GGML_BACKEND_API struct ggml_tensor * haswell_ggml_set_i32 (struct ggml_tensor * tensor, int32_t value);
GGML_BACKEND_API struct ggml_tensor * haswell_ggml_set_f32 (struct ggml_tensor * tensor, float value);

GGML_BACKEND_API int32_t haswell_ggml_get_i32_1d(const struct ggml_tensor * tensor, int i);
GGML_BACKEND_API void    haswell_ggml_set_i32_1d(const struct ggml_tensor * tensor, int i, int32_t value);

GGML_BACKEND_API int32_t haswell_ggml_get_i32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3);
GGML_BACKEND_API void    haswell_ggml_set_i32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3, int32_t value);

GGML_BACKEND_API float   haswell_ggml_get_f32_1d(const struct ggml_tensor * tensor, int i);
GGML_BACKEND_API void    haswell_ggml_set_f32_1d(const struct ggml_tensor * tensor, int i, float value);

GGML_BACKEND_API float   haswell_ggml_get_f32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3);
GGML_BACKEND_API void    haswell_ggml_set_f32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3, float value);

GGML_BACKEND_API struct ggml_threadpool *      haswell_ggml_threadpool_new           (struct ggml_threadpool_params  * params);
GGML_BACKEND_API void                          haswell_ggml_threadpool_free          (struct ggml_threadpool * threadpool);
// GGML_BACKEND_API int                           haswell_ggml_threadpool_get_n_threads (struct ggml_threadpool * threadpool);
GGML_BACKEND_API void                          haswell_ggml_threadpool_pause         (struct ggml_threadpool * threadpool);
GGML_BACKEND_API void                          haswell_ggml_threadpool_resume        (struct ggml_threadpool * threadpool);

GGML_BACKEND_API struct ggml_cplan haswell_ggml_graph_plan(
              const struct ggml_cgraph * cgraph,
                                   int   n_threads, /* = GGML_DEFAULT_N_THREADS */
                struct ggml_threadpool * threadpool /* = NULL */ );
GGML_BACKEND_API enum ggml_status  haswell_ggml_graph_compute(struct ggml_cgraph * cgraph, struct ggml_cplan * cplan);

GGML_BACKEND_API enum ggml_status  haswell_ggml_graph_compute_with_ctx(struct ggml_context * ctx, struct ggml_cgraph * cgraph, int n_threads);



GGML_BACKEND_API const struct ggml_type_traits_cpu * haswell_ggml_get_type_traits_cpu(enum ggml_type type);

GGML_BACKEND_API void haswell_ggml_cpu_init(void);

//
// CPU backend
//

GGML_BACKEND_API ggml_backend_t haswell_ggml_backend_cpu_init(void);

GGML_BACKEND_API bool haswell_ggml_backend_is_cpu                (ggml_backend_t backend);
GGML_BACKEND_API void haswell_ggml_backend_cpu_set_n_threads     (ggml_backend_t backend_cpu, int n_threads);
GGML_BACKEND_API void haswell_ggml_backend_cpu_set_threadpool    (ggml_backend_t backend_cpu, ggml_threadpool_t threadpool);
GGML_BACKEND_API void haswell_ggml_backend_cpu_set_abort_callback(ggml_backend_t backend_cpu, ggml_abort_callback abort_callback, void * abort_callback_data);

GGML_BACKEND_API ggml_backend_reg_t haswell_ggml_backend_cpu_reg(void);




// API declarations for CPU variant: skylakex


GGML_BACKEND_API void    skylakex_ggml_numa_init(enum ggml_numa_strategy numa); // call once for better performance on NUMA systems
GGML_BACKEND_API bool    skylakex_ggml_is_numa(void); // true if init detected that system has >1 NUMA node

GGML_BACKEND_API struct ggml_tensor * skylakex_ggml_new_i32(struct ggml_context * ctx, int32_t value);
GGML_BACKEND_API struct ggml_tensor * skylakex_ggml_new_f32(struct ggml_context * ctx, float value);

GGML_BACKEND_API struct ggml_tensor * skylakex_ggml_set_i32 (struct ggml_tensor * tensor, int32_t value);
GGML_BACKEND_API struct ggml_tensor * skylakex_ggml_set_f32 (struct ggml_tensor * tensor, float value);

GGML_BACKEND_API int32_t skylakex_ggml_get_i32_1d(const struct ggml_tensor * tensor, int i);
GGML_BACKEND_API void    skylakex_ggml_set_i32_1d(const struct ggml_tensor * tensor, int i, int32_t value);

GGML_BACKEND_API int32_t skylakex_ggml_get_i32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3);
GGML_BACKEND_API void    skylakex_ggml_set_i32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3, int32_t value);

GGML_BACKEND_API float   skylakex_ggml_get_f32_1d(const struct ggml_tensor * tensor, int i);
GGML_BACKEND_API void    skylakex_ggml_set_f32_1d(const struct ggml_tensor * tensor, int i, float value);

GGML_BACKEND_API float   skylakex_ggml_get_f32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3);
GGML_BACKEND_API void    skylakex_ggml_set_f32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3, float value);

GGML_BACKEND_API struct ggml_threadpool *      skylakex_ggml_threadpool_new           (struct ggml_threadpool_params  * params);
GGML_BACKEND_API void                          skylakex_ggml_threadpool_free          (struct ggml_threadpool * threadpool);
// GGML_BACKEND_API int                           skylakex_ggml_threadpool_get_n_threads (struct ggml_threadpool * threadpool);
GGML_BACKEND_API void                          skylakex_ggml_threadpool_pause         (struct ggml_threadpool * threadpool);
GGML_BACKEND_API void                          skylakex_ggml_threadpool_resume        (struct ggml_threadpool * threadpool);

GGML_BACKEND_API struct ggml_cplan skylakex_ggml_graph_plan(
              const struct ggml_cgraph * cgraph,
                                   int   n_threads, /* = GGML_DEFAULT_N_THREADS */
                struct ggml_threadpool * threadpool /* = NULL */ );
GGML_BACKEND_API enum ggml_status  skylakex_ggml_graph_compute(struct ggml_cgraph * cgraph, struct ggml_cplan * cplan);

GGML_BACKEND_API enum ggml_status  skylakex_ggml_graph_compute_with_ctx(struct ggml_context * ctx, struct ggml_cgraph * cgraph, int n_threads);



GGML_BACKEND_API const struct ggml_type_traits_cpu * skylakex_ggml_get_type_traits_cpu(enum ggml_type type);

GGML_BACKEND_API void skylakex_ggml_cpu_init(void);

//
// CPU backend
//

GGML_BACKEND_API ggml_backend_t skylakex_ggml_backend_cpu_init(void);

GGML_BACKEND_API bool skylakex_ggml_backend_is_cpu                (ggml_backend_t backend);
GGML_BACKEND_API void skylakex_ggml_backend_cpu_set_n_threads     (ggml_backend_t backend_cpu, int n_threads);
GGML_BACKEND_API void skylakex_ggml_backend_cpu_set_threadpool    (ggml_backend_t backend_cpu, ggml_threadpool_t threadpool);
GGML_BACKEND_API void skylakex_ggml_backend_cpu_set_abort_callback(ggml_backend_t backend_cpu, ggml_abort_callback abort_callback, void * abort_callback_data);

GGML_BACKEND_API ggml_backend_reg_t skylakex_ggml_backend_cpu_reg(void);




// API declarations for CPU variant: icelake


GGML_BACKEND_API void    icelake_ggml_numa_init(enum ggml_numa_strategy numa); // call once for better performance on NUMA systems
GGML_BACKEND_API bool    icelake_ggml_is_numa(void); // true if init detected that system has >1 NUMA node

GGML_BACKEND_API struct ggml_tensor * icelake_ggml_new_i32(struct ggml_context * ctx, int32_t value);
GGML_BACKEND_API struct ggml_tensor * icelake_ggml_new_f32(struct ggml_context * ctx, float value);

GGML_BACKEND_API struct ggml_tensor * icelake_ggml_set_i32 (struct ggml_tensor * tensor, int32_t value);
GGML_BACKEND_API struct ggml_tensor * icelake_ggml_set_f32 (struct ggml_tensor * tensor, float value);

GGML_BACKEND_API int32_t icelake_ggml_get_i32_1d(const struct ggml_tensor * tensor, int i);
GGML_BACKEND_API void    icelake_ggml_set_i32_1d(const struct ggml_tensor * tensor, int i, int32_t value);

GGML_BACKEND_API int32_t icelake_ggml_get_i32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3);
GGML_BACKEND_API void    icelake_ggml_set_i32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3, int32_t value);

GGML_BACKEND_API float   icelake_ggml_get_f32_1d(const struct ggml_tensor * tensor, int i);
GGML_BACKEND_API void    icelake_ggml_set_f32_1d(const struct ggml_tensor * tensor, int i, float value);

GGML_BACKEND_API float   icelake_ggml_get_f32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3);
GGML_BACKEND_API void    icelake_ggml_set_f32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3, float value);

GGML_BACKEND_API struct ggml_threadpool *      icelake_ggml_threadpool_new           (struct ggml_threadpool_params  * params);
GGML_BACKEND_API void                          icelake_ggml_threadpool_free          (struct ggml_threadpool * threadpool);
// GGML_BACKEND_API int                           icelake_ggml_threadpool_get_n_threads (struct ggml_threadpool * threadpool);
GGML_BACKEND_API void                          icelake_ggml_threadpool_pause         (struct ggml_threadpool * threadpool);
GGML_BACKEND_API void                          icelake_ggml_threadpool_resume        (struct ggml_threadpool * threadpool);

GGML_BACKEND_API struct ggml_cplan icelake_ggml_graph_plan(
              const struct ggml_cgraph * cgraph,
                                   int   n_threads, /* = GGML_DEFAULT_N_THREADS */
                struct ggml_threadpool * threadpool /* = NULL */ );
GGML_BACKEND_API enum ggml_status  icelake_ggml_graph_compute(struct ggml_cgraph * cgraph, struct ggml_cplan * cplan);

GGML_BACKEND_API enum ggml_status  icelake_ggml_graph_compute_with_ctx(struct ggml_context * ctx, struct ggml_cgraph * cgraph, int n_threads);



GGML_BACKEND_API const struct ggml_type_traits_cpu * icelake_ggml_get_type_traits_cpu(enum ggml_type type);

GGML_BACKEND_API void icelake_ggml_cpu_init(void);

//
// CPU backend
//

GGML_BACKEND_API ggml_backend_t icelake_ggml_backend_cpu_init(void);

GGML_BACKEND_API bool icelake_ggml_backend_is_cpu                (ggml_backend_t backend);
GGML_BACKEND_API void icelake_ggml_backend_cpu_set_n_threads     (ggml_backend_t backend_cpu, int n_threads);
GGML_BACKEND_API void icelake_ggml_backend_cpu_set_threadpool    (ggml_backend_t backend_cpu, ggml_threadpool_t threadpool);
GGML_BACKEND_API void icelake_ggml_backend_cpu_set_abort_callback(ggml_backend_t backend_cpu, ggml_abort_callback abort_callback, void * abort_callback_data);

GGML_BACKEND_API ggml_backend_reg_t icelake_ggml_backend_cpu_reg(void);




// API declarations for CPU variant: alderlake


GGML_BACKEND_API void    alderlake_ggml_numa_init(enum ggml_numa_strategy numa); // call once for better performance on NUMA systems
GGML_BACKEND_API bool    alderlake_ggml_is_numa(void); // true if init detected that system has >1 NUMA node

GGML_BACKEND_API struct ggml_tensor * alderlake_ggml_new_i32(struct ggml_context * ctx, int32_t value);
GGML_BACKEND_API struct ggml_tensor * alderlake_ggml_new_f32(struct ggml_context * ctx, float value);

GGML_BACKEND_API struct ggml_tensor * alderlake_ggml_set_i32 (struct ggml_tensor * tensor, int32_t value);
GGML_BACKEND_API struct ggml_tensor * alderlake_ggml_set_f32 (struct ggml_tensor * tensor, float value);

GGML_BACKEND_API int32_t alderlake_ggml_get_i32_1d(const struct ggml_tensor * tensor, int i);
GGML_BACKEND_API void    alderlake_ggml_set_i32_1d(const struct ggml_tensor * tensor, int i, int32_t value);

GGML_BACKEND_API int32_t alderlake_ggml_get_i32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3);
GGML_BACKEND_API void    alderlake_ggml_set_i32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3, int32_t value);

GGML_BACKEND_API float   alderlake_ggml_get_f32_1d(const struct ggml_tensor * tensor, int i);
GGML_BACKEND_API void    alderlake_ggml_set_f32_1d(const struct ggml_tensor * tensor, int i, float value);

GGML_BACKEND_API float   alderlake_ggml_get_f32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3);
GGML_BACKEND_API void    alderlake_ggml_set_f32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3, float value);

GGML_BACKEND_API struct ggml_threadpool *      alderlake_ggml_threadpool_new           (struct ggml_threadpool_params  * params);
GGML_BACKEND_API void                          alderlake_ggml_threadpool_free          (struct ggml_threadpool * threadpool);
// GGML_BACKEND_API int                           alderlake_ggml_threadpool_get_n_threads (struct ggml_threadpool * threadpool);
GGML_BACKEND_API void                          alderlake_ggml_threadpool_pause         (struct ggml_threadpool * threadpool);
GGML_BACKEND_API void                          alderlake_ggml_threadpool_resume        (struct ggml_threadpool * threadpool);

GGML_BACKEND_API struct ggml_cplan alderlake_ggml_graph_plan(
              const struct ggml_cgraph * cgraph,
                                   int   n_threads, /* = GGML_DEFAULT_N_THREADS */
                struct ggml_threadpool * threadpool /* = NULL */ );
GGML_BACKEND_API enum ggml_status  alderlake_ggml_graph_compute(struct ggml_cgraph * cgraph, struct ggml_cplan * cplan);

GGML_BACKEND_API enum ggml_status  alderlake_ggml_graph_compute_with_ctx(struct ggml_context * ctx, struct ggml_cgraph * cgraph, int n_threads);



GGML_BACKEND_API const struct ggml_type_traits_cpu * alderlake_ggml_get_type_traits_cpu(enum ggml_type type);

GGML_BACKEND_API void alderlake_ggml_cpu_init(void);

//
// CPU backend
//

GGML_BACKEND_API ggml_backend_t alderlake_ggml_backend_cpu_init(void);

GGML_BACKEND_API bool alderlake_ggml_backend_is_cpu                (ggml_backend_t backend);
GGML_BACKEND_API void alderlake_ggml_backend_cpu_set_n_threads     (ggml_backend_t backend_cpu, int n_threads);
GGML_BACKEND_API void alderlake_ggml_backend_cpu_set_threadpool    (ggml_backend_t backend_cpu, ggml_threadpool_t threadpool);
GGML_BACKEND_API void alderlake_ggml_backend_cpu_set_abort_callback(ggml_backend_t backend_cpu, ggml_abort_callback abort_callback, void * abort_callback_data);

GGML_BACKEND_API ggml_backend_reg_t alderlake_ggml_backend_cpu_reg(void);




// API declarations for CPU variant: sapphirerapids


GGML_BACKEND_API void    sapphirerapids_ggml_numa_init(enum ggml_numa_strategy numa); // call once for better performance on NUMA systems
GGML_BACKEND_API bool    sapphirerapids_ggml_is_numa(void); // true if init detected that system has >1 NUMA node

GGML_BACKEND_API struct ggml_tensor * sapphirerapids_ggml_new_i32(struct ggml_context * ctx, int32_t value);
GGML_BACKEND_API struct ggml_tensor * sapphirerapids_ggml_new_f32(struct ggml_context * ctx, float value);

GGML_BACKEND_API struct ggml_tensor * sapphirerapids_ggml_set_i32 (struct ggml_tensor * tensor, int32_t value);
GGML_BACKEND_API struct ggml_tensor * sapphirerapids_ggml_set_f32 (struct ggml_tensor * tensor, float value);

GGML_BACKEND_API int32_t sapphirerapids_ggml_get_i32_1d(const struct ggml_tensor * tensor, int i);
GGML_BACKEND_API void    sapphirerapids_ggml_set_i32_1d(const struct ggml_tensor * tensor, int i, int32_t value);

GGML_BACKEND_API int32_t sapphirerapids_ggml_get_i32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3);
GGML_BACKEND_API void    sapphirerapids_ggml_set_i32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3, int32_t value);

GGML_BACKEND_API float   sapphirerapids_ggml_get_f32_1d(const struct ggml_tensor * tensor, int i);
GGML_BACKEND_API void    sapphirerapids_ggml_set_f32_1d(const struct ggml_tensor * tensor, int i, float value);

GGML_BACKEND_API float   sapphirerapids_ggml_get_f32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3);
GGML_BACKEND_API void    sapphirerapids_ggml_set_f32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3, float value);

GGML_BACKEND_API struct ggml_threadpool *      sapphirerapids_ggml_threadpool_new           (struct ggml_threadpool_params  * params);
GGML_BACKEND_API void                          sapphirerapids_ggml_threadpool_free          (struct ggml_threadpool * threadpool);
// GGML_BACKEND_API int                           sapphirerapids_ggml_threadpool_get_n_threads (struct ggml_threadpool * threadpool);
GGML_BACKEND_API void                          sapphirerapids_ggml_threadpool_pause         (struct ggml_threadpool * threadpool);
GGML_BACKEND_API void                          sapphirerapids_ggml_threadpool_resume        (struct ggml_threadpool * threadpool);

GGML_BACKEND_API struct ggml_cplan sapphirerapids_ggml_graph_plan(
              const struct ggml_cgraph * cgraph,
                                   int   n_threads, /* = GGML_DEFAULT_N_THREADS */
                struct ggml_threadpool * threadpool /* = NULL */ );
GGML_BACKEND_API enum ggml_status  sapphirerapids_ggml_graph_compute(struct ggml_cgraph * cgraph, struct ggml_cplan * cplan);

GGML_BACKEND_API enum ggml_status  sapphirerapids_ggml_graph_compute_with_ctx(struct ggml_context * ctx, struct ggml_cgraph * cgraph, int n_threads);



GGML_BACKEND_API const struct ggml_type_traits_cpu * sapphirerapids_ggml_get_type_traits_cpu(enum ggml_type type);

GGML_BACKEND_API void sapphirerapids_ggml_cpu_init(void);

//
// CPU backend
//

GGML_BACKEND_API ggml_backend_t sapphirerapids_ggml_backend_cpu_init(void);

GGML_BACKEND_API bool sapphirerapids_ggml_backend_is_cpu                (ggml_backend_t backend);
GGML_BACKEND_API void sapphirerapids_ggml_backend_cpu_set_n_threads     (ggml_backend_t backend_cpu, int n_threads);
GGML_BACKEND_API void sapphirerapids_ggml_backend_cpu_set_threadpool    (ggml_backend_t backend_cpu, ggml_threadpool_t threadpool);
GGML_BACKEND_API void sapphirerapids_ggml_backend_cpu_set_abort_callback(ggml_backend_t backend_cpu, ggml_abort_callback abort_callback, void * abort_callback_data);

GGML_BACKEND_API ggml_backend_reg_t sapphirerapids_ggml_backend_cpu_reg(void);





// pointers to selected CPU backend API (except constant CPU feature functions,
// replaced by implementations with runtime CPU feature detection below)
// from ggml-cpu.h converted to pointers with same name for exporting
// with exeception of ggml_cpu_init, ggml_backend_cpu_init ggml_backend_cpu_reg,
// which are implemented as wrapper functions below that trigger CPU backend selection

void    (*ggml_numa_init)(enum ggml_numa_strategy numa); // call once for better performance on NUMA systems
bool    (*ggml_is_numa)(void); // true if init detected that system has >1 NUMA node

struct ggml_tensor * (*ggml_new_i32)(struct ggml_context * ctx, int32_t value);
struct ggml_tensor * (*ggml_new_f32)(struct ggml_context * ctx, float value);

struct ggml_tensor * (*ggml_set_i32)(struct ggml_tensor * tensor, int32_t value);
struct ggml_tensor * (*ggml_set_f32)(struct ggml_tensor * tensor, float value);

int32_t (*ggml_get_i32_1d)(const struct ggml_tensor * tensor, int i);
void    (*ggml_set_i32_1d)(const struct ggml_tensor * tensor, int i, int32_t value);

int32_t (*ggml_get_i32_nd)(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3);
void    (*ggml_set_i32_nd)(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3, int32_t value);

float   (*ggml_get_f32_1d)(const struct ggml_tensor * tensor, int i);
void    (*ggml_set_f32_1d)(const struct ggml_tensor * tensor, int i, float value);

float   (*ggml_get_f32_nd)(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3);
void    (*ggml_set_f32_nd)(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3, float value);

struct ggml_threadpool *      (*ggml_threadpool_new          )(struct ggml_threadpool_params  * params);
void                          (*ggml_threadpool_free         )(struct ggml_threadpool * threadpool);
// int                           (*ggml_threadpool_get_n_threads)(struct ggml_threadpool * threadpool);
void                          (*ggml_threadpool_pause        )(struct ggml_threadpool * threadpool);
void                          (*ggml_threadpool_resume       )(struct ggml_threadpool * threadpool);

// ggml_graph_plan() has to be called before ggml_graph_compute()
// when plan.work_size > 0, caller must allocate memory for plan.work_data
struct ggml_cplan (*ggml_graph_plan)(
              const struct ggml_cgraph * cgraph,
                                   int   n_threads, /* = GGML_DEFAULT_N_THREADS */
                struct ggml_threadpool * threadpool /* = NULL */ );
enum ggml_status  (*ggml_graph_compute)(struct ggml_cgraph * cgraph, struct ggml_cplan * cplan);

// same as ggml_graph_compute() but the work data is allocated as a part of the context
// note: the drawback of this API is that you must have ensured that the context has enough memory for the work data
enum ggml_status  (*ggml_graph_compute_with_ctx)(struct ggml_context * ctx, struct ggml_cgraph * cgraph, int n_threads);

const struct ggml_type_traits_cpu * (*ggml_get_type_traits_cpu)(enum ggml_type type);

void (*_ggml_cpu_init)(void);

//
// CPU backend
//

ggml_backend_t (*_ggml_backend_cpu_init)(void);

bool (*ggml_backend_is_cpu                )(ggml_backend_t backend);
void (*ggml_backend_cpu_set_n_threads     )(ggml_backend_t backend_cpu, int n_threads);
void (*ggml_backend_cpu_set_threadpool    )(ggml_backend_t backend_cpu, ggml_threadpool_t threadpool);
void (*ggml_backend_cpu_set_abort_callback)(ggml_backend_t backend_cpu, ggml_abort_callback abort_callback, void * abort_callback_data);

ggml_backend_reg_t (*_ggml_backend_cpu_reg)(void);





// implementations

static struct cpuid_x86& ggml_cpu_dispatch_cpuid() {
    static struct cpuid_x86 cpu_features;
    return cpu_features;
}

bool ggml_cpu_backend_selected = false;

void ggml_select_cpu_backend();



// x86
GGML_BACKEND_API int ggml_cpu_has_sse3       (void) { return ggml_cpu_dispatch_cpuid().SSE3(); }
GGML_BACKEND_API int ggml_cpu_has_ssse3      (void) { return ggml_cpu_dispatch_cpuid().SSSE3(); }
GGML_BACKEND_API int ggml_cpu_has_avx        (void) { return ggml_cpu_dispatch_cpuid().AVX(); }
GGML_BACKEND_API int ggml_cpu_has_avx_vnni   (void) { return ggml_cpu_dispatch_cpuid().AVX_VNNI(); }
GGML_BACKEND_API int ggml_cpu_has_avx2       (void) { return ggml_cpu_dispatch_cpuid().AVX2(); }
GGML_BACKEND_API int ggml_cpu_has_f16c       (void) { return ggml_cpu_dispatch_cpuid().F16C(); }
GGML_BACKEND_API int ggml_cpu_has_fma        (void) { return ggml_cpu_dispatch_cpuid().FMA(); }
GGML_BACKEND_API int ggml_cpu_has_avx512     (void) { return ggml_cpu_dispatch_cpuid().AVX512F(); }
GGML_BACKEND_API int ggml_cpu_has_avx512_vbmi(void) { return ggml_cpu_dispatch_cpuid().AVX512_VBMI(); }
GGML_BACKEND_API int ggml_cpu_has_avx512_vnni(void) { return ggml_cpu_dispatch_cpuid().AVX512_VNNI(); }
GGML_BACKEND_API int ggml_cpu_has_avx512_bf16(void) { return ggml_cpu_dispatch_cpuid().AVX512_BF16(); }
GGML_BACKEND_API int ggml_cpu_has_amx_int8   (void) { return ggml_cpu_dispatch_cpuid().AMX_INT8(); }
// ARM
GGML_BACKEND_API int ggml_cpu_has_neon       (void) { return 0; }
GGML_BACKEND_API int ggml_cpu_has_arm_fma    (void) { return 0; }
GGML_BACKEND_API int ggml_cpu_has_fp16_va    (void) { return 0; }
GGML_BACKEND_API int ggml_cpu_has_dotprod    (void) { return 0; }
GGML_BACKEND_API int ggml_cpu_has_matmul_int8(void) { return 0; }
GGML_BACKEND_API int ggml_cpu_has_sve        (void) { return 0; }
GGML_BACKEND_API int ggml_cpu_get_sve_cnt    (void) { return 0; }  // sve vector length in bytes
// other
GGML_BACKEND_API int ggml_cpu_has_riscv_v    (void) { return 0; }
GGML_BACKEND_API int ggml_cpu_has_vsx        (void) { return 0; }
GGML_BACKEND_API int ggml_cpu_has_wasm_simd  (void) { return 0; }
GGML_BACKEND_API int ggml_cpu_has_llamafile  (void) { return 0; }


// these wapper functions will trigger CPU backend selection

GGML_BACKEND_API void ggml_cpu_init(void) {
    ggml_select_cpu_backend();
    _ggml_cpu_init();
}

GGML_BACKEND_API ggml_backend_t ggml_backend_cpu_init(void) {
    ggml_select_cpu_backend();
    return _ggml_backend_cpu_init();
}

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_cpu_reg(void) {
    ggml_select_cpu_backend();
    return _ggml_backend_cpu_reg();
}


// selects detected CPU backend

void ggml_select_cpu_backend() {
    if (ggml_cpu_backend_selected)
        return;

    // CPU variants to support from CMakeLists.txt:
    // ggml_add_cpu_backend_variant(sapphirerapids AVX F16C AVX2 FMA AVX512 AVX512_VBMI AVX512_VNNI AVX512_BF16 AMX_TILE AMX_INT8)
    // ggml_add_cpu_backend_variant(icelake        AVX F16C AVX2 FMA AVX512 AVX512_VBMI AVX512_VNNI)
    // ggml_add_cpu_backend_variant(skylakex       AVX F16C AVX2 FMA AVX512)
    // ggml_add_cpu_backend_variant(alderlake      AVX F16C AVX2 FMA AVX_VNNI)
    // ggml_add_cpu_backend_variant(haswell        AVX F16C AVX2 FMA)
    // ggml_add_cpu_backend_variant(sandybridge    AVX)

    auto& cpu_features = ggml_cpu_dispatch_cpuid();

    bool sandybridge = cpu_features.AVX();
    bool haswell = cpu_features.AVX() && cpu_features.F16C() && cpu_features.AVX2() && cpu_features.FMA();
    bool alderlake = haswell && cpu_features.AVX_VNNI();
    bool skylakex = haswell && cpu_features.AVX512F();
    bool icelake = skylakex && cpu_features.AVX512_VBMI() && cpu_features.AVX512_VNNI();
    bool sapphirerapids = icelake && cpu_features.AVX512_BF16() && cpu_features.AMX_TILE() && cpu_features.AMX_INT8();

    if (sapphirerapids) {

        PRINTF("ggml_cpu_backend: selecting CPU backend: sapphirerapids\n");

        ggml_numa_init = sapphirerapids_ggml_numa_init;
        ggml_is_numa = sapphirerapids_ggml_is_numa;

        ggml_new_i32 = sapphirerapids_ggml_new_i32;
        ggml_new_f32 = sapphirerapids_ggml_new_f32;

        ggml_set_i32 = sapphirerapids_ggml_set_i32;
        ggml_set_f32 = sapphirerapids_ggml_set_f32;

        ggml_get_i32_1d = sapphirerapids_ggml_get_i32_1d;
        ggml_set_i32_1d = sapphirerapids_ggml_set_i32_1d;

        ggml_get_i32_nd = sapphirerapids_ggml_get_i32_nd;
        ggml_set_i32_nd = sapphirerapids_ggml_set_i32_nd;

        ggml_get_f32_1d = sapphirerapids_ggml_get_f32_1d;
        ggml_set_f32_1d = sapphirerapids_ggml_set_f32_1d;

        ggml_get_f32_nd = sapphirerapids_ggml_get_f32_nd;
        ggml_set_f32_nd = sapphirerapids_ggml_set_f32_nd;

        ggml_threadpool_new = sapphirerapids_ggml_threadpool_new;
        ggml_threadpool_free = sapphirerapids_ggml_threadpool_free;
        // ggml_threadpool_get_n_threads = sapphirerapids_ggml_threadpool_get_n_threads;
        ggml_threadpool_pause = sapphirerapids_ggml_threadpool_pause;
        ggml_threadpool_resume = sapphirerapids_ggml_threadpool_resume;

        ggml_graph_plan = sapphirerapids_ggml_graph_plan;
        ggml_graph_compute = sapphirerapids_ggml_graph_compute;
        ggml_graph_compute_with_ctx = sapphirerapids_ggml_graph_compute_with_ctx;

        ggml_get_type_traits_cpu = sapphirerapids_ggml_get_type_traits_cpu;

        _ggml_cpu_init = sapphirerapids_ggml_cpu_init;

        _ggml_backend_cpu_init = sapphirerapids_ggml_backend_cpu_init;

        ggml_backend_is_cpu = sapphirerapids_ggml_backend_is_cpu;
        ggml_backend_cpu_set_n_threads = sapphirerapids_ggml_backend_cpu_set_n_threads;
        ggml_backend_cpu_set_threadpool = sapphirerapids_ggml_backend_cpu_set_threadpool;
        ggml_backend_cpu_set_abort_callback = sapphirerapids_ggml_backend_cpu_set_abort_callback;

        _ggml_backend_cpu_reg = sapphirerapids_ggml_backend_cpu_reg;

    } else if (icelake) {

        PRINTF("ggml_cpu_backend: selecting CPU backend: icelake\n");

        ggml_numa_init = icelake_ggml_numa_init;
        ggml_is_numa = icelake_ggml_is_numa;

        ggml_new_i32 = icelake_ggml_new_i32;
        ggml_new_f32 = icelake_ggml_new_f32;

        ggml_set_i32 = icelake_ggml_set_i32;
        ggml_set_f32 = icelake_ggml_set_f32;

        ggml_get_i32_1d = icelake_ggml_get_i32_1d;
        ggml_set_i32_1d = icelake_ggml_set_i32_1d;

        ggml_get_i32_nd = icelake_ggml_get_i32_nd;
        ggml_set_i32_nd = icelake_ggml_set_i32_nd;

        ggml_get_f32_1d = icelake_ggml_get_f32_1d;
        ggml_set_f32_1d = icelake_ggml_set_f32_1d;

        ggml_get_f32_nd = icelake_ggml_get_f32_nd;
        ggml_set_f32_nd = icelake_ggml_set_f32_nd;

        ggml_threadpool_new = icelake_ggml_threadpool_new;
        ggml_threadpool_free = icelake_ggml_threadpool_free;
        // ggml_threadpool_get_n_threads = icelake_ggml_threadpool_get_n_threads;
        ggml_threadpool_pause = icelake_ggml_threadpool_pause;
        ggml_threadpool_resume = icelake_ggml_threadpool_resume;

        ggml_graph_plan = icelake_ggml_graph_plan;
        ggml_graph_compute = icelake_ggml_graph_compute;
        ggml_graph_compute_with_ctx = icelake_ggml_graph_compute_with_ctx;

        ggml_get_type_traits_cpu = icelake_ggml_get_type_traits_cpu;

        _ggml_cpu_init = icelake_ggml_cpu_init;

        _ggml_backend_cpu_init = icelake_ggml_backend_cpu_init;

        ggml_backend_is_cpu = icelake_ggml_backend_is_cpu;
        ggml_backend_cpu_set_n_threads = icelake_ggml_backend_cpu_set_n_threads;
        ggml_backend_cpu_set_threadpool = icelake_ggml_backend_cpu_set_threadpool;
        ggml_backend_cpu_set_abort_callback = icelake_ggml_backend_cpu_set_abort_callback;

        _ggml_backend_cpu_reg = icelake_ggml_backend_cpu_reg;

    } else if (skylakex) {

        PRINTF("ggml_cpu_backend: selecting CPU backend: skylakex\n");

        ggml_numa_init = skylakex_ggml_numa_init;
        ggml_is_numa = skylakex_ggml_is_numa;

        ggml_new_i32 = skylakex_ggml_new_i32;
        ggml_new_f32 = skylakex_ggml_new_f32;

        ggml_set_i32 = skylakex_ggml_set_i32;
        ggml_set_f32 = skylakex_ggml_set_f32;

        ggml_get_i32_1d = skylakex_ggml_get_i32_1d;
        ggml_set_i32_1d = skylakex_ggml_set_i32_1d;

        ggml_get_i32_nd = skylakex_ggml_get_i32_nd;
        ggml_set_i32_nd = skylakex_ggml_set_i32_nd;

        ggml_get_f32_1d = skylakex_ggml_get_f32_1d;
        ggml_set_f32_1d = skylakex_ggml_set_f32_1d;

        ggml_get_f32_nd = skylakex_ggml_get_f32_nd;
        ggml_set_f32_nd = skylakex_ggml_set_f32_nd;

        ggml_threadpool_new = skylakex_ggml_threadpool_new;
        ggml_threadpool_free = skylakex_ggml_threadpool_free;
        // ggml_threadpool_get_n_threads = skylakex_ggml_threadpool_get_n_threads;
        ggml_threadpool_pause = skylakex_ggml_threadpool_pause;
        ggml_threadpool_resume = skylakex_ggml_threadpool_resume;

        ggml_graph_plan = skylakex_ggml_graph_plan;
        ggml_graph_compute = skylakex_ggml_graph_compute;
        ggml_graph_compute_with_ctx = skylakex_ggml_graph_compute_with_ctx;

        ggml_get_type_traits_cpu = skylakex_ggml_get_type_traits_cpu;

        _ggml_cpu_init = skylakex_ggml_cpu_init;

        _ggml_backend_cpu_init = skylakex_ggml_backend_cpu_init;

        ggml_backend_is_cpu = skylakex_ggml_backend_is_cpu;
        ggml_backend_cpu_set_n_threads = skylakex_ggml_backend_cpu_set_n_threads;
        ggml_backend_cpu_set_threadpool = skylakex_ggml_backend_cpu_set_threadpool;
        ggml_backend_cpu_set_abort_callback = skylakex_ggml_backend_cpu_set_abort_callback;

        _ggml_backend_cpu_reg = skylakex_ggml_backend_cpu_reg;

    } else if (alderlake) {

        PRINTF("ggml_cpu_backend: selecting CPU backend: alderlake\n");

        ggml_numa_init = alderlake_ggml_numa_init;
        ggml_is_numa = alderlake_ggml_is_numa;

        ggml_new_i32 = alderlake_ggml_new_i32;
        ggml_new_f32 = alderlake_ggml_new_f32;

        ggml_set_i32 = alderlake_ggml_set_i32;
        ggml_set_f32 = alderlake_ggml_set_f32;

        ggml_get_i32_1d = alderlake_ggml_get_i32_1d;
        ggml_set_i32_1d = alderlake_ggml_set_i32_1d;

        ggml_get_i32_nd = alderlake_ggml_get_i32_nd;
        ggml_set_i32_nd = alderlake_ggml_set_i32_nd;

        ggml_get_f32_1d = alderlake_ggml_get_f32_1d;
        ggml_set_f32_1d = alderlake_ggml_set_f32_1d;

        ggml_get_f32_nd = alderlake_ggml_get_f32_nd;
        ggml_set_f32_nd = alderlake_ggml_set_f32_nd;

        ggml_threadpool_new = alderlake_ggml_threadpool_new;
        ggml_threadpool_free = alderlake_ggml_threadpool_free;
        // ggml_threadpool_get_n_threads = alderlake_ggml_threadpool_get_n_threads;
        ggml_threadpool_pause = alderlake_ggml_threadpool_pause;
        ggml_threadpool_resume = alderlake_ggml_threadpool_resume;

        ggml_graph_plan = alderlake_ggml_graph_plan;
        ggml_graph_compute = alderlake_ggml_graph_compute;
        ggml_graph_compute_with_ctx = alderlake_ggml_graph_compute_with_ctx;

        ggml_get_type_traits_cpu = alderlake_ggml_get_type_traits_cpu;

        _ggml_cpu_init = alderlake_ggml_cpu_init;

        _ggml_backend_cpu_init = alderlake_ggml_backend_cpu_init;

        ggml_backend_is_cpu = alderlake_ggml_backend_is_cpu;
        ggml_backend_cpu_set_n_threads = alderlake_ggml_backend_cpu_set_n_threads;
        ggml_backend_cpu_set_threadpool = alderlake_ggml_backend_cpu_set_threadpool;
        ggml_backend_cpu_set_abort_callback = alderlake_ggml_backend_cpu_set_abort_callback;

        _ggml_backend_cpu_reg = alderlake_ggml_backend_cpu_reg;

    } else if (haswell) {

        PRINTF("ggml_cpu_backend: selecting CPU backend: haswell\n");

        ggml_numa_init = haswell_ggml_numa_init;
        ggml_is_numa = haswell_ggml_is_numa;

        ggml_new_i32 = haswell_ggml_new_i32;
        ggml_new_f32 = haswell_ggml_new_f32;

        ggml_set_i32 = haswell_ggml_set_i32;
        ggml_set_f32 = haswell_ggml_set_f32;

        ggml_get_i32_1d = haswell_ggml_get_i32_1d;
        ggml_set_i32_1d = haswell_ggml_set_i32_1d;

        ggml_get_i32_nd = haswell_ggml_get_i32_nd;
        ggml_set_i32_nd = haswell_ggml_set_i32_nd;

        ggml_get_f32_1d = haswell_ggml_get_f32_1d;
        ggml_set_f32_1d = haswell_ggml_set_f32_1d;

        ggml_get_f32_nd = haswell_ggml_get_f32_nd;
        ggml_set_f32_nd = haswell_ggml_set_f32_nd;

        ggml_threadpool_new = haswell_ggml_threadpool_new;
        ggml_threadpool_free = haswell_ggml_threadpool_free;
        // ggml_threadpool_get_n_threads = haswell_ggml_threadpool_get_n_threads;
        ggml_threadpool_pause = haswell_ggml_threadpool_pause;
        ggml_threadpool_resume = haswell_ggml_threadpool_resume;

        ggml_graph_plan = haswell_ggml_graph_plan;
        ggml_graph_compute = haswell_ggml_graph_compute;
        ggml_graph_compute_with_ctx = haswell_ggml_graph_compute_with_ctx;

        ggml_get_type_traits_cpu = haswell_ggml_get_type_traits_cpu;

        _ggml_cpu_init = haswell_ggml_cpu_init;

        _ggml_backend_cpu_init = haswell_ggml_backend_cpu_init;

        ggml_backend_is_cpu = haswell_ggml_backend_is_cpu;
        ggml_backend_cpu_set_n_threads = haswell_ggml_backend_cpu_set_n_threads;
        ggml_backend_cpu_set_threadpool = haswell_ggml_backend_cpu_set_threadpool;
        ggml_backend_cpu_set_abort_callback = haswell_ggml_backend_cpu_set_abort_callback;

        _ggml_backend_cpu_reg = haswell_ggml_backend_cpu_reg;

    } else if (sandybridge) {

        PRINTF("ggml_cpu_backend: selecting CPU backend: sandybridge\n");

        ggml_numa_init = sandybridge_ggml_numa_init;
        ggml_is_numa = sandybridge_ggml_is_numa;

        ggml_new_i32 = sandybridge_ggml_new_i32;
        ggml_new_f32 = sandybridge_ggml_new_f32;

        ggml_set_i32 = sandybridge_ggml_set_i32;
        ggml_set_f32 = sandybridge_ggml_set_f32;

        ggml_get_i32_1d = sandybridge_ggml_get_i32_1d;
        ggml_set_i32_1d = sandybridge_ggml_set_i32_1d;

        ggml_get_i32_nd = sandybridge_ggml_get_i32_nd;
        ggml_set_i32_nd = sandybridge_ggml_set_i32_nd;

        ggml_get_f32_1d = sandybridge_ggml_get_f32_1d;
        ggml_set_f32_1d = sandybridge_ggml_set_f32_1d;

        ggml_get_f32_nd = sandybridge_ggml_get_f32_nd;
        ggml_set_f32_nd = sandybridge_ggml_set_f32_nd;

        ggml_threadpool_new = sandybridge_ggml_threadpool_new;
        ggml_threadpool_free = sandybridge_ggml_threadpool_free;
        // ggml_threadpool_get_n_threads = sandybridge_ggml_threadpool_get_n_threads;
        ggml_threadpool_pause = sandybridge_ggml_threadpool_pause;
        ggml_threadpool_resume = sandybridge_ggml_threadpool_resume;

        ggml_graph_plan = sandybridge_ggml_graph_plan;
        ggml_graph_compute = sandybridge_ggml_graph_compute;
        ggml_graph_compute_with_ctx = sandybridge_ggml_graph_compute_with_ctx;

        ggml_get_type_traits_cpu = sandybridge_ggml_get_type_traits_cpu;

        _ggml_cpu_init = sandybridge_ggml_cpu_init;

        _ggml_backend_cpu_init = sandybridge_ggml_backend_cpu_init;

        ggml_backend_is_cpu = sandybridge_ggml_backend_is_cpu;
        ggml_backend_cpu_set_n_threads = sandybridge_ggml_backend_cpu_set_n_threads;
        ggml_backend_cpu_set_threadpool = sandybridge_ggml_backend_cpu_set_threadpool;
        ggml_backend_cpu_set_abort_callback = sandybridge_ggml_backend_cpu_set_abort_callback;

        _ggml_backend_cpu_reg = sandybridge_ggml_backend_cpu_reg;

    }

    ggml_cpu_backend_selected = true;
}

#ifdef  __cplusplus
}
#endif

// for reference: CPU API defined in ggml-cpu.h

// GGML_BACKEND_API void    ggml_numa_init(enum ggml_numa_strategy numa); // call once for better performance on NUMA systems
// GGML_BACKEND_API bool    ggml_is_numa(void); // true if init detected that system has >1 NUMA node
//
// GGML_BACKEND_API struct ggml_tensor * ggml_new_i32(struct ggml_context * ctx, int32_t value);
// GGML_BACKEND_API struct ggml_tensor * ggml_new_f32(struct ggml_context * ctx, float value);
//
// GGML_BACKEND_API struct ggml_tensor * ggml_set_i32 (struct ggml_tensor * tensor, int32_t value);
// GGML_BACKEND_API struct ggml_tensor * ggml_set_f32 (struct ggml_tensor * tensor, float value);
//
// GGML_BACKEND_API int32_t ggml_get_i32_1d(const struct ggml_tensor * tensor, int i);
// GGML_BACKEND_API void    ggml_set_i32_1d(const struct ggml_tensor * tensor, int i, int32_t value);
//
// GGML_BACKEND_API int32_t ggml_get_i32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3);
// GGML_BACKEND_API void    ggml_set_i32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3, int32_t value);
//
// GGML_BACKEND_API float   ggml_get_f32_1d(const struct ggml_tensor * tensor, int i);
// GGML_BACKEND_API void    ggml_set_f32_1d(const struct ggml_tensor * tensor, int i, float value);
//
// GGML_BACKEND_API float   ggml_get_f32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3);
// GGML_BACKEND_API void    ggml_set_f32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3, float value);
//
// GGML_BACKEND_API struct ggml_threadpool *      ggml_threadpool_new           (struct ggml_threadpool_params  * params);
// GGML_BACKEND_API void                          ggml_threadpool_free          (struct ggml_threadpool * threadpool);
// GGML_BACKEND_API int                           ggml_threadpool_get_n_threads (struct ggml_threadpool * threadpool);
// GGML_BACKEND_API void                          ggml_threadpool_pause         (struct ggml_threadpool * threadpool);
// GGML_BACKEND_API void                          ggml_threadpool_resume        (struct ggml_threadpool * threadpool);
//
// // ggml_graph_plan() has to be called before ggml_graph_compute()
// // when plan.work_size > 0, caller must allocate memory for plan.work_data
// GGML_BACKEND_API struct ggml_cplan ggml_graph_plan(
//               const struct ggml_cgraph * cgraph,
//                                    int   n_threads, [# = GGML_DEFAULT_N_THREADS #]
//                 struct ggml_threadpool * threadpool [# = NULL #] );
// GGML_BACKEND_API enum ggml_status  ggml_graph_compute(struct ggml_cgraph * cgraph, struct ggml_cplan * cplan);
//
// // same as ggml_graph_compute() but the work data is allocated as a part of the context
// // note: the drawback of this API is that you must have ensured that the context has enough memory for the work data
// GGML_BACKEND_API enum ggml_status  ggml_graph_compute_with_ctx(struct ggml_context * ctx, struct ggml_cgraph * cgraph, int n_threads);
//
// //
// // system info
// //
//
// // x86
// GGML_BACKEND_API int ggml_cpu_has_sse3       (void);
// GGML_BACKEND_API int ggml_cpu_has_ssse3      (void);
// GGML_BACKEND_API int ggml_cpu_has_avx        (void);
// GGML_BACKEND_API int ggml_cpu_has_avx_vnni   (void);
// GGML_BACKEND_API int ggml_cpu_has_avx2       (void);
// GGML_BACKEND_API int ggml_cpu_has_f16c       (void);
// GGML_BACKEND_API int ggml_cpu_has_fma        (void);
// GGML_BACKEND_API int ggml_cpu_has_avx512     (void);
// GGML_BACKEND_API int ggml_cpu_has_avx512_vbmi(void);
// GGML_BACKEND_API int ggml_cpu_has_avx512_vnni(void);
// GGML_BACKEND_API int ggml_cpu_has_avx512_bf16(void);
// GGML_BACKEND_API int ggml_cpu_has_amx_int8   (void);
// // ARM
// GGML_BACKEND_API int ggml_cpu_has_neon       (void);
// GGML_BACKEND_API int ggml_cpu_has_arm_fma    (void);
// GGML_BACKEND_API int ggml_cpu_has_fp16_va    (void);
// GGML_BACKEND_API int ggml_cpu_has_dotprod    (void);
// GGML_BACKEND_API int ggml_cpu_has_matmul_int8(void);
// GGML_BACKEND_API int ggml_cpu_has_sve        (void);
// GGML_BACKEND_API int ggml_cpu_get_sve_cnt    (void);  // sve vector length in bytes
// // other
// GGML_BACKEND_API int ggml_cpu_has_riscv_v    (void);
// GGML_BACKEND_API int ggml_cpu_has_vsx        (void);
// GGML_BACKEND_API int ggml_cpu_has_wasm_simd  (void);
// GGML_BACKEND_API int ggml_cpu_has_llamafile  (void);
//
// GGML_BACKEND_API const struct ggml_type_traits_cpu * ggml_get_type_traits_cpu(enum ggml_type type);
//
// GGML_BACKEND_API void ggml_cpu_init(void);
//
// //
// // CPU backend
// //
//
// GGML_BACKEND_API ggml_backend_t ggml_backend_cpu_init(void);
//
// GGML_BACKEND_API bool ggml_backend_is_cpu                (ggml_backend_t backend);
// GGML_BACKEND_API void ggml_backend_cpu_set_n_threads     (ggml_backend_t backend_cpu, int n_threads);
// GGML_BACKEND_API void ggml_backend_cpu_set_threadpool    (ggml_backend_t backend_cpu, ggml_threadpool_t threadpool);
// GGML_BACKEND_API void ggml_backend_cpu_set_abort_callback(ggml_backend_t backend_cpu, ggml_abort_callback abort_callback, void * abort_callback_data);
//
// GGML_BACKEND_API ggml_backend_reg_t ggml_backend_cpu_reg(void);

#else
#pragma GCC diagnostic ignored "-Wempty-translation-unit"
#endif
