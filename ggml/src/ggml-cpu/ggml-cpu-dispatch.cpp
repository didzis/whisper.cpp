#ifdef __x86_64__

// disable things
#ifdef GGML_BACKEND_DL
#undef GGML_BACKEND_DL
#endif
#define ggml_backend_cpu_x86_score __xyz_hidden_ggml_backend_cpu_x86_score
// a quick hack
#include "ggml-cpu/arch/x86/cpu-feats.cpp"

#include <cstdio> // for fprintf()

#define PRINTF(...) fprintf(stderr, __VA_ARGS__)


#ifdef  __cplusplus
extern "C" {
#endif

// some definitions and declarations from ggml-cpu.h

    // the compute plan that needs to be prepared for ggml_graph_compute()
    // since https://github.com/ggml-org/ggml/issues/287
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
    GGML_BACKEND_API int ggml_cpu_has_bmi2       (void);
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
    GGML_BACKEND_API int ggml_cpu_has_sme        (void);
    // other
    GGML_BACKEND_API int ggml_cpu_has_riscv_v    (void);
    GGML_BACKEND_API int ggml_cpu_get_rvv_vlen   (void);
    GGML_BACKEND_API int ggml_cpu_has_vsx        (void);
    GGML_BACKEND_API int ggml_cpu_has_vxe        (void);
    GGML_BACKEND_API int ggml_cpu_has_nnpa       (void);
    GGML_BACKEND_API int ggml_cpu_has_wasm_simd  (void);
    GGML_BACKEND_API int ggml_cpu_has_llamafile  (void);

    GGML_BACKEND_API void ggml_cpu_init(void);
    GGML_BACKEND_API ggml_backend_t ggml_backend_cpu_init(void);
    GGML_BACKEND_API ggml_backend_reg_t ggml_backend_cpu_reg(void);




// ===========================================================
// API declarations for CPU variant: x64
// ===========================================================

#define CPU_NAME x64
#include "ggml-cpu-api-funcs.h"

// ===========================================================
// API declarations for CPU variant: sse42
// ===========================================================

#define CPU_NAME sse42
#include "ggml-cpu-api-funcs.h"

// ===========================================================
// API declarations for CPU variant: sandybridge
// ===========================================================

#define CPU_NAME sandybridge
#include "ggml-cpu-api-funcs.h"

// ===========================================================
// API declarations for CPU variant: haswell
// ===========================================================

#define CPU_NAME haswell
#include "ggml-cpu-api-funcs.h"

// ===========================================================
// API declarations for CPU variant: skylakex
// ===========================================================

#define CPU_NAME skylakex
#include "ggml-cpu-api-funcs.h"

// ===========================================================
// API declarations for CPU variant: ivybridge
// ===========================================================

#define CPU_NAME ivybridge
#include "ggml-cpu-api-funcs.h"

// ===========================================================
// API declarations for CPU variant: piledriver
// ===========================================================

#define CPU_NAME piledriver
#include "ggml-cpu-api-funcs.h"

// ===========================================================
// API declarations for CPU variant: cannonlake
// ===========================================================

#define CPU_NAME cannonlake
#include "ggml-cpu-api-funcs.h"

// ===========================================================
// API declarations for CPU variant: cascadelake
// ===========================================================

#define CPU_NAME cascadelake
#include "ggml-cpu-api-funcs.h"

// ===========================================================
// API declarations for CPU variant: cooperlake
// ===========================================================

#define CPU_NAME cooperlake
#include "ggml-cpu-api-funcs.h"

// ===========================================================
// API declarations for CPU variant: zen4
// ===========================================================

#define CPU_NAME zen4
#include "ggml-cpu-api-funcs.h"

// ===========================================================
// API declarations for CPU variant: icelake
// ===========================================================

#define CPU_NAME icelake
#include "ggml-cpu-api-funcs.h"

// ===========================================================
// API declarations for CPU variant: alderlake
// ===========================================================

#define CPU_NAME alderlake
#include "ggml-cpu-api-funcs.h"

// ===========================================================
// API declarations for CPU variant: sapphirerapids
// ===========================================================

#define CPU_NAME sapphirerapids
#include "ggml-cpu-api-funcs.h"






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

void (*ggml_cpu_fp32_to_i32)(const float *, int32_t *, int64_t);
void (*ggml_cpu_fp32_to_fp32)(const float *, float *, int64_t);
void (*ggml_cpu_fp32_to_fp16)(const float *, ggml_fp16_t *, int64_t);
void (*ggml_cpu_fp16_to_fp32)(const ggml_fp16_t *, float *, int64_t);
void (*ggml_cpu_fp32_to_bf16)(const float *, ggml_bf16_t *, int64_t);
void (*ggml_cpu_bf16_to_fp32)(const ggml_bf16_t *, float *, int64_t);





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
GGML_BACKEND_API int ggml_cpu_has_bmi2       (void) { return ggml_cpu_dispatch_cpuid().BMI2(); }
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
GGML_BACKEND_API int ggml_cpu_has_sme        (void) { return 0; }
// other
GGML_BACKEND_API int ggml_cpu_has_riscv_v    (void) { return 0; }
GGML_BACKEND_API int ggml_cpu_get_rvv_vlen   (void) { return 0; }
GGML_BACKEND_API int ggml_cpu_has_vsx        (void) { return 0; }
GGML_BACKEND_API int ggml_cpu_has_vxe        (void) { return 0; }
GGML_BACKEND_API int ggml_cpu_has_nnpa       (void) { return 0; }
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
    // ggml_add_cpu_backend_variant(sapphirerapids SSE42 AVX F16C FMA AVX2 BMI2 AVX512 AVX512_VBMI AVX512_VNNI AVX512_BF16 AMX_TILE AMX_INT8)
    // ggml_add_cpu_backend_variant(zen4           SSE42 AVX F16C FMA AVX2 BMI2 AVX512 AVX512_VBMI AVX512_VNNI AVX512_BF16)
    // ggml_add_cpu_backend_variant(icelake        SSE42 AVX F16C FMA AVX2 BMI2 AVX512 AVX512_VBMI AVX512_VNNI)
    // ggml_add_cpu_backend_variant(cannonlake     SSE42 AVX F16C FMA AVX2 BMI2 AVX512 AVX512_VBMI)
    // ggml_add_cpu_backend_variant(cooperlake     SSE42 AVX F16C FMA AVX2 BMI2 AVX512 AVX512_VNNI AVX512_BF16)
    // ggml_add_cpu_backend_variant(cascadelake    SSE42 AVX F16C FMA AVX2 BMI2 AVX512 AVX512_VNNI)
    // ggml_add_cpu_backend_variant(skylakex       SSE42 AVX F16C FMA AVX2 BMI2 AVX512)
    // ggml_add_cpu_backend_variant(alderlake      SSE42 AVX F16C FMA AVX2 BMI2 AVX_VNNI)
    // ggml_add_cpu_backend_variant(haswell        SSE42 AVX F16C FMA AVX2 BMI2)
    // ggml_add_cpu_backend_variant(piledriver     SSE42 AVX F16C FMA)
    // ggml_add_cpu_backend_variant(ivybridge      SSE42 AVX F16C)
    // ggml_add_cpu_backend_variant(sandybridge    SSE42 AVX)
    // ggml_add_cpu_backend_variant(sse42          SSE42)
    // ggml_add_cpu_backend_variant(x64            )

    auto& cpu_features = ggml_cpu_dispatch_cpuid();

    bool sse42 = cpu_features.SSE42();
    bool avx = cpu_features.AVX();
    bool f16c = cpu_features.F16C();
    bool fma = cpu_features.FMA();
    bool avx2 = cpu_features.AVX2();
    bool bmi2 = cpu_features.BMI2();
    bool avx512 = cpu_features.AVX512F() && cpu_features.AVX512CD() && cpu_features.AVX512VL() \
                                         && cpu_features.AVX512DQ() && cpu_features.AVX512BW();
#ifndef GGML_DISABLE_RECENT_CPU_ARCHITECTURES
    bool avx_vnni = cpu_features.AVX_VNNI();
    bool avx512_vnni = cpu_features.AVX512_VNNI();
    bool avx512_vbmi = cpu_features.AVX512_VBMI();
    bool avx512_bf16 = cpu_features.AVX512_BF16();
    bool amx_tile = cpu_features.AMX_TILE();
    bool amx_int8 = cpu_features.AMX_INT8();
#endif

    bool sandybridge    = sse42 && avx;
    bool ivybridge      = sse42 && avx && f16c;
    bool piledriver     = sse42 && avx && f16c && fma;
    bool haswell        = sse42 && avx && f16c && fma && avx2 && bmi2;
    bool skylakex       = sse42 && avx && f16c && fma && avx2 && bmi2 && avx512;
#ifndef GGML_DISABLE_RECENT_CPU_ARCHITECTURES
    bool alderlake      = sse42 && avx && f16c && fma && avx2 && bmi2 && avx_vnni;
    bool cascadelake    = sse42 && avx && f16c && fma && avx2 && bmi2 && avx512 && avx512_vnni;
    bool cooperlake     = sse42 && avx && f16c && fma && avx2 && bmi2 && avx512 && avx512_vnni && avx512_bf16;
    bool cannonlake     = sse42 && avx && f16c && fma && avx2 && bmi2 && avx512 && avx512_vbmi;
    bool icelake        = sse42 && avx && f16c && fma && avx2 && bmi2 && avx512 && avx512_vbmi && avx512_vnni;
    bool zen4           = sse42 && avx && f16c && fma && avx2 && bmi2 && avx512 && avx512_vbmi && avx512_vnni && avx512_bf16;
    bool sapphirerapids = sse42 && avx && f16c && fma && avx2 && bmi2 && avx512 && avx512_vbmi && avx512_vnni && avx512_bf16 && amx_tile && amx_int8;
#endif

    if (false) {
#ifndef GGML_DISABLE_RECENT_CPU_ARCHITECTURES
    } else if (sapphirerapids) {

#define CPU_NAME sapphirerapids
#include "ggml-cpu-select-api-funcs.h"

    } else if (zen4) {

#define CPU_NAME zen4
#include "ggml-cpu-select-api-funcs.h"

    } else if (icelake) {

#define CPU_NAME icelake
#include "ggml-cpu-select-api-funcs.h"

    } else if (cannonlake) {

#define CPU_NAME cannonlake
#include "ggml-cpu-select-api-funcs.h"

    } else if (cooperlake) {

#define CPU_NAME cooperlake
#include "ggml-cpu-select-api-funcs.h"

    } else if (cascadelake) {

#define CPU_NAME cascadelake
#include "ggml-cpu-select-api-funcs.h"

    } else if (alderlake) {

#define CPU_NAME alderlake
#include "ggml-cpu-select-api-funcs.h"

#endif
    } else if (skylakex) {

#define CPU_NAME skylakex
#include "ggml-cpu-select-api-funcs.h"

    } else if (haswell) {

#define CPU_NAME haswell
#include "ggml-cpu-select-api-funcs.h"

    } else if (piledriver) {

#define CPU_NAME piledriver
#include "ggml-cpu-select-api-funcs.h"

    } else if (ivybridge) {

#define CPU_NAME ivybridge
#include "ggml-cpu-select-api-funcs.h"

    } else if (sandybridge) {

#define CPU_NAME sandybridge
#include "ggml-cpu-select-api-funcs.h"

    } else if (sse42) {

#define CPU_NAME sse42
#include "ggml-cpu-select-api-funcs.h"

    } else {

#define CPU_NAME x64
#include "ggml-cpu-select-api-funcs.h"

    }

    ggml_cpu_backend_selected = true;
}

#ifdef  __cplusplus
}
#endif


#else
#pragma GCC diagnostic ignored "-Wempty-translation-unit"
#endif
