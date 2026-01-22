
bool ggml_cpu_sse42_detected = false;
bool ggml_cpu_avx_detected = false;
bool ggml_cpu_f16c_detected = false;
bool ggml_cpu_fma_detected = false;
bool ggml_cpu_avx2_detected = false;
bool ggml_cpu_bmi2_detected = false;
bool ggml_cpu_avx_vnni_detected = false;
bool ggml_cpu_avx512_detected = false;
bool ggml_cpu_avx512f_detected = false;
bool ggml_cpu_avx512cd_detected = false;
bool ggml_cpu_avx512vl_detected = false;
bool ggml_cpu_avx512dq_detected = false;
bool ggml_cpu_avx512bw_detected = false;
bool ggml_cpu_avx512_vbmi_detected = false;
bool ggml_cpu_avx512_vnni_detected = false;
bool ggml_cpu_avx512_bf16_detected = false;
bool ggml_cpu_amx_tile_detected = false;
bool ggml_cpu_amx_int8_detected = false;

#if defined(__x86_64__) || (defined(_MSC_VER) && defined(_M_AMD64))

// disable things
#ifdef GGML_BACKEND_DL
#undef GGML_BACKEND_DL
#endif
#define ggml_backend_cpu_x86_score __xyz_hidden_ggml_backend_cpu_x86_score
// a quick hack
#include "ggml-cpu/arch/x86/cpu-feats.cpp"


struct _ggml_cpu_features {
    cpuid_x86 cpu_features;

    _ggml_cpu_features() {
        ggml_cpu_sse42_detected = cpu_features.SSE42();
        ggml_cpu_avx_detected = cpu_features.AVX();
        ggml_cpu_f16c_detected = cpu_features.F16C();
        ggml_cpu_fma_detected = cpu_features.FMA();
        ggml_cpu_avx2_detected = cpu_features.AVX2();
        ggml_cpu_bmi2_detected = cpu_features.BMI2();
        ggml_cpu_avx_vnni_detected = cpu_features.AVX_VNNI();
        ggml_cpu_avx512_detected = cpu_features.AVX512F() && cpu_features.AVX512CD() && cpu_features.AVX512VL() \
                                                          && cpu_features.AVX512DQ() && cpu_features.AVX512BW();
        ggml_cpu_avx512f_detected = cpu_features.AVX512F();
        ggml_cpu_avx512cd_detected = cpu_features.AVX512CD();
        ggml_cpu_avx512vl_detected = cpu_features.AVX512VL();
        ggml_cpu_avx512dq_detected = cpu_features.AVX512DQ();
        ggml_cpu_avx512bw_detected = cpu_features.AVX512BW();
        ggml_cpu_avx512_vbmi_detected = cpu_features.AVX512_VBMI();
        ggml_cpu_avx512_vnni_detected = cpu_features.AVX512_VNNI();
        ggml_cpu_avx512_bf16_detected = cpu_features.AVX512_BF16();
        ggml_cpu_amx_tile_detected = cpu_features.AMX_TILE();
        ggml_cpu_amx_int8_detected = cpu_features.AMX_INT8();
    }

} __ggml_cpu_features;

#endif
