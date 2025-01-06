
bool ggml_cpu_f16c_detected = false;
bool ggml_cpu_avx2_detected = false;
bool ggml_cpu_avx512f_detected = false;
bool ggml_cpu_avx512bf16_detected = false;

#if defined(__x86_64__) || (defined(_MSC_VER) && defined(_M_AMD64))

// disable things
#ifdef GGML_BACKEND_DL
#undef GGML_BACKEND_DL
#endif
#define ggml_backend_cpu_x86_score __xyz_hidden_ggml_backend_cpu_x86_score
// a quick hack
#include "ggml-cpu/cpu-feats-x86.cpp"


struct _ggml_cpu_features {
    cpuid_x86 cpu_features;

    _ggml_cpu_features() {
        ggml_cpu_f16c_detected = cpu_features.F16C();
        ggml_cpu_avx2_detected = cpu_features.AVX2();
        ggml_cpu_avx512f_detected = cpu_features.AVX512F();
        ggml_cpu_avx512bf16_detected = cpu_features.AVX512_BF16();
    }

} __ggml_cpu_features;

#endif
