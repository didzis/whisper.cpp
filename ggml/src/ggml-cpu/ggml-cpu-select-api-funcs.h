
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

#ifdef STR
#undef STR
#endif

#ifdef STR2
#undef STR2
#endif

#define STR2(x) #x
#define STR(x) STR2(x)

    PRINTF("ggml_cpu_backend: selecting CPU backend: " STR(CPU_NAME) "\n");

#undef STR
#undef STR2

    // included from ggml-cpu-dispatch.cpp
    // we are assigning functions declared by ggml-cpu-api-funcs.h
    // to function pointers defined in ggml-cpu-dispatch.cpp

    ggml_numa_init = CPU_PREFIX(ggml_numa_init);
    ggml_is_numa = CPU_PREFIX(ggml_is_numa);

    ggml_new_i32 = CPU_PREFIX(ggml_new_i32);
    ggml_new_f32 = CPU_PREFIX(ggml_new_f32);

    ggml_set_i32 = CPU_PREFIX(ggml_set_i32);
    ggml_set_f32 = CPU_PREFIX(ggml_set_f32);

    ggml_get_i32_1d = CPU_PREFIX(ggml_get_i32_1d);
    ggml_set_i32_1d = CPU_PREFIX(ggml_set_i32_1d);

    ggml_get_i32_nd = CPU_PREFIX(ggml_get_i32_nd);
    ggml_set_i32_nd = CPU_PREFIX(ggml_set_i32_nd);

    ggml_get_f32_1d = CPU_PREFIX(ggml_get_f32_1d);
    ggml_set_f32_1d = CPU_PREFIX(ggml_set_f32_1d);

    ggml_get_f32_nd = CPU_PREFIX(ggml_get_f32_nd);
    ggml_set_f32_nd = CPU_PREFIX(ggml_set_f32_nd);

    ggml_threadpool_new = CPU_PREFIX(ggml_threadpool_new);
    ggml_threadpool_free = CPU_PREFIX(ggml_threadpool_free);
    // ggml_threadpool_get_n_threads = CPU_PREFIX(ggml_threadpool_get_n_threads);
    ggml_threadpool_pause = CPU_PREFIX(ggml_threadpool_pause);
    ggml_threadpool_resume = CPU_PREFIX(ggml_threadpool_resume);

    ggml_graph_plan = CPU_PREFIX(ggml_graph_plan);
    ggml_graph_compute = CPU_PREFIX(ggml_graph_compute);
    ggml_graph_compute_with_ctx = CPU_PREFIX(ggml_graph_compute_with_ctx);

    ggml_get_type_traits_cpu = CPU_PREFIX(ggml_get_type_traits_cpu);

    _ggml_cpu_init = CPU_PREFIX(ggml_cpu_init);

    _ggml_backend_cpu_init = CPU_PREFIX(ggml_backend_cpu_init);

    ggml_backend_is_cpu = CPU_PREFIX(ggml_backend_is_cpu);
    ggml_backend_cpu_set_n_threads = CPU_PREFIX(ggml_backend_cpu_set_n_threads);
    ggml_backend_cpu_set_threadpool = CPU_PREFIX(ggml_backend_cpu_set_threadpool);
    ggml_backend_cpu_set_abort_callback = CPU_PREFIX(ggml_backend_cpu_set_abort_callback);

    _ggml_backend_cpu_reg = CPU_PREFIX(ggml_backend_cpu_reg);

    ggml_cpu_fp32_to_i32 = CPU_PREFIX(ggml_cpu_fp32_to_i32);
    ggml_cpu_fp32_to_fp32 = CPU_PREFIX(ggml_cpu_fp32_to_fp32);
    ggml_cpu_fp32_to_fp16 = CPU_PREFIX(ggml_cpu_fp32_to_fp16);
    ggml_cpu_fp16_to_fp32 = CPU_PREFIX(ggml_cpu_fp16_to_fp32);
    ggml_cpu_fp32_to_bf16 = CPU_PREFIX(ggml_cpu_fp32_to_bf16);
    ggml_cpu_bf16_to_fp32 = CPU_PREFIX(ggml_cpu_bf16_to_fp32);


#undef CPU_PREFIX
#undef CAT
#undef CAT2
#undef CAT3
#undef CPU_NAME
