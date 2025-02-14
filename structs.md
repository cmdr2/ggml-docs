# Structs

## Index

- [ggml_backend_buffer_type](#ggml_backend_buffer_type)
- [ggml_backend](#ggml_backend)
- [ggml_tallocr](#ggml_tallocr)
- [ggml_gallocr](#ggml_gallocr)
- [ggml_backend_event](#ggml_backend_event)
- [ggml_backend_reg](#ggml_backend_reg)
- [ggml_backend_device](#ggml_backend_device)
- [ggml_backend_dev_caps](#ggml_backend_dev_caps)
- [ggml_backend_dev_props](#ggml_backend_dev_props)
- [ggml_backend_feature](#ggml_backend_feature)
- [ggml_backend_sched](#ggml_backend_sched)
- [ggml_backend_graph_copy](#ggml_backend_graph_copy)
- [ggml_context_deleter](#ggml_context_deleter)
- [gguf_context_deleter](#gguf_context_deleter)
- [ggml_gallocr_deleter](#ggml_gallocr_deleter)
- [ggml_backend_deleter](#ggml_backend_deleter)
- [ggml_backend_buffer_deleter](#ggml_backend_buffer_deleter)
- [ggml_backend_event_deleter](#ggml_backend_event_deleter)
- [ggml_backend_sched_deleter](#ggml_backend_sched_deleter)
- [ggml_cplan](#ggml_cplan)
- [ggml_type_traits_cpu](#ggml_type_traits_cpu)
- [ggml_vk_device](#ggml_vk_device)
- [ggml_tensor](#ggml_tensor)
- [ggml_cgraph](#ggml_cgraph)
- [ggml_opt_dataset](#ggml_opt_dataset)
- [ggml_opt_context](#ggml_opt_context)
- [ggml_opt_result](#ggml_opt_result)
- [ggml_opt_optimizer_params](#ggml_opt_optimizer_params)
- [struct (unnamed at ggml-opt.h:71:9)](#struct (unnamed at ggml-opt.h:71:9))
- [ggml_opt_params](#ggml_opt_params)
- [ggml_bf16_t](#ggml_bf16_t)
- [ggml_object](#ggml_object)
- [ggml_context](#ggml_context)
- [ggml_init_params](#ggml_init_params)
- [ggml_backend_buffer](#ggml_backend_buffer)
- [ggml_type_traits](#ggml_type_traits)
- [ggml_threadpool_params](#ggml_threadpool_params)
- [ggml_threadpool](#ggml_threadpool)
- [gguf_context](#gguf_context)
- [gguf_init_params](#gguf_init_params)


## Detailed Definitions

### ggml_backend_buffer_type

```c
struct ggml_backend_buffer_type;
```

Source: [ggml-alloc.h#L9](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-alloc.h#L9)

### ggml_backend

```c
struct ggml_backend;
```

Source: [ggml-alloc.h#L11](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-alloc.h#L11)

### ggml_tallocr

```c
struct ggml_tallocr {
    ggml_backend_buffer_t buffer;
    void * base;
    int alignment;
    int offset;
};
```

Source: [ggml-alloc.h#L14](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-alloc.h#L14)

### ggml_gallocr

```c
struct ggml_gallocr;
```

Source: [ggml-alloc.h#L46](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-alloc.h#L46)

### ggml_backend_event

```c
struct ggml_backend_event;
```

Source: [ggml-backend.h#L26](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-backend.h#L26)

### ggml_backend_reg

```c
struct ggml_backend_reg;
```

Source: [ggml-backend.h#L29](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-backend.h#L29)

### ggml_backend_device

```c
struct ggml_backend_device;
```

Source: [ggml-backend.h#L30](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-backend.h#L30)

### ggml_backend_dev_caps

```c
struct ggml_backend_dev_caps {
    bool async;
    bool host_buffer;
    bool buffer_from_host_ptr;
    bool events;
};
```

Source: [ggml-backend.h#L140](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-backend.h#L140)

### ggml_backend_dev_props

```c
struct ggml_backend_dev_props {
    const char * name;
    const char * description;
    int memory_free;
    int memory_total;
    enum ggml_backend_dev_type type;
    struct ggml_backend_dev_caps caps;
};
```

Source: [ggml-backend.h#L152](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-backend.h#L152)

### ggml_backend_feature

```c
struct ggml_backend_feature {
    const char * name;
    const char * value;
};
```

Source: [ggml-backend.h#L196](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-backend.h#L196)

### ggml_backend_sched

```c
struct ggml_backend_sched;
```

Source: [ggml-backend.h#L280](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-backend.h#L280)

### ggml_backend_graph_copy

```c
struct ggml_backend_graph_copy {
    ggml_backend_buffer_t buffer;
    struct ggml_context * ctx_allocated;
    struct ggml_context * ctx_unallocated;
    struct ggml_cgraph * graph;
};
```

Source: [ggml-backend.h#L328](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-backend.h#L328)

### ggml_context_deleter

```c
struct ggml_context_deleter {
    int () operator;
};
```

Source: [ggml-cpp.h#L17](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-cpp.h#L17)

### gguf_context_deleter

```c
struct gguf_context_deleter {
    int () operator;
};
```

Source: [ggml-cpp.h#L18](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-cpp.h#L18)

### ggml_gallocr_deleter

```c
struct ggml_gallocr_deleter {
    int () operator;
};
```

Source: [ggml-cpp.h#L25](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-cpp.h#L25)

### ggml_backend_deleter

```c
struct ggml_backend_deleter {
    int () operator;
};
```

Source: [ggml-cpp.h#L31](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-cpp.h#L31)

### ggml_backend_buffer_deleter

```c
struct ggml_backend_buffer_deleter {
    int () operator;
};
```

Source: [ggml-cpp.h#L32](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-cpp.h#L32)

### ggml_backend_event_deleter

```c
struct ggml_backend_event_deleter {
    int () operator;
};
```

Source: [ggml-cpp.h#L33](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-cpp.h#L33)

### ggml_backend_sched_deleter

```c
struct ggml_backend_sched_deleter {
    int () operator;
};
```

Source: [ggml-cpp.h#L34](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-cpp.h#L34)

### ggml_cplan

```c
struct ggml_cplan {
    int work_size;
    uint8_t * work_data;
    int n_threads;
    struct ggml_threadpool * threadpool;
    int abort_callback;
    void * abort_callback_data;
};
```

Source: [ggml-cpu.h#L12](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-cpu.h#L12)

### ggml_type_traits_cpu

```c
struct ggml_type_traits_cpu {
    ggml_from_float_t from_float;
    ggml_vec_dot_t vec_dot;
    enum ggml_type vec_dot_type;
    int64_t nrows;
};
```

Source: [ggml-cpu.h#L109](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-cpu.h#L109)

### ggml_vk_device

```c
struct ggml_vk_device {
    int index;
    int type;
    int heapSize;
    const char * name;
    const char * vendor;
    int subgroupSize;
    uint64_t bufferAlignment;
    uint64_t maxAlloc;
};
```

Source: [ggml-kompute.h#L16](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-kompute.h#L16)

### ggml_tensor

```c
struct ggml_tensor {
    enum ggml_type type;
    struct ggml_backend_buffer * buffer;
    int64_t[4] ne;
    int[4] nb;
    enum ggml_op op;
    int32_t[16] op_params;
    int32_t flags;
    struct ggml_tensor *[10] src;
    struct ggml_tensor * view_src;
    int view_offs;
    void * data;
    char[64] name;
    void * extra;
    char[8] padding;
};
```

Source: [ggml.h#L576](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L576)

### ggml_cgraph

```c
struct ggml_cgraph;
```

Source: [ggml.h#L348](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L348)

### ggml_opt_dataset

```c
struct ggml_opt_dataset;
```

Source: [ggml-opt.h#L18](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-opt.h#L18)

### ggml_opt_context

```c
struct ggml_opt_context;
```

Source: [ggml-opt.h#L19](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-opt.h#L19)

### ggml_opt_result

```c
struct ggml_opt_result;
```

Source: [ggml-opt.h#L20](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-opt.h#L20)

### ggml_opt_optimizer_params

```c
struct ggml_opt_optimizer_params {
    struct (unnamed struct at ggml-opt.h:71:9) adamw;
};
```

Source: [ggml-opt.h#L69](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-opt.h#L69)

### struct (unnamed at ggml-opt.h:71:9)

```c
struct struct (unnamed at ggml-opt.h:71:9) {
    float alpha;
    float beta1;
    float beta2;
    float eps;
    float wd;
};
```

Source: [ggml-opt.h#L71](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-opt.h#L71)

### ggml_opt_params

```c
struct ggml_opt_params {
    ggml_backend_sched_t backend_sched;
    struct ggml_context * ctx_compute;
    struct ggml_tensor * inputs;
    struct ggml_tensor * outputs;
    enum ggml_opt_loss_type loss_type;
    enum ggml_opt_build_type build_type;
    int32_t opt_period;
    ggml_opt_get_optimizer_params get_opt_pars;
    void * get_opt_pars_ud;
};
```

Source: [ggml-opt.h#L89](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-opt.h#L89)

### ggml_bf16_t

```c
struct ggml_bf16_t {
    uint16_t bits;
};
```

Source: [ggml.h#L339](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L339)

### ggml_object

```c
struct ggml_object;
```

Source: [ggml.h#L346](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L346)

### ggml_context

```c
struct ggml_context;
```

Source: [ggml.h#L347](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L347)

### ggml_init_params

```c
struct ggml_init_params {
    int mem_size;
    void * mem_buffer;
    int no_alloc;
};
```

Source: [ggml.h#L568](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L568)

### ggml_backend_buffer

```c
struct ggml_backend_buffer;
```

Source: [ggml.h#L579](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L579)

### ggml_type_traits

```c
struct ggml_type_traits {
    const char * type_name;
    int64_t blck_size;
    int64_t blck_size_interleave;
    int type_size;
    bool is_quantized;
    ggml_to_float_t to_float;
    ggml_from_float_t from_float_ref;
};
```

Source: [ggml.h#L2148](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L2148)

### ggml_threadpool_params

```c
struct ggml_threadpool_params {
    int cpumask;
    int n_threads;
    enum ggml_sched_priority prio;
    uint32_t poll;
    bool strict_cpu;
    bool paused;
};
```

Source: [ggml.h#L2174](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L2174)

### ggml_threadpool

```c
struct ggml_threadpool;
```

Source: [ggml.h#L2183](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L2183)

### gguf_context

```c
struct gguf_context;
```

Source: [gguf.h#L70](https://github.com/ggml-org/ggml/blob/9a4acb3/include/gguf.h#L70)

### gguf_init_params

```c
struct gguf_init_params {
    bool no_alloc;
    struct ggml_context ** ctx;
};
```

Source: [gguf.h#L72](https://github.com/ggml-org/ggml/blob/9a4acb3/include/gguf.h#L72)
