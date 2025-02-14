# Typedefs

## Index

- [ggml_backend_buffer_type_t](#ggml_backend_buffer_type_t)
- [ggml_backend_buffer_t](#ggml_backend_buffer_t)
- [ggml_backend_t](#ggml_backend_t)
- [ggml_gallocr_t](#ggml_gallocr_t)
- [ggml_backend_event_t](#ggml_backend_event_t)
- [ggml_backend_graph_plan_t](#ggml_backend_graph_plan_t)
- [ggml_backend_reg_t](#ggml_backend_reg_t)
- [ggml_backend_dev_t](#ggml_backend_dev_t)
- [ggml_backend_split_buffer_type_t](#ggml_backend_split_buffer_type_t)
- [ggml_backend_set_n_threads_t](#ggml_backend_set_n_threads_t)
- [ggml_backend_dev_get_extra_bufts_t](#ggml_backend_dev_get_extra_bufts_t)
- [ggml_backend_set_abort_callback_t](#ggml_backend_set_abort_callback_t)
- [ggml_backend_get_features_t](#ggml_backend_get_features_t)
- [ggml_backend_sched_t](#ggml_backend_sched_t)
- [ggml_backend_sched_eval_callback](#ggml_backend_sched_eval_callback)
- [ggml_backend_eval_callback](#ggml_backend_eval_callback)
- [ggml_vec_dot_t](#ggml_vec_dot_t)
- [ggml_opt_dataset_t](#ggml_opt_dataset_t)
- [ggml_opt_context_t](#ggml_opt_context_t)
- [ggml_opt_result_t](#ggml_opt_result_t)
- [ggml_opt_get_optimizer_params](#ggml_opt_get_optimizer_params)
- [ggml_opt_epoch_callback](#ggml_opt_epoch_callback)
- [ggml_fp16_t](#ggml_fp16_t)
- [ggml_bf16_t](#ggml_bf16_t)
- [bool](#bool)
- [ggml_guid](#ggml_guid)
- [ggml_guid_t](#ggml_guid_t)
- [ggml_unary_op_f32_t](#ggml_unary_op_f32_t)
- [ggml_binary_op_f32_t](#ggml_binary_op_f32_t)
- [ggml_custom1_op_f32_t](#ggml_custom1_op_f32_t)
- [ggml_custom2_op_f32_t](#ggml_custom2_op_f32_t)
- [ggml_custom3_op_f32_t](#ggml_custom3_op_f32_t)
- [ggml_custom1_op_t](#ggml_custom1_op_t)
- [ggml_custom2_op_t](#ggml_custom2_op_t)
- [ggml_custom3_op_t](#ggml_custom3_op_t)
- [ggml_log_callback](#ggml_log_callback)
- [ggml_to_float_t](#ggml_to_float_t)
- [ggml_from_float_t](#ggml_from_float_t)
- [ggml_threadpool_t](#ggml_threadpool_t)


## Detailed Definitions

### ggml_backend_buffer_type_t

```c
typedef struct ggml_backend_buffer_type * ggml_backend_buffer_type_t;
```

Source: [ggml-backend.h#L24](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-backend.h#L24)

### ggml_backend_buffer_t

```c
typedef struct ggml_backend_buffer * ggml_backend_buffer_t;
```

Source: [ggml-backend.h#L25](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-backend.h#L25)

### ggml_backend_t

```c
typedef struct ggml_backend * ggml_backend_t;
```

Source: [ggml-kompute.h#L38](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-kompute.h#L38)

### ggml_gallocr_t

```c
typedef struct ggml_gallocr * ggml_gallocr_t;
```

Source: [ggml-alloc.h#L46](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-alloc.h#L46)

### ggml_backend_event_t

```c
typedef struct ggml_backend_event * ggml_backend_event_t;
```

Source: [ggml-backend.h#L26](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-backend.h#L26)

### ggml_backend_graph_plan_t

```c
typedef void * ggml_backend_graph_plan_t;
```

Source: [ggml-backend.h#L28](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-backend.h#L28)

### ggml_backend_reg_t

```c
typedef struct ggml_backend_reg * ggml_backend_reg_t;
```

Source: [ggml-backend.h#L29](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-backend.h#L29)

### ggml_backend_dev_t

```c
typedef struct ggml_backend_device * ggml_backend_dev_t;
```

Source: [ggml-backend.h#L30](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-backend.h#L30)

### ggml_backend_split_buffer_type_t

```c
typedef ggml_backend_buffer_type_t (*)(int, const float *) ggml_backend_split_buffer_type_t;
```

Source: [ggml-backend.h#L188](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-backend.h#L188)

### ggml_backend_set_n_threads_t

```c
typedef void (*)(ggml_backend_t, int) ggml_backend_set_n_threads_t;
```

Source: [ggml-backend.h#L190](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-backend.h#L190)

### ggml_backend_dev_get_extra_bufts_t

```c
typedef ggml_backend_buffer_type_t *(*)(ggml_backend_dev_t) ggml_backend_dev_get_extra_bufts_t;
```

Source: [ggml-backend.h#L192](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-backend.h#L192)

### ggml_backend_set_abort_callback_t

```c
typedef void (*)(ggml_backend_t, int, void *) ggml_backend_set_abort_callback_t;
```

Source: [ggml-backend.h#L194](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-backend.h#L194)

### ggml_backend_get_features_t

```c
typedef struct ggml_backend_feature *(*)(ggml_backend_reg_t) ggml_backend_get_features_t;
```

Source: [ggml-backend.h#L200](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-backend.h#L200)

### ggml_backend_sched_t

```c
typedef struct ggml_backend_sched * ggml_backend_sched_t;
```

Source: [ggml-backend.h#L280](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-backend.h#L280)

### ggml_backend_sched_eval_callback

```c
typedef int (*)(struct ggml_tensor *, bool *, void *) ggml_backend_sched_eval_callback;
```

Source: [ggml-backend.h#L289](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-backend.h#L289)

### ggml_backend_eval_callback

```c
typedef int (*)(int, struct ggml_tensor *, struct ggml_tensor *, void *) ggml_backend_eval_callback;
```

Source: [ggml-backend.h#L339](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-backend.h#L339)

### ggml_vec_dot_t

```c
typedef void (*)(int, float *restrict, int, const void *restrict, int, const void *restrict, int, int) ggml_vec_dot_t;
```

Source: [ggml-cpu.h#L106](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-cpu.h#L106)

### ggml_opt_dataset_t

```c
typedef struct ggml_opt_dataset * ggml_opt_dataset_t;
```

Source: [ggml-opt.h#L22](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-opt.h#L22)

### ggml_opt_context_t

```c
typedef struct ggml_opt_context * ggml_opt_context_t;
```

Source: [ggml-opt.h#L23](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-opt.h#L23)

### ggml_opt_result_t

```c
typedef struct ggml_opt_result * ggml_opt_result_t;
```

Source: [ggml-opt.h#L24](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-opt.h#L24)

### ggml_opt_get_optimizer_params

```c
typedef struct ggml_opt_optimizer_params (*)(void *) ggml_opt_get_optimizer_params;
```

Source: [ggml-opt.h#L82](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-opt.h#L82)

### ggml_opt_epoch_callback

```c
typedef void (*)(bool *, ggml_opt_context_t, ggml_opt_dataset_t, ggml_opt_result_t, int64_t, int64_t, int64_t) ggml_opt_epoch_callback;
```

Source: [ggml-opt.h#L171](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-opt.h#L171)

### ggml_fp16_t

```c
typedef uint16_t ggml_fp16_t;
```

Source: [ggml.h#L332](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L332)

### ggml_bf16_t

```c
typedef struct ggml_bf16_t ggml_bf16_t;
```

Source: [ggml.h#L339](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L339)

### bool

```c
typedef int (int *) bool;
```

Source: [ggml.h#L615](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L615)

### ggml_guid

```c
typedef uint8_t[16] ggml_guid;
```

Source: [ggml.h#L623](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L623)

### ggml_guid_t

```c
typedef ggml_guid * ggml_guid_t;
```

Source: [ggml.h#L624](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L624)

### ggml_unary_op_f32_t

```c
typedef void (*)(const int, float *, const float *) ggml_unary_op_f32_t;
```

Source: [ggml.h#L1895](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L1895)

### ggml_binary_op_f32_t

```c
typedef void (*)(const int, float *, const float *, const float *) ggml_binary_op_f32_t;
```

Source: [ggml.h#L1896](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L1896)

### ggml_custom1_op_f32_t

```c
typedef void (*)(struct ggml_tensor *, const struct ggml_tensor *) ggml_custom1_op_f32_t;
```

Source: [ggml.h#L1898](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L1898)

### ggml_custom2_op_f32_t

```c
typedef void (*)(struct ggml_tensor *, const struct ggml_tensor *, const struct ggml_tensor *) ggml_custom2_op_f32_t;
```

Source: [ggml.h#L1899](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L1899)

### ggml_custom3_op_f32_t

```c
typedef void (*)(struct ggml_tensor *, const struct ggml_tensor *, const struct ggml_tensor *, const struct ggml_tensor *) ggml_custom3_op_f32_t;
```

Source: [ggml.h#L1900](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L1900)

### ggml_custom1_op_t

```c
typedef void (*)(struct ggml_tensor *, const struct ggml_tensor *, int, int, void *) ggml_custom1_op_t;
```

Source: [ggml.h#L1972](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L1972)

### ggml_custom2_op_t

```c
typedef void (*)(struct ggml_tensor *, const struct ggml_tensor *, const struct ggml_tensor *, int, int, void *) ggml_custom2_op_t;
```

Source: [ggml.h#L1973](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L1973)

### ggml_custom3_op_t

```c
typedef void (*)(struct ggml_tensor *, const struct ggml_tensor *, const struct ggml_tensor *, const struct ggml_tensor *, int, int, void *) ggml_custom3_op_t;
```

Source: [ggml.h#L1974](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L1974)

### ggml_log_callback

```c
typedef void (*)(enum ggml_log_level, const char *, void *) ggml_log_callback;
```

Source: [ggml.h#L2094](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L2094)

### ggml_to_float_t

```c
typedef void (*)(const void *restrict, float *restrict, int64_t) ggml_to_float_t;
```

Source: [ggml.h#L2145](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L2145)

### ggml_from_float_t

```c
typedef void (*)(const float *restrict, void *restrict, int64_t) ggml_from_float_t;
```

Source: [ggml.h#L2146](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L2146)

### ggml_threadpool_t

```c
typedef struct ggml_threadpool * ggml_threadpool_t;
```

Source: [ggml.h#L2185](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L2185)
