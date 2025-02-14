# Enums

## Index

- [ggml_backend_buffer_usage](#ggml_backend_buffer_usage)
- [ggml_backend_dev_type](#ggml_backend_dev_type)
- [ggml_numa_strategy](#ggml_numa_strategy)
- [ggml_opt_loss_type](#ggml_opt_loss_type)
- [ggml_opt_build_type](#ggml_opt_build_type)
- [ggml_status](#ggml_status)
- [ggml_type](#ggml_type)
- [ggml_prec](#ggml_prec)
- [ggml_ftype](#ggml_ftype)
- [ggml_op](#ggml_op)
- [ggml_unary_op](#ggml_unary_op)
- [ggml_object_type](#ggml_object_type)
- [ggml_log_level](#ggml_log_level)
- [ggml_tensor_flag](#ggml_tensor_flag)
- [ggml_op_pool](#ggml_op_pool)
- [ggml_sort_order](#ggml_sort_order)
- [ggml_sched_priority](#ggml_sched_priority)
- [gguf_type](#gguf_type)


## Detailed Definitions

### ggml_backend_buffer_usage

```c
enum ggml_backend_buffer_usage {
    GGML_BACKEND_BUFFER_USAGE_ANY = 0
    GGML_BACKEND_BUFFER_USAGE_WEIGHTS = 1
    GGML_BACKEND_BUFFER_USAGE_COMPUTE = 2
};
```

Source: [ggml-backend.h#L49](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-backend.h#L49)

### ggml_backend_dev_type

```c
enum ggml_backend_dev_type {
    GGML_BACKEND_DEVICE_TYPE_CPU = 0
    GGML_BACKEND_DEVICE_TYPE_GPU = 1
    GGML_BACKEND_DEVICE_TYPE_ACCEL = 2
};
```

Source: [ggml-backend.h#L130](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-backend.h#L130)

### ggml_numa_strategy

```c
enum ggml_numa_strategy {
    GGML_NUMA_STRATEGY_DISABLED = 0
    GGML_NUMA_STRATEGY_DISTRIBUTE = 1
    GGML_NUMA_STRATEGY_ISOLATE = 2
    GGML_NUMA_STRATEGY_NUMACTL = 3
    GGML_NUMA_STRATEGY_MIRROR = 4
    GGML_NUMA_STRATEGY_COUNT = 5
};
```

Source: [ggml-cpu.h#L25](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-cpu.h#L25)

### ggml_opt_loss_type

```c
enum ggml_opt_loss_type {
    GGML_OPT_LOSS_TYPE_MEAN = 0
    GGML_OPT_LOSS_TYPE_SUM = 1
    GGML_OPT_LOSS_TYPE_CROSS_ENTROPY = 2
    GGML_OPT_LOSS_TYPE_MEAN_SQUARED_ERROR = 3
};
```

Source: [ggml-opt.h#L30](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-opt.h#L30)

### ggml_opt_build_type

```c
enum ggml_opt_build_type {
    GGML_OPT_BUILD_TYPE_FORWARD = 0
    GGML_OPT_BUILD_TYPE_GRAD = 1
    GGML_OPT_BUILD_TYPE_OPT = 2
};
```

Source: [ggml-opt.h#L62](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-opt.h#L62)

### ggml_status

```c
enum ggml_status {
    GGML_STATUS_ALLOC_FAILED = -2
    GGML_STATUS_FAILED = -1
    GGML_STATUS_SUCCESS = 0
    GGML_STATUS_ABORTED = 1
};
```

Source: [ggml.h#L320](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L320)

### ggml_type

```c
enum ggml_type {
    GGML_TYPE_F32 = 0
    GGML_TYPE_F16 = 1
    GGML_TYPE_Q4_0 = 2
    GGML_TYPE_Q4_1 = 3
    GGML_TYPE_Q5_0 = 6
    GGML_TYPE_Q5_1 = 7
    GGML_TYPE_Q8_0 = 8
    GGML_TYPE_Q8_1 = 9
    GGML_TYPE_Q2_K = 10
    GGML_TYPE_Q3_K = 11
    GGML_TYPE_Q4_K = 12
    GGML_TYPE_Q5_K = 13
    GGML_TYPE_Q6_K = 14
    GGML_TYPE_Q8_K = 15
    GGML_TYPE_IQ2_XXS = 16
    GGML_TYPE_IQ2_XS = 17
    GGML_TYPE_IQ3_XXS = 18
    GGML_TYPE_IQ1_S = 19
    GGML_TYPE_IQ4_NL = 20
    GGML_TYPE_IQ3_S = 21
    GGML_TYPE_IQ2_S = 22
    GGML_TYPE_IQ4_XS = 23
    GGML_TYPE_I8 = 24
    GGML_TYPE_I16 = 25
    GGML_TYPE_I32 = 26
    GGML_TYPE_I64 = 27
    GGML_TYPE_F64 = 28
    GGML_TYPE_IQ1_M = 29
    GGML_TYPE_BF16 = 30
    GGML_TYPE_TQ1_0 = 34
    GGML_TYPE_TQ2_0 = 35
    GGML_TYPE_COUNT = 39
};
```

Source: [ggml.h#L351](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L351)

### ggml_prec

```c
enum ggml_prec {
    GGML_PREC_DEFAULT = 0
    GGML_PREC_F32 = 1
};
```

Source: [ggml.h#L395](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L395)

### ggml_ftype

```c
enum ggml_ftype {
    GGML_FTYPE_UNKNOWN = -1
    GGML_FTYPE_ALL_F32 = 0
    GGML_FTYPE_MOSTLY_F16 = 1
    GGML_FTYPE_MOSTLY_Q4_0 = 2
    GGML_FTYPE_MOSTLY_Q4_1 = 3
    GGML_FTYPE_MOSTLY_Q4_1_SOME_F16 = 4
    GGML_FTYPE_MOSTLY_Q8_0 = 7
    GGML_FTYPE_MOSTLY_Q5_0 = 8
    GGML_FTYPE_MOSTLY_Q5_1 = 9
    GGML_FTYPE_MOSTLY_Q2_K = 10
    GGML_FTYPE_MOSTLY_Q3_K = 11
    GGML_FTYPE_MOSTLY_Q4_K = 12
    GGML_FTYPE_MOSTLY_Q5_K = 13
    GGML_FTYPE_MOSTLY_Q6_K = 14
    GGML_FTYPE_MOSTLY_IQ2_XXS = 15
    GGML_FTYPE_MOSTLY_IQ2_XS = 16
    GGML_FTYPE_MOSTLY_IQ3_XXS = 17
    GGML_FTYPE_MOSTLY_IQ1_S = 18
    GGML_FTYPE_MOSTLY_IQ4_NL = 19
    GGML_FTYPE_MOSTLY_IQ3_S = 20
    GGML_FTYPE_MOSTLY_IQ2_S = 21
    GGML_FTYPE_MOSTLY_IQ4_XS = 22
    GGML_FTYPE_MOSTLY_IQ1_M = 23
    GGML_FTYPE_MOSTLY_BF16 = 24
};
```

Source: [ggml.h#L401](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L401)

### ggml_op

```c
enum ggml_op {
    GGML_OP_NONE = 0
    GGML_OP_DUP = 1
    GGML_OP_ADD = 2
    GGML_OP_ADD1 = 3
    GGML_OP_ACC = 4
    GGML_OP_SUB = 5
    GGML_OP_MUL = 6
    GGML_OP_DIV = 7
    GGML_OP_SQR = 8
    GGML_OP_SQRT = 9
    GGML_OP_LOG = 10
    GGML_OP_SIN = 11
    GGML_OP_COS = 12
    GGML_OP_SUM = 13
    GGML_OP_SUM_ROWS = 14
    GGML_OP_MEAN = 15
    GGML_OP_ARGMAX = 16
    GGML_OP_COUNT_EQUAL = 17
    GGML_OP_REPEAT = 18
    GGML_OP_REPEAT_BACK = 19
    GGML_OP_CONCAT = 20
    GGML_OP_SILU_BACK = 21
    GGML_OP_NORM = 22
    GGML_OP_RMS_NORM = 23
    GGML_OP_RMS_NORM_BACK = 24
    GGML_OP_GROUP_NORM = 25
    GGML_OP_MUL_MAT = 26
    GGML_OP_MUL_MAT_ID = 27
    GGML_OP_OUT_PROD = 28
    GGML_OP_SCALE = 29
    GGML_OP_SET = 30
    GGML_OP_CPY = 31
    GGML_OP_CONT = 32
    GGML_OP_RESHAPE = 33
    GGML_OP_VIEW = 34
    GGML_OP_PERMUTE = 35
    GGML_OP_TRANSPOSE = 36
    GGML_OP_GET_ROWS = 37
    GGML_OP_GET_ROWS_BACK = 38
    GGML_OP_DIAG = 39
    GGML_OP_DIAG_MASK_INF = 40
    GGML_OP_DIAG_MASK_ZERO = 41
    GGML_OP_SOFT_MAX = 42
    GGML_OP_SOFT_MAX_BACK = 43
    GGML_OP_ROPE = 44
    GGML_OP_ROPE_BACK = 45
    GGML_OP_CLAMP = 46
    GGML_OP_CONV_TRANSPOSE_1D = 47
    GGML_OP_IM2COL = 48
    GGML_OP_IM2COL_BACK = 49
    GGML_OP_CONV_TRANSPOSE_2D = 50
    GGML_OP_POOL_1D = 51
    GGML_OP_POOL_2D = 52
    GGML_OP_POOL_2D_BACK = 53
    GGML_OP_UPSCALE = 54
    GGML_OP_PAD = 55
    GGML_OP_PAD_REFLECT_1D = 56
    GGML_OP_ARANGE = 57
    GGML_OP_TIMESTEP_EMBEDDING = 58
    GGML_OP_ARGSORT = 59
    GGML_OP_LEAKY_RELU = 60
    GGML_OP_FLASH_ATTN_EXT = 61
    GGML_OP_FLASH_ATTN_BACK = 62
    GGML_OP_SSM_CONV = 63
    GGML_OP_SSM_SCAN = 64
    GGML_OP_WIN_PART = 65
    GGML_OP_WIN_UNPART = 66
    GGML_OP_GET_REL_POS = 67
    GGML_OP_ADD_REL_POS = 68
    GGML_OP_RWKV_WKV6 = 69
    GGML_OP_GATED_LINEAR_ATTN = 70
    GGML_OP_UNARY = 71
    GGML_OP_MAP_UNARY = 72
    GGML_OP_MAP_BINARY = 73
    GGML_OP_MAP_CUSTOM1_F32 = 74
    GGML_OP_MAP_CUSTOM2_F32 = 75
    GGML_OP_MAP_CUSTOM3_F32 = 76
    GGML_OP_MAP_CUSTOM1 = 77
    GGML_OP_MAP_CUSTOM2 = 78
    GGML_OP_MAP_CUSTOM3 = 79
    GGML_OP_CROSS_ENTROPY_LOSS = 80
    GGML_OP_CROSS_ENTROPY_LOSS_BACK = 81
    GGML_OP_OPT_STEP_ADAMW = 82
    GGML_OP_COUNT = 83
};
```

Source: [ggml.h#L429](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L429)

### ggml_unary_op

```c
enum ggml_unary_op {
    GGML_UNARY_OP_ABS = 0
    GGML_UNARY_OP_SGN = 1
    GGML_UNARY_OP_NEG = 2
    GGML_UNARY_OP_STEP = 3
    GGML_UNARY_OP_TANH = 4
    GGML_UNARY_OP_ELU = 5
    GGML_UNARY_OP_RELU = 6
    GGML_UNARY_OP_SIGMOID = 7
    GGML_UNARY_OP_GELU = 8
    GGML_UNARY_OP_GELU_QUICK = 9
    GGML_UNARY_OP_SILU = 10
    GGML_UNARY_OP_HARDSWISH = 11
    GGML_UNARY_OP_HARDSIGMOID = 12
    GGML_UNARY_OP_EXP = 13
    GGML_UNARY_OP_COUNT = 14
};
```

Source: [ggml.h#L526](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L526)

### ggml_object_type

```c
enum ggml_object_type {
    GGML_OBJECT_TYPE_TENSOR = 0
    GGML_OBJECT_TYPE_GRAPH = 1
    GGML_OBJECT_TYPE_WORK_BUFFER = 2
};
```

Source: [ggml.h#L545](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L545)

### ggml_log_level

```c
enum ggml_log_level {
    GGML_LOG_LEVEL_NONE = 0
    GGML_LOG_LEVEL_DEBUG = 1
    GGML_LOG_LEVEL_INFO = 2
    GGML_LOG_LEVEL_WARN = 3
    GGML_LOG_LEVEL_ERROR = 4
    GGML_LOG_LEVEL_CONT = 5
};
```

Source: [ggml.h#L551](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L551)

### ggml_tensor_flag

```c
enum ggml_tensor_flag {
    GGML_TENSOR_FLAG_INPUT = 1
    GGML_TENSOR_FLAG_OUTPUT = 2
    GGML_TENSOR_FLAG_PARAM = 4
    GGML_TENSOR_FLAG_LOSS = 8
};
```

Source: [ggml.h#L561](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L561)

### ggml_op_pool

```c
enum ggml_op_pool {
    GGML_OP_POOL_MAX = 0
    GGML_OP_POOL_AVG = 1
    GGML_OP_POOL_COUNT = 2
};
```

Source: [ggml.h#L1672](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L1672)

### ggml_sort_order

```c
enum ggml_sort_order {
    GGML_SORT_ORDER_ASC = 0
    GGML_SORT_ORDER_DESC = 1
};
```

Source: [ggml.h#L1756](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L1756)

### ggml_sched_priority

```c
enum ggml_sched_priority {
    GGML_SCHED_PRIO_NORMAL = 0
    GGML_SCHED_PRIO_MEDIUM = 1
    GGML_SCHED_PRIO_HIGH = 2
    GGML_SCHED_PRIO_REALTIME = 3
};
```

Source: [ggml.h#L2165](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L2165)

### gguf_type

```c
enum gguf_type {
    GGUF_TYPE_UINT8 = 0
    GGUF_TYPE_INT8 = 1
    GGUF_TYPE_UINT16 = 2
    GGUF_TYPE_INT16 = 3
    GGUF_TYPE_UINT32 = 4
    GGUF_TYPE_INT32 = 5
    GGUF_TYPE_FLOAT32 = 6
    GGUF_TYPE_BOOL = 7
    GGUF_TYPE_STRING = 8
    GGUF_TYPE_ARRAY = 9
    GGUF_TYPE_UINT64 = 10
    GGUF_TYPE_INT64 = 11
    GGUF_TYPE_FLOAT64 = 12
    GGUF_TYPE_COUNT = 13
};
```

Source: [gguf.h#L53](https://github.com/ggml-org/ggml/blob/9a4acb3/include/gguf.h#L53)
