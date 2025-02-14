# Functions

## Index

<details>
<summary>Context Management</summary>

- [ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft)](#ggml-backend-alloc-ctx-tensors-from-buft) - Allocates context tensors from a given backend buffer type.
- [ggml_backend_alloc_ctx_tensors(ctx, backend)](#ggml-backend-alloc-ctx-tensors) - Allocates context tensors for the current backend.
- [ggml_init(params)](#ggml-init) - Creates a ggml_context.
- [ggml_reset(ctx)](#ggml-reset) - Resets the internal state of ggml.
- [ggml_free(ctx)](#ggml-free) - Frees resources allocated by ggml.
- [ggml_get_no_alloc(ctx)](#ggml-get-no-alloc) - Checks if memory allocation is disabled in ggml.
- [ggml_set_no_alloc(ctx, no_alloc)](#ggml-set-no-alloc) - Sets the flag to disable memory allocation in ggml.
</details>


<details>
<summary>Tensor and Graph Allocators</summary>

- [ggml_tallocr_new(buffer)](#ggml-tallocr-new) - Creates a new tensor allocator instance.
- [ggml_tallocr_alloc(talloc, tensor)](#ggml-tallocr-alloc) - Allocates resources for a tensor allocator instance.
- [ggml_gallocr_new(buft)](#ggml-gallocr-new) - Creates a new graph allocator instance.
- [ggml_gallocr_new_n(bufts, n_bufs)](#ggml-gallocr-new-n) - Creates a new graph allocator instance with a specified count.
- [ggml_gallocr_free(galloc)](#ggml-gallocr-free) - Frees a graph allocator instance.
- [ggml_gallocr_reserve(galloc, graph)](#ggml-gallocr-reserve) - Reserves resources for a graph allocator instance.
- [ggml_gallocr_reserve_n(galloc, graph, node_buffer_ids, leaf_buffer_ids)](#ggml-gallocr-reserve-n) - Reserves multiple slots for a graph allocator instance.
- [ggml_gallocr_alloc_graph(galloc, graph)](#ggml-gallocr-alloc-graph) - Allocates a computation graph for a graph allocator instance.
- [ggml_gallocr_get_buffer_size(galloc, buffer_id)](#ggml-gallocr-get-buffer-size) - Returns the required buffer size for a graph allocator instance.
</details>


<details>
<summary>Graph Construction & Execution</summary>

- [ggml_graph_plan(cgraph, n_threads, threadpool)](#ggml-graph-plan) - Plans the execution order of a computation graph.
- [ggml_graph_compute(cgraph, cplan)](#ggml-graph-compute) - Executes the computation graph.
- [ggml_graph_compute_with_ctx(ctx, cgraph, n_threads)](#ggml-graph-compute-with-ctx) - Computes a graph with an associated execution context.
- [ggml_new_graph(ctx)](#ggml-new-graph) - Creates a new computation graph.
- [ggml_new_graph_custom(ctx, size, grads)](#ggml-new-graph-custom) - Creates a custom computation graph.
- [ggml_graph_dup(ctx, cgraph)](#ggml-graph-dup) - Duplicates an existing computation graph.
- [ggml_graph_cpy(src, dst)](#ggml-graph-cpy) - Copies a computation graph.
- [ggml_graph_reset(cgraph)](#ggml-graph-reset) - Resets the state of a computation graph.
- [ggml_graph_clear(cgraph)](#ggml-graph-clear) - Clears all nodes from a computation graph.
- [ggml_graph_size(cgraph)](#ggml-graph-size) - Returns the size of a computation graph.
- [ggml_graph_node(cgraph, i)](#ggml-graph-node) - Retrieves a specific node from a computation graph.
- [ggml_graph_nodes(cgraph)](#ggml-graph-nodes) - Returns all nodes within a computation graph.
- [ggml_graph_n_nodes(cgraph)](#ggml-graph-n-nodes) - Returns the total number of nodes in a computation graph.
- [ggml_graph_add_node(cgraph, tensor)](#ggml-graph-add-node) - Adds a node to a computation graph.
- [ggml_graph_overhead()](#ggml-graph-overhead) - Returns the memory overhead of a computation graph.
- [ggml_graph_overhead_custom(size, grads)](#ggml-graph-overhead-custom) - Returns custom overhead metrics for a computation graph.
- [ggml_graph_get_tensor(cgraph, name)](#ggml-graph-get-tensor) - Retrieves a tensor from a graph node.
- [ggml_graph_get_grad(cgraph, node)](#ggml-graph-get-grad) - Retrieves the gradient tensor from a graph node.
- [ggml_graph_get_grad_acc(cgraph, node)](#ggml-graph-get-grad-acc) - Retrieves the accumulated gradients from a graph node.
- [ggml_graph_export(cgraph, fname)](#ggml-graph-export) - Exports a computation graph to a file or format.
- [ggml_graph_import(fname, ctx_data, ctx_eval)](#ggml-graph-import) - Imports a computation graph from a file or format.
- [ggml_graph_print(cgraph)](#ggml-graph-print) - Prints a human-readable representation of a computation graph.
- [ggml_graph_dump_dot(gb, gf, filename)](#ggml-graph-dump-dot) - Dumps the computation graph in DOT format.
</details>


<details>
<summary>Backend Tensor, Graph & Event Operations</summary>

- [ggml_backend_tensor_copy(src, dst)](#ggml-backend-tensor-copy) - Copies a tensor using backend operations.
- [ggml_backend_guid(backend)](#ggml-backend-guid) - Returns the unique identifier for the backend.
- [ggml_backend_name(backend)](#ggml-backend-name) - Retrieves the name of the current backend.
- [ggml_backend_free(backend)](#ggml-backend-free) - Frees resources associated with the backend.
- [ggml_backend_get_default_buffer_type(backend)](#ggml-backend-get-default-buffer-type) - Returns the default buffer type for the backend.
- [ggml_backend_alloc_buffer(backend, size)](#ggml-backend-alloc-buffer) - Allocates a buffer within the backend.
- [ggml_backend_get_alignment(backend)](#ggml-backend-get-alignment) - Retrieves the alignment requirement for the backend.
- [ggml_backend_get_max_size(backend)](#ggml-backend-get-max-size) - Returns the maximum buffer size supported by the backend.
- [ggml_backend_tensor_set_async(backend, tensor, data, offset, size)](#ggml-backend-tensor-set-async) - Enables asynchronous execution for a backend tensor.
- [ggml_backend_tensor_get_async(backend, tensor, data, offset, size)](#ggml-backend-tensor-get-async) - Retrieves the asynchronous status of a backend tensor.
- [ggml_backend_tensor_set(tensor, data, offset, size)](#ggml-backend-tensor-set) - Sets properties for a backend tensor.
- [ggml_backend_tensor_get(tensor, data, offset, size)](#ggml-backend-tensor-get) - Gets properties of a backend tensor.
- [ggml_backend_tensor_memset(tensor, value, offset, size)](#ggml-backend-tensor-memset) - Fills a backend tensor with a constant value.
- [ggml_backend_synchronize(backend)](#ggml-backend-synchronize) - Synchronizes operations on the backend.
- [ggml_backend_graph_plan_create(backend, cgraph)](#ggml-backend-graph-plan-create) - Creates a computation graph plan for the backend.
- [ggml_backend_graph_plan_free(backend, plan)](#ggml-backend-graph-plan-free) - Frees a previously created backend graph plan.
- [ggml_backend_graph_plan_compute(backend, plan)](#ggml-backend-graph-plan-compute) - Executes a computation graph plan on the backend.
- [ggml_backend_graph_compute(backend, cgraph)](#ggml-backend-graph-compute) - Computes a graph on the backend synchronously.
- [ggml_backend_graph_compute_async(backend, cgraph)](#ggml-backend-graph-compute-async) - Initiates asynchronous graph computation on the backend.
- [ggml_backend_supports_op(backend, op)](#ggml-backend-supports-op) - Checks if the backend supports a specific operation.
- [ggml_backend_supports_buft(backend, buft)](#ggml-backend-supports-buft) - Verifies if the backend supports a specific buffer type.
- [ggml_backend_offload_op(backend, op)](#ggml-backend-offload-op) - Offloads an operation to a backend device.
- [ggml_backend_tensor_copy_async(backend_src, backend_dst, src, dst)](#ggml-backend-tensor-copy-async) - Asynchronously copies a tensor using the backend.
- [ggml_backend_get_device(backend)](#ggml-backend-get-device) - Retrieves the current device used by the backend.
- [ggml_backend_event_new(device)](#ggml-backend-event-new) - Creates a new event for backend synchronization.
- [ggml_backend_event_free(event)](#ggml-backend-event-free) - Frees a backend event.
- [ggml_backend_event_record(event, backend)](#ggml-backend-event-record) - Records a timestamp in a backend event.
- [ggml_backend_event_synchronize(event)](#ggml-backend-event-synchronize) - Synchronizes a backend event.
- [ggml_backend_event_wait(backend, event)](#ggml-backend-event-wait) - Waits for a backend event to complete.
- [ggml_backend_dev_name(device)](#ggml-backend-dev-name) - Returns the name of a backend device.
- [ggml_backend_dev_description(device)](#ggml-backend-dev-description) - Retrieves the description of a backend device.
- [ggml_backend_dev_memory(device, free, total)](#ggml-backend-dev-memory) - Returns the memory capacity of a backend device.
- [ggml_backend_dev_type(device)](#ggml-backend-dev-type) - Retrieves the type of a backend device.
- [ggml_backend_dev_get_props(device, props)](#ggml-backend-dev-get-props) - Gets the properties of a backend device.
- [ggml_backend_dev_backend_reg(device)](#ggml-backend-dev-backend-reg) - Retrieves the registry entry for a backend device.
- [ggml_backend_dev_init(device, params)](#ggml-backend-dev-init) - Initializes a backend device.
- [ggml_backend_dev_buffer_type(device)](#ggml-backend-dev-buffer-type) - Returns the buffer type used by a backend device.
- [ggml_backend_dev_host_buffer_type(device)](#ggml-backend-dev-host-buffer-type) - Returns the host buffer type for a backend device.
- [ggml_backend_dev_buffer_from_host_ptr(device, ptr, size, max_tensor_size)](#ggml-backend-dev-buffer-from-host-ptr) - Creates a device buffer from a host pointer.
- [ggml_backend_dev_supports_op(device, op)](#ggml-backend-dev-supports-op) - Checks if a backend device supports a specific operation.
- [ggml_backend_dev_supports_buft(device, buft)](#ggml-backend-dev-supports-buft) - Verifies if a backend device supports a buffer type.
- [ggml_backend_dev_offload_op(device, op)](#ggml-backend-dev-offload-op) - Offloads an operation specifically to a backend device.
- [ggml_backend_reg_name(reg)](#ggml-backend-reg-name) - Retrieves the name of a backend registry.
- [ggml_backend_reg_dev_count(reg)](#ggml-backend-reg-dev-count) - Returns the number of devices in a backend registry.
- [ggml_backend_reg_dev_get(reg, index)](#ggml-backend-reg-dev-get) - Gets a device from the backend registry by index.
- [ggml_backend_reg_get_proc_address(reg, name)](#ggml-backend-reg-get-proc-address) - Retrieves a procedure address from the backend registry.
- [ggml_backend_device_register(device)](#ggml-backend-device-register) - Registers a new device with the backend.
- [ggml_backend_reg_count()](#ggml-backend-reg-count) - Returns the total count of backend registries.
- [ggml_backend_reg_get(index)](#ggml-backend-reg-get) - Retrieves an entry from the backend registry.
- [ggml_backend_reg_by_name(name)](#ggml-backend-reg-by-name) - Finds a backend registry entry by name.
- [ggml_backend_dev_count()](#ggml-backend-dev-count) - Returns the number of available backend devices.
- [ggml_backend_dev_get(index)](#ggml-backend-dev-get) - Retrieves a backend device by index.
- [ggml_backend_dev_by_name(name)](#ggml-backend-dev-by-name) - Finds a backend device by its name.
- [ggml_backend_dev_by_type(type)](#ggml-backend-dev-by-type) - Finds a backend device by its type.
- [ggml_backend_init_by_name(name, params)](#ggml-backend-init-by-name) - Initializes the backend using its name.
- [ggml_backend_init_by_type(type, params)](#ggml-backend-init-by-type) - Initializes the backend based on device type.
- [ggml_backend_init_best()](#ggml-backend-init-best) - Initializes the best available backend automatically.
- [ggml_backend_load(path)](#ggml-backend-load) - Loads a backend module.
- [ggml_backend_unload(reg)](#ggml-backend-unload) - Unloads a backend module.
- [ggml_backend_load_all()](#ggml-backend-load-all) - Loads all available backend modules.
- [ggml_backend_load_all_from_path(dir_path)](#ggml-backend-load-all-from-path) - Loads all backend modules from a specified path.
- [ggml_backend_sched_new(backends, bufts, n_backends, graph_size, parallel)](#ggml-backend-sched-new) - Creates a new scheduler for backend operations.
- [ggml_backend_sched_free(sched)](#ggml-backend-sched-free) - Frees a backend scheduler.
- [ggml_backend_sched_reserve(sched, measure_graph)](#ggml-backend-sched-reserve) - Reserves resources within the backend scheduler.
- [ggml_backend_sched_get_n_backends(sched)](#ggml-backend-sched-get-n-backends) - Returns the number of backends managed by the scheduler.
- [ggml_backend_sched_get_backend(sched, i)](#ggml-backend-sched-get-backend) - Retrieves a specific backend from the scheduler.
- [ggml_backend_sched_get_n_splits(sched)](#ggml-backend-sched-get-n-splits) - Gets the number of splits configured in the scheduler.
- [ggml_backend_sched_get_n_copies(sched)](#ggml-backend-sched-get-n-copies) - Returns the number of tensor copies scheduled.
- [ggml_backend_sched_get_buffer_size(sched, backend)](#ggml-backend-sched-get-buffer-size) - Retrieves the buffer size allocated by the scheduler.
- [ggml_backend_sched_set_tensor_backend(sched, node, backend)](#ggml-backend-sched-set-tensor-backend) - Assigns a tensor to a specific backend in the scheduler.
- [ggml_backend_sched_get_tensor_backend(sched, node)](#ggml-backend-sched-get-tensor-backend) - Retrieves the backend associated with a tensor in the scheduler.
- [ggml_backend_sched_alloc_graph(sched, graph)](#ggml-backend-sched-alloc-graph) - Allocates a computation graph for the scheduler.
- [ggml_backend_sched_graph_compute(sched, graph)](#ggml-backend-sched-graph-compute) - Computes a graph using the scheduler synchronously.
- [ggml_backend_sched_graph_compute_async(sched, graph)](#ggml-backend-sched-graph-compute-async) - Initiates asynchronous graph computation via the scheduler.
- [ggml_backend_sched_synchronize(sched)](#ggml-backend-sched-synchronize) - Synchronizes all scheduled backend operations.
- [ggml_backend_sched_reset(sched)](#ggml-backend-sched-reset) - Resets the state of the backend scheduler.
- [ggml_backend_sched_set_eval_callback(sched, callback, user_data)](#ggml-backend-sched-set-eval-callback) - Sets an evaluation callback function in the scheduler.
- [ggml_backend_graph_copy(backend, graph)](#ggml-backend-graph-copy) - Creates a duplicate of a backend computation graph.
- [ggml_backend_graph_copy_free(copy)](#ggml-backend-graph-copy-free) - Frees a duplicated backend graph.
- [ggml_backend_compare_graph_backend(backend1, backend2, graph, callback, user_data)](#ggml-backend-compare-graph-backend) - Compares two backend graph implementations.
- [ggml_backend_tensor_alloc(buffer, tensor, addr)](#ggml-backend-tensor-alloc) - Allocates memory for a new backend tensor.
- [ggml_backend_view_init(tensor)](#ggml-backend-view-init) - Initializes a tensor view for the backend.
</details>


<details>
<summary>Mathematical Operations & Element-wise Arithmetic</summary>

- [ggml_new_i32(ctx, value)](#ggml-new-i32) - Creates a new 32-bit integer tensor.
- [ggml_new_f32(ctx, value)](#ggml-new-f32) - Creates a new 32-bit floating-point tensor.
- [ggml_set_i32(tensor, value)](#ggml-set-i32) - Sets the value of a 32-bit integer tensor.
- [ggml_set_f32(tensor, value)](#ggml-set-f32) - Sets the value of a 32-bit floating-point tensor.
- [ggml_get_i32_1d(tensor, i)](#ggml-get-i32-1d) - Retrieves one-dimensional data from a 32-bit integer tensor.
- [ggml_set_i32_1d(tensor, i, value)](#ggml-set-i32-1d) - Sets one-dimensional data for a 32-bit integer tensor.
- [ggml_get_i32_nd(tensor, i0, i1, i2, i3)](#ggml-get-i32-nd) - Retrieves N-dimensional data from a 32-bit integer tensor.
- [ggml_set_i32_nd(tensor, i0, i1, i2, i3, value)](#ggml-set-i32-nd) - Sets N-dimensional data for a 32-bit integer tensor.
- [ggml_get_f32_1d(tensor, i)](#ggml-get-f32-1d) - Retrieves one-dimensional data from a 32-bit floating-point tensor.
- [ggml_set_f32_1d(tensor, i, value)](#ggml-set-f32-1d) - Sets one-dimensional data for a 32-bit floating-point tensor.
- [ggml_get_f32_nd(tensor, i0, i1, i2, i3)](#ggml-get-f32-nd) - Retrieves N-dimensional data from a 32-bit floating-point tensor.
- [ggml_set_f32_nd(tensor, i0, i1, i2, i3, value)](#ggml-set-f32-nd) - Sets N-dimensional data for a 32-bit floating-point tensor.
- [ggml_add(ctx, a, b)](#ggml-add) - Adds two tensors element-wise.
- [ggml_add_inplace(ctx, a, b)](#ggml-add-inplace) - Adds one tensor to another in-place.
- [ggml_add_cast(ctx, a, b, type)](#ggml-add-cast) - Performs tensor addition with type casting.
- [ggml_add1(ctx, a, b)](#ggml-add1) - Adds a scalar value to a tensor.
- [ggml_add1_inplace(ctx, a, b)](#ggml-add1-inplace) - Adds a scalar value to a tensor in-place.
- [ggml_acc(ctx, a, b, nb1, nb2, nb3, offset)](#ggml-acc) - Accumulates values from one tensor into another.
- [ggml_acc_inplace(ctx, a, b, nb1, nb2, nb3, offset)](#ggml-acc-inplace) - Accumulates tensor values in-place.
- [ggml_sub(ctx, a, b)](#ggml-sub) - Subtracts one tensor from another element-wise.
- [ggml_sub_inplace(ctx, a, b)](#ggml-sub-inplace) - Subtracts one tensor from another in-place.
- [ggml_mul(ctx, a, b)](#ggml-mul) - Multiplies two tensors element-wise.
- [ggml_mul_inplace(ctx, a, b)](#ggml-mul-inplace) - Multiplies two tensors element-wise in-place.
- [ggml_div(ctx, a, b)](#ggml-div) - Divides one tensor by another element-wise.
- [ggml_div_inplace(ctx, a, b)](#ggml-div-inplace) - Divides one tensor by another in-place.
- [ggml_sqr(ctx, a)](#ggml-sqr) - Squares each element of a tensor.
- [ggml_sqr_inplace(ctx, a)](#ggml-sqr-inplace) - Squares each tensor element in-place.
- [ggml_sqrt(ctx, a)](#ggml-sqrt) - Computes the square root of each tensor element.
- [ggml_sqrt_inplace(ctx, a)](#ggml-sqrt-inplace) - Computes square roots of tensor elements in-place.
- [ggml_log(ctx, a)](#ggml-log) - Computes the natural logarithm of each tensor element.
- [ggml_log_inplace(ctx, a)](#ggml-log-inplace) - Computes natural logarithms in-place on a tensor.
- [ggml_sin(ctx, a)](#ggml-sin) - Computes the sine of each tensor element.
- [ggml_sin_inplace(ctx, a)](#ggml-sin-inplace) - Computes sine in-place on a tensor.
- [ggml_cos(ctx, a)](#ggml-cos) - Computes the cosine of each tensor element.
- [ggml_cos_inplace(ctx, a)](#ggml-cos-inplace) - Computes cosine in-place on a tensor.
- [ggml_exp(ctx, a)](#ggml-exp) - Computes the exponential of each tensor element.
- [ggml_exp_inplace(ctx, a)](#ggml-exp-inplace) - Computes exponentials in-place on a tensor.
- [ggml_scale(ctx, a, s)](#ggml-scale) - Scales a tensor by a constant factor.
- [ggml_scale_inplace(ctx, a, s)](#ggml-scale-inplace) - Scales a tensor in-place by a constant factor.
</details>


<details>
<summary>Activation Functions & Normalization</summary>

- [ggml_tanh(ctx, a)](#ggml-tanh) - Computes the hyperbolic tangent of each tensor element.
- [ggml_tanh_inplace(ctx, a)](#ggml-tanh-inplace) - Computes hyperbolic tangent in-place on a tensor.
- [ggml_elu(ctx, a)](#ggml-elu) - Applies the ELU activation function to a tensor.
- [ggml_elu_inplace(ctx, a)](#ggml-elu-inplace) - Applies the ELU activation function in-place on a tensor.
- [ggml_relu(ctx, a)](#ggml-relu) - Applies the ReLU activation function to a tensor.
- [ggml_leaky_relu(ctx, a, negative_slope, inplace)](#ggml-leaky-relu) - Applies the Leaky ReLU activation function to a tensor.
- [ggml_relu_inplace(ctx, a)](#ggml-relu-inplace) - Applies the ReLU activation function in-place on a tensor.
- [ggml_sigmoid(ctx, a)](#ggml-sigmoid) - Applies the sigmoid activation function to a tensor.
- [ggml_sigmoid_inplace(ctx, a)](#ggml-sigmoid-inplace) - Applies the sigmoid activation function in-place on a tensor.
- [ggml_gelu(ctx, a)](#ggml-gelu) - Applies the GELU activation function to a tensor.
- [ggml_gelu_inplace(ctx, a)](#ggml-gelu-inplace) - Applies the GELU activation function in-place on a tensor.
- [ggml_gelu_quick(ctx, a)](#ggml-gelu-quick) - Applies a fast approximation of the GELU activation.
- [ggml_gelu_quick_inplace(ctx, a)](#ggml-gelu-quick-inplace) - Applies a fast GELU approximation in-place on a tensor.
- [ggml_silu(ctx, a)](#ggml-silu) - Applies the SiLU (swish) activation function to a tensor.
- [ggml_silu_inplace(ctx, a)](#ggml-silu-inplace) - Applies the SiLU activation function in-place on a tensor.
- [ggml_silu_back(ctx, a, b)](#ggml-silu-back) - Computes the backward pass for the SiLU activation.
- [ggml_hardswish(ctx, a)](#ggml-hardswish) - Applies the hard-swish activation function to a tensor.
- [ggml_hardsigmoid(ctx, a)](#ggml-hardsigmoid) - Applies the hard-sigmoid activation function to a tensor.
- [ggml_norm(ctx, a, eps)](#ggml-norm) - Normalizes the elements of a tensor.
- [ggml_norm_inplace(ctx, a, eps)](#ggml-norm-inplace) - Normalizes a tensor in-place.
- [ggml_rms_norm(ctx, a, eps)](#ggml-rms-norm) - Applies RMS normalization to a tensor.
- [ggml_rms_norm_inplace(ctx, a, eps)](#ggml-rms-norm-inplace) - Applies RMS normalization in-place on a tensor.
- [ggml_group_norm(ctx, a, n_groups, eps)](#ggml-group-norm) - Applies group normalization to a tensor.
- [ggml_group_norm_inplace(ctx, a, n_groups, eps)](#ggml-group-norm-inplace) - Applies group normalization in-place on a tensor.
- [ggml_rms_norm_back(ctx, a, b, eps)](#ggml-rms-norm-back) - Computes the backward pass for RMS normalization.
</details>


<details>
<summary>Tensor Duplication, Copy & Reshaping</summary>

- [ggml_dup_tensor(ctx, src)](#ggml-dup-tensor) - Duplicates an existing tensor.
- [ggml_view_tensor(ctx, src)](#ggml-view-tensor) - Creates a view into an existing tensor.
- [ggml_get_first_tensor(ctx)](#ggml-get-first-tensor) - Retrieves the first tensor in a tensor list.
- [ggml_get_next_tensor(ctx, tensor)](#ggml-get-next-tensor) - Retrieves the next tensor in a tensor list.
- [ggml_get_tensor(ctx, name)](#ggml-get-tensor) - Retrieves a tensor by its index or identifier.
- [ggml_new_tensor(ctx, type, n_dims, ne)](#ggml-new-tensor) - Creates a new tensor with specified parameters.
- [ggml_new_tensor_1d(ctx, type, ne0)](#ggml-new-tensor-1d) - Creates a new one-dimensional tensor.
- [ggml_new_tensor_2d(ctx, type, ne0, ne1)](#ggml-new-tensor-2d) - Creates a new two-dimensional tensor.
- [ggml_new_tensor_3d(ctx, type, ne0, ne1, ne2)](#ggml-new-tensor-3d) - Creates a new three-dimensional tensor.
- [ggml_new_tensor_4d(ctx, type, ne0, ne1, ne2, ne3)](#ggml-new-tensor-4d) - Creates a new four-dimensional tensor.
- [ggml_new_buffer(ctx, nbytes)](#ggml-new-buffer) - Creates a new memory buffer for tensor operations.
- [ggml_dup(ctx, a)](#ggml-dup) - Duplicates a tensor (shallow copy).
- [ggml_dup_inplace(ctx, a)](#ggml-dup-inplace) - Duplicates a tensor in-place.
- [ggml_set(ctx, a, b, nb1, nb2, nb3, offset)](#ggml-set) - Sets tensor elements to specified values.
- [ggml_set_inplace(ctx, a, b, nb1, nb2, nb3, offset)](#ggml-set-inplace) - Sets tensor elements in-place to specified values.
- [ggml_set_1d(ctx, a, b, offset)](#ggml-set-1d) - Sets elements of a one-dimensional tensor.
- [ggml_set_1d_inplace(ctx, a, b, offset)](#ggml-set-1d-inplace) - Sets one-dimensional tensor elements in-place.
- [ggml_set_2d(ctx, a, b, nb1, offset)](#ggml-set-2d) - Sets elements of a two-dimensional tensor.
- [ggml_set_2d_inplace(ctx, a, b, nb1, offset)](#ggml-set-2d-inplace) - Sets two-dimensional tensor elements in-place.
- [ggml_cpy(ctx, a, b)](#ggml-cpy) - Copies data from one tensor to another.
- [ggml_cast(ctx, a, type)](#ggml-cast) - Casts a tensor to a different data type.
- [ggml_cont(ctx, a)](#ggml-cont) - Ensures a tensor is stored contiguously in memory.
- [ggml_cont_1d(ctx, a, ne0)](#ggml-cont-1d) - Ensures a one-dimensional tensor is contiguous in memory.
- [ggml_cont_2d(ctx, a, ne0, ne1)](#ggml-cont-2d) - Ensures a two-dimensional tensor is contiguous in memory.
- [ggml_cont_3d(ctx, a, ne0, ne1, ne2)](#ggml-cont-3d) - Ensures a three-dimensional tensor is contiguous in memory.
- [ggml_cont_4d(ctx, a, ne0, ne1, ne2, ne3)](#ggml-cont-4d) - Ensures a four-dimensional tensor is contiguous in memory.
- [ggml_reshape(ctx, a, b)](#ggml-reshape) - Reshapes a tensor to new dimensions.
- [ggml_reshape_1d(ctx, a, ne0)](#ggml-reshape-1d) - Reshapes a tensor into one dimension.
- [ggml_reshape_2d(ctx, a, ne0, ne1)](#ggml-reshape-2d) - Reshapes a tensor into two dimensions.
- [ggml_reshape_3d(ctx, a, ne0, ne1, ne2)](#ggml-reshape-3d) - Reshapes a tensor into three dimensions.
- [ggml_reshape_4d(ctx, a, ne0, ne1, ne2, ne3)](#ggml-reshape-4d) - Reshapes a tensor into four dimensions.
- [ggml_view_1d(ctx, a, ne0, offset)](#ggml-view-1d) - Creates a one-dimensional view of a tensor.
- [ggml_view_2d(ctx, a, ne0, ne1, nb1, offset)](#ggml-view-2d) - Creates a two-dimensional view of a tensor.
- [ggml_view_3d(ctx, a, ne0, ne1, ne2, nb1, nb2, offset)](#ggml-view-3d) - Creates a three-dimensional view of a tensor.
- [ggml_view_4d(ctx, a, ne0, ne1, ne2, ne3, nb1, nb2, nb3, offset)](#ggml-view-4d) - Creates a four-dimensional view of a tensor.
- [ggml_permute(ctx, a, axis0, axis1, axis2, axis3)](#ggml-permute) - Permutes the dimensions of a tensor.
- [ggml_transpose(ctx, a)](#ggml-transpose) - Transposes a tensor.
</details>


<details>
<summary>Indexing, Reduction & Repetition Operations</summary>

- [ggml_sum(ctx, a)](#ggml-sum) - Sums all elements of a tensor.
- [ggml_sum_rows(ctx, a)](#ggml-sum-rows) - Sums the elements across each row of a tensor.
- [ggml_mean(ctx, a)](#ggml-mean) - Computes the mean value of a tensor.
- [ggml_argmax(ctx, a)](#ggml-argmax) - Finds the index of the maximum element in a tensor.
- [ggml_count_equal(ctx, a, b)](#ggml-count-equal) - Counts elements equal to a specified value in a tensor.
- [ggml_repeat(ctx, a, b)](#ggml-repeat) - Repeats a tensor along specified dimensions.
- [ggml_repeat_back(ctx, a, b)](#ggml-repeat-back) - Repeats a tensor in reverse along specified dimensions.
- [ggml_concat(ctx, a, b, dim)](#ggml-concat) - Concatenates multiple tensors along a given dimension.
- [ggml_get_rows(ctx, a, b)](#ggml-get-rows) - Retrieves specific rows from a tensor.
- [ggml_get_rows_back(ctx, a, b, c)](#ggml-get-rows-back) - Retrieves rows from a tensor in reverse order.
- [ggml_diag(ctx, a)](#ggml-diag) - Extracts the diagonal elements of a tensor.
- [ggml_diag_mask_inf(ctx, a, n_past)](#ggml-diag-mask-inf) - Masks the diagonal of a tensor with negative infinity.
- [ggml_diag_mask_inf_inplace(ctx, a, n_past)](#ggml-diag-mask-inf-inplace) - Masks the diagonal with negative infinity in-place.
- [ggml_diag_mask_zero(ctx, a, n_past)](#ggml-diag-mask-zero) - Masks the diagonal of a tensor with zero.
- [ggml_diag_mask_zero_inplace(ctx, a, n_past)](#ggml-diag-mask-zero-inplace) - Masks the diagonal with zero in-place.
- [ggml_soft_max(ctx, a)](#ggml-soft-max) - Applies the softmax function to a tensor.
- [ggml_soft_max_inplace(ctx, a)](#ggml-soft-max-inplace) - Applies the softmax function in-place on a tensor.
- [ggml_soft_max_ext(ctx, a, mask, scale, max_bias)](#ggml-soft-max-ext) - Applies an extended softmax function to a tensor.
- [ggml_soft_max_ext_back(ctx, a, b, scale, max_bias)](#ggml-soft-max-ext-back) - Computes the backward pass for the extended softmax.
- [ggml_soft_max_ext_back_inplace(ctx, a, b, scale, max_bias)](#ggml-soft-max-ext-back-inplace) - Computes the backward pass for extended softmax in-place.
- [ggml_argsort(ctx, a, order)](#ggml-argsort) - Returns indices that sort the tensor.
- [ggml_arange(ctx, start, stop, step)](#ggml-arange) - Generates a tensor with sequential values.
- [ggml_top_k(ctx, a, k)](#ggml-top-k) - Selects the top k elements from a tensor.
</details>


<details>
<summary>Convolution, Pooling & Image Operations</summary>

- [ggml_im2col(ctx, a, b, s0, s1, p0, p1, d0, d1, is_2D, dst_type)](#ggml-im2col) - Transforms image data into a columnar matrix format.
- [ggml_im2col_back(ctx, a, b, ne, s0, s1, p0, p1, d0, d1, is_2D)](#ggml-im2col-back) - Reconstructs image data from its columnar representation.
- [ggml_conv_1d(ctx, a, b, s0, p0, d0)](#ggml-conv-1d) - Performs a one-dimensional convolution operation.
- [ggml_conv_1d_ph(ctx, a, b, s, d)](#ggml-conv-1d-ph) - Performs a phase-based one-dimensional convolution.
- [ggml_conv_1d_dw(ctx, a, b, s0, p0, d0)](#ggml-conv-1d-dw) - Performs a depthwise one-dimensional convolution.
- [ggml_conv_1d_dw_ph(ctx, a, b, s0, d0)](#ggml-conv-1d-dw-ph) - Performs a phase-based depthwise one-dimensional convolution.
- [ggml_conv_transpose_1d(ctx, a, b, s0, p0, d0)](#ggml-conv-transpose-1d) - Performs a transposed one-dimensional convolution.
- [ggml_conv_2d(ctx, a, b, s0, s1, p0, p1, d0, d1)](#ggml-conv-2d) - Performs a two-dimensional convolution operation.
- [ggml_conv_2d_sk_p0(ctx, a, b)](#ggml-conv-2d-sk-p0) - Performs a two-dimensional convolution with stride kernel padding of 0.
- [ggml_conv_2d_s1_ph(ctx, a, b)](#ggml-conv-2d-s1-ph) - Performs a phase-based two-dimensional convolution with stride 1.
- [ggml_conv_2d_dw(ctx, a, b, s0, s1, p0, p1, d0, d1)](#ggml-conv-2d-dw) - Performs a depthwise two-dimensional convolution.
- [ggml_conv_transpose_2d_p0(ctx, a, b, stride)](#ggml-conv-transpose-2d-p0) - Performs a transposed two-dimensional convolution with padding 0.
- [ggml_pool_1d(ctx, a, op, k0, s0, p0)](#ggml-pool-1d) - Performs a one-dimensional pooling operation.
- [ggml_pool_2d(ctx, a, op, k0, k1, s0, s1, p0, p1)](#ggml-pool-2d) - Performs a two-dimensional pooling operation.
- [ggml_pool_2d_back(ctx, a, af, op, k0, k1, s0, s1, p0, p1)](#ggml-pool-2d-back) - Computes the backward pass for a two-dimensional pooling operation.
- [ggml_upscale(ctx, a, scale_factor)](#ggml-upscale) - Upscales a tensor by a specified factor.
- [ggml_upscale_ext(ctx, a, ne0, ne1, ne2, ne3)](#ggml-upscale-ext) - Upscales a tensor with extended options.
- [ggml_pad(ctx, a, p0, p1, p2, p3)](#ggml-pad) - Pads a tensor with a specified value.
- [ggml_pad_reflect_1d(ctx, a, p0, p1)](#ggml-pad-reflect-1d) - Pads a one-dimensional tensor using reflection.
</details>


<details>
<summary>Sequence & Attention Operations</summary>

- [ggml_timestep_embedding(ctx, timesteps, dim, max_period)](#ggml-timestep-embedding) - Generates timestep embeddings for a tensor.
- [ggml_flash_attn_ext(ctx, q, k, v, mask, scale, max_bias, logit_softcap)](#ggml-flash-attn-ext) - Performs an extended flash attention operation.
- [ggml_flash_attn_ext_set_prec(a, prec)](#ggml-flash-attn-ext-set-prec) - Sets the precision for extended flash attention.
- [ggml_flash_attn_ext_get_prec(a)](#ggml-flash-attn-ext-get-prec) - Gets the current precision setting for extended flash attention.
- [ggml_flash_attn_back(ctx, q, k, v, d, masked)](#ggml-flash-attn-back) - Computes the backward pass for flash attention.
- [ggml_ssm_conv(ctx, sx, c)](#ggml-ssm-conv) - Performs a state-space model convolution.
- [ggml_ssm_scan(ctx, s, x, dt, A, B, C)](#ggml-ssm-scan) - Performs a state-space model scan operation.
- [ggml_win_part(ctx, a, w)](#ggml-win-part) - Partitions a tensor into windows.
- [ggml_win_unpart(ctx, a, w0, h0, w)](#ggml-win-unpart) - Reassembles a tensor from its window partitions.
- [ggml_unary(ctx, a, op)](#ggml-unary) - Applies a unary function to all tensor elements.
- [ggml_unary_inplace(ctx, a, op)](#ggml-unary-inplace) - Applies a unary function to a tensor in-place.
- [ggml_get_rel_pos(ctx, a, qh, kh)](#ggml-get-rel-pos) - Retrieves relative positional information from a tensor.
- [ggml_add_rel_pos(ctx, a, pw, ph)](#ggml-add-rel-pos) - Adds relative positional embeddings to a tensor.
- [ggml_add_rel_pos_inplace(ctx, a, pw, ph)](#ggml-add-rel-pos-inplace) - Adds relative positional embeddings in-place to a tensor.
- [ggml_rwkv_wkv6(ctx, k, v, r, tf, td, state)](#ggml-rwkv-wkv6) - Computes the RWKV WKV6 operation on a tensor.
- [ggml_gated_linear_attn(ctx, k, v, q, g, state, scale)](#ggml-gated-linear-attn) - Applies gated linear attention to a tensor.
</details>


<details>
<summary>Backend Buffer Management</summary>

- [ggml_backend_buft_name(buft)](#ggml-backend-buft-name) - Retrieves the name of a backend buffer type.
- [ggml_backend_buft_alloc_buffer(buft, size)](#ggml-backend-buft-alloc-buffer) - Allocates a buffer for a backend buffer type.
- [ggml_backend_buft_get_alignment(buft)](#ggml-backend-buft-get-alignment) - Gets the memory alignment for a backend buffer type.
- [ggml_backend_buft_get_max_size(buft)](#ggml-backend-buft-get-max-size) - Returns the maximum size allowed for a backend buffer type.
- [ggml_backend_buft_get_alloc_size(buft, tensor)](#ggml-backend-buft-get-alloc-size) - Retrieves the allocation size for a backend buffer type.
- [ggml_backend_buft_is_host(buft)](#ggml-backend-buft-is-host) - Checks if the backend buffer type resides in host memory.
- [ggml_backend_buft_get_device(buft)](#ggml-backend-buft-get-device) - Gets the device associated with a backend buffer type.
- [ggml_backend_buffer_name(buffer)](#ggml-backend-buffer-name) - Retrieves the name of a backend buffer.
- [ggml_backend_buffer_free(buffer)](#ggml-backend-buffer-free) - Frees a previously allocated backend buffer.
- [ggml_backend_buffer_get_base(buffer)](#ggml-backend-buffer-get-base) - Returns the base pointer of a backend buffer.
- [ggml_backend_buffer_get_size(buffer)](#ggml-backend-buffer-get-size) - Retrieves the size of a backend buffer.
- [ggml_backend_buffer_init_tensor(buffer, tensor)](#ggml-backend-buffer-init-tensor) - Initializes a tensor within a backend buffer.
- [ggml_backend_buffer_get_alignment(buffer)](#ggml-backend-buffer-get-alignment) - Gets the alignment property of a backend buffer.
- [ggml_backend_buffer_get_max_size(buffer)](#ggml-backend-buffer-get-max-size) - Returns the maximum size supported by a backend buffer.
- [ggml_backend_buffer_get_alloc_size(buffer, tensor)](#ggml-backend-buffer-get-alloc-size) - Retrieves the allocated size of a backend buffer.
- [ggml_backend_buffer_clear(buffer, value)](#ggml-backend-buffer-clear) - Clears the data stored in a backend buffer.
- [ggml_backend_buffer_is_host(buffer)](#ggml-backend-buffer-is-host) - Checks if a backend buffer is located in host memory.
- [ggml_backend_buffer_set_usage(buffer, usage)](#ggml-backend-buffer-set-usage) - Sets the usage flags for a backend buffer.
- [ggml_backend_buffer_get_usage(buffer)](#ggml-backend-buffer-get-usage) - Retrieves the usage flags of a backend buffer.
- [ggml_backend_buffer_get_type(buffer)](#ggml-backend-buffer-get-type) - Returns the type of a backend buffer.
- [ggml_backend_buffer_reset(buffer)](#ggml-backend-buffer-reset) - Resets a backend buffer to its initial state.
</details>


<details>
<summary>Tensor Optimization & Training</summary>

- [ggml_opt_dataset_init(ne_datapoint, ne_label, ndata, ndata_shard)](#ggml-opt-dataset-init) - Initializes an optimization dataset.
- [ggml_opt_dataset_free(dataset)](#ggml-opt-dataset-free) - Frees an optimization dataset.
- [ggml_opt_dataset_data(dataset)](#ggml-opt-dataset-data) - Retrieves the data from an optimization dataset.
- [ggml_opt_dataset_labels(dataset)](#ggml-opt-dataset-labels) - Retrieves the labels from an optimization dataset.
- [ggml_opt_dataset_shuffle(opt_ctx, dataset, idata)](#ggml-opt-dataset-shuffle) - Shuffles the entries in an optimization dataset.
- [ggml_opt_dataset_get_batch(dataset, data_batch, labels_batch, ibatch)](#ggml-opt-dataset-get-batch) - Gets a batch of data from an optimization dataset.
- [ggml_opt_get_default_optimizer_params(userdata)](#ggml-opt-get-default-optimizer-params) - Retrieves the default parameters for the optimizer.
- [ggml_opt_default_params(backend_sched, ctx_compute, inputs, outputs, loss_type)](#ggml-opt-default-params) - Returns default optimization parameters.
- [ggml_opt_init(params)](#ggml-opt-init) - Initializes an optimizer instance.
- [ggml_opt_free(opt_ctx)](#ggml-opt-free) - Frees an optimizer instance.
- [ggml_opt_reset(opt_ctx, optimizer)](#ggml-opt-reset) - Resets the optimizer to its initial state.
- [ggml_opt_inputs(opt_ctx)](#ggml-opt-inputs) - Retrieves the input tensors for the optimizer.
- [ggml_opt_outputs(opt_ctx)](#ggml-opt-outputs) - Retrieves the output tensors for the optimizer.
- [ggml_opt_labels(opt_ctx)](#ggml-opt-labels) - Retrieves the label tensors used in optimization.
- [ggml_opt_loss(opt_ctx)](#ggml-opt-loss) - Returns the computed loss from the optimizer.
- [ggml_opt_pred(opt_ctx)](#ggml-opt-pred) - Retrieves prediction outputs from the optimizer.
- [ggml_opt_ncorrect(opt_ctx)](#ggml-opt-ncorrect) - Returns the number of correct predictions.
- [ggml_opt_grad_acc(opt_ctx, node)](#ggml-opt-grad-acc) - Retrieves the accumulated gradients from the optimizer.
- [ggml_opt_result_init()](#ggml-opt-result-init) - Initializes a structure for storing optimizer results.
- [ggml_opt_result_free(result)](#ggml-opt-result-free) - Frees an optimizer result structure.
- [ggml_opt_result_reset(result)](#ggml-opt-result-reset) - Resets the optimizer result structure.
- [ggml_opt_result_ndata(result, ndata)](#ggml-opt-result-ndata) - Returns the number of data points in the optimizer results.
- [ggml_opt_result_loss(result, loss, unc)](#ggml-opt-result-loss) - Retrieves the loss value from the optimizer results.
- [ggml_opt_result_pred(result, pred)](#ggml-opt-result-pred) - Retrieves predictions from the optimizer results.
- [ggml_opt_result_accuracy(result, accuracy, unc)](#ggml-opt-result-accuracy) - Calculates accuracy from the optimizer results.
- [ggml_opt_forward(opt_ctx, result)](#ggml-opt-forward) - Performs a forward pass using the optimizer.
- [ggml_opt_forward_backward(opt_ctx, result)](#ggml-opt-forward-backward) - Performs both forward and backward passes in optimization.
- [ggml_opt_epoch(opt_ctx, dataset, result_train, result_eval, idata_split, callback_train, callback_eval)](#ggml-opt-epoch) - Executes one optimization epoch.
- [ggml_opt_epoch_callback_progress_bar(train, opt_ctx, dataset, result, ibatch, ibatch_max, t_start_us)](#ggml-opt-epoch-callback-progress-bar) - Updates a progress bar during an optimization epoch.
- [ggml_opt_fit(backend_sched, ctx_compute, inputs, outputs, dataset, loss_type, get_opt_pars, nepoch, nbatch_logical, val_split, silent)](#ggml-opt-fit) - Trains a model using the optimizer.
- [ggml_opt_step_adamw(ctx, a, grad, m, v, adamw_params)](#ggml-opt-step-adamw) - Performs an optimization step using the AdamW algorithm.
</details>


<details>
<summary>Custom Mapping & Loss Functions</summary>

- [ggml_map_unary_f32(ctx, a, fun)](#ggml-map-unary-f32) - Maps a unary function over a float tensor.
- [ggml_map_unary_inplace_f32(ctx, a, fun)](#ggml-map-unary-inplace-f32) - Applies a unary function to a float tensor in-place.
- [ggml_map_binary_f32(ctx, a, b, fun)](#ggml-map-binary-f32) - Applies a binary function to two float tensors.
- [ggml_map_binary_inplace_f32(ctx, a, b, fun)](#ggml-map-binary-inplace-f32) - Applies a binary function in-place on two float tensors.
- [ggml_map_custom1_f32(ctx, a, fun)](#ggml-map-custom1-f32) - Applies a custom unary function on a float tensor.
- [ggml_map_custom1_inplace_f32(ctx, a, fun)](#ggml-map-custom1-inplace-f32) - Applies a custom unary function on a float tensor in-place.
- [ggml_map_custom2_f32(ctx, a, b, fun)](#ggml-map-custom2-f32) - Applies a custom binary function on a float tensor.
- [ggml_map_custom2_inplace_f32(ctx, a, b, fun)](#ggml-map-custom2-inplace-f32) - Applies a custom binary function on a float tensor in-place.
- [ggml_map_custom3_f32(ctx, a, b, c, fun)](#ggml-map-custom3-f32) - Applies a custom ternary function on a float tensor.
- [ggml_map_custom3_inplace_f32(ctx, a, b, c, fun)](#ggml-map-custom3-inplace-f32) - Applies a custom ternary function on a float tensor in-place.
- [ggml_map_custom1(ctx, a, fun, n_tasks, userdata)](#ggml-map-custom1) - Applies a custom unary function to a tensor.
- [ggml_map_custom1_inplace(ctx, a, fun, n_tasks, userdata)](#ggml-map-custom1-inplace) - Applies a custom unary function to a tensor in-place.
- [ggml_map_custom2(ctx, a, b, fun, n_tasks, userdata)](#ggml-map-custom2) - Applies a custom binary function to a tensor.
- [ggml_map_custom2_inplace(ctx, a, b, fun, n_tasks, userdata)](#ggml-map-custom2-inplace) - Applies a custom binary function to a tensor in-place.
- [ggml_map_custom3(ctx, a, b, c, fun, n_tasks, userdata)](#ggml-map-custom3) - Applies a custom ternary function to a tensor.
- [ggml_map_custom3_inplace(ctx, a, b, c, fun, n_tasks, userdata)](#ggml-map-custom3-inplace) - Applies a custom ternary function to a tensor in-place.
- [ggml_cross_entropy_loss(ctx, a, b)](#ggml-cross-entropy-loss) - Computes the cross-entropy loss for a tensor.
- [ggml_cross_entropy_loss_back(ctx, a, b, c)](#ggml-cross-entropy-loss-back) - Computes the backward pass for cross-entropy loss.
</details>


<details>
<summary>Utility & Information Functions</summary>

- [ggml_fp16_to_fp32()](#ggml-fp16-to-fp32) - Converts half-precision floats to single-precision.
- [ggml_fp32_to_fp16()](#ggml-fp32-to-fp16) - Converts single-precision floats to half-precision.
- [ggml_fp16_to_fp32_row(, , )](#ggml-fp16-to-fp32-row) - Converts a row of half-precision floats to single-precision.
- [ggml_fp32_to_fp16_row(, , )](#ggml-fp32-to-fp16-row) - Converts a row of single-precision floats to half-precision.
- [ggml_fp32_to_bf16()](#ggml-fp32-to-bf16) - Converts single-precision floats to bfloat16 format.
- [ggml_bf16_to_fp32()](#ggml-bf16-to-fp32) - Converts bfloat16 values to single-precision floats.
- [ggml_bf16_to_fp32_row(, , )](#ggml-bf16-to-fp32-row) - Converts a row of bfloat16 values to single-precision.
- [ggml_fp32_to_bf16_row_ref(, , )](#ggml-fp32-to-bf16-row-ref) - Converts a reference row of single-precision floats to bfloat16.
- [ggml_fp32_to_bf16_row(, , )](#ggml-fp32-to-bf16-row) - Converts a row of single-precision floats to bfloat16.
- [ggml_guid_matches(guid_a, guid_b)](#ggml-guid-matches) - Checks if two GUIDs match.
- [ggml_time_init()](#ggml-time-init) - Initializes the ggml timing system.
- [ggml_time_ms()](#ggml-time-ms) - Returns the current time in milliseconds.
- [ggml_time_us()](#ggml-time-us) - Returns the current time in microseconds.
- [ggml_cycles()](#ggml-cycles) - Returns the current CPU cycle count.
- [ggml_cycles_per_ms()](#ggml-cycles-per-ms) - Calculates the number of CPU cycles per millisecond.
- [ggml_fopen(fname, mode)](#ggml-fopen) - Opens a file with ggml-specific settings.
- [ggml_print_object(obj)](#ggml-print-object) - Prints detailed information of a ggml object.
- [ggml_print_objects(ctx)](#ggml-print-objects) - Prints details of multiple ggml objects.
- [ggml_nelements(tensor)](#ggml-nelements) - Returns the number of elements in a tensor.
- [ggml_nrows(tensor)](#ggml-nrows) - Returns the number of rows in a tensor.
- [ggml_nbytes(tensor)](#ggml-nbytes) - Returns the total number of bytes occupied by a tensor.
- [ggml_nbytes_pad(tensor)](#ggml-nbytes-pad) - Returns the padded byte size of a tensor.
- [ggml_blck_size(type)](#ggml-blck-size) - Returns the block size used in tensor operations.
- [ggml_type_size(type)](#ggml-type-size) - Returns the size in bytes of a tensor type.
- [ggml_row_size(type, ne)](#ggml-row-size) - Returns the size of a tensor row in bytes.
- [ggml_type_sizef(type)](#ggml-type-sizef) - Returns the floating-point size in bytes for a tensor type.
- [ggml_type_name(type)](#ggml-type-name) - Returns the name of a tensor type.
- [ggml_op_name(op)](#ggml-op-name) - Returns the name of an operation.
- [ggml_op_symbol(op)](#ggml-op-symbol) - Returns the symbol representing an operation.
- [ggml_unary_op_name(op)](#ggml-unary-op-name) - Returns the name of a unary operation.
- [ggml_op_desc(t)](#ggml-op-desc) - Provides a description of an operation.
- [ggml_element_size(tensor)](#ggml-element-size) - Returns the size in bytes of a single tensor element.
- [ggml_is_quantized(type)](#ggml-is-quantized) - Checks if a tensor is quantized.
- [ggml_ftype_to_ggml_type(ftype)](#ggml-ftype-to-ggml-type) - Converts a file type to a ggml tensor type.
- [ggml_is_transposed(tensor)](#ggml-is-transposed) - Checks if a tensor has been transposed.
- [ggml_is_permuted(tensor)](#ggml-is-permuted) - Checks if a tensor's dimensions are permuted.
- [ggml_is_empty(tensor)](#ggml-is-empty) - Checks if a tensor contains no data.
- [ggml_is_scalar(tensor)](#ggml-is-scalar) - Checks if a tensor represents a scalar.
- [ggml_is_vector(tensor)](#ggml-is-vector) - Checks if a tensor is a vector.
- [ggml_is_matrix(tensor)](#ggml-is-matrix) - Checks if a tensor is a matrix.
- [ggml_is_3d(tensor)](#ggml-is-3d) - Checks if a tensor is three-dimensional.
- [ggml_n_dims(tensor)](#ggml-n-dims) - Returns the number of dimensions of a tensor.
- [ggml_is_contiguous(tensor)](#ggml-is-contiguous) - Checks if a tensor is stored contiguously in memory.
- [ggml_is_contiguous_0(tensor)](#ggml-is-contiguous-0) - Checks if the first dimension of a tensor is contiguous.
- [ggml_is_contiguous_1(tensor)](#ggml-is-contiguous-1) - Checks if the second dimension of a tensor is contiguous.
- [ggml_is_contiguous_2(tensor)](#ggml-is-contiguous-2) - Checks if the third dimension of a tensor is contiguous.
- [ggml_are_same_shape(t0, t1)](#ggml-are-same-shape) - Checks if two tensors have identical shapes.
- [ggml_are_same_stride(t0, t1)](#ggml-are-same-stride) - Checks if two tensors have identical memory strides.
- [ggml_can_repeat(t0, t1)](#ggml-can-repeat) - Checks if a tensor can be repeated along its dimensions.
- [ggml_tensor_overhead()](#ggml-tensor-overhead) - Returns the memory overhead of a tensor.
- [ggml_validate_row_data(type, data, nbytes)](#ggml-validate-row-data) - Validates the data contained in a tensor row.
- [ggml_used_mem(ctx)](#ggml-used-mem) - Returns the amount of memory currently used by ggml.
- [ggml_get_mem_buffer(ctx)](#ggml-get-mem-buffer) - Retrieves the current memory buffer pointer used by ggml.
- [ggml_get_mem_size(ctx)](#ggml-get-mem-size) - Returns the size of the allocated memory buffer in ggml.
- [ggml_get_max_tensor_size(ctx)](#ggml-get-max-tensor-size) - Returns the maximum allowable size for a tensor.
- [ggml_get_data(tensor)](#ggml-get-data) - Returns a pointer to the raw data of a tensor.
- [ggml_get_data_f32(tensor)](#ggml-get-data-f32) - Returns a pointer to the float data of a tensor.
- [ggml_get_name(tensor)](#ggml-get-name) - Retrieves the name of a tensor.
- [ggml_set_name(tensor, name)](#ggml-set-name) - Assigns a name to a tensor.
- [ggml_format_name(tensor, fmt)](#ggml-format-name) - Formats a tensor name for display purposes.
- [ggml_log_set(log_callback, user_data)](#ggml-log-set) - Sets the logging level or output for ggml.
- [ggml_set_zero(tensor)](#ggml-set-zero) - Sets all elements of a tensor to zero.
</details>


<details>
<summary>Quantization</summary>

- [ggml_quantize_init(type)](#ggml-quantize-init) - Initializes quantization parameters for tensors.
- [ggml_quantize_free()](#ggml-quantize-free) - Frees resources used for quantization.
- [ggml_quantize_requires_imatrix(type)](#ggml-quantize-requires-imatrix) - Checks if quantization requires an integer matrix.
- [ggml_quantize_chunk(type, src, dst, start, nrows, n_per_row, imatrix)](#ggml-quantize-chunk) - Quantizes a chunk of tensor data.
</details>


<details>
<summary>GGUF File Operations</summary>

- [gguf_init_empty()](#gguf-init-empty) - Initializes an empty GGUF structure.
- [gguf_init_from_file(fname, params)](#gguf-init-from-file) - Initializes a GGUF structure from a file.
- [gguf_free(ctx)](#gguf-free) - Frees a GGUF structure.
- [gguf_type_name(type)](#gguf-type-name) - Returns the name of a GGUF type.
- [gguf_get_version(ctx)](#gguf-get-version) - Retrieves the version of the GGUF format.
- [gguf_get_alignment(ctx)](#gguf-get-alignment) - Returns the alignment requirement for GGUF data.
- [gguf_get_data_offset(ctx)](#gguf-get-data-offset) - Retrieves the data offset within a GGUF file.
- [gguf_get_n_kv(ctx)](#gguf-get-n-kv) - Returns the number of key-value pairs in a GGUF structure.
- [gguf_find_key(ctx, key)](#gguf-find-key) - Searches for a key within a GGUF structure.
- [gguf_get_key(ctx, key_id)](#gguf-get-key) - Retrieves a key from a GGUF structure.
- [gguf_get_kv_type(ctx, key_id)](#gguf-get-kv-type) - Returns the type of a key-value pair in GGUF.
- [gguf_get_arr_type(ctx, key_id)](#gguf-get-arr-type) - Returns the array type for a GGUF structure.
- [gguf_get_val_u8(ctx, key_id)](#gguf-get-val-u8) - Retrieves an unsigned 8-bit value from GGUF.
- [gguf_get_val_i8(ctx, key_id)](#gguf-get-val-i8) - Retrieves a signed 8-bit value from GGUF.
- [gguf_get_val_u16(ctx, key_id)](#gguf-get-val-u16) - Retrieves an unsigned 16-bit value from GGUF.
- [gguf_get_val_i16(ctx, key_id)](#gguf-get-val-i16) - Retrieves a signed 16-bit value from GGUF.
- [gguf_get_val_u32(ctx, key_id)](#gguf-get-val-u32) - Retrieves an unsigned 32-bit value from GGUF.
- [gguf_get_val_i32(ctx, key_id)](#gguf-get-val-i32) - Retrieves a signed 32-bit value from GGUF.
- [gguf_get_val_f32(ctx, key_id)](#gguf-get-val-f32) - Retrieves a 32-bit floating-point value from GGUF.
- [gguf_get_val_u64(ctx, key_id)](#gguf-get-val-u64) - Retrieves an unsigned 64-bit value from GGUF.
- [gguf_get_val_i64(ctx, key_id)](#gguf-get-val-i64) - Retrieves a signed 64-bit value from GGUF.
- [gguf_get_val_f64(ctx, key_id)](#gguf-get-val-f64) - Retrieves a 64-bit floating-point value from GGUF.
- [gguf_get_val_bool(ctx, key_id)](#gguf-get-val-bool) - Retrieves a boolean value from GGUF.
- [gguf_get_val_str(ctx, key_id)](#gguf-get-val-str) - Retrieves a string value from GGUF.
- [gguf_get_val_data(ctx, key_id)](#gguf-get-val-data) - Retrieves raw data associated with a key in GGUF.
- [gguf_get_arr_n(ctx, key_id)](#gguf-get-arr-n) - Returns the number of elements in a GGUF array.
- [gguf_get_arr_data(ctx, key_id)](#gguf-get-arr-data) - Retrieves the data from a GGUF array.
- [gguf_get_arr_str(ctx, key_id, i)](#gguf-get-arr-str) - Retrieves an array of strings from GGUF.
- [gguf_get_n_tensors(ctx)](#gguf-get-n-tensors) - Returns the number of tensors stored in a GGUF file.
- [gguf_find_tensor(ctx, name)](#gguf-find-tensor) - Searches for a tensor by name in a GGUF structure.
- [gguf_get_tensor_offset(ctx, tensor_id)](#gguf-get-tensor-offset) - Retrieves the data offset for a tensor in GGUF.
- [gguf_get_tensor_name(ctx, tensor_id)](#gguf-get-tensor-name) - Retrieves the name of a tensor from GGUF.
- [gguf_get_tensor_type(ctx, tensor_id)](#gguf-get-tensor-type) - Returns the type of a tensor stored in GGUF.
- [gguf_get_tensor_size(ctx, tensor_id)](#gguf-get-tensor-size) - Returns the size of a tensor in a GGUF file.
- [gguf_remove_key(ctx, key)](#gguf-remove-key) - Removes a key-value pair from a GGUF structure.
- [gguf_set_val_u8(ctx, key, val)](#gguf-set-val-u8) - Sets an unsigned 8-bit value in GGUF.
- [gguf_set_val_i8(ctx, key, val)](#gguf-set-val-i8) - Sets a signed 8-bit value in GGUF.
- [gguf_set_val_u16(ctx, key, val)](#gguf-set-val-u16) - Sets an unsigned 16-bit value in GGUF.
- [gguf_set_val_i16(ctx, key, val)](#gguf-set-val-i16) - Sets a signed 16-bit value in GGUF.
- [gguf_set_val_u32(ctx, key, val)](#gguf-set-val-u32) - Sets an unsigned 32-bit value in GGUF.
- [gguf_set_val_i32(ctx, key, val)](#gguf-set-val-i32) - Sets a signed 32-bit value in GGUF.
- [gguf_set_val_f32(ctx, key, val)](#gguf-set-val-f32) - Sets a 32-bit floating-point value in GGUF.
- [gguf_set_val_u64(ctx, key, val)](#gguf-set-val-u64) - Sets an unsigned 64-bit value in GGUF.
- [gguf_set_val_i64(ctx, key, val)](#gguf-set-val-i64) - Sets a signed 64-bit value in GGUF.
- [gguf_set_val_f64(ctx, key, val)](#gguf-set-val-f64) - Sets a 64-bit floating-point value in GGUF.
- [gguf_set_val_bool(ctx, key, val)](#gguf-set-val-bool) - Sets a boolean value in GGUF.
- [gguf_set_val_str(ctx, key, val)](#gguf-set-val-str) - Sets a string value in GGUF.
- [gguf_set_arr_data(ctx, key, type, data, n)](#gguf-set-arr-data) - Sets the data for an array in GGUF.
- [gguf_set_arr_str(ctx, key, data, n)](#gguf-set-arr-str) - Sets an array of strings in GGUF.
- [gguf_set_kv(ctx, src)](#gguf-set-kv) - Sets a key-value pair in GGUF.
- [gguf_add_tensor(ctx, tensor)](#gguf-add-tensor) - Adds a tensor to a GGUF structure.
- [gguf_set_tensor_type(ctx, name, type)](#gguf-set-tensor-type) - Sets the type for a tensor in GGUF.
- [gguf_set_tensor_data(ctx, name, data)](#gguf-set-tensor-data) - Sets the data for a tensor in GGUF.
- [gguf_write_to_file(ctx, fname, only_meta)](#gguf-write-to-file) - Writes a GGUF structure to a file.
- [gguf_get_meta_size(ctx)](#gguf-get-meta-size) - Returns the size of the metadata in a GGUF file.
- [gguf_get_meta_data(ctx, data)](#gguf-get-meta-data) - Retrieves the metadata from a GGUF file.
</details>


<details>
<summary>Threadpool Management</summary>

- [ggml_threadpool_new(params)](#ggml-threadpool-new) - Creates a new thread pool for parallel operations.
- [ggml_threadpool_free(threadpool)](#ggml-threadpool-free) - Frees a previously created thread pool.
- [ggml_threadpool_get_n_threads(threadpool)](#ggml-threadpool-get-n-threads) - Returns the number of threads in the thread pool.
- [ggml_threadpool_pause(threadpool)](#ggml-threadpool-pause) - Pauses execution of the thread pool.
- [ggml_threadpool_resume(threadpool)](#ggml-threadpool-resume) - Resumes execution of the thread pool.
- [ggml_threadpool_params_default(n_threads)](#ggml-threadpool-params-default) - Returns default parameters for thread pool configuration.
- [ggml_threadpool_params_init(p, n_threads)](#ggml-threadpool-params-init) - Initializes parameters for a thread pool.
- [ggml_threadpool_params_match(p0, p1)](#ggml-threadpool-params-match) - Checks if two thread pool parameter sets match.
</details>


<details>
<summary>CPU Backend Operations</summary>

- [ggml_cpu_has_sse3()](#ggml-cpu-has-sse3) - Checks if the CPU supports SSE3 instructions.
- [ggml_cpu_has_ssse3()](#ggml-cpu-has-ssse3) - Checks if the CPU supports SSSE3 instructions.
- [ggml_cpu_has_avx()](#ggml-cpu-has-avx) - Checks if the CPU supports AVX instructions.
- [ggml_cpu_has_avx_vnni()](#ggml-cpu-has-avx-vnni) - Checks if the CPU supports AVX VNNI instructions.
- [ggml_cpu_has_avx2()](#ggml-cpu-has-avx2) - Checks if the CPU supports AVX2 instructions.
- [ggml_cpu_has_f16c()](#ggml-cpu-has-f16c) - Checks if the CPU supports F16C instructions.
- [ggml_cpu_has_fma()](#ggml-cpu-has-fma) - Checks if the CPU supports FMA (Fused Multiply-Add) instructions.
- [ggml_cpu_has_avx512()](#ggml-cpu-has-avx512) - Checks if the CPU supports AVX512 instructions.
- [ggml_cpu_has_avx512_vbmi()](#ggml-cpu-has-avx512-vbmi) - Checks if the CPU supports AVX512 VBMI instructions.
- [ggml_cpu_has_avx512_vnni()](#ggml-cpu-has-avx512-vnni) - Checks if the CPU supports AVX512 VNNI instructions.
- [ggml_cpu_has_avx512_bf16()](#ggml-cpu-has-avx512-bf16) - Checks if the CPU supports AVX512 BF16 instructions.
- [ggml_cpu_has_amx_int8()](#ggml-cpu-has-amx-int8) - Checks if the CPU supports AMX INT8 instructions.
- [ggml_cpu_has_neon()](#ggml-cpu-has-neon) - Checks if the CPU supports NEON instructions.
- [ggml_cpu_has_arm_fma()](#ggml-cpu-has-arm-fma) - Checks if the CPU supports ARM FMA instructions.
- [ggml_cpu_has_fp16_va()](#ggml-cpu-has-fp16-va) - Checks if the CPU supports FP16 vector arithmetic.
- [ggml_cpu_has_dotprod()](#ggml-cpu-has-dotprod) - Checks if the CPU supports dot product operations.
- [ggml_cpu_has_matmul_int8()](#ggml-cpu-has-matmul-int8) - Checks if the CPU supports int8 matrix multiplication.
- [ggml_cpu_has_sve()](#ggml-cpu-has-sve) - Checks if the CPU supports SVE (Scalable Vector Extension).
- [ggml_cpu_get_sve_cnt()](#ggml-cpu-get-sve-cnt) - Returns the number of SVE registers available on the CPU.
- [ggml_cpu_has_riscv_v()](#ggml-cpu-has-riscv-v) - Checks if a RISC-V CPU supports vector instructions.
- [ggml_cpu_has_vsx()](#ggml-cpu-has-vsx) - Checks if the CPU supports VSX instructions.
- [ggml_cpu_has_wasm_simd()](#ggml-cpu-has-wasm-simd) - Checks if the CPU supports WebAssembly SIMD.
- [ggml_cpu_has_llamafile()](#ggml-cpu-has-llamafile) - Checks if the CPU supports llama file optimizations.
- [ggml_get_type_traits_cpu(type)](#ggml-get-type-traits-cpu) - Retrieves CPU type traits for tensor operations.
- [ggml_cpu_init()](#ggml-cpu-init) - Initializes CPU-specific settings for ggml.
- [ggml_backend_cpu_init()](#ggml-backend-cpu-init) - Initializes the CPU backend.
- [ggml_backend_is_cpu(backend)](#ggml-backend-is-cpu) - Checks if the active backend is CPU-based.
- [ggml_backend_cpu_set_n_threads(backend_cpu, n_threads)](#ggml-backend-cpu-set-n-threads) - Sets the number of threads for the CPU backend.
- [ggml_backend_cpu_set_threadpool(backend_cpu, threadpool)](#ggml-backend-cpu-set-threadpool) - Assigns a thread pool to the CPU backend.
- [ggml_backend_cpu_set_abort_callback(backend_cpu, abort_callback, abort_callback_data)](#ggml-backend-cpu-set-abort-callback) - Sets an abort callback for CPU backend operations.
- [ggml_backend_cpu_reg()](#ggml-backend-cpu-reg) - Registers the CPU backend.
- [ggml_backend_cpu_buffer_from_ptr(ptr, size)](#ggml-backend-cpu-buffer-from-ptr) - Creates a CPU backend buffer from an existing pointer.
- [ggml_backend_cpu_buffer_type()](#ggml-backend-cpu-buffer-type) - Returns the buffer type used by the CPU backend.
</details>


<details>
<summary>CUDA Backend Operations</summary>

- [ggml_backend_cuda_init(device)](#ggml-backend-cuda-init) - Initializes the CUDA backend.
- [ggml_backend_is_cuda(backend)](#ggml-backend-is-cuda) - Checks if the active backend is CUDA.
- [ggml_backend_cuda_buffer_type(device)](#ggml-backend-cuda-buffer-type) - Returns the buffer type for the CUDA backend.
- [ggml_backend_cuda_split_buffer_type(main_device, tensor_split)](#ggml-backend-cuda-split-buffer-type) - Returns the split buffer type for CUDA operations.
- [ggml_backend_cuda_host_buffer_type()](#ggml-backend-cuda-host-buffer-type) - Returns the host buffer type for the CUDA backend.
- [ggml_backend_cuda_get_device_count()](#ggml-backend-cuda-get-device-count) - Returns the number of available CUDA devices.
- [ggml_backend_cuda_get_device_description(device, description, description_size)](#ggml-backend-cuda-get-device-description) - Retrieves the description of a CUDA device.
- [ggml_backend_cuda_get_device_memory(device, free, total)](#ggml-backend-cuda-get-device-memory) - Returns the memory capacity of a CUDA device.
- [ggml_backend_cuda_register_host_buffer(buffer, size)](#ggml-backend-cuda-register-host-buffer) - Registers a host buffer with the CUDA backend.
- [ggml_backend_cuda_unregister_host_buffer(buffer)](#ggml-backend-cuda-unregister-host-buffer) - Unregisters a host buffer from the CUDA backend.
- [ggml_backend_cuda_reg()](#ggml-backend-cuda-reg) - Registers the CUDA backend.
</details>


<details>
<summary>Vulkan Backend Operations</summary>

- [ggml_vk_available_devices(memoryRequired, count)](#ggml-vk-available-devices) - Lists the available Vulkan devices.
- [ggml_vk_get_device(device, memoryRequired, name)](#ggml-vk-get-device) - Retrieves a Vulkan device by index.
- [ggml_vk_has_vulkan()](#ggml-vk-has-vulkan) - Checks if Vulkan is supported on the system.
- [ggml_vk_has_device()](#ggml-vk-has-device) - Checks if a specific Vulkan device is available.
- [ggml_vk_current_device()](#ggml-vk-current-device) - Returns the currently active Vulkan device.
- [ggml_backend_vk_init(dev_num)](#ggml-backend-vk-init) - Initializes the Vulkan backend.
- [ggml_backend_is_vk(backend)](#ggml-backend-is-vk) - Checks if the active backend is Vulkan.
- [ggml_backend_vk_get_device_count()](#ggml-backend-vk-get-device-count) - Returns the number of available Vulkan devices.
- [ggml_backend_vk_get_device_description(device, description, description_size)](#ggml-backend-vk-get-device-description) - Retrieves the description of a Vulkan device.
- [ggml_backend_vk_get_device_memory(device, free, total)](#ggml-backend-vk-get-device-memory) - Returns the memory capacity of a Vulkan device.
- [ggml_backend_vk_buffer_type(dev_num)](#ggml-backend-vk-buffer-type) - Returns the buffer type for the Vulkan backend.
- [ggml_backend_vk_host_buffer_type()](#ggml-backend-vk-host-buffer-type) - Returns the host buffer type for the Vulkan backend.
- [ggml_backend_vk_reg()](#ggml-backend-vk-reg) - Registers the Vulkan backend.
</details>


<details>
<summary>Metal Backend Operations</summary>

- [ggml_backend_metal_init()](#ggml-backend-metal-init) - Initializes the Metal backend.
- [ggml_backend_is_metal(backend)](#ggml-backend-is-metal) - Checks if the active backend is Metal.
- [ggml_backend_metal_buffer_from_ptr(data, size, max_size)](#ggml-backend-metal-buffer-from-ptr) - Creates a Metal buffer from a host pointer.
- [ggml_backend_metal_set_abort_callback(backend, abort_callback, user_data)](#ggml-backend-metal-set-abort-callback) - Sets an abort callback for Metal operations.
- [ggml_backend_metal_buffer_type()](#ggml-backend-metal-buffer-type) - Returns the buffer type for the Metal backend.
- [ggml_backend_metal_supports_family(backend, family)](#ggml-backend-metal-supports-family) - Checks if the Metal backend supports a specific family.
- [ggml_backend_metal_capture_next_compute(backend)](#ggml-backend-metal-capture-next-compute) - Captures the next compute command in Metal.
- [ggml_backend_metal_reg()](#ggml-backend-metal-reg) - Registers the Metal backend.
</details>


<details>
<summary>OpenCL Backend Operations</summary>

- [ggml_backend_opencl_init()](#ggml-backend-opencl-init) - Initializes the OpenCL backend.
- [ggml_backend_is_opencl(backend)](#ggml-backend-is-opencl) - Checks if the active backend is OpenCL.
- [ggml_backend_opencl_buffer_type()](#ggml-backend-opencl-buffer-type) - Returns the buffer type for the OpenCL backend.
- [ggml_backend_opencl_host_buffer_type()](#ggml-backend-opencl-host-buffer-type) - Returns the host buffer type for OpenCL.
- [ggml_backend_opencl_reg()](#ggml-backend-opencl-reg) - Registers the OpenCL backend.
</details>


<details>
<summary>CANN Backend Operations</summary>

- [ggml_backend_cann_reg()](#ggml-backend-cann-reg) - Registers the CANN backend.
- [ggml_backend_cann_init(device)](#ggml-backend-cann-init) - Initializes the CANN backend.
- [ggml_backend_is_cann(backend)](#ggml-backend-is-cann) - Checks if the CANN backend is active.
- [ggml_backend_cann_buffer_type(device)](#ggml-backend-cann-buffer-type) - Returns the buffer type for the CANN backend.
- [ggml_backend_cann_get_device_count()](#ggml-backend-cann-get-device-count) - Returns the number of CANN devices available.
- [ggml_backend_cann_host_buffer_type()](#ggml-backend-cann-host-buffer-type) - Returns the host buffer type for the CANN backend.
- [ggml_backend_cann_get_device_description(device, description, description_size)](#ggml-backend-cann-get-device-description) - Retrieves the description of a CANN device.
- [ggml_backend_cann_get_device_memory(device, free, total)](#ggml-backend-cann-get-device-memory) - Returns the memory capacity of a CANN device.
</details>


<details>
<summary>Kompute Backend Operations</summary>

- [ggml_backend_kompute_init(device)](#ggml-backend-kompute-init) - Initializes the Kompute backend.
- [ggml_backend_is_kompute(backend)](#ggml-backend-is-kompute) - Checks if the active backend is Kompute.
- [ggml_backend_kompute_buffer_type(device)](#ggml-backend-kompute-buffer-type) - Returns the buffer type for the Kompute backend.
- [ggml_backend_kompute_reg()](#ggml-backend-kompute-reg) - Registers the Kompute backend.
</details>


<details>
<summary>RPC Backend Operations</summary>

- [ggml_backend_rpc_init(endpoint)](#ggml-backend-rpc-init) - Initializes the RPC backend.
- [ggml_backend_is_rpc(backend)](#ggml-backend-is-rpc) - Checks if the active backend is RPC.
- [ggml_backend_rpc_buffer_type(endpoint)](#ggml-backend-rpc-buffer-type) - Returns the buffer type for the RPC backend.
- [ggml_backend_rpc_get_device_memory(endpoint, free, total)](#ggml-backend-rpc-get-device-memory) - Returns the device memory for an RPC backend device.
- [ggml_backend_rpc_start_server(backend, endpoint, free_mem, total_mem)](#ggml-backend-rpc-start-server) - Starts an RPC server for backend communication.
- [ggml_backend_rpc_reg()](#ggml-backend-rpc-reg) - Registers the RPC backend.
- [ggml_backend_rpc_add_device(endpoint)](#ggml-backend-rpc-add-device) - Adds a device to the RPC backend.
</details>


<details>
<summary>SYCL Backend Operations</summary>

- [ggml_backend_sycl_init(device)](#ggml-backend-sycl-init) - Initializes the SYCL backend.
- [ggml_backend_is_sycl(backend)](#ggml-backend-is-sycl) - Checks if the active backend is SYCL.
- [ggml_backend_sycl_buffer_type(device)](#ggml-backend-sycl-buffer-type) - Returns the buffer type for the SYCL backend.
- [ggml_backend_sycl_split_buffer_type(tensor_split)](#ggml-backend-sycl-split-buffer-type) - Returns the split buffer type for SYCL operations.
- [ggml_backend_sycl_host_buffer_type()](#ggml-backend-sycl-host-buffer-type) - Returns the host buffer type for the SYCL backend.
- [ggml_backend_sycl_print_sycl_devices()](#ggml-backend-sycl-print-sycl-devices) - Prints available SYCL devices.
- [ggml_backend_sycl_get_gpu_list(id_list, max_len)](#ggml-backend-sycl-get-gpu-list) - Retrieves a list of SYCL GPU devices.
- [ggml_backend_sycl_get_device_description(device, description, description_size)](#ggml-backend-sycl-get-device-description) - Gets the description of a SYCL device.
- [ggml_backend_sycl_get_device_count()](#ggml-backend-sycl-get-device-count) - Returns the number of available SYCL devices.
- [ggml_backend_sycl_get_device_memory(device, free, total)](#ggml-backend-sycl-get-device-memory) - Returns the memory capacity of a SYCL device.
- [ggml_backend_sycl_reg()](#ggml-backend-sycl-reg) - Registers the SYCL backend.
</details>


## Detailed Definitions

### ggml_backend_alloc_ctx_tensors_from_buft

Allocates context tensors from a given backend buffer type.

```c
struct ggml_backend_buffer * ggml_backend_alloc_ctx_tensors_from_buft(struct ggml_context * ctx, ggml_backend_buffer_type_t buft)
```

Source: [ggml-alloc.h#L71](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-alloc.h#L71)

### ggml_backend_alloc_ctx_tensors

Allocates context tensors for the current backend.

```c
struct ggml_backend_buffer * ggml_backend_alloc_ctx_tensors(struct ggml_context * ctx, ggml_backend_t backend)
```

Source: [ggml-alloc.h#L72](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-alloc.h#L72)

### ggml_init

Creates a ggml_context.

```c
struct ggml_context * ggml_init(struct ggml_init_params params)
```

Source: [ggml.h#L695](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L695)

### ggml_reset

Resets the internal state of ggml.

```c
void ggml_reset(struct ggml_context * ctx)
```

Source: [ggml.h#L696](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L696)

### ggml_free

Frees resources allocated by ggml.

```c
void ggml_free(struct ggml_context * ctx)
```

Source: [ggml.h#L697](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L697)

### ggml_get_no_alloc

Checks if memory allocation is disabled in ggml.

```c
int ggml_get_no_alloc(struct ggml_context * ctx)
```

Source: [ggml.h#L701](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L701)

### ggml_set_no_alloc

Sets the flag to disable memory allocation in ggml.

```c
void ggml_set_no_alloc(struct ggml_context * ctx, bool no_alloc)
```

Source: [ggml.h#L702](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L702)

### ggml_tallocr_new

Creates a new tensor allocator instance.

```c
struct ggml_tallocr ggml_tallocr_new(ggml_backend_buffer_t buffer)
```

Source: [ggml-alloc.h#L21](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-alloc.h#L21)

### ggml_tallocr_alloc

Allocates resources for a tensor allocator instance.

```c
void ggml_tallocr_alloc(struct ggml_tallocr * talloc, struct ggml_tensor * tensor)
```

Source: [ggml-alloc.h#L22](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-alloc.h#L22)

### ggml_gallocr_new

Creates a new graph allocator instance.

```c
ggml_gallocr_t ggml_gallocr_new(ggml_backend_buffer_type_t buft)
```

Source: [ggml-alloc.h#L48](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-alloc.h#L48)

### ggml_gallocr_new_n

Creates a new graph allocator instance with a specified count.

```c
ggml_gallocr_t ggml_gallocr_new_n(ggml_backend_buffer_type_t * bufts, int n_bufs)
```

Source: [ggml-alloc.h#L49](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-alloc.h#L49)

### ggml_gallocr_free

Frees a graph allocator instance.

```c
void ggml_gallocr_free(ggml_gallocr_t galloc)
```

Source: [ggml-alloc.h#L50](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-alloc.h#L50)

### ggml_gallocr_reserve

Reserves resources for a graph allocator instance.

```c
int ggml_gallocr_reserve(ggml_gallocr_t galloc, struct ggml_cgraph * graph)
```

Source: [ggml-alloc.h#L56](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-alloc.h#L56)

### ggml_gallocr_reserve_n

Reserves multiple slots for a graph allocator instance.

```c
int ggml_gallocr_reserve_n(ggml_gallocr_t galloc, struct ggml_cgraph * graph, const int * node_buffer_ids, const int * leaf_buffer_ids)
```

Source: [ggml-alloc.h#L57](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-alloc.h#L57)

### ggml_gallocr_alloc_graph

Allocates a computation graph for a graph allocator instance.

```c
int ggml_gallocr_alloc_graph(ggml_gallocr_t galloc, struct ggml_cgraph * graph)
```

Source: [ggml-alloc.h#L65](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-alloc.h#L65)

### ggml_gallocr_get_buffer_size

Returns the required buffer size for a graph allocator instance.

```c
int ggml_gallocr_get_buffer_size(ggml_gallocr_t galloc, int buffer_id)
```

Source: [ggml-alloc.h#L67](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-alloc.h#L67)

### ggml_graph_plan

Plans the execution order of a computation graph.

```c
struct ggml_cplan ggml_graph_plan(const struct ggml_cgraph * cgraph, int n_threads, struct ggml_threadpool * threadpool)
```

Source: [ggml-cpu.h#L63](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-cpu.h#L63)

### ggml_graph_compute

Executes the computation graph.

```c
enum ggml_status ggml_graph_compute(struct ggml_cgraph * cgraph, struct ggml_cplan * cplan)
```

Source: [ggml-cpu.h#L67](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-cpu.h#L67)

### ggml_graph_compute_with_ctx

Computes a graph with an associated execution context.

```c
enum ggml_status ggml_graph_compute_with_ctx(struct ggml_context * ctx, struct ggml_cgraph * cgraph, int n_threads)
```

Source: [ggml-cpu.h#L71](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-cpu.h#L71)

### ggml_new_graph

Creates a new computation graph.

```c
struct ggml_cgraph * ggml_new_graph(struct ggml_context * ctx)
```

Source: [ggml.h#L2063](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L2063)

### ggml_new_graph_custom

Creates a custom computation graph.

```c
struct ggml_cgraph * ggml_new_graph_custom(struct ggml_context * ctx, int size, bool grads)
```

Source: [ggml.h#L2064](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L2064)

### ggml_graph_dup

Duplicates an existing computation graph.

```c
struct ggml_cgraph * ggml_graph_dup(struct ggml_context * ctx, struct ggml_cgraph * cgraph)
```

Source: [ggml.h#L2065](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L2065)

### ggml_graph_cpy

Copies a computation graph.

```c
void ggml_graph_cpy(struct ggml_cgraph * src, struct ggml_cgraph * dst)
```

Source: [ggml.h#L2066](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L2066)

### ggml_graph_reset

Resets the state of a computation graph.

```c
void ggml_graph_reset(struct ggml_cgraph * cgraph)
```

Source: [ggml.h#L2067](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L2067)

### ggml_graph_clear

Clears all nodes from a computation graph.

```c
void ggml_graph_clear(struct ggml_cgraph * cgraph)
```

Source: [ggml.h#L2068](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L2068)

### ggml_graph_size

Returns the size of a computation graph.

```c
int ggml_graph_size(struct ggml_cgraph * cgraph)
```

Source: [ggml.h#L2070](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L2070)

### ggml_graph_node

Retrieves a specific node from a computation graph.

```c
struct ggml_tensor * ggml_graph_node(struct ggml_cgraph * cgraph, int i)
```

Source: [ggml.h#L2071](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L2071)

### ggml_graph_nodes

Returns all nodes within a computation graph.

```c
struct ggml_tensor ** ggml_graph_nodes(struct ggml_cgraph * cgraph)
```

Source: [ggml.h#L2072](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L2072)

### ggml_graph_n_nodes

Returns the total number of nodes in a computation graph.

```c
int ggml_graph_n_nodes(struct ggml_cgraph * cgraph)
```

Source: [ggml.h#L2073](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L2073)

### ggml_graph_add_node

Adds a node to a computation graph.

```c
void ggml_graph_add_node(struct ggml_cgraph * cgraph, struct ggml_tensor * tensor)
```

Source: [ggml.h#L2075](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L2075)

### ggml_graph_overhead

Returns the memory overhead of a computation graph.

```c
int ggml_graph_overhead()
```

Source: [ggml.h#L2077](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L2077)

### ggml_graph_overhead_custom

Returns custom overhead metrics for a computation graph.

```c
int ggml_graph_overhead_custom(int size, bool grads)
```

Source: [ggml.h#L2078](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L2078)

### ggml_graph_get_tensor

Retrieves a tensor from a graph node.

```c
struct ggml_tensor * ggml_graph_get_tensor(const struct ggml_cgraph * cgraph, const char * name)
```

Source: [ggml.h#L2080](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L2080)

### ggml_graph_get_grad

Retrieves the gradient tensor from a graph node.

```c
struct ggml_tensor * ggml_graph_get_grad(const struct ggml_cgraph * cgraph, const struct ggml_tensor * node)
```

Source: [ggml.h#L2081](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L2081)

### ggml_graph_get_grad_acc

Retrieves the accumulated gradients from a graph node.

```c
struct ggml_tensor * ggml_graph_get_grad_acc(const struct ggml_cgraph * cgraph, const struct ggml_tensor * node)
```

Source: [ggml.h#L2082](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L2082)

### ggml_graph_export

Exports a computation graph to a file or format.

```c
void ggml_graph_export(const struct ggml_cgraph * cgraph, const char * fname)
```

Source: [ggml.h#L2084](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L2084)

### ggml_graph_import

Imports a computation graph from a file or format.

```c
struct ggml_cgraph * ggml_graph_import(const char * fname, struct ggml_context ** ctx_data, struct ggml_context ** ctx_eval)
```

Source: [ggml.h#L2085](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L2085)

### ggml_graph_print

Prints a human-readable representation of a computation graph.

```c
void ggml_graph_print(const struct ggml_cgraph * cgraph)
```

Source: [ggml.h#L2088](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L2088)

### ggml_graph_dump_dot

Dumps the computation graph in DOT format.

```c
void ggml_graph_dump_dot(const struct ggml_cgraph * gb, const struct ggml_cgraph * gf, const char * filename)
```

Source: [ggml.h#L2091](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L2091)

### ggml_backend_tensor_copy

Copies a tensor using backend operations.

```c
void ggml_backend_tensor_copy(struct ggml_tensor * src, struct ggml_tensor * dst)
```

Source: [ggml-backend.h#L71](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-backend.h#L71)

### ggml_backend_guid

Returns the unique identifier for the backend.

```c
ggml_guid_t ggml_backend_guid(ggml_backend_t backend)
```

Source: [ggml-backend.h#L77](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-backend.h#L77)

### ggml_backend_name

Retrieves the name of the current backend.

```c
const char * ggml_backend_name(ggml_backend_t backend)
```

Source: [ggml-backend.h#L78](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-backend.h#L78)

### ggml_backend_free

Frees resources associated with the backend.

```c
void ggml_backend_free(ggml_backend_t backend)
```

Source: [ggml-backend.h#L79](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-backend.h#L79)

### ggml_backend_get_default_buffer_type

Returns the default buffer type for the backend.

```c
ggml_backend_buffer_type_t ggml_backend_get_default_buffer_type(ggml_backend_t backend)
```

Source: [ggml-backend.h#L81](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-backend.h#L81)

### ggml_backend_alloc_buffer

Allocates a buffer within the backend.

```c
ggml_backend_buffer_t ggml_backend_alloc_buffer(ggml_backend_t backend, int size)
```

Source: [ggml-backend.h#L82](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-backend.h#L82)

### ggml_backend_get_alignment

Retrieves the alignment requirement for the backend.

```c
int ggml_backend_get_alignment(ggml_backend_t backend)
```

Source: [ggml-backend.h#L83](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-backend.h#L83)

### ggml_backend_get_max_size

Returns the maximum buffer size supported by the backend.

```c
int ggml_backend_get_max_size(ggml_backend_t backend)
```

Source: [ggml-backend.h#L84](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-backend.h#L84)

### ggml_backend_tensor_set_async

Enables asynchronous execution for a backend tensor.

```c
void ggml_backend_tensor_set_async(ggml_backend_t backend, struct ggml_tensor * tensor, const void * data, int offset, int size)
```

Source: [ggml-backend.h#L86](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-backend.h#L86)

### ggml_backend_tensor_get_async

Retrieves the asynchronous status of a backend tensor.

```c
void ggml_backend_tensor_get_async(ggml_backend_t backend, const struct ggml_tensor * tensor, void * data, int offset, int size)
```

Source: [ggml-backend.h#L87](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-backend.h#L87)

### ggml_backend_tensor_set

Sets properties for a backend tensor.

```c
void ggml_backend_tensor_set(struct ggml_tensor * tensor, const void * data, int offset, int size)
```

Source: [ggml-backend.h#L90](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-backend.h#L90)

### ggml_backend_tensor_get

Gets properties of a backend tensor.

```c
void ggml_backend_tensor_get(const struct ggml_tensor * tensor, void * data, int offset, int size)
```

Source: [ggml-backend.h#L91](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-backend.h#L91)

### ggml_backend_tensor_memset

Fills a backend tensor with a constant value.

```c
void ggml_backend_tensor_memset(struct ggml_tensor * tensor, uint8_t value, int offset, int size)
```

Source: [ggml-backend.h#L92](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-backend.h#L92)

### ggml_backend_synchronize

Synchronizes operations on the backend.

```c
void ggml_backend_synchronize(ggml_backend_t backend)
```

Source: [ggml-backend.h#L94](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-backend.h#L94)

### ggml_backend_graph_plan_create

Creates a computation graph plan for the backend.

```c
ggml_backend_graph_plan_t ggml_backend_graph_plan_create(ggml_backend_t backend, struct ggml_cgraph * cgraph)
```

Source: [ggml-backend.h#L96](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-backend.h#L96)

### ggml_backend_graph_plan_free

Frees a previously created backend graph plan.

```c
void ggml_backend_graph_plan_free(ggml_backend_t backend, ggml_backend_graph_plan_t plan)
```

Source: [ggml-backend.h#L97](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-backend.h#L97)

### ggml_backend_graph_plan_compute

Executes a computation graph plan on the backend.

```c
enum ggml_status ggml_backend_graph_plan_compute(ggml_backend_t backend, ggml_backend_graph_plan_t plan)
```

Source: [ggml-backend.h#L99](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-backend.h#L99)

### ggml_backend_graph_compute

Computes a graph on the backend synchronously.

```c
enum ggml_status ggml_backend_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph)
```

Source: [ggml-backend.h#L100](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-backend.h#L100)

### ggml_backend_graph_compute_async

Initiates asynchronous graph computation on the backend.

```c
enum ggml_status ggml_backend_graph_compute_async(ggml_backend_t backend, struct ggml_cgraph * cgraph)
```

Source: [ggml-backend.h#L101](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-backend.h#L101)

### ggml_backend_supports_op

Checks if the backend supports a specific operation.

```c
int ggml_backend_supports_op(ggml_backend_t backend, const struct ggml_tensor * op)
```

Source: [ggml-backend.h#L104](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-backend.h#L104)

### ggml_backend_supports_buft

Verifies if the backend supports a specific buffer type.

```c
int ggml_backend_supports_buft(ggml_backend_t backend, ggml_backend_buffer_type_t buft)
```

Source: [ggml-backend.h#L105](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-backend.h#L105)

### ggml_backend_offload_op

Offloads an operation to a backend device.

```c
int ggml_backend_offload_op(ggml_backend_t backend, const struct ggml_tensor * op)
```

Source: [ggml-backend.h#L106](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-backend.h#L106)

### ggml_backend_tensor_copy_async

Asynchronously copies a tensor using the backend.

```c
void ggml_backend_tensor_copy_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, struct ggml_tensor * src, struct ggml_tensor * dst)
```

Source: [ggml-backend.h#L112](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-backend.h#L112)

### ggml_backend_get_device

Retrieves the current device used by the backend.

```c
ggml_backend_dev_t ggml_backend_get_device(ggml_backend_t backend)
```

Source: [ggml-backend.h#L114](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-backend.h#L114)

### ggml_backend_event_new

Creates a new event for backend synchronization.

```c
ggml_backend_event_t ggml_backend_event_new(ggml_backend_dev_t device)
```

Source: [ggml-backend.h#L120](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-backend.h#L120)

### ggml_backend_event_free

Frees a backend event.

```c
void ggml_backend_event_free(ggml_backend_event_t event)
```

Source: [ggml-backend.h#L121](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-backend.h#L121)

### ggml_backend_event_record

Records a timestamp in a backend event.

```c
void ggml_backend_event_record(ggml_backend_event_t event, ggml_backend_t backend)
```

Source: [ggml-backend.h#L122](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-backend.h#L122)

### ggml_backend_event_synchronize

Synchronizes a backend event.

```c
void ggml_backend_event_synchronize(ggml_backend_event_t event)
```

Source: [ggml-backend.h#L123](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-backend.h#L123)

### ggml_backend_event_wait

Waits for a backend event to complete.

```c
void ggml_backend_event_wait(ggml_backend_t backend, ggml_backend_event_t event)
```

Source: [ggml-backend.h#L124](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-backend.h#L124)

### ggml_backend_dev_name

Returns the name of a backend device.

```c
const char * ggml_backend_dev_name(ggml_backend_dev_t device)
```

Source: [ggml-backend.h#L161](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-backend.h#L161)

### ggml_backend_dev_description

Retrieves the description of a backend device.

```c
const char * ggml_backend_dev_description(ggml_backend_dev_t device)
```

Source: [ggml-backend.h#L162](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-backend.h#L162)

### ggml_backend_dev_memory

Returns the memory capacity of a backend device.

```c
void ggml_backend_dev_memory(ggml_backend_dev_t device, int * free, int * total)
```

Source: [ggml-backend.h#L163](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-backend.h#L163)

### ggml_backend_dev_type

Retrieves the type of a backend device.

```c
enum ggml_backend_dev_type ggml_backend_dev_type(ggml_backend_dev_t device)
```

Source: [ggml-backend.h#L164](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-backend.h#L164)

### ggml_backend_dev_get_props

Gets the properties of a backend device.

```c
void ggml_backend_dev_get_props(ggml_backend_dev_t device, struct ggml_backend_dev_props * props)
```

Source: [ggml-backend.h#L165](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-backend.h#L165)

### ggml_backend_dev_backend_reg

Retrieves the registry entry for a backend device.

```c
ggml_backend_reg_t ggml_backend_dev_backend_reg(ggml_backend_dev_t device)
```

Source: [ggml-backend.h#L166](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-backend.h#L166)

### ggml_backend_dev_init

Initializes a backend device.

```c
ggml_backend_t ggml_backend_dev_init(ggml_backend_dev_t device, const char * params)
```

Source: [ggml-backend.h#L167](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-backend.h#L167)

### ggml_backend_dev_buffer_type

Returns the buffer type used by a backend device.

```c
ggml_backend_buffer_type_t ggml_backend_dev_buffer_type(ggml_backend_dev_t device)
```

Source: [ggml-backend.h#L168](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-backend.h#L168)

### ggml_backend_dev_host_buffer_type

Returns the host buffer type for a backend device.

```c
ggml_backend_buffer_type_t ggml_backend_dev_host_buffer_type(ggml_backend_dev_t device)
```

Source: [ggml-backend.h#L169](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-backend.h#L169)

### ggml_backend_dev_buffer_from_host_ptr

Creates a device buffer from a host pointer.

```c
ggml_backend_buffer_t ggml_backend_dev_buffer_from_host_ptr(ggml_backend_dev_t device, void * ptr, int size, int max_tensor_size)
```

Source: [ggml-backend.h#L170](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-backend.h#L170)

### ggml_backend_dev_supports_op

Checks if a backend device supports a specific operation.

```c
int ggml_backend_dev_supports_op(ggml_backend_dev_t device, const struct ggml_tensor * op)
```

Source: [ggml-backend.h#L172](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-backend.h#L172)

### ggml_backend_dev_supports_buft

Verifies if a backend device supports a buffer type.

```c
int ggml_backend_dev_supports_buft(ggml_backend_dev_t device, ggml_backend_buffer_type_t buft)
```

Source: [ggml-backend.h#L173](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-backend.h#L173)

### ggml_backend_dev_offload_op

Offloads an operation specifically to a backend device.

```c
int ggml_backend_dev_offload_op(ggml_backend_dev_t device, const struct ggml_tensor * op)
```

Source: [ggml-backend.h#L174](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-backend.h#L174)

### ggml_backend_reg_name

Retrieves the name of a backend registry.

```c
const char * ggml_backend_reg_name(ggml_backend_reg_t reg)
```

Source: [ggml-backend.h#L180](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-backend.h#L180)

### ggml_backend_reg_dev_count

Returns the number of devices in a backend registry.

```c
int ggml_backend_reg_dev_count(ggml_backend_reg_t reg)
```

Source: [ggml-backend.h#L181](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-backend.h#L181)

### ggml_backend_reg_dev_get

Gets a device from the backend registry by index.

```c
ggml_backend_dev_t ggml_backend_reg_dev_get(ggml_backend_reg_t reg, int index)
```

Source: [ggml-backend.h#L182](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-backend.h#L182)

### ggml_backend_reg_get_proc_address

Retrieves a procedure address from the backend registry.

```c
void * ggml_backend_reg_get_proc_address(ggml_backend_reg_t reg, const char * name)
```

Source: [ggml-backend.h#L183](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-backend.h#L183)

### ggml_backend_device_register

Registers a new device with the backend.

```c
void ggml_backend_device_register(ggml_backend_dev_t device)
```

Source: [ggml-backend.h#L206](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-backend.h#L206)

### ggml_backend_reg_count

Returns the total count of backend registries.

```c
int ggml_backend_reg_count()
```

Source: [ggml-backend.h#L209](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-backend.h#L209)

### ggml_backend_reg_get

Retrieves an entry from the backend registry.

```c
ggml_backend_reg_t ggml_backend_reg_get(int index)
```

Source: [ggml-backend.h#L210](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-backend.h#L210)

### ggml_backend_reg_by_name

Finds a backend registry entry by name.

```c
ggml_backend_reg_t ggml_backend_reg_by_name(const char * name)
```

Source: [ggml-backend.h#L211](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-backend.h#L211)

### ggml_backend_dev_count

Returns the number of available backend devices.

```c
int ggml_backend_dev_count()
```

Source: [ggml-backend.h#L214](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-backend.h#L214)

### ggml_backend_dev_get

Retrieves a backend device by index.

```c
ggml_backend_dev_t ggml_backend_dev_get(int index)
```

Source: [ggml-backend.h#L215](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-backend.h#L215)

### ggml_backend_dev_by_name

Finds a backend device by its name.

```c
ggml_backend_dev_t ggml_backend_dev_by_name(const char * name)
```

Source: [ggml-backend.h#L216](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-backend.h#L216)

### ggml_backend_dev_by_type

Finds a backend device by its type.

```c
ggml_backend_dev_t ggml_backend_dev_by_type(enum ggml_backend_dev_type type)
```

Source: [ggml-backend.h#L217](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-backend.h#L217)

### ggml_backend_init_by_name

Initializes the backend using its name.

```c
ggml_backend_t ggml_backend_init_by_name(const char * name, const char * params)
```

Source: [ggml-backend.h#L221](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-backend.h#L221)

### ggml_backend_init_by_type

Initializes the backend based on device type.

```c
ggml_backend_t ggml_backend_init_by_type(enum ggml_backend_dev_type type, const char * params)
```

Source: [ggml-backend.h#L223](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-backend.h#L223)

### ggml_backend_init_best

Initializes the best available backend automatically.

```c
ggml_backend_t ggml_backend_init_best()
```

Source: [ggml-backend.h#L225](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-backend.h#L225)

### ggml_backend_load

Loads a backend module.

```c
ggml_backend_reg_t ggml_backend_load(const char * path)
```

Source: [ggml-backend.h#L228](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-backend.h#L228)

### ggml_backend_unload

Unloads a backend module.

```c
void ggml_backend_unload(ggml_backend_reg_t reg)
```

Source: [ggml-backend.h#L230](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-backend.h#L230)

### ggml_backend_load_all

Loads all available backend modules.

```c
void ggml_backend_load_all()
```

Source: [ggml-backend.h#L232](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-backend.h#L232)

### ggml_backend_load_all_from_path

Loads all backend modules from a specified path.

```c
void ggml_backend_load_all_from_path(const char * dir_path)
```

Source: [ggml-backend.h#L233](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-backend.h#L233)

### ggml_backend_sched_new

Creates a new scheduler for backend operations.

```c
ggml_backend_sched_t ggml_backend_sched_new(ggml_backend_t * backends, ggml_backend_buffer_type_t * bufts, int n_backends, int graph_size, bool parallel)
```

Source: [ggml-backend.h#L292](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-backend.h#L292)

### ggml_backend_sched_free

Frees a backend scheduler.

```c
void ggml_backend_sched_free(ggml_backend_sched_t sched)
```

Source: [ggml-backend.h#L293](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-backend.h#L293)

### ggml_backend_sched_reserve

Reserves resources within the backend scheduler.

```c
int ggml_backend_sched_reserve(ggml_backend_sched_t sched, struct ggml_cgraph * measure_graph)
```

Source: [ggml-backend.h#L296](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-backend.h#L296)

### ggml_backend_sched_get_n_backends

Returns the number of backends managed by the scheduler.

```c
int ggml_backend_sched_get_n_backends(ggml_backend_sched_t sched)
```

Source: [ggml-backend.h#L298](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-backend.h#L298)

### ggml_backend_sched_get_backend

Retrieves a specific backend from the scheduler.

```c
ggml_backend_t ggml_backend_sched_get_backend(ggml_backend_sched_t sched, int i)
```

Source: [ggml-backend.h#L299](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-backend.h#L299)

### ggml_backend_sched_get_n_splits

Gets the number of splits configured in the scheduler.

```c
int ggml_backend_sched_get_n_splits(ggml_backend_sched_t sched)
```

Source: [ggml-backend.h#L302](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-backend.h#L302)

### ggml_backend_sched_get_n_copies

Returns the number of tensor copies scheduled.

```c
int ggml_backend_sched_get_n_copies(ggml_backend_sched_t sched)
```

Source: [ggml-backend.h#L303](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-backend.h#L303)

### ggml_backend_sched_get_buffer_size

Retrieves the buffer size allocated by the scheduler.

```c
int ggml_backend_sched_get_buffer_size(ggml_backend_sched_t sched, ggml_backend_t backend)
```

Source: [ggml-backend.h#L305](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-backend.h#L305)

### ggml_backend_sched_set_tensor_backend

Assigns a tensor to a specific backend in the scheduler.

```c
void ggml_backend_sched_set_tensor_backend(ggml_backend_sched_t sched, struct ggml_tensor * node, ggml_backend_t backend)
```

Source: [ggml-backend.h#L307](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-backend.h#L307)

### ggml_backend_sched_get_tensor_backend

Retrieves the backend associated with a tensor in the scheduler.

```c
ggml_backend_t ggml_backend_sched_get_tensor_backend(ggml_backend_sched_t sched, struct ggml_tensor * node)
```

Source: [ggml-backend.h#L308](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-backend.h#L308)

### ggml_backend_sched_alloc_graph

Allocates a computation graph for the scheduler.

```c
int ggml_backend_sched_alloc_graph(ggml_backend_sched_t sched, struct ggml_cgraph * graph)
```

Source: [ggml-backend.h#L311](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-backend.h#L311)

### ggml_backend_sched_graph_compute

Computes a graph using the scheduler synchronously.

```c
enum ggml_status ggml_backend_sched_graph_compute(ggml_backend_sched_t sched, struct ggml_cgraph * graph)
```

Source: [ggml-backend.h#L312](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-backend.h#L312)

### ggml_backend_sched_graph_compute_async

Initiates asynchronous graph computation via the scheduler.

```c
enum ggml_status ggml_backend_sched_graph_compute_async(ggml_backend_sched_t sched, struct ggml_cgraph * graph)
```

Source: [ggml-backend.h#L313](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-backend.h#L313)

### ggml_backend_sched_synchronize

Synchronizes all scheduled backend operations.

```c
void ggml_backend_sched_synchronize(ggml_backend_sched_t sched)
```

Source: [ggml-backend.h#L314](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-backend.h#L314)

### ggml_backend_sched_reset

Resets the state of the backend scheduler.

```c
void ggml_backend_sched_reset(ggml_backend_sched_t sched)
```

Source: [ggml-backend.h#L319](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-backend.h#L319)

### ggml_backend_sched_set_eval_callback

Sets an evaluation callback function in the scheduler.

```c
void ggml_backend_sched_set_eval_callback(ggml_backend_sched_t sched, ggml_backend_sched_eval_callback callback, void * user_data)
```

Source: [ggml-backend.h#L322](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-backend.h#L322)

### ggml_backend_graph_copy

Creates a duplicate of a backend computation graph.

```c
struct ggml_backend_graph_copy ggml_backend_graph_copy(ggml_backend_t backend, struct ggml_cgraph * graph)
```

Source: [ggml-backend.h#L336](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-backend.h#L336)

### ggml_backend_graph_copy_free

Frees a duplicated backend graph.

```c
void ggml_backend_graph_copy_free(struct ggml_backend_graph_copy copy)
```

Source: [ggml-backend.h#L337](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-backend.h#L337)

### ggml_backend_compare_graph_backend

Compares two backend graph implementations.

```c
int ggml_backend_compare_graph_backend(ggml_backend_t backend1, ggml_backend_t backend2, struct ggml_cgraph * graph, ggml_backend_eval_callback callback, void * user_data)
```

Source: [ggml-backend.h#L342](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-backend.h#L342)

### ggml_backend_tensor_alloc

Allocates memory for a new backend tensor.

```c
void ggml_backend_tensor_alloc(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, void * addr)
```

Source: [ggml-backend.h#L345](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-backend.h#L345)

### ggml_backend_view_init

Initializes a tensor view for the backend.

```c
void ggml_backend_view_init(struct ggml_tensor * tensor)
```

Source: [ggml-backend.h#L346](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-backend.h#L346)

### ggml_new_i32

Creates a new 32-bit integer tensor.

```c
struct ggml_tensor * ggml_new_i32(struct ggml_context * ctx, int32_t value)
```

Source: [ggml-cpu.h#L37](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-cpu.h#L37)

### ggml_new_f32

Creates a new 32-bit floating-point tensor.

```c
struct ggml_tensor * ggml_new_f32(struct ggml_context * ctx, float value)
```

Source: [ggml-cpu.h#L38](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-cpu.h#L38)

### ggml_set_i32

Sets the value of a 32-bit integer tensor.

```c
struct ggml_tensor * ggml_set_i32(struct ggml_tensor * tensor, int32_t value)
```

Source: [ggml-cpu.h#L40](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-cpu.h#L40)

### ggml_set_f32

Sets the value of a 32-bit floating-point tensor.

```c
struct ggml_tensor * ggml_set_f32(struct ggml_tensor * tensor, float value)
```

Source: [ggml-cpu.h#L41](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-cpu.h#L41)

### ggml_get_i32_1d

Retrieves one-dimensional data from a 32-bit integer tensor.

```c
int32_t ggml_get_i32_1d(const struct ggml_tensor * tensor, int i)
```

Source: [ggml-cpu.h#L43](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-cpu.h#L43)

### ggml_set_i32_1d

Sets one-dimensional data for a 32-bit integer tensor.

```c
void ggml_set_i32_1d(const struct ggml_tensor * tensor, int i, int32_t value)
```

Source: [ggml-cpu.h#L44](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-cpu.h#L44)

### ggml_get_i32_nd

Retrieves N-dimensional data from a 32-bit integer tensor.

```c
int32_t ggml_get_i32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3)
```

Source: [ggml-cpu.h#L46](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-cpu.h#L46)

### ggml_set_i32_nd

Sets N-dimensional data for a 32-bit integer tensor.

```c
void ggml_set_i32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3, int32_t value)
```

Source: [ggml-cpu.h#L47](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-cpu.h#L47)

### ggml_get_f32_1d

Retrieves one-dimensional data from a 32-bit floating-point tensor.

```c
float ggml_get_f32_1d(const struct ggml_tensor * tensor, int i)
```

Source: [ggml-cpu.h#L49](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-cpu.h#L49)

### ggml_set_f32_1d

Sets one-dimensional data for a 32-bit floating-point tensor.

```c
void ggml_set_f32_1d(const struct ggml_tensor * tensor, int i, float value)
```

Source: [ggml-cpu.h#L50](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-cpu.h#L50)

### ggml_get_f32_nd

Retrieves N-dimensional data from a 32-bit floating-point tensor.

```c
float ggml_get_f32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3)
```

Source: [ggml-cpu.h#L52](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-cpu.h#L52)

### ggml_set_f32_nd

Sets N-dimensional data for a 32-bit floating-point tensor.

```c
void ggml_set_f32_nd(const struct ggml_tensor * tensor, int i0, int i1, int i2, int i3, float value)
```

Source: [ggml-cpu.h#L53](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-cpu.h#L53)

### ggml_add

Adds two tensors element-wise.

```c
struct ggml_tensor * ggml_add(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b)
```

Source: [ggml.h#L782](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L782)

### ggml_add_inplace

Adds one tensor to another in-place.

```c
struct ggml_tensor * ggml_add_inplace(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b)
```

Source: [ggml.h#L787](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L787)

### ggml_add_cast

Performs tensor addition with type casting.

```c
struct ggml_tensor * ggml_add_cast(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, enum ggml_type type)
```

Source: [ggml.h#L792](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L792)

### ggml_add1

Adds a scalar value to a tensor.

```c
struct ggml_tensor * ggml_add1(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b)
```

Source: [ggml.h#L798](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L798)

### ggml_add1_inplace

Adds a scalar value to a tensor in-place.

```c
struct ggml_tensor * ggml_add1_inplace(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b)
```

Source: [ggml.h#L803](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L803)

### ggml_acc

Accumulates values from one tensor into another.

```c
struct ggml_tensor * ggml_acc(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, int nb1, int nb2, int nb3, int offset)
```

Source: [ggml.h#L811](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L811)

### ggml_acc_inplace

Accumulates tensor values in-place.

```c
struct ggml_tensor * ggml_acc_inplace(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, int nb1, int nb2, int nb3, int offset)
```

Source: [ggml.h#L820](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L820)

### ggml_sub

Subtracts one tensor from another element-wise.

```c
struct ggml_tensor * ggml_sub(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b)
```

Source: [ggml.h#L829](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L829)

### ggml_sub_inplace

Subtracts one tensor from another in-place.

```c
struct ggml_tensor * ggml_sub_inplace(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b)
```

Source: [ggml.h#L834](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L834)

### ggml_mul

Multiplies two tensors element-wise.

```c
struct ggml_tensor * ggml_mul(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b)
```

Source: [ggml.h#L839](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L839)

### ggml_mul_inplace

Multiplies two tensors element-wise in-place.

```c
struct ggml_tensor * ggml_mul_inplace(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b)
```

Source: [ggml.h#L844](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L844)

### ggml_div

Divides one tensor by another element-wise.

```c
struct ggml_tensor * ggml_div(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b)
```

Source: [ggml.h#L849](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L849)

### ggml_div_inplace

Divides one tensor by another in-place.

```c
struct ggml_tensor * ggml_div_inplace(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b)
```

Source: [ggml.h#L854](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L854)

### ggml_sqr

Squares each element of a tensor.

```c
struct ggml_tensor * ggml_sqr(struct ggml_context * ctx, struct ggml_tensor * a)
```

Source: [ggml.h#L859](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L859)

### ggml_sqr_inplace

Squares each tensor element in-place.

```c
struct ggml_tensor * ggml_sqr_inplace(struct ggml_context * ctx, struct ggml_tensor * a)
```

Source: [ggml.h#L863](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L863)

### ggml_sqrt

Computes the square root of each tensor element.

```c
struct ggml_tensor * ggml_sqrt(struct ggml_context * ctx, struct ggml_tensor * a)
```

Source: [ggml.h#L867](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L867)

### ggml_sqrt_inplace

Computes square roots of tensor elements in-place.

```c
struct ggml_tensor * ggml_sqrt_inplace(struct ggml_context * ctx, struct ggml_tensor * a)
```

Source: [ggml.h#L871](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L871)

### ggml_log

Computes the natural logarithm of each tensor element.

```c
struct ggml_tensor * ggml_log(struct ggml_context * ctx, struct ggml_tensor * a)
```

Source: [ggml.h#L875](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L875)

### ggml_log_inplace

Computes natural logarithms in-place on a tensor.

```c
struct ggml_tensor * ggml_log_inplace(struct ggml_context * ctx, struct ggml_tensor * a)
```

Source: [ggml.h#L879](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L879)

### ggml_sin

Computes the sine of each tensor element.

```c
struct ggml_tensor * ggml_sin(struct ggml_context * ctx, struct ggml_tensor * a)
```

Source: [ggml.h#L883](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L883)

### ggml_sin_inplace

Computes sine in-place on a tensor.

```c
struct ggml_tensor * ggml_sin_inplace(struct ggml_context * ctx, struct ggml_tensor * a)
```

Source: [ggml.h#L887](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L887)

### ggml_cos

Computes the cosine of each tensor element.

```c
struct ggml_tensor * ggml_cos(struct ggml_context * ctx, struct ggml_tensor * a)
```

Source: [ggml.h#L891](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L891)

### ggml_cos_inplace

Computes cosine in-place on a tensor.

```c
struct ggml_tensor * ggml_cos_inplace(struct ggml_context * ctx, struct ggml_tensor * a)
```

Source: [ggml.h#L895](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L895)

### ggml_exp

Computes the exponential of each tensor element.

```c
struct ggml_tensor * ggml_exp(struct ggml_context * ctx, struct ggml_tensor * a)
```

Source: [ggml.h#L1055](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L1055)

### ggml_exp_inplace

Computes exponentials in-place on a tensor.

```c
struct ggml_tensor * ggml_exp_inplace(struct ggml_context * ctx, struct ggml_tensor * a)
```

Source: [ggml.h#L1059](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L1059)

### ggml_scale

Scales a tensor by a constant factor.

```c
struct ggml_tensor * ggml_scale(struct ggml_context * ctx, struct ggml_tensor * a, float s)
```

Source: [ggml.h#L1139](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L1139)

### ggml_scale_inplace

Scales a tensor in-place by a constant factor.

```c
struct ggml_tensor * ggml_scale_inplace(struct ggml_context * ctx, struct ggml_tensor * a, float s)
```

Source: [ggml.h#L1145](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L1145)

### ggml_tanh

Computes the hyperbolic tangent of each tensor element.

```c
struct ggml_tensor * ggml_tanh(struct ggml_context * ctx, struct ggml_tensor * a)
```

Source: [ggml.h#L978](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L978)

### ggml_tanh_inplace

Computes hyperbolic tangent in-place on a tensor.

```c
struct ggml_tensor * ggml_tanh_inplace(struct ggml_context * ctx, struct ggml_tensor * a)
```

Source: [ggml.h#L982](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L982)

### ggml_elu

Applies the ELU activation function to a tensor.

```c
struct ggml_tensor * ggml_elu(struct ggml_context * ctx, struct ggml_tensor * a)
```

Source: [ggml.h#L986](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L986)

### ggml_elu_inplace

Applies the ELU activation function in-place on a tensor.

```c
struct ggml_tensor * ggml_elu_inplace(struct ggml_context * ctx, struct ggml_tensor * a)
```

Source: [ggml.h#L990](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L990)

### ggml_relu

Applies the ReLU activation function to a tensor.

```c
struct ggml_tensor * ggml_relu(struct ggml_context * ctx, struct ggml_tensor * a)
```

Source: [ggml.h#L994](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L994)

### ggml_leaky_relu

Applies the Leaky ReLU activation function to a tensor.

```c
struct ggml_tensor * ggml_leaky_relu(struct ggml_context * ctx, struct ggml_tensor * a, float negative_slope, bool inplace)
```

Source: [ggml.h#L998](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L998)

### ggml_relu_inplace

Applies the ReLU activation function in-place on a tensor.

```c
struct ggml_tensor * ggml_relu_inplace(struct ggml_context * ctx, struct ggml_tensor * a)
```

Source: [ggml.h#L1002](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L1002)

### ggml_sigmoid

Applies the sigmoid activation function to a tensor.

```c
struct ggml_tensor * ggml_sigmoid(struct ggml_context * ctx, struct ggml_tensor * a)
```

Source: [ggml.h#L1006](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L1006)

### ggml_sigmoid_inplace

Applies the sigmoid activation function in-place on a tensor.

```c
struct ggml_tensor * ggml_sigmoid_inplace(struct ggml_context * ctx, struct ggml_tensor * a)
```

Source: [ggml.h#L1010](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L1010)

### ggml_gelu

Applies the GELU activation function to a tensor.

```c
struct ggml_tensor * ggml_gelu(struct ggml_context * ctx, struct ggml_tensor * a)
```

Source: [ggml.h#L1014](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L1014)

### ggml_gelu_inplace

Applies the GELU activation function in-place on a tensor.

```c
struct ggml_tensor * ggml_gelu_inplace(struct ggml_context * ctx, struct ggml_tensor * a)
```

Source: [ggml.h#L1018](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L1018)

### ggml_gelu_quick

Applies a fast approximation of the GELU activation.

```c
struct ggml_tensor * ggml_gelu_quick(struct ggml_context * ctx, struct ggml_tensor * a)
```

Source: [ggml.h#L1022](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L1022)

### ggml_gelu_quick_inplace

Applies a fast GELU approximation in-place on a tensor.

```c
struct ggml_tensor * ggml_gelu_quick_inplace(struct ggml_context * ctx, struct ggml_tensor * a)
```

Source: [ggml.h#L1026](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L1026)

### ggml_silu

Applies the SiLU (swish) activation function to a tensor.

```c
struct ggml_tensor * ggml_silu(struct ggml_context * ctx, struct ggml_tensor * a)
```

Source: [ggml.h#L1030](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L1030)

### ggml_silu_inplace

Applies the SiLU activation function in-place on a tensor.

```c
struct ggml_tensor * ggml_silu_inplace(struct ggml_context * ctx, struct ggml_tensor * a)
```

Source: [ggml.h#L1034](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L1034)

### ggml_silu_back

Computes the backward pass for the SiLU activation.

```c
struct ggml_tensor * ggml_silu_back(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b)
```

Source: [ggml.h#L1040](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L1040)

### ggml_hardswish

Applies the hard-swish activation function to a tensor.

```c
struct ggml_tensor * ggml_hardswish(struct ggml_context * ctx, struct ggml_tensor * a)
```

Source: [ggml.h#L1046](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L1046)

### ggml_hardsigmoid

Applies the hard-sigmoid activation function to a tensor.

```c
struct ggml_tensor * ggml_hardsigmoid(struct ggml_context * ctx, struct ggml_tensor * a)
```

Source: [ggml.h#L1051](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L1051)

### ggml_norm

Normalizes the elements of a tensor.

```c
struct ggml_tensor * ggml_norm(struct ggml_context * ctx, struct ggml_tensor * a, float eps)
```

Source: [ggml.h#L1064](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L1064)

### ggml_norm_inplace

Normalizes a tensor in-place.

```c
struct ggml_tensor * ggml_norm_inplace(struct ggml_context * ctx, struct ggml_tensor * a, float eps)
```

Source: [ggml.h#L1069](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L1069)

### ggml_rms_norm

Applies RMS normalization to a tensor.

```c
struct ggml_tensor * ggml_rms_norm(struct ggml_context * ctx, struct ggml_tensor * a, float eps)
```

Source: [ggml.h#L1074](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L1074)

### ggml_rms_norm_inplace

Applies RMS normalization in-place on a tensor.

```c
struct ggml_tensor * ggml_rms_norm_inplace(struct ggml_context * ctx, struct ggml_tensor * a, float eps)
```

Source: [ggml.h#L1079](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L1079)

### ggml_group_norm

Applies group normalization to a tensor.

```c
struct ggml_tensor * ggml_group_norm(struct ggml_context * ctx, struct ggml_tensor * a, int n_groups, float eps)
```

Source: [ggml.h#L1086](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L1086)

### ggml_group_norm_inplace

Applies group normalization in-place on a tensor.

```c
struct ggml_tensor * ggml_group_norm_inplace(struct ggml_context * ctx, struct ggml_tensor * a, int n_groups, float eps)
```

Source: [ggml.h#L1092](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L1092)

### ggml_rms_norm_back

Computes the backward pass for RMS normalization.

```c
struct ggml_tensor * ggml_rms_norm_back(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, float eps)
```

Source: [ggml.h#L1100](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L1100)

### ggml_dup_tensor

Duplicates an existing tensor.

```c
struct ggml_tensor * ggml_dup_tensor(struct ggml_context * ctx, const struct ggml_tensor * src)
```

Source: [ggml.h#L742](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L742)

### ggml_view_tensor

Creates a view into an existing tensor.

```c
struct ggml_tensor * ggml_view_tensor(struct ggml_context * ctx, struct ggml_tensor * src)
```

Source: [ggml.h#L743](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L743)

### ggml_get_first_tensor

Retrieves the first tensor in a tensor list.

```c
struct ggml_tensor * ggml_get_first_tensor(const struct ggml_context * ctx)
```

Source: [ggml.h#L746](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L746)

### ggml_get_next_tensor

Retrieves the next tensor in a tensor list.

```c
struct ggml_tensor * ggml_get_next_tensor(const struct ggml_context * ctx, struct ggml_tensor * tensor)
```

Source: [ggml.h#L747](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L747)

### ggml_get_tensor

Retrieves a tensor by its index or identifier.

```c
struct ggml_tensor * ggml_get_tensor(struct ggml_context * ctx, const char * name)
```

Source: [ggml.h#L748](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L748)

### ggml_new_tensor

Creates a new tensor with specified parameters.

```c
struct ggml_tensor * ggml_new_tensor(struct ggml_context * ctx, enum ggml_type type, int n_dims, const int64_t * ne)
```

Source: [ggml.h#L708](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L708)

### ggml_new_tensor_1d

Creates a new one-dimensional tensor.

```c
struct ggml_tensor * ggml_new_tensor_1d(struct ggml_context * ctx, enum ggml_type type, int64_t ne0)
```

Source: [ggml.h#L714](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L714)

### ggml_new_tensor_2d

Creates a new two-dimensional tensor.

```c
struct ggml_tensor * ggml_new_tensor_2d(struct ggml_context * ctx, enum ggml_type type, int64_t ne0, int64_t ne1)
```

Source: [ggml.h#L719](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L719)

### ggml_new_tensor_3d

Creates a new three-dimensional tensor.

```c
struct ggml_tensor * ggml_new_tensor_3d(struct ggml_context * ctx, enum ggml_type type, int64_t ne0, int64_t ne1, int64_t ne2)
```

Source: [ggml.h#L725](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L725)

### ggml_new_tensor_4d

Creates a new four-dimensional tensor.

```c
struct ggml_tensor * ggml_new_tensor_4d(struct ggml_context * ctx, enum ggml_type type, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3)
```

Source: [ggml.h#L732](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L732)

### ggml_new_buffer

Creates a new memory buffer for tensor operations.

```c
void * ggml_new_buffer(struct ggml_context * ctx, int nbytes)
```

Source: [ggml.h#L740](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L740)

### ggml_dup

Duplicates a tensor (shallow copy).

```c
struct ggml_tensor * ggml_dup(struct ggml_context * ctx, struct ggml_tensor * a)
```

Source: [ggml.h#L773](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L773)

### ggml_dup_inplace

Duplicates a tensor in-place.

```c
struct ggml_tensor * ggml_dup_inplace(struct ggml_context * ctx, struct ggml_tensor * a)
```

Source: [ggml.h#L778](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L778)

### ggml_set

Sets tensor elements to specified values.

```c
struct ggml_tensor * ggml_set(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, int nb1, int nb2, int nb3, int offset)
```

Source: [ggml.h#L1151](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L1151)

### ggml_set_inplace

Sets tensor elements in-place to specified values.

```c
struct ggml_tensor * ggml_set_inplace(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, int nb1, int nb2, int nb3, int offset)
```

Source: [ggml.h#L1161](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L1161)

### ggml_set_1d

Sets elements of a one-dimensional tensor.

```c
struct ggml_tensor * ggml_set_1d(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, int offset)
```

Source: [ggml.h#L1170](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L1170)

### ggml_set_1d_inplace

Sets one-dimensional tensor elements in-place.

```c
struct ggml_tensor * ggml_set_1d_inplace(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, int offset)
```

Source: [ggml.h#L1176](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L1176)

### ggml_set_2d

Sets elements of a two-dimensional tensor.

```c
struct ggml_tensor * ggml_set_2d(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, int nb1, int offset)
```

Source: [ggml.h#L1183](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L1183)

### ggml_set_2d_inplace

Sets two-dimensional tensor elements in-place.

```c
struct ggml_tensor * ggml_set_2d_inplace(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, int nb1, int offset)
```

Source: [ggml.h#L1191](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L1191)

### ggml_cpy

Copies data from one tensor to another.

```c
struct ggml_tensor * ggml_cpy(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b)
```

Source: [ggml.h#L1199](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L1199)

### ggml_cast

Casts a tensor to a different data type.

```c
struct ggml_tensor * ggml_cast(struct ggml_context * ctx, struct ggml_tensor * a, enum ggml_type type)
```

Source: [ggml.h#L1204](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L1204)

### ggml_cont

Ensures a tensor is stored contiguously in memory.

```c
struct ggml_tensor * ggml_cont(struct ggml_context * ctx, struct ggml_tensor * a)
```

Source: [ggml.h#L1210](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L1210)

### ggml_cont_1d

Ensures a one-dimensional tensor is contiguous in memory.

```c
struct ggml_tensor * ggml_cont_1d(struct ggml_context * ctx, struct ggml_tensor * a, int64_t ne0)
```

Source: [ggml.h#L1215](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L1215)

### ggml_cont_2d

Ensures a two-dimensional tensor is contiguous in memory.

```c
struct ggml_tensor * ggml_cont_2d(struct ggml_context * ctx, struct ggml_tensor * a, int64_t ne0, int64_t ne1)
```

Source: [ggml.h#L1220](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L1220)

### ggml_cont_3d

Ensures a three-dimensional tensor is contiguous in memory.

```c
struct ggml_tensor * ggml_cont_3d(struct ggml_context * ctx, struct ggml_tensor * a, int64_t ne0, int64_t ne1, int64_t ne2)
```

Source: [ggml.h#L1226](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L1226)

### ggml_cont_4d

Ensures a four-dimensional tensor is contiguous in memory.

```c
struct ggml_tensor * ggml_cont_4d(struct ggml_context * ctx, struct ggml_tensor * a, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3)
```

Source: [ggml.h#L1233](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L1233)

### ggml_reshape

Reshapes a tensor to new dimensions.

```c
struct ggml_tensor * ggml_reshape(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b)
```

Source: [ggml.h#L1243](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L1243)

### ggml_reshape_1d

Reshapes a tensor into one dimension.

```c
struct ggml_tensor * ggml_reshape_1d(struct ggml_context * ctx, struct ggml_tensor * a, int64_t ne0)
```

Source: [ggml.h#L1250](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L1250)

### ggml_reshape_2d

Reshapes a tensor into two dimensions.

```c
struct ggml_tensor * ggml_reshape_2d(struct ggml_context * ctx, struct ggml_tensor * a, int64_t ne0, int64_t ne1)
```

Source: [ggml.h#L1255](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L1255)

### ggml_reshape_3d

Reshapes a tensor into three dimensions.

```c
struct ggml_tensor * ggml_reshape_3d(struct ggml_context * ctx, struct ggml_tensor * a, int64_t ne0, int64_t ne1, int64_t ne2)
```

Source: [ggml.h#L1263](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L1263)

### ggml_reshape_4d

Reshapes a tensor into four dimensions.

```c
struct ggml_tensor * ggml_reshape_4d(struct ggml_context * ctx, struct ggml_tensor * a, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3)
```

Source: [ggml.h#L1270](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L1270)

### ggml_view_1d

Creates a one-dimensional view of a tensor.

```c
struct ggml_tensor * ggml_view_1d(struct ggml_context * ctx, struct ggml_tensor * a, int64_t ne0, int offset)
```

Source: [ggml.h#L1279](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L1279)

### ggml_view_2d

Creates a two-dimensional view of a tensor.

```c
struct ggml_tensor * ggml_view_2d(struct ggml_context * ctx, struct ggml_tensor * a, int64_t ne0, int64_t ne1, int nb1, int offset)
```

Source: [ggml.h#L1285](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L1285)

### ggml_view_3d

Creates a three-dimensional view of a tensor.

```c
struct ggml_tensor * ggml_view_3d(struct ggml_context * ctx, struct ggml_tensor * a, int64_t ne0, int64_t ne1, int64_t ne2, int nb1, int nb2, int offset)
```

Source: [ggml.h#L1293](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L1293)

### ggml_view_4d

Creates a four-dimensional view of a tensor.

```c
struct ggml_tensor * ggml_view_4d(struct ggml_context * ctx, struct ggml_tensor * a, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, int nb1, int nb2, int nb3, int offset)
```

Source: [ggml.h#L1303](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L1303)

### ggml_permute

Permutes the dimensions of a tensor.

```c
struct ggml_tensor * ggml_permute(struct ggml_context * ctx, struct ggml_tensor * a, int axis0, int axis1, int axis2, int axis3)
```

Source: [ggml.h#L1315](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L1315)

### ggml_transpose

Transposes a tensor.

```c
struct ggml_tensor * ggml_transpose(struct ggml_context * ctx, struct ggml_tensor * a)
```

Source: [ggml.h#L1324](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L1324)

### ggml_sum

Sums all elements of a tensor.

```c
struct ggml_tensor * ggml_sum(struct ggml_context * ctx, struct ggml_tensor * a)
```

Source: [ggml.h#L900](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L900)

### ggml_sum_rows

Sums the elements across each row of a tensor.

```c
struct ggml_tensor * ggml_sum_rows(struct ggml_context * ctx, struct ggml_tensor * a)
```

Source: [ggml.h#L905](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L905)

### ggml_mean

Computes the mean value of a tensor.

```c
struct ggml_tensor * ggml_mean(struct ggml_context * ctx, struct ggml_tensor * a)
```

Source: [ggml.h#L910](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L910)

### ggml_argmax

Finds the index of the maximum element in a tensor.

```c
struct ggml_tensor * ggml_argmax(struct ggml_context * ctx, struct ggml_tensor * a)
```

Source: [ggml.h#L915](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L915)

### ggml_count_equal

Counts elements equal to a specified value in a tensor.

```c
struct ggml_tensor * ggml_count_equal(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b)
```

Source: [ggml.h#L920](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L920)

### ggml_repeat

Repeats a tensor along specified dimensions.

```c
struct ggml_tensor * ggml_repeat(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b)
```

Source: [ggml.h#L927](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L927)

### ggml_repeat_back

Repeats a tensor in reverse along specified dimensions.

```c
struct ggml_tensor * ggml_repeat_back(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b)
```

Source: [ggml.h#L933](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L933)

### ggml_concat

Concatenates multiple tensors along a given dimension.

```c
struct ggml_tensor * ggml_concat(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, int dim)
```

Source: [ggml.h#L940](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L940)

### ggml_get_rows

Retrieves specific rows from a tensor.

```c
struct ggml_tensor * ggml_get_rows(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b)
```

Source: [ggml.h#L1329](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L1329)

### ggml_get_rows_back

Retrieves rows from a tensor in reverse order.

```c
struct ggml_tensor * ggml_get_rows_back(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, struct ggml_tensor * c)
```

Source: [ggml.h#L1334](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L1334)

### ggml_diag

Extracts the diagonal elements of a tensor.

```c
struct ggml_tensor * ggml_diag(struct ggml_context * ctx, struct ggml_tensor * a)
```

Source: [ggml.h#L1340](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L1340)

### ggml_diag_mask_inf

Masks the diagonal of a tensor with negative infinity.

```c
struct ggml_tensor * ggml_diag_mask_inf(struct ggml_context * ctx, struct ggml_tensor * a, int n_past)
```

Source: [ggml.h#L1345](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L1345)

### ggml_diag_mask_inf_inplace

Masks the diagonal with negative infinity in-place.

```c
struct ggml_tensor * ggml_diag_mask_inf_inplace(struct ggml_context * ctx, struct ggml_tensor * a, int n_past)
```

Source: [ggml.h#L1351](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L1351)

### ggml_diag_mask_zero

Masks the diagonal of a tensor with zero.

```c
struct ggml_tensor * ggml_diag_mask_zero(struct ggml_context * ctx, struct ggml_tensor * a, int n_past)
```

Source: [ggml.h#L1357](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L1357)

### ggml_diag_mask_zero_inplace

Masks the diagonal with zero in-place.

```c
struct ggml_tensor * ggml_diag_mask_zero_inplace(struct ggml_context * ctx, struct ggml_tensor * a, int n_past)
```

Source: [ggml.h#L1363](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L1363)

### ggml_soft_max

Applies the softmax function to a tensor.

```c
struct ggml_tensor * ggml_soft_max(struct ggml_context * ctx, struct ggml_tensor * a)
```

Source: [ggml.h#L1368](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L1368)

### ggml_soft_max_inplace

Applies the softmax function in-place on a tensor.

```c
struct ggml_tensor * ggml_soft_max_inplace(struct ggml_context * ctx, struct ggml_tensor * a)
```

Source: [ggml.h#L1373](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L1373)

### ggml_soft_max_ext

Applies an extended softmax function to a tensor.

```c
struct ggml_tensor * ggml_soft_max_ext(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * mask, float scale, float max_bias)
```

Source: [ggml.h#L1380](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L1380)

### ggml_soft_max_ext_back

Computes the backward pass for the extended softmax.

```c
struct ggml_tensor * ggml_soft_max_ext_back(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, float scale, float max_bias)
```

Source: [ggml.h#L1387](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L1387)

### ggml_soft_max_ext_back_inplace

Computes the backward pass for extended softmax in-place.

```c
struct ggml_tensor * ggml_soft_max_ext_back_inplace(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, float scale, float max_bias)
```

Source: [ggml.h#L1395](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L1395)

### ggml_argsort

Returns indices that sort the tensor.

```c
struct ggml_tensor * ggml_argsort(struct ggml_context * ctx, struct ggml_tensor * a, enum ggml_sort_order order)
```

Source: [ggml.h#L1761](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L1761)

### ggml_arange

Generates a tensor with sequential values.

```c
struct ggml_tensor * ggml_arange(struct ggml_context * ctx, float start, float stop, float step)
```

Source: [ggml.h#L1766](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L1766)

### ggml_top_k

Selects the top k elements from a tensor.

```c
struct ggml_tensor * ggml_top_k(struct ggml_context * ctx, struct ggml_tensor * a, int k)
```

Source: [ggml.h#L1773](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L1773)

### ggml_im2col

Transforms image data into a columnar matrix format.

```c
struct ggml_tensor * ggml_im2col(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, int s0, int s1, int p0, int p1, int d0, int d1, bool is_2D, enum ggml_type dst_type)
```

Source: [ggml.h#L1549](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L1549)

### ggml_im2col_back

Reconstructs image data from its columnar representation.

```c
struct ggml_tensor * ggml_im2col_back(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, int64_t * ne, int s0, int s1, int p0, int p1, int d0, int d1, bool is_2D)
```

Source: [ggml.h#L1562](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L1562)

### ggml_conv_1d

Performs a one-dimensional convolution operation.

```c
struct ggml_tensor * ggml_conv_1d(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, int s0, int p0, int d0)
```

Source: [ggml.h#L1575](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L1575)

### ggml_conv_1d_ph

Performs a phase-based one-dimensional convolution.

```c
struct ggml_tensor * ggml_conv_1d_ph(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, int s, int d)
```

Source: [ggml.h#L1585](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L1585)

### ggml_conv_1d_dw

Performs a depthwise one-dimensional convolution.

```c
struct ggml_tensor * ggml_conv_1d_dw(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, int s0, int p0, int d0)
```

Source: [ggml.h#L1594](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L1594)

### ggml_conv_1d_dw_ph

Performs a phase-based depthwise one-dimensional convolution.

```c
struct ggml_tensor * ggml_conv_1d_dw_ph(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, int s0, int d0)
```

Source: [ggml.h#L1602](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L1602)

### ggml_conv_transpose_1d

Performs a transposed one-dimensional convolution.

```c
struct ggml_tensor * ggml_conv_transpose_1d(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, int s0, int p0, int d0)
```

Source: [ggml.h#L1609](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L1609)

### ggml_conv_2d

Performs a two-dimensional convolution operation.

```c
struct ggml_tensor * ggml_conv_2d(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, int s0, int s1, int p0, int p1, int d0, int d1)
```

Source: [ggml.h#L1617](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L1617)

### ggml_conv_2d_sk_p0

Performs a two-dimensional convolution with stride kernel padding of 0.

```c
struct ggml_tensor * ggml_conv_2d_sk_p0(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b)
```

Source: [ggml.h#L1636](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L1636)

### ggml_conv_2d_s1_ph

Performs a phase-based two-dimensional convolution with stride 1.

```c
struct ggml_tensor * ggml_conv_2d_s1_ph(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b)
```

Source: [ggml.h#L1649](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L1649)

### ggml_conv_2d_dw

Performs a depthwise two-dimensional convolution.

```c
struct ggml_tensor * ggml_conv_2d_dw(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, int s0, int s1, int p0, int p1, int d0, int d1)
```

Source: [ggml.h#L1655](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L1655)

### ggml_conv_transpose_2d_p0

Performs a transposed two-dimensional convolution with padding 0.

```c
struct ggml_tensor * ggml_conv_transpose_2d_p0(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, int stride)
```

Source: [ggml.h#L1666](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L1666)

### ggml_pool_1d

Performs a one-dimensional pooling operation.

```c
struct ggml_tensor * ggml_pool_1d(struct ggml_context * ctx, struct ggml_tensor * a, enum ggml_op_pool op, int k0, int s0, int p0)
```

Source: [ggml.h#L1678](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L1678)

### ggml_pool_2d

Performs a two-dimensional pooling operation.

```c
struct ggml_tensor * ggml_pool_2d(struct ggml_context * ctx, struct ggml_tensor * a, enum ggml_op_pool op, int k0, int k1, int s0, int s1, float p0, float p1)
```

Source: [ggml.h#L1688](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L1688)

### ggml_pool_2d_back

Computes the backward pass for a two-dimensional pooling operation.

```c
struct ggml_tensor * ggml_pool_2d_back(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * af, enum ggml_op_pool op, int k0, int k1, int s0, int s1, float p0, float p1)
```

Source: [ggml.h#L1699](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L1699)

### ggml_upscale

Upscales a tensor by a specified factor.

```c
struct ggml_tensor * ggml_upscale(struct ggml_context * ctx, struct ggml_tensor * a, int scale_factor)
```

Source: [ggml.h#L1714](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L1714)

### ggml_upscale_ext

Upscales a tensor with extended options.

```c
struct ggml_tensor * ggml_upscale_ext(struct ggml_context * ctx, struct ggml_tensor * a, int ne0, int ne1, int ne2, int ne3)
```

Source: [ggml.h#L1722](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L1722)

### ggml_pad

Pads a tensor with a specified value.

```c
struct ggml_tensor * ggml_pad(struct ggml_context * ctx, struct ggml_tensor * a, int p0, int p1, int p2, int p3)
```

Source: [ggml.h#L1731](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L1731)

### ggml_pad_reflect_1d

Pads a one-dimensional tensor using reflection.

```c
struct ggml_tensor * ggml_pad_reflect_1d(struct ggml_context * ctx, struct ggml_tensor * a, int p0, int p1)
```

Source: [ggml.h#L1740](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L1740)

### ggml_timestep_embedding

Generates timestep embeddings for a tensor.

```c
struct ggml_tensor * ggml_timestep_embedding(struct ggml_context * ctx, struct ggml_tensor * timesteps, int dim, int max_period)
```

Source: [ggml.h#L1749](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L1749)

### ggml_flash_attn_ext

Performs an extended flash attention operation.

```c
struct ggml_tensor * ggml_flash_attn_ext(struct ggml_context * ctx, struct ggml_tensor * q, struct ggml_tensor * k, struct ggml_tensor * v, struct ggml_tensor * mask, float scale, float max_bias, float logit_softcap)
```

Source: [ggml.h#L1785](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L1785)

### ggml_flash_attn_ext_set_prec

Sets the precision for extended flash attention.

```c
void ggml_flash_attn_ext_set_prec(struct ggml_tensor * a, enum ggml_prec prec)
```

Source: [ggml.h#L1795](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L1795)

### ggml_flash_attn_ext_get_prec

Gets the current precision setting for extended flash attention.

```c
enum ggml_prec ggml_flash_attn_ext_get_prec(const struct ggml_tensor * a)
```

Source: [ggml.h#L1799](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L1799)

### ggml_flash_attn_back

Computes the backward pass for flash attention.

```c
struct ggml_tensor * ggml_flash_attn_back(struct ggml_context * ctx, struct ggml_tensor * q, struct ggml_tensor * k, struct ggml_tensor * v, struct ggml_tensor * d, bool masked)
```

Source: [ggml.h#L1803](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L1803)

### ggml_ssm_conv

Performs a state-space model convolution.

```c
struct ggml_tensor * ggml_ssm_conv(struct ggml_context * ctx, struct ggml_tensor * sx, struct ggml_tensor * c)
```

Source: [ggml.h#L1811](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L1811)

### ggml_ssm_scan

Performs a state-space model scan operation.

```c
struct ggml_tensor * ggml_ssm_scan(struct ggml_context * ctx, struct ggml_tensor * s, struct ggml_tensor * x, struct ggml_tensor * dt, struct ggml_tensor * A, struct ggml_tensor * B, struct ggml_tensor * C)
```

Source: [ggml.h#L1816](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L1816)

### ggml_win_part

Partitions a tensor into windows.

```c
struct ggml_tensor * ggml_win_part(struct ggml_context * ctx, struct ggml_tensor * a, int w)
```

Source: [ggml.h#L1831](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L1831)

### ggml_win_unpart

Reassembles a tensor from its window partitions.

```c
struct ggml_tensor * ggml_win_unpart(struct ggml_context * ctx, struct ggml_tensor * a, int w0, int h0, int w)
```

Source: [ggml.h#L1838](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L1838)

### ggml_unary

Applies a unary function to all tensor elements.

```c
struct ggml_tensor * ggml_unary(struct ggml_context * ctx, struct ggml_tensor * a, enum ggml_unary_op op)
```

Source: [ggml.h#L1845](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L1845)

### ggml_unary_inplace

Applies a unary function to a tensor in-place.

```c
struct ggml_tensor * ggml_unary_inplace(struct ggml_context * ctx, struct ggml_tensor * a, enum ggml_unary_op op)
```

Source: [ggml.h#L1850](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L1850)

### ggml_get_rel_pos

Retrieves relative positional information from a tensor.

```c
struct ggml_tensor * ggml_get_rel_pos(struct ggml_context * ctx, struct ggml_tensor * a, int qh, int kh)
```

Source: [ggml.h#L1856](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L1856)

### ggml_add_rel_pos

Adds relative positional embeddings to a tensor.

```c
struct ggml_tensor * ggml_add_rel_pos(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * pw, struct ggml_tensor * ph)
```

Source: [ggml.h#L1863](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L1863)

### ggml_add_rel_pos_inplace

Adds relative positional embeddings in-place to a tensor.

```c
struct ggml_tensor * ggml_add_rel_pos_inplace(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * pw, struct ggml_tensor * ph)
```

Source: [ggml.h#L1869](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L1869)

### ggml_rwkv_wkv6

Computes the RWKV WKV6 operation on a tensor.

```c
struct ggml_tensor * ggml_rwkv_wkv6(struct ggml_context * ctx, struct ggml_tensor * k, struct ggml_tensor * v, struct ggml_tensor * r, struct ggml_tensor * tf, struct ggml_tensor * td, struct ggml_tensor * state)
```

Source: [ggml.h#L1875](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L1875)

### ggml_gated_linear_attn

Applies gated linear attention to a tensor.

```c
struct ggml_tensor * ggml_gated_linear_attn(struct ggml_context * ctx, struct ggml_tensor * k, struct ggml_tensor * v, struct ggml_tensor * q, struct ggml_tensor * g, struct ggml_tensor * state, float scale)
```

Source: [ggml.h#L1884](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L1884)

### ggml_backend_buft_name

Retrieves the name of a backend buffer type.

```c
const char * ggml_backend_buft_name(ggml_backend_buffer_type_t buft)
```

Source: [ggml-backend.h#L37](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-backend.h#L37)

### ggml_backend_buft_alloc_buffer

Allocates a buffer for a backend buffer type.

```c
ggml_backend_buffer_t ggml_backend_buft_alloc_buffer(ggml_backend_buffer_type_t buft, int size)
```

Source: [ggml-backend.h#L38](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-backend.h#L38)

### ggml_backend_buft_get_alignment

Gets the memory alignment for a backend buffer type.

```c
int ggml_backend_buft_get_alignment(ggml_backend_buffer_type_t buft)
```

Source: [ggml-backend.h#L39](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-backend.h#L39)

### ggml_backend_buft_get_max_size

Returns the maximum size allowed for a backend buffer type.

```c
int ggml_backend_buft_get_max_size(ggml_backend_buffer_type_t buft)
```

Source: [ggml-backend.h#L40](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-backend.h#L40)

### ggml_backend_buft_get_alloc_size

Retrieves the allocation size for a backend buffer type.

```c
int ggml_backend_buft_get_alloc_size(ggml_backend_buffer_type_t buft, struct ggml_tensor * tensor)
```

Source: [ggml-backend.h#L41](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-backend.h#L41)

### ggml_backend_buft_is_host

Checks if the backend buffer type resides in host memory.

```c
int ggml_backend_buft_is_host(ggml_backend_buffer_type_t buft)
```

Source: [ggml-backend.h#L42](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-backend.h#L42)

### ggml_backend_buft_get_device

Gets the device associated with a backend buffer type.

```c
ggml_backend_dev_t ggml_backend_buft_get_device(ggml_backend_buffer_type_t buft)
```

Source: [ggml-backend.h#L43](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-backend.h#L43)

### ggml_backend_buffer_name

Retrieves the name of a backend buffer.

```c
const char * ggml_backend_buffer_name(ggml_backend_buffer_t buffer)
```

Source: [ggml-backend.h#L55](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-backend.h#L55)

### ggml_backend_buffer_free

Frees a previously allocated backend buffer.

```c
void ggml_backend_buffer_free(ggml_backend_buffer_t buffer)
```

Source: [ggml-backend.h#L56](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-backend.h#L56)

### ggml_backend_buffer_get_base

Returns the base pointer of a backend buffer.

```c
void * ggml_backend_buffer_get_base(ggml_backend_buffer_t buffer)
```

Source: [ggml-backend.h#L57](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-backend.h#L57)

### ggml_backend_buffer_get_size

Retrieves the size of a backend buffer.

```c
int ggml_backend_buffer_get_size(ggml_backend_buffer_t buffer)
```

Source: [ggml-backend.h#L58](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-backend.h#L58)

### ggml_backend_buffer_init_tensor

Initializes a tensor within a backend buffer.

```c
void ggml_backend_buffer_init_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor)
```

Source: [ggml-backend.h#L59](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-backend.h#L59)

### ggml_backend_buffer_get_alignment

Gets the alignment property of a backend buffer.

```c
int ggml_backend_buffer_get_alignment(ggml_backend_buffer_t buffer)
```

Source: [ggml-backend.h#L60](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-backend.h#L60)

### ggml_backend_buffer_get_max_size

Returns the maximum size supported by a backend buffer.

```c
int ggml_backend_buffer_get_max_size(ggml_backend_buffer_t buffer)
```

Source: [ggml-backend.h#L61](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-backend.h#L61)

### ggml_backend_buffer_get_alloc_size

Retrieves the allocated size of a backend buffer.

```c
int ggml_backend_buffer_get_alloc_size(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor)
```

Source: [ggml-backend.h#L62](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-backend.h#L62)

### ggml_backend_buffer_clear

Clears the data stored in a backend buffer.

```c
void ggml_backend_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value)
```

Source: [ggml-backend.h#L63](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-backend.h#L63)

### ggml_backend_buffer_is_host

Checks if a backend buffer is located in host memory.

```c
int ggml_backend_buffer_is_host(ggml_backend_buffer_t buffer)
```

Source: [ggml-backend.h#L64](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-backend.h#L64)

### ggml_backend_buffer_set_usage

Sets the usage flags for a backend buffer.

```c
void ggml_backend_buffer_set_usage(ggml_backend_buffer_t buffer, enum ggml_backend_buffer_usage usage)
```

Source: [ggml-backend.h#L65](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-backend.h#L65)

### ggml_backend_buffer_get_usage

Retrieves the usage flags of a backend buffer.

```c
enum ggml_backend_buffer_usage ggml_backend_buffer_get_usage(ggml_backend_buffer_t buffer)
```

Source: [ggml-backend.h#L66](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-backend.h#L66)

### ggml_backend_buffer_get_type

Returns the type of a backend buffer.

```c
ggml_backend_buffer_type_t ggml_backend_buffer_get_type(ggml_backend_buffer_t buffer)
```

Source: [ggml-backend.h#L67](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-backend.h#L67)

### ggml_backend_buffer_reset

Resets a backend buffer to its initial state.

```c
void ggml_backend_buffer_reset(ggml_backend_buffer_t buffer)
```

Source: [ggml-backend.h#L68](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-backend.h#L68)

### ggml_opt_dataset_init

Initializes an optimization dataset.

```c
ggml_opt_dataset_t ggml_opt_dataset_init(int64_t ne_datapoint, int64_t ne_label, int64_t ndata, int64_t ndata_shard)
```

Source: [ggml-opt.h#L39](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-opt.h#L39)

### ggml_opt_dataset_free

Frees an optimization dataset.

```c
void ggml_opt_dataset_free(ggml_opt_dataset_t dataset)
```

Source: [ggml-opt.h#L44](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-opt.h#L44)

### ggml_opt_dataset_data

Retrieves the data from an optimization dataset.

```c
struct ggml_tensor * ggml_opt_dataset_data(ggml_opt_dataset_t dataset)
```

Source: [ggml-opt.h#L47](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-opt.h#L47)

### ggml_opt_dataset_labels

Retrieves the labels from an optimization dataset.

```c
struct ggml_tensor * ggml_opt_dataset_labels(ggml_opt_dataset_t dataset)
```

Source: [ggml-opt.h#L48](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-opt.h#L48)

### ggml_opt_dataset_shuffle

Shuffles the entries in an optimization dataset.

```c
void ggml_opt_dataset_shuffle(ggml_opt_context_t opt_ctx, ggml_opt_dataset_t dataset, int64_t idata)
```

Source: [ggml-opt.h#L51](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-opt.h#L51)

### ggml_opt_dataset_get_batch

Gets a batch of data from an optimization dataset.

```c
void ggml_opt_dataset_get_batch(ggml_opt_dataset_t dataset, struct ggml_tensor * data_batch, struct ggml_tensor * labels_batch, int64_t ibatch)
```

Source: [ggml-opt.h#L54](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-opt.h#L54)

### ggml_opt_get_default_optimizer_params

Retrieves the default parameters for the optimizer.

```c
struct ggml_opt_optimizer_params ggml_opt_get_default_optimizer_params(void * userdata)
```

Source: [ggml-opt.h#L86](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-opt.h#L86)

### ggml_opt_default_params

Returns default optimization parameters.

```c
struct ggml_opt_params ggml_opt_default_params(ggml_backend_sched_t backend_sched, struct ggml_context * ctx_compute, struct ggml_tensor * inputs, struct ggml_tensor * outputs, enum ggml_opt_loss_type loss_type)
```

Source: [ggml-opt.h#L110](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-opt.h#L110)

### ggml_opt_init

Initializes an optimizer instance.

```c
ggml_opt_context_t ggml_opt_init(struct ggml_opt_params params)
```

Source: [ggml-opt.h#L117](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-opt.h#L117)

### ggml_opt_free

Frees an optimizer instance.

```c
void ggml_opt_free(ggml_opt_context_t opt_ctx)
```

Source: [ggml-opt.h#L118](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-opt.h#L118)

### ggml_opt_reset

Resets the optimizer to its initial state.

```c
void ggml_opt_reset(ggml_opt_context_t opt_ctx, bool optimizer)
```

Source: [ggml-opt.h#L121](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-opt.h#L121)

### ggml_opt_inputs

Retrieves the input tensors for the optimizer.

```c
struct ggml_tensor * ggml_opt_inputs(ggml_opt_context_t opt_ctx)
```

Source: [ggml-opt.h#L124](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-opt.h#L124)

### ggml_opt_outputs

Retrieves the output tensors for the optimizer.

```c
struct ggml_tensor * ggml_opt_outputs(ggml_opt_context_t opt_ctx)
```

Source: [ggml-opt.h#L125](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-opt.h#L125)

### ggml_opt_labels

Retrieves the label tensors used in optimization.

```c
struct ggml_tensor * ggml_opt_labels(ggml_opt_context_t opt_ctx)
```

Source: [ggml-opt.h#L126](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-opt.h#L126)

### ggml_opt_loss

Returns the computed loss from the optimizer.

```c
struct ggml_tensor * ggml_opt_loss(ggml_opt_context_t opt_ctx)
```

Source: [ggml-opt.h#L127](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-opt.h#L127)

### ggml_opt_pred

Retrieves prediction outputs from the optimizer.

```c
struct ggml_tensor * ggml_opt_pred(ggml_opt_context_t opt_ctx)
```

Source: [ggml-opt.h#L128](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-opt.h#L128)

### ggml_opt_ncorrect

Returns the number of correct predictions.

```c
struct ggml_tensor * ggml_opt_ncorrect(ggml_opt_context_t opt_ctx)
```

Source: [ggml-opt.h#L129](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-opt.h#L129)

### ggml_opt_grad_acc

Retrieves the accumulated gradients from the optimizer.

```c
struct ggml_tensor * ggml_opt_grad_acc(ggml_opt_context_t opt_ctx, struct ggml_tensor * node)
```

Source: [ggml-opt.h#L131](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-opt.h#L131)

### ggml_opt_result_init

Initializes a structure for storing optimizer results.

```c
ggml_opt_result_t ggml_opt_result_init()
```

Source: [ggml-opt.h#L135](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-opt.h#L135)

### ggml_opt_result_free

Frees an optimizer result structure.

```c
void ggml_opt_result_free(ggml_opt_result_t result)
```

Source: [ggml-opt.h#L136](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-opt.h#L136)

### ggml_opt_result_reset

Resets the optimizer result structure.

```c
void ggml_opt_result_reset(ggml_opt_result_t result)
```

Source: [ggml-opt.h#L137](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-opt.h#L137)

### ggml_opt_result_ndata

Returns the number of data points in the optimizer results.

```c
void ggml_opt_result_ndata(ggml_opt_result_t result, int64_t * ndata)
```

Source: [ggml-opt.h#L140](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-opt.h#L140)

### ggml_opt_result_loss

Retrieves the loss value from the optimizer results.

```c
void ggml_opt_result_loss(ggml_opt_result_t result, double * loss, double * unc)
```

Source: [ggml-opt.h#L141](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-opt.h#L141)

### ggml_opt_result_pred

Retrieves predictions from the optimizer results.

```c
void ggml_opt_result_pred(ggml_opt_result_t result, int32_t * pred)
```

Source: [ggml-opt.h#L142](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-opt.h#L142)

### ggml_opt_result_accuracy

Calculates accuracy from the optimizer results.

```c
void ggml_opt_result_accuracy(ggml_opt_result_t result, double * accuracy, double * unc)
```

Source: [ggml-opt.h#L143](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-opt.h#L143)

### ggml_opt_forward

Performs a forward pass using the optimizer.

```c
void ggml_opt_forward(ggml_opt_context_t opt_ctx, ggml_opt_result_t result)
```

Source: [ggml-opt.h#L148](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-opt.h#L148)

### ggml_opt_forward_backward

Performs both forward and backward passes in optimization.

```c
void ggml_opt_forward_backward(ggml_opt_context_t opt_ctx, ggml_opt_result_t result)
```

Source: [ggml-opt.h#L151](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-opt.h#L151)

### ggml_opt_epoch

Executes one optimization epoch.

```c
void ggml_opt_epoch(ggml_opt_context_t opt_ctx, ggml_opt_dataset_t dataset, ggml_opt_result_t result_train, ggml_opt_result_t result_eval, int64_t idata_split, ggml_opt_epoch_callback callback_train, ggml_opt_epoch_callback callback_eval)
```

Source: [ggml-opt.h#L181](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-opt.h#L181)

### ggml_opt_epoch_callback_progress_bar

Updates a progress bar during an optimization epoch.

```c
void ggml_opt_epoch_callback_progress_bar(bool train, ggml_opt_context_t opt_ctx, ggml_opt_dataset_t dataset, ggml_opt_result_t result, int64_t ibatch, int64_t ibatch_max, int64_t t_start_us)
```

Source: [ggml-opt.h#L191](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-opt.h#L191)

### ggml_opt_fit

Trains a model using the optimizer.

```c
void ggml_opt_fit(ggml_backend_sched_t backend_sched, struct ggml_context * ctx_compute, struct ggml_tensor * inputs, struct ggml_tensor * outputs, ggml_opt_dataset_t dataset, enum ggml_opt_loss_type loss_type, ggml_opt_get_optimizer_params get_opt_pars, int64_t nepoch, int64_t nbatch_logical, float val_split, bool silent)
```

Source: [ggml-opt.h#L201](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-opt.h#L201)

### ggml_opt_step_adamw

Performs an optimization step using the AdamW algorithm.

```c
struct ggml_tensor * ggml_opt_step_adamw(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * grad, struct ggml_tensor * m, struct ggml_tensor * v, struct ggml_tensor * adamw_params)
```

Source: [ggml.h#L2043](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L2043)

### ggml_map_unary_f32

Maps a unary function over a float tensor.

```c
struct ggml_tensor * ggml_map_unary_f32(struct ggml_context * ctx, struct ggml_tensor * a, ggml_unary_op_f32_t fun)
```

Source: [ggml.h#L1902](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L1902)

### ggml_map_unary_inplace_f32

Applies a unary function to a float tensor in-place.

```c
struct ggml_tensor * ggml_map_unary_inplace_f32(struct ggml_context * ctx, struct ggml_tensor * a, ggml_unary_op_f32_t fun)
```

Source: [ggml.h#L1908](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L1908)

### ggml_map_binary_f32

Applies a binary function to two float tensors.

```c
struct ggml_tensor * ggml_map_binary_f32(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, ggml_binary_op_f32_t fun)
```

Source: [ggml.h#L1914](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L1914)

### ggml_map_binary_inplace_f32

Applies a binary function in-place on two float tensors.

```c
struct ggml_tensor * ggml_map_binary_inplace_f32(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, ggml_binary_op_f32_t fun)
```

Source: [ggml.h#L1921](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L1921)

### ggml_map_custom1_f32

Applies a custom unary function on a float tensor.

```c
struct ggml_tensor * ggml_map_custom1_f32(struct ggml_context * ctx, struct ggml_tensor * a, ggml_custom1_op_f32_t fun)
```

Source: [ggml.h#L1928](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L1928)

### ggml_map_custom1_inplace_f32

Applies a custom unary function on a float tensor in-place.

```c
struct ggml_tensor * ggml_map_custom1_inplace_f32(struct ggml_context * ctx, struct ggml_tensor * a, ggml_custom1_op_f32_t fun)
```

Source: [ggml.h#L1934](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L1934)

### ggml_map_custom2_f32

Applies a custom binary function on a float tensor.

```c
struct ggml_tensor * ggml_map_custom2_f32(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, ggml_custom2_op_f32_t fun)
```

Source: [ggml.h#L1940](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L1940)

### ggml_map_custom2_inplace_f32

Applies a custom binary function on a float tensor in-place.

```c
struct ggml_tensor * ggml_map_custom2_inplace_f32(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, ggml_custom2_op_f32_t fun)
```

Source: [ggml.h#L1947](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L1947)

### ggml_map_custom3_f32

Applies a custom ternary function on a float tensor.

```c
struct ggml_tensor * ggml_map_custom3_f32(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, struct ggml_tensor * c, ggml_custom3_op_f32_t fun)
```

Source: [ggml.h#L1954](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L1954)

### ggml_map_custom3_inplace_f32

Applies a custom ternary function on a float tensor in-place.

```c
struct ggml_tensor * ggml_map_custom3_inplace_f32(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, struct ggml_tensor * c, ggml_custom3_op_f32_t fun)
```

Source: [ggml.h#L1962](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L1962)

### ggml_map_custom1

Applies a custom unary function to a tensor.

```c
struct ggml_tensor * ggml_map_custom1(struct ggml_context * ctx, struct ggml_tensor * a, ggml_custom1_op_t fun, int n_tasks, void * userdata)
```

Source: [ggml.h#L1979](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L1979)

### ggml_map_custom1_inplace

Applies a custom unary function to a tensor in-place.

```c
struct ggml_tensor * ggml_map_custom1_inplace(struct ggml_context * ctx, struct ggml_tensor * a, ggml_custom1_op_t fun, int n_tasks, void * userdata)
```

Source: [ggml.h#L1986](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L1986)

### ggml_map_custom2

Applies a custom binary function to a tensor.

```c
struct ggml_tensor * ggml_map_custom2(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, ggml_custom2_op_t fun, int n_tasks, void * userdata)
```

Source: [ggml.h#L1993](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L1993)

### ggml_map_custom2_inplace

Applies a custom binary function to a tensor in-place.

```c
struct ggml_tensor * ggml_map_custom2_inplace(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, ggml_custom2_op_t fun, int n_tasks, void * userdata)
```

Source: [ggml.h#L2001](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L2001)

### ggml_map_custom3

Applies a custom ternary function to a tensor.

```c
struct ggml_tensor * ggml_map_custom3(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, struct ggml_tensor * c, ggml_custom3_op_t fun, int n_tasks, void * userdata)
```

Source: [ggml.h#L2009](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L2009)

### ggml_map_custom3_inplace

Applies a custom ternary function to a tensor in-place.

```c
struct ggml_tensor * ggml_map_custom3_inplace(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, struct ggml_tensor * c, ggml_custom3_op_t fun, int n_tasks, void * userdata)
```

Source: [ggml.h#L2018](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L2018)

### ggml_cross_entropy_loss

Computes the cross-entropy loss for a tensor.

```c
struct ggml_tensor * ggml_cross_entropy_loss(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b)
```

Source: [ggml.h#L2029](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L2029)

### ggml_cross_entropy_loss_back

Computes the backward pass for cross-entropy loss.

```c
struct ggml_tensor * ggml_cross_entropy_loss_back(struct ggml_context * ctx, struct ggml_tensor * a, struct ggml_tensor * b, struct ggml_tensor * c)
```

Source: [ggml.h#L2034](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L2034)

### ggml_fp16_to_fp32

Converts half-precision floats to single-precision.

```c
float ggml_fp16_to_fp32(ggml_fp16_t )
```

Source: [ggml.h#L333](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L333)

### ggml_fp32_to_fp16

Converts single-precision floats to half-precision.

```c
ggml_fp16_t ggml_fp32_to_fp16(float )
```

Source: [ggml.h#L334](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L334)

### ggml_fp16_to_fp32_row

Converts a row of half-precision floats to single-precision.

```c
void ggml_fp16_to_fp32_row(const ggml_fp16_t * , float * , int64_t )
```

Source: [ggml.h#L335](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L335)

### ggml_fp32_to_fp16_row

Converts a row of single-precision floats to half-precision.

```c
void ggml_fp32_to_fp16_row(const float * , ggml_fp16_t * , int64_t )
```

Source: [ggml.h#L336](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L336)

### ggml_fp32_to_bf16

Converts single-precision floats to bfloat16 format.

```c
ggml_bf16_t ggml_fp32_to_bf16(float )
```

Source: [ggml.h#L340](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L340)

### ggml_bf16_to_fp32

Converts bfloat16 values to single-precision floats.

```c
float ggml_bf16_to_fp32(ggml_bf16_t )
```

Source: [ggml.h#L341](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L341)

### ggml_bf16_to_fp32_row

Converts a row of bfloat16 values to single-precision.

```c
void ggml_bf16_to_fp32_row(const ggml_bf16_t * , float * , int64_t )
```

Source: [ggml.h#L342](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L342)

### ggml_fp32_to_bf16_row_ref

Converts a reference row of single-precision floats to bfloat16.

```c
void ggml_fp32_to_bf16_row_ref(const float * , ggml_bf16_t * , int64_t )
```

Source: [ggml.h#L343](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L343)

### ggml_fp32_to_bf16_row

Converts a row of single-precision floats to bfloat16.

```c
void ggml_fp32_to_bf16_row(const float * , ggml_bf16_t * , int64_t )
```

Source: [ggml.h#L344](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L344)

### ggml_guid_matches

Checks if two GUIDs match.

```c
int ggml_guid_matches(ggml_guid_t guid_a, ggml_guid_t guid_b)
```

Source: [ggml.h#L626](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L626)

### ggml_time_init

Initializes the ggml timing system.

```c
void ggml_time_init()
```

Source: [ggml.h#L630](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L630)

### ggml_time_ms

Returns the current time in milliseconds.

```c
int64_t ggml_time_ms()
```

Source: [ggml.h#L631](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L631)

### ggml_time_us

Returns the current time in microseconds.

```c
int64_t ggml_time_us()
```

Source: [ggml.h#L632](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L632)

### ggml_cycles

Returns the current CPU cycle count.

```c
int64_t ggml_cycles()
```

Source: [ggml.h#L633](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L633)

### ggml_cycles_per_ms

Calculates the number of CPU cycles per millisecond.

```c
int64_t ggml_cycles_per_ms()
```

Source: [ggml.h#L634](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L634)

### ggml_fopen

Opens a file with ggml-specific settings.

```c
FILE * ggml_fopen(const char * fname, const char * mode)
```

Source: [ggml.h#L637](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L637)

### ggml_print_object

Prints detailed information of a ggml object.

```c
void ggml_print_object(const struct ggml_object * obj)
```

Source: [ggml.h#L639](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L639)

### ggml_print_objects

Prints details of multiple ggml objects.

```c
void ggml_print_objects(const struct ggml_context * ctx)
```

Source: [ggml.h#L640](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L640)

### ggml_nelements

Returns the number of elements in a tensor.

```c
int64_t ggml_nelements(const struct ggml_tensor * tensor)
```

Source: [ggml.h#L642](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L642)

### ggml_nrows

Returns the number of rows in a tensor.

```c
int64_t ggml_nrows(const struct ggml_tensor * tensor)
```

Source: [ggml.h#L643](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L643)

### ggml_nbytes

Returns the total number of bytes occupied by a tensor.

```c
int ggml_nbytes(const struct ggml_tensor * tensor)
```

Source: [ggml.h#L644](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L644)

### ggml_nbytes_pad

Returns the padded byte size of a tensor.

```c
int ggml_nbytes_pad(const struct ggml_tensor * tensor)
```

Source: [ggml.h#L645](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L645)

### ggml_blck_size

Returns the block size used in tensor operations.

```c
int64_t ggml_blck_size(enum ggml_type type)
```

Source: [ggml.h#L647](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L647)

### ggml_type_size

Returns the size in bytes of a tensor type.

```c
int ggml_type_size(enum ggml_type type)
```

Source: [ggml.h#L648](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L648)

### ggml_row_size

Returns the size of a tensor row in bytes.

```c
int ggml_row_size(enum ggml_type type, int64_t ne)
```

Source: [ggml.h#L649](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L649)

### ggml_type_sizef

Returns the floating-point size in bytes for a tensor type.

```c
double ggml_type_sizef(enum ggml_type type)
```

Source: [ggml.h#L651](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L651)

### ggml_type_name

Returns the name of a tensor type.

```c
const char * ggml_type_name(enum ggml_type type)
```

Source: [ggml.h#L655](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L655)

### ggml_op_name

Returns the name of an operation.

```c
const char * ggml_op_name(enum ggml_op op)
```

Source: [ggml.h#L656](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L656)

### ggml_op_symbol

Returns the symbol representing an operation.

```c
const char * ggml_op_symbol(enum ggml_op op)
```

Source: [ggml.h#L657](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L657)

### ggml_unary_op_name

Returns the name of a unary operation.

```c
const char * ggml_unary_op_name(enum ggml_unary_op op)
```

Source: [ggml.h#L659](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L659)

### ggml_op_desc

Provides a description of an operation.

```c
const char * ggml_op_desc(const struct ggml_tensor * t)
```

Source: [ggml.h#L660](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L660)

### ggml_element_size

Returns the size in bytes of a single tensor element.

```c
int ggml_element_size(const struct ggml_tensor * tensor)
```

Source: [ggml.h#L662](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L662)

### ggml_is_quantized

Checks if a tensor is quantized.

```c
int ggml_is_quantized(enum ggml_type type)
```

Source: [ggml.h#L664](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L664)

### ggml_ftype_to_ggml_type

Converts a file type to a ggml tensor type.

```c
enum ggml_type ggml_ftype_to_ggml_type(enum ggml_ftype ftype)
```

Source: [ggml.h#L667](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L667)

### ggml_is_transposed

Checks if a tensor has been transposed.

```c
int ggml_is_transposed(const struct ggml_tensor * tensor)
```

Source: [ggml.h#L669](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L669)

### ggml_is_permuted

Checks if a tensor's dimensions are permuted.

```c
int ggml_is_permuted(const struct ggml_tensor * tensor)
```

Source: [ggml.h#L670](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L670)

### ggml_is_empty

Checks if a tensor contains no data.

```c
int ggml_is_empty(const struct ggml_tensor * tensor)
```

Source: [ggml.h#L671](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L671)

### ggml_is_scalar

Checks if a tensor represents a scalar.

```c
int ggml_is_scalar(const struct ggml_tensor * tensor)
```

Source: [ggml.h#L672](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L672)

### ggml_is_vector

Checks if a tensor is a vector.

```c
int ggml_is_vector(const struct ggml_tensor * tensor)
```

Source: [ggml.h#L673](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L673)

### ggml_is_matrix

Checks if a tensor is a matrix.

```c
int ggml_is_matrix(const struct ggml_tensor * tensor)
```

Source: [ggml.h#L674](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L674)

### ggml_is_3d

Checks if a tensor is three-dimensional.

```c
int ggml_is_3d(const struct ggml_tensor * tensor)
```

Source: [ggml.h#L675](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L675)

### ggml_n_dims

Returns the number of dimensions of a tensor.

```c
int ggml_n_dims(const struct ggml_tensor * tensor)
```

Source: [ggml.h#L676](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L676)

### ggml_is_contiguous

Checks if a tensor is stored contiguously in memory.

```c
int ggml_is_contiguous(const struct ggml_tensor * tensor)
```

Source: [ggml.h#L678](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L678)

### ggml_is_contiguous_0

Checks if the first dimension of a tensor is contiguous.

```c
int ggml_is_contiguous_0(const struct ggml_tensor * tensor)
```

Source: [ggml.h#L679](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L679)

### ggml_is_contiguous_1

Checks if the second dimension of a tensor is contiguous.

```c
int ggml_is_contiguous_1(const struct ggml_tensor * tensor)
```

Source: [ggml.h#L680](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L680)

### ggml_is_contiguous_2

Checks if the third dimension of a tensor is contiguous.

```c
int ggml_is_contiguous_2(const struct ggml_tensor * tensor)
```

Source: [ggml.h#L681](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L681)

### ggml_are_same_shape

Checks if two tensors have identical shapes.

```c
int ggml_are_same_shape(const struct ggml_tensor * t0, const struct ggml_tensor * t1)
```

Source: [ggml.h#L683](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L683)

### ggml_are_same_stride

Checks if two tensors have identical memory strides.

```c
int ggml_are_same_stride(const struct ggml_tensor * t0, const struct ggml_tensor * t1)
```

Source: [ggml.h#L684](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L684)

### ggml_can_repeat

Checks if a tensor can be repeated along its dimensions.

```c
int ggml_can_repeat(const struct ggml_tensor * t0, const struct ggml_tensor * t1)
```

Source: [ggml.h#L686](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L686)

### ggml_tensor_overhead

Returns the memory overhead of a tensor.

```c
int ggml_tensor_overhead()
```

Source: [ggml.h#L689](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L689)

### ggml_validate_row_data

Validates the data contained in a tensor row.

```c
int ggml_validate_row_data(enum ggml_type type, const void * data, int nbytes)
```

Source: [ggml.h#L691](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L691)

### ggml_used_mem

Returns the amount of memory currently used by ggml.

```c
int ggml_used_mem(const struct ggml_context * ctx)
```

Source: [ggml.h#L699](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L699)

### ggml_get_mem_buffer

Retrieves the current memory buffer pointer used by ggml.

```c
void * ggml_get_mem_buffer(const struct ggml_context * ctx)
```

Source: [ggml.h#L704](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L704)

### ggml_get_mem_size

Returns the size of the allocated memory buffer in ggml.

```c
int ggml_get_mem_size(const struct ggml_context * ctx)
```

Source: [ggml.h#L705](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L705)

### ggml_get_max_tensor_size

Returns the maximum allowable size for a tensor.

```c
int ggml_get_max_tensor_size(const struct ggml_context * ctx)
```

Source: [ggml.h#L706](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L706)

### ggml_get_data

Returns a pointer to the raw data of a tensor.

```c
void * ggml_get_data(const struct ggml_tensor * tensor)
```

Source: [ggml.h#L755](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L755)

### ggml_get_data_f32

Returns a pointer to the float data of a tensor.

```c
float * ggml_get_data_f32(const struct ggml_tensor * tensor)
```

Source: [ggml.h#L756](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L756)

### ggml_get_name

Retrieves the name of a tensor.

```c
const char * ggml_get_name(const struct ggml_tensor * tensor)
```

Source: [ggml.h#L758](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L758)

### ggml_set_name

Assigns a name to a tensor.

```c
struct ggml_tensor * ggml_set_name(struct ggml_tensor * tensor, const char * name)
```

Source: [ggml.h#L759](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L759)

### ggml_format_name

Formats a tensor name for display purposes.

```c
struct ggml_tensor * ggml_format_name(struct ggml_tensor * tensor, const char * fmt)
```

Source: [ggml.h#L761](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L761)

### ggml_log_set

Sets the logging level or output for ggml.

```c
void ggml_log_set(ggml_log_callback log_callback, void * user_data)
```

Source: [ggml.h#L2098](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L2098)

### ggml_set_zero

Sets all elements of a tensor to zero.

```c
struct ggml_tensor * ggml_set_zero(struct ggml_tensor * tensor)
```

Source: [ggml.h#L2100](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L2100)

### ggml_quantize_init

Initializes quantization parameters for tensors.

```c
void ggml_quantize_init(enum ggml_type type)
```

Source: [ggml.h#L2115](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L2115)

### ggml_quantize_free

Frees resources used for quantization.

```c
void ggml_quantize_free()
```

Source: [ggml.h#L2116](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L2116)

### ggml_quantize_requires_imatrix

Checks if quantization requires an integer matrix.

```c
int ggml_quantize_requires_imatrix(enum ggml_type type)
```

Source: [ggml.h#L2119](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L2119)

### ggml_quantize_chunk

Quantizes a chunk of tensor data.

```c
int ggml_quantize_chunk(enum ggml_type type, const float * src, void * dst, int64_t start, int64_t nrows, int64_t n_per_row, const float * imatrix)
```

Source: [ggml.h#L2122](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L2122)

### gguf_init_empty

Initializes an empty GGUF structure.

```c
struct gguf_context * gguf_init_empty()
```

Source: [gguf.h#L79](https://github.com/ggml-org/ggml/blob/9a4acb3/include/gguf.h#L79)

### gguf_init_from_file

Initializes a GGUF structure from a file.

```c
struct gguf_context * gguf_init_from_file(const char * fname, struct gguf_init_params params)
```

Source: [gguf.h#L80](https://github.com/ggml-org/ggml/blob/9a4acb3/include/gguf.h#L80)

### gguf_free

Frees a GGUF structure.

```c
void gguf_free(struct gguf_context * ctx)
```

Source: [gguf.h#L83](https://github.com/ggml-org/ggml/blob/9a4acb3/include/gguf.h#L83)

### gguf_type_name

Returns the name of a GGUF type.

```c
const char * gguf_type_name(enum gguf_type type)
```

Source: [gguf.h#L85](https://github.com/ggml-org/ggml/blob/9a4acb3/include/gguf.h#L85)

### gguf_get_version

Retrieves the version of the GGUF format.

```c
uint32_t gguf_get_version(const struct gguf_context * ctx)
```

Source: [gguf.h#L87](https://github.com/ggml-org/ggml/blob/9a4acb3/include/gguf.h#L87)

### gguf_get_alignment

Returns the alignment requirement for GGUF data.

```c
int gguf_get_alignment(const struct gguf_context * ctx)
```

Source: [gguf.h#L88](https://github.com/ggml-org/ggml/blob/9a4acb3/include/gguf.h#L88)

### gguf_get_data_offset

Retrieves the data offset within a GGUF file.

```c
int gguf_get_data_offset(const struct gguf_context * ctx)
```

Source: [gguf.h#L89](https://github.com/ggml-org/ggml/blob/9a4acb3/include/gguf.h#L89)

### gguf_get_n_kv

Returns the number of key-value pairs in a GGUF structure.

```c
int64_t gguf_get_n_kv(const struct gguf_context * ctx)
```

Source: [gguf.h#L91](https://github.com/ggml-org/ggml/blob/9a4acb3/include/gguf.h#L91)

### gguf_find_key

Searches for a key within a GGUF structure.

```c
int64_t gguf_find_key(const struct gguf_context * ctx, const char * key)
```

Source: [gguf.h#L92](https://github.com/ggml-org/ggml/blob/9a4acb3/include/gguf.h#L92)

### gguf_get_key

Retrieves a key from a GGUF structure.

```c
const char * gguf_get_key(const struct gguf_context * ctx, int64_t key_id)
```

Source: [gguf.h#L93](https://github.com/ggml-org/ggml/blob/9a4acb3/include/gguf.h#L93)

### gguf_get_kv_type

Returns the type of a key-value pair in GGUF.

```c
enum gguf_type gguf_get_kv_type(const struct gguf_context * ctx, int64_t key_id)
```

Source: [gguf.h#L95](https://github.com/ggml-org/ggml/blob/9a4acb3/include/gguf.h#L95)

### gguf_get_arr_type

Returns the array type for a GGUF structure.

```c
enum gguf_type gguf_get_arr_type(const struct gguf_context * ctx, int64_t key_id)
```

Source: [gguf.h#L96](https://github.com/ggml-org/ggml/blob/9a4acb3/include/gguf.h#L96)

### gguf_get_val_u8

Retrieves an unsigned 8-bit value from GGUF.

```c
uint8_t gguf_get_val_u8(const struct gguf_context * ctx, int64_t key_id)
```

Source: [gguf.h#L99](https://github.com/ggml-org/ggml/blob/9a4acb3/include/gguf.h#L99)

### gguf_get_val_i8

Retrieves a signed 8-bit value from GGUF.

```c
int8_t gguf_get_val_i8(const struct gguf_context * ctx, int64_t key_id)
```

Source: [gguf.h#L100](https://github.com/ggml-org/ggml/blob/9a4acb3/include/gguf.h#L100)

### gguf_get_val_u16

Retrieves an unsigned 16-bit value from GGUF.

```c
uint16_t gguf_get_val_u16(const struct gguf_context * ctx, int64_t key_id)
```

Source: [gguf.h#L101](https://github.com/ggml-org/ggml/blob/9a4acb3/include/gguf.h#L101)

### gguf_get_val_i16

Retrieves a signed 16-bit value from GGUF.

```c
int16_t gguf_get_val_i16(const struct gguf_context * ctx, int64_t key_id)
```

Source: [gguf.h#L102](https://github.com/ggml-org/ggml/blob/9a4acb3/include/gguf.h#L102)

### gguf_get_val_u32

Retrieves an unsigned 32-bit value from GGUF.

```c
uint32_t gguf_get_val_u32(const struct gguf_context * ctx, int64_t key_id)
```

Source: [gguf.h#L103](https://github.com/ggml-org/ggml/blob/9a4acb3/include/gguf.h#L103)

### gguf_get_val_i32

Retrieves a signed 32-bit value from GGUF.

```c
int32_t gguf_get_val_i32(const struct gguf_context * ctx, int64_t key_id)
```

Source: [gguf.h#L104](https://github.com/ggml-org/ggml/blob/9a4acb3/include/gguf.h#L104)

### gguf_get_val_f32

Retrieves a 32-bit floating-point value from GGUF.

```c
float gguf_get_val_f32(const struct gguf_context * ctx, int64_t key_id)
```

Source: [gguf.h#L105](https://github.com/ggml-org/ggml/blob/9a4acb3/include/gguf.h#L105)

### gguf_get_val_u64

Retrieves an unsigned 64-bit value from GGUF.

```c
uint64_t gguf_get_val_u64(const struct gguf_context * ctx, int64_t key_id)
```

Source: [gguf.h#L106](https://github.com/ggml-org/ggml/blob/9a4acb3/include/gguf.h#L106)

### gguf_get_val_i64

Retrieves a signed 64-bit value from GGUF.

```c
int64_t gguf_get_val_i64(const struct gguf_context * ctx, int64_t key_id)
```

Source: [gguf.h#L107](https://github.com/ggml-org/ggml/blob/9a4acb3/include/gguf.h#L107)

### gguf_get_val_f64

Retrieves a 64-bit floating-point value from GGUF.

```c
double gguf_get_val_f64(const struct gguf_context * ctx, int64_t key_id)
```

Source: [gguf.h#L108](https://github.com/ggml-org/ggml/blob/9a4acb3/include/gguf.h#L108)

### gguf_get_val_bool

Retrieves a boolean value from GGUF.

```c
int gguf_get_val_bool(const struct gguf_context * ctx, int64_t key_id)
```

Source: [gguf.h#L109](https://github.com/ggml-org/ggml/blob/9a4acb3/include/gguf.h#L109)

### gguf_get_val_str

Retrieves a string value from GGUF.

```c
const char * gguf_get_val_str(const struct gguf_context * ctx, int64_t key_id)
```

Source: [gguf.h#L110](https://github.com/ggml-org/ggml/blob/9a4acb3/include/gguf.h#L110)

### gguf_get_val_data

Retrieves raw data associated with a key in GGUF.

```c
const void * gguf_get_val_data(const struct gguf_context * ctx, int64_t key_id)
```

Source: [gguf.h#L111](https://github.com/ggml-org/ggml/blob/9a4acb3/include/gguf.h#L111)

### gguf_get_arr_n

Returns the number of elements in a GGUF array.

```c
int gguf_get_arr_n(const struct gguf_context * ctx, int64_t key_id)
```

Source: [gguf.h#L112](https://github.com/ggml-org/ggml/blob/9a4acb3/include/gguf.h#L112)

### gguf_get_arr_data

Retrieves the data from a GGUF array.

```c
const void * gguf_get_arr_data(const struct gguf_context * ctx, int64_t key_id)
```

Source: [gguf.h#L116](https://github.com/ggml-org/ggml/blob/9a4acb3/include/gguf.h#L116)

### gguf_get_arr_str

Retrieves an array of strings from GGUF.

```c
const char * gguf_get_arr_str(const struct gguf_context * ctx, int64_t key_id, int i)
```

Source: [gguf.h#L119](https://github.com/ggml-org/ggml/blob/9a4acb3/include/gguf.h#L119)

### gguf_get_n_tensors

Returns the number of tensors stored in a GGUF file.

```c
int64_t gguf_get_n_tensors(const struct gguf_context * ctx)
```

Source: [gguf.h#L121](https://github.com/ggml-org/ggml/blob/9a4acb3/include/gguf.h#L121)

### gguf_find_tensor

Searches for a tensor by name in a GGUF structure.

```c
int64_t gguf_find_tensor(const struct gguf_context * ctx, const char * name)
```

Source: [gguf.h#L122](https://github.com/ggml-org/ggml/blob/9a4acb3/include/gguf.h#L122)

### gguf_get_tensor_offset

Retrieves the data offset for a tensor in GGUF.

```c
int gguf_get_tensor_offset(const struct gguf_context * ctx, int64_t tensor_id)
```

Source: [gguf.h#L123](https://github.com/ggml-org/ggml/blob/9a4acb3/include/gguf.h#L123)

### gguf_get_tensor_name

Retrieves the name of a tensor from GGUF.

```c
const char * gguf_get_tensor_name(const struct gguf_context * ctx, int64_t tensor_id)
```

Source: [gguf.h#L124](https://github.com/ggml-org/ggml/blob/9a4acb3/include/gguf.h#L124)

### gguf_get_tensor_type

Returns the type of a tensor stored in GGUF.

```c
enum ggml_type gguf_get_tensor_type(const struct gguf_context * ctx, int64_t tensor_id)
```

Source: [gguf.h#L125](https://github.com/ggml-org/ggml/blob/9a4acb3/include/gguf.h#L125)

### gguf_get_tensor_size

Returns the size of a tensor in a GGUF file.

```c
int gguf_get_tensor_size(const struct gguf_context * ctx, int64_t tensor_id)
```

Source: [gguf.h#L126](https://github.com/ggml-org/ggml/blob/9a4acb3/include/gguf.h#L126)

### gguf_remove_key

Removes a key-value pair from a GGUF structure.

```c
int64_t gguf_remove_key(struct gguf_context * ctx, const char * key)
```

Source: [gguf.h#L129](https://github.com/ggml-org/ggml/blob/9a4acb3/include/gguf.h#L129)

### gguf_set_val_u8

Sets an unsigned 8-bit value in GGUF.

```c
void gguf_set_val_u8(struct gguf_context * ctx, const char * key, uint8_t val)
```

Source: [gguf.h#L132](https://github.com/ggml-org/ggml/blob/9a4acb3/include/gguf.h#L132)

### gguf_set_val_i8

Sets a signed 8-bit value in GGUF.

```c
void gguf_set_val_i8(struct gguf_context * ctx, const char * key, int8_t val)
```

Source: [gguf.h#L133](https://github.com/ggml-org/ggml/blob/9a4acb3/include/gguf.h#L133)

### gguf_set_val_u16

Sets an unsigned 16-bit value in GGUF.

```c
void gguf_set_val_u16(struct gguf_context * ctx, const char * key, uint16_t val)
```

Source: [gguf.h#L134](https://github.com/ggml-org/ggml/blob/9a4acb3/include/gguf.h#L134)

### gguf_set_val_i16

Sets a signed 16-bit value in GGUF.

```c
void gguf_set_val_i16(struct gguf_context * ctx, const char * key, int16_t val)
```

Source: [gguf.h#L135](https://github.com/ggml-org/ggml/blob/9a4acb3/include/gguf.h#L135)

### gguf_set_val_u32

Sets an unsigned 32-bit value in GGUF.

```c
void gguf_set_val_u32(struct gguf_context * ctx, const char * key, uint32_t val)
```

Source: [gguf.h#L136](https://github.com/ggml-org/ggml/blob/9a4acb3/include/gguf.h#L136)

### gguf_set_val_i32

Sets a signed 32-bit value in GGUF.

```c
void gguf_set_val_i32(struct gguf_context * ctx, const char * key, int32_t val)
```

Source: [gguf.h#L137](https://github.com/ggml-org/ggml/blob/9a4acb3/include/gguf.h#L137)

### gguf_set_val_f32

Sets a 32-bit floating-point value in GGUF.

```c
void gguf_set_val_f32(struct gguf_context * ctx, const char * key, float val)
```

Source: [gguf.h#L138](https://github.com/ggml-org/ggml/blob/9a4acb3/include/gguf.h#L138)

### gguf_set_val_u64

Sets an unsigned 64-bit value in GGUF.

```c
void gguf_set_val_u64(struct gguf_context * ctx, const char * key, uint64_t val)
```

Source: [gguf.h#L139](https://github.com/ggml-org/ggml/blob/9a4acb3/include/gguf.h#L139)

### gguf_set_val_i64

Sets a signed 64-bit value in GGUF.

```c
void gguf_set_val_i64(struct gguf_context * ctx, const char * key, int64_t val)
```

Source: [gguf.h#L140](https://github.com/ggml-org/ggml/blob/9a4acb3/include/gguf.h#L140)

### gguf_set_val_f64

Sets a 64-bit floating-point value in GGUF.

```c
void gguf_set_val_f64(struct gguf_context * ctx, const char * key, double val)
```

Source: [gguf.h#L141](https://github.com/ggml-org/ggml/blob/9a4acb3/include/gguf.h#L141)

### gguf_set_val_bool

Sets a boolean value in GGUF.

```c
void gguf_set_val_bool(struct gguf_context * ctx, const char * key, bool val)
```

Source: [gguf.h#L142](https://github.com/ggml-org/ggml/blob/9a4acb3/include/gguf.h#L142)

### gguf_set_val_str

Sets a string value in GGUF.

```c
void gguf_set_val_str(struct gguf_context * ctx, const char * key, const char * val)
```

Source: [gguf.h#L143](https://github.com/ggml-org/ggml/blob/9a4acb3/include/gguf.h#L143)

### gguf_set_arr_data

Sets the data for an array in GGUF.

```c
void gguf_set_arr_data(struct gguf_context * ctx, const char * key, enum gguf_type type, const void * data, int n)
```

Source: [gguf.h#L146](https://github.com/ggml-org/ggml/blob/9a4acb3/include/gguf.h#L146)

### gguf_set_arr_str

Sets an array of strings in GGUF.

```c
void gguf_set_arr_str(struct gguf_context * ctx, const char * key, const char ** data, int n)
```

Source: [gguf.h#L149](https://github.com/ggml-org/ggml/blob/9a4acb3/include/gguf.h#L149)

### gguf_set_kv

Sets a key-value pair in GGUF.

```c
void gguf_set_kv(struct gguf_context * ctx, const struct gguf_context * src)
```

Source: [gguf.h#L152](https://github.com/ggml-org/ggml/blob/9a4acb3/include/gguf.h#L152)

### gguf_add_tensor

Adds a tensor to a GGUF structure.

```c
void gguf_add_tensor(struct gguf_context * ctx, const struct ggml_tensor * tensor)
```

Source: [gguf.h#L155](https://github.com/ggml-org/ggml/blob/9a4acb3/include/gguf.h#L155)

### gguf_set_tensor_type

Sets the type for a tensor in GGUF.

```c
void gguf_set_tensor_type(struct gguf_context * ctx, const char * name, enum ggml_type type)
```

Source: [gguf.h#L159](https://github.com/ggml-org/ggml/blob/9a4acb3/include/gguf.h#L159)

### gguf_set_tensor_data

Sets the data for a tensor in GGUF.

```c
void gguf_set_tensor_data(struct gguf_context * ctx, const char * name, const void * data)
```

Source: [gguf.h#L162](https://github.com/ggml-org/ggml/blob/9a4acb3/include/gguf.h#L162)

### gguf_write_to_file

Writes a GGUF structure to a file.

```c
int gguf_write_to_file(const struct gguf_context * ctx, const char * fname, bool only_meta)
```

Source: [gguf.h#L192](https://github.com/ggml-org/ggml/blob/9a4acb3/include/gguf.h#L192)

### gguf_get_meta_size

Returns the size of the metadata in a GGUF file.

```c
int gguf_get_meta_size(const struct gguf_context * ctx)
```

Source: [gguf.h#L195](https://github.com/ggml-org/ggml/blob/9a4acb3/include/gguf.h#L195)

### gguf_get_meta_data

Retrieves the metadata from a GGUF file.

```c
void gguf_get_meta_data(const struct gguf_context * ctx, void * data)
```

Source: [gguf.h#L198](https://github.com/ggml-org/ggml/blob/9a4acb3/include/gguf.h#L198)

### ggml_threadpool_new

Creates a new thread pool for parallel operations.

```c
struct ggml_threadpool * ggml_threadpool_new(struct ggml_threadpool_params * params)
```

Source: [ggml-cpu.h#L55](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-cpu.h#L55)

### ggml_threadpool_free

Frees a previously created thread pool.

```c
void ggml_threadpool_free(struct ggml_threadpool * threadpool)
```

Source: [ggml-cpu.h#L56](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-cpu.h#L56)

### ggml_threadpool_get_n_threads

Returns the number of threads in the thread pool.

```c
int ggml_threadpool_get_n_threads(struct ggml_threadpool * threadpool)
```

Source: [ggml-cpu.h#L57](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-cpu.h#L57)

### ggml_threadpool_pause

Pauses execution of the thread pool.

```c
void ggml_threadpool_pause(struct ggml_threadpool * threadpool)
```

Source: [ggml-cpu.h#L58](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-cpu.h#L58)

### ggml_threadpool_resume

Resumes execution of the thread pool.

```c
void ggml_threadpool_resume(struct ggml_threadpool * threadpool)
```

Source: [ggml-cpu.h#L59](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-cpu.h#L59)

### ggml_threadpool_params_default

Returns default parameters for thread pool configuration.

```c
struct ggml_threadpool_params ggml_threadpool_params_default(int n_threads)
```

Source: [ggml.h#L2187](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L2187)

### ggml_threadpool_params_init

Initializes parameters for a thread pool.

```c
void ggml_threadpool_params_init(struct ggml_threadpool_params * p, int n_threads)
```

Source: [ggml.h#L2188](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L2188)

### ggml_threadpool_params_match

Checks if two thread pool parameter sets match.

```c
int ggml_threadpool_params_match(const struct ggml_threadpool_params * p0, const struct ggml_threadpool_params * p1)
```

Source: [ggml.h#L2189](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml.h#L2189)

### ggml_cpu_has_sse3

Checks if the CPU supports SSE3 instructions.

```c
int ggml_cpu_has_sse3()
```

Source: [ggml-cpu.h#L78](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-cpu.h#L78)

### ggml_cpu_has_ssse3

Checks if the CPU supports SSSE3 instructions.

```c
int ggml_cpu_has_ssse3()
```

Source: [ggml-cpu.h#L79](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-cpu.h#L79)

### ggml_cpu_has_avx

Checks if the CPU supports AVX instructions.

```c
int ggml_cpu_has_avx()
```

Source: [ggml-cpu.h#L80](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-cpu.h#L80)

### ggml_cpu_has_avx_vnni

Checks if the CPU supports AVX VNNI instructions.

```c
int ggml_cpu_has_avx_vnni()
```

Source: [ggml-cpu.h#L81](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-cpu.h#L81)

### ggml_cpu_has_avx2

Checks if the CPU supports AVX2 instructions.

```c
int ggml_cpu_has_avx2()
```

Source: [ggml-cpu.h#L82](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-cpu.h#L82)

### ggml_cpu_has_f16c

Checks if the CPU supports F16C instructions.

```c
int ggml_cpu_has_f16c()
```

Source: [ggml-cpu.h#L83](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-cpu.h#L83)

### ggml_cpu_has_fma

Checks if the CPU supports FMA (Fused Multiply-Add) instructions.

```c
int ggml_cpu_has_fma()
```

Source: [ggml-cpu.h#L84](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-cpu.h#L84)

### ggml_cpu_has_avx512

Checks if the CPU supports AVX512 instructions.

```c
int ggml_cpu_has_avx512()
```

Source: [ggml-cpu.h#L85](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-cpu.h#L85)

### ggml_cpu_has_avx512_vbmi

Checks if the CPU supports AVX512 VBMI instructions.

```c
int ggml_cpu_has_avx512_vbmi()
```

Source: [ggml-cpu.h#L86](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-cpu.h#L86)

### ggml_cpu_has_avx512_vnni

Checks if the CPU supports AVX512 VNNI instructions.

```c
int ggml_cpu_has_avx512_vnni()
```

Source: [ggml-cpu.h#L87](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-cpu.h#L87)

### ggml_cpu_has_avx512_bf16

Checks if the CPU supports AVX512 BF16 instructions.

```c
int ggml_cpu_has_avx512_bf16()
```

Source: [ggml-cpu.h#L88](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-cpu.h#L88)

### ggml_cpu_has_amx_int8

Checks if the CPU supports AMX INT8 instructions.

```c
int ggml_cpu_has_amx_int8()
```

Source: [ggml-cpu.h#L89](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-cpu.h#L89)

### ggml_cpu_has_neon

Checks if the CPU supports NEON instructions.

```c
int ggml_cpu_has_neon()
```

Source: [ggml-cpu.h#L91](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-cpu.h#L91)

### ggml_cpu_has_arm_fma

Checks if the CPU supports ARM FMA instructions.

```c
int ggml_cpu_has_arm_fma()
```

Source: [ggml-cpu.h#L92](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-cpu.h#L92)

### ggml_cpu_has_fp16_va

Checks if the CPU supports FP16 vector arithmetic.

```c
int ggml_cpu_has_fp16_va()
```

Source: [ggml-cpu.h#L93](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-cpu.h#L93)

### ggml_cpu_has_dotprod

Checks if the CPU supports dot product operations.

```c
int ggml_cpu_has_dotprod()
```

Source: [ggml-cpu.h#L94](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-cpu.h#L94)

### ggml_cpu_has_matmul_int8

Checks if the CPU supports int8 matrix multiplication.

```c
int ggml_cpu_has_matmul_int8()
```

Source: [ggml-cpu.h#L95](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-cpu.h#L95)

### ggml_cpu_has_sve

Checks if the CPU supports SVE (Scalable Vector Extension).

```c
int ggml_cpu_has_sve()
```

Source: [ggml-cpu.h#L96](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-cpu.h#L96)

### ggml_cpu_get_sve_cnt

Returns the number of SVE registers available on the CPU.

```c
int ggml_cpu_get_sve_cnt()
```

Source: [ggml-cpu.h#L97](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-cpu.h#L97)

### ggml_cpu_has_riscv_v

Checks if a RISC-V CPU supports vector instructions.

```c
int ggml_cpu_has_riscv_v()
```

Source: [ggml-cpu.h#L99](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-cpu.h#L99)

### ggml_cpu_has_vsx

Checks if the CPU supports VSX instructions.

```c
int ggml_cpu_has_vsx()
```

Source: [ggml-cpu.h#L100](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-cpu.h#L100)

### ggml_cpu_has_wasm_simd

Checks if the CPU supports WebAssembly SIMD.

```c
int ggml_cpu_has_wasm_simd()
```

Source: [ggml-cpu.h#L101](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-cpu.h#L101)

### ggml_cpu_has_llamafile

Checks if the CPU supports llama file optimizations.

```c
int ggml_cpu_has_llamafile()
```

Source: [ggml-cpu.h#L102](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-cpu.h#L102)

### ggml_get_type_traits_cpu

Retrieves CPU type traits for tensor operations.

```c
const struct ggml_type_traits_cpu * ggml_get_type_traits_cpu(enum ggml_type type)
```

Source: [ggml-cpu.h#L116](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-cpu.h#L116)

### ggml_cpu_init

Initializes CPU-specific settings for ggml.

```c
void ggml_cpu_init()
```

Source: [ggml-cpu.h#L118](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-cpu.h#L118)

### ggml_backend_cpu_init

Initializes the CPU backend.

```c
ggml_backend_t ggml_backend_cpu_init()
```

Source: [ggml-cpu.h#L124](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-cpu.h#L124)

### ggml_backend_is_cpu

Checks if the active backend is CPU-based.

```c
int ggml_backend_is_cpu(ggml_backend_t backend)
```

Source: [ggml-cpu.h#L126](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-cpu.h#L126)

### ggml_backend_cpu_set_n_threads

Sets the number of threads for the CPU backend.

```c
void ggml_backend_cpu_set_n_threads(ggml_backend_t backend_cpu, int n_threads)
```

Source: [ggml-cpu.h#L127](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-cpu.h#L127)

### ggml_backend_cpu_set_threadpool

Assigns a thread pool to the CPU backend.

```c
void ggml_backend_cpu_set_threadpool(ggml_backend_t backend_cpu, ggml_threadpool_t threadpool)
```

Source: [ggml-cpu.h#L128](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-cpu.h#L128)

### ggml_backend_cpu_set_abort_callback

Sets an abort callback for CPU backend operations.

```c
void ggml_backend_cpu_set_abort_callback(ggml_backend_t backend_cpu, int abort_callback, void * abort_callback_data)
```

Source: [ggml-cpu.h#L129](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-cpu.h#L129)

### ggml_backend_cpu_reg

Registers the CPU backend.

```c
ggml_backend_reg_t ggml_backend_cpu_reg()
```

Source: [ggml-cpu.h#L131](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-cpu.h#L131)

### ggml_backend_cpu_buffer_from_ptr

Creates a CPU backend buffer from an existing pointer.

```c
ggml_backend_buffer_t ggml_backend_cpu_buffer_from_ptr(void * ptr, int size)
```

Source: [ggml-backend.h#L349](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-backend.h#L349)

### ggml_backend_cpu_buffer_type

Returns the buffer type used by the CPU backend.

```c
ggml_backend_buffer_type_t ggml_backend_cpu_buffer_type()
```

Source: [ggml-backend.h#L350](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-backend.h#L350)

### ggml_backend_cuda_init

Initializes the CUDA backend.

```c
ggml_backend_t ggml_backend_cuda_init(int device)
```

Source: [ggml-cuda.h#L23](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-cuda.h#L23)

### ggml_backend_is_cuda

Checks if the active backend is CUDA.

```c
int ggml_backend_is_cuda(ggml_backend_t backend)
```

Source: [ggml-cuda.h#L25](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-cuda.h#L25)

### ggml_backend_cuda_buffer_type

Returns the buffer type for the CUDA backend.

```c
ggml_backend_buffer_type_t ggml_backend_cuda_buffer_type(int device)
```

Source: [ggml-cuda.h#L28](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-cuda.h#L28)

### ggml_backend_cuda_split_buffer_type

Returns the split buffer type for CUDA operations.

```c
ggml_backend_buffer_type_t ggml_backend_cuda_split_buffer_type(int main_device, const float * tensor_split)
```

Source: [ggml-cuda.h#L31](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-cuda.h#L31)

### ggml_backend_cuda_host_buffer_type

Returns the host buffer type for the CUDA backend.

```c
ggml_backend_buffer_type_t ggml_backend_cuda_host_buffer_type()
```

Source: [ggml-cuda.h#L34](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-cuda.h#L34)

### ggml_backend_cuda_get_device_count

Returns the number of available CUDA devices.

```c
int ggml_backend_cuda_get_device_count()
```

Source: [ggml-cuda.h#L36](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-cuda.h#L36)

### ggml_backend_cuda_get_device_description

Retrieves the description of a CUDA device.

```c
void ggml_backend_cuda_get_device_description(int device, char * description, int description_size)
```

Source: [ggml-cuda.h#L37](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-cuda.h#L37)

### ggml_backend_cuda_get_device_memory

Returns the memory capacity of a CUDA device.

```c
void ggml_backend_cuda_get_device_memory(int device, int * free, int * total)
```

Source: [ggml-cuda.h#L38](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-cuda.h#L38)

### ggml_backend_cuda_register_host_buffer

Registers a host buffer with the CUDA backend.

```c
int ggml_backend_cuda_register_host_buffer(void * buffer, int size)
```

Source: [ggml-cuda.h#L40](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-cuda.h#L40)

### ggml_backend_cuda_unregister_host_buffer

Unregisters a host buffer from the CUDA backend.

```c
void ggml_backend_cuda_unregister_host_buffer(void * buffer)
```

Source: [ggml-cuda.h#L41](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-cuda.h#L41)

### ggml_backend_cuda_reg

Registers the CUDA backend.

```c
ggml_backend_reg_t ggml_backend_cuda_reg()
```

Source: [ggml-cuda.h#L43](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-cuda.h#L43)

### ggml_vk_available_devices

Lists the available Vulkan devices.

```c
struct ggml_vk_device * ggml_vk_available_devices(int memoryRequired, int * count)
```

Source: [ggml-kompute.h#L27](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-kompute.h#L27)

### ggml_vk_get_device

Retrieves a Vulkan device by index.

```c
int ggml_vk_get_device(struct ggml_vk_device * device, int memoryRequired, const char * name)
```

Source: [ggml-kompute.h#L28](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-kompute.h#L28)

### ggml_vk_has_vulkan

Checks if Vulkan is supported on the system.

```c
int ggml_vk_has_vulkan()
```

Source: [ggml-kompute.h#L29](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-kompute.h#L29)

### ggml_vk_has_device

Checks if a specific Vulkan device is available.

```c
int ggml_vk_has_device()
```

Source: [ggml-kompute.h#L30](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-kompute.h#L30)

### ggml_vk_current_device

Returns the currently active Vulkan device.

```c
struct ggml_vk_device ggml_vk_current_device()
```

Source: [ggml-kompute.h#L31](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-kompute.h#L31)

### ggml_backend_vk_init

Initializes the Vulkan backend.

```c
ggml_backend_t ggml_backend_vk_init(int dev_num)
```

Source: [ggml-vulkan.h#L14](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-vulkan.h#L14)

### ggml_backend_is_vk

Checks if the active backend is Vulkan.

```c
int ggml_backend_is_vk(ggml_backend_t backend)
```

Source: [ggml-vulkan.h#L16](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-vulkan.h#L16)

### ggml_backend_vk_get_device_count

Returns the number of available Vulkan devices.

```c
int ggml_backend_vk_get_device_count()
```

Source: [ggml-vulkan.h#L17](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-vulkan.h#L17)

### ggml_backend_vk_get_device_description

Retrieves the description of a Vulkan device.

```c
void ggml_backend_vk_get_device_description(int device, char * description, int description_size)
```

Source: [ggml-vulkan.h#L18](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-vulkan.h#L18)

### ggml_backend_vk_get_device_memory

Returns the memory capacity of a Vulkan device.

```c
void ggml_backend_vk_get_device_memory(int device, int * free, int * total)
```

Source: [ggml-vulkan.h#L19](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-vulkan.h#L19)

### ggml_backend_vk_buffer_type

Returns the buffer type for the Vulkan backend.

```c
ggml_backend_buffer_type_t ggml_backend_vk_buffer_type(int dev_num)
```

Source: [ggml-vulkan.h#L21](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-vulkan.h#L21)

### ggml_backend_vk_host_buffer_type

Returns the host buffer type for the Vulkan backend.

```c
ggml_backend_buffer_type_t ggml_backend_vk_host_buffer_type()
```

Source: [ggml-vulkan.h#L23](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-vulkan.h#L23)

### ggml_backend_vk_reg

Registers the Vulkan backend.

```c
ggml_backend_reg_t ggml_backend_vk_reg()
```

Source: [ggml-vulkan.h#L25](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-vulkan.h#L25)

### ggml_backend_metal_init

Initializes the Metal backend.

```c
ggml_backend_t ggml_backend_metal_init()
```

Source: [ggml-metal.h#L42](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-metal.h#L42)

### ggml_backend_is_metal

Checks if the active backend is Metal.

```c
int ggml_backend_is_metal(ggml_backend_t backend)
```

Source: [ggml-metal.h#L44](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-metal.h#L44)

### ggml_backend_metal_buffer_from_ptr

Creates a Metal buffer from a host pointer.

```c
ggml_backend_buffer_t ggml_backend_metal_buffer_from_ptr(void * data, int size, int max_size)
```

Source: [ggml-metal.h#L46](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-metal.h#L46)

### ggml_backend_metal_set_abort_callback

Sets an abort callback for Metal operations.

```c
void ggml_backend_metal_set_abort_callback(ggml_backend_t backend, int abort_callback, void * user_data)
```

Source: [ggml-metal.h#L50](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-metal.h#L50)

### ggml_backend_metal_buffer_type

Returns the buffer type for the Metal backend.

```c
ggml_backend_buffer_type_t ggml_backend_metal_buffer_type()
```

Source: [ggml-metal.h#L52](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-metal.h#L52)

### ggml_backend_metal_supports_family

Checks if the Metal backend supports a specific family.

```c
int ggml_backend_metal_supports_family(ggml_backend_t backend, int family)
```

Source: [ggml-metal.h#L57](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-metal.h#L57)

### ggml_backend_metal_capture_next_compute

Captures the next compute command in Metal.

```c
void ggml_backend_metal_capture_next_compute(ggml_backend_t backend)
```

Source: [ggml-metal.h#L60](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-metal.h#L60)

### ggml_backend_metal_reg

Registers the Metal backend.

```c
ggml_backend_reg_t ggml_backend_metal_reg()
```

Source: [ggml-metal.h#L62](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-metal.h#L62)

### ggml_backend_opencl_init

Initializes the OpenCL backend.

```c
ggml_backend_t ggml_backend_opencl_init()
```

Source: [ggml-opencl.h#L14](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-opencl.h#L14)

### ggml_backend_is_opencl

Checks if the active backend is OpenCL.

```c
int ggml_backend_is_opencl(ggml_backend_t backend)
```

Source: [ggml-opencl.h#L15](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-opencl.h#L15)

### ggml_backend_opencl_buffer_type

Returns the buffer type for the OpenCL backend.

```c
ggml_backend_buffer_type_t ggml_backend_opencl_buffer_type()
```

Source: [ggml-opencl.h#L17](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-opencl.h#L17)

### ggml_backend_opencl_host_buffer_type

Returns the host buffer type for OpenCL.

```c
ggml_backend_buffer_type_t ggml_backend_opencl_host_buffer_type()
```

Source: [ggml-opencl.h#L18](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-opencl.h#L18)

### ggml_backend_opencl_reg

Registers the OpenCL backend.

```c
ggml_backend_reg_t ggml_backend_opencl_reg()
```

Source: [ggml-opencl.h#L20](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-opencl.h#L20)

### ggml_backend_cann_reg

Registers the CANN backend.

```c
ggml_backend_reg_t ggml_backend_cann_reg()
```

Source: [ggml-cann.h#L37](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-cann.h#L37)

### ggml_backend_cann_init

Initializes the CANN backend.

```c
ggml_backend_t ggml_backend_cann_init(int32_t device)
```

Source: [ggml-cann.h#L49](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-cann.h#L49)

### ggml_backend_is_cann

Checks if the CANN backend is active.

```c
int ggml_backend_is_cann(ggml_backend_t backend)
```

Source: [ggml-cann.h#L60](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-cann.h#L60)

### ggml_backend_cann_buffer_type

Returns the buffer type for the CANN backend.

```c
ggml_backend_buffer_type_t ggml_backend_cann_buffer_type(int32_t device)
```

Source: [ggml-cann.h#L73](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-cann.h#L73)

### ggml_backend_cann_get_device_count

Returns the number of CANN devices available.

```c
int32_t ggml_backend_cann_get_device_count()
```

Source: [ggml-cann.h#L83](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-cann.h#L83)

### ggml_backend_cann_host_buffer_type

Returns the host buffer type for the CANN backend.

```c
ggml_backend_buffer_type_t ggml_backend_cann_host_buffer_type()
```

Source: [ggml-cann.h#L90](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-cann.h#L90)

### ggml_backend_cann_get_device_description

Retrieves the description of a CANN device.

```c
void ggml_backend_cann_get_device_description(int32_t device, char * description, int description_size)
```

Source: [ggml-cann.h#L102](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-cann.h#L102)

### ggml_backend_cann_get_device_memory

Returns the memory capacity of a CANN device.

```c
void ggml_backend_cann_get_device_memory(int32_t device, int * free, int * total)
```

Source: [ggml-cann.h#L117](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-cann.h#L117)

### ggml_backend_kompute_init

Initializes the Kompute backend.

```c
ggml_backend_t ggml_backend_kompute_init(int device)
```

Source: [ggml-kompute.h#L40](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-kompute.h#L40)

### ggml_backend_is_kompute

Checks if the active backend is Kompute.

```c
int ggml_backend_is_kompute(ggml_backend_t backend)
```

Source: [ggml-kompute.h#L42](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-kompute.h#L42)

### ggml_backend_kompute_buffer_type

Returns the buffer type for the Kompute backend.

```c
ggml_backend_buffer_type_t ggml_backend_kompute_buffer_type(int device)
```

Source: [ggml-kompute.h#L44](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-kompute.h#L44)

### ggml_backend_kompute_reg

Registers the Kompute backend.

```c
ggml_backend_reg_t ggml_backend_kompute_reg()
```

Source: [ggml-kompute.h#L46](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-kompute.h#L46)

### ggml_backend_rpc_init

Initializes the RPC backend.

```c
ggml_backend_t ggml_backend_rpc_init(const char * endpoint)
```

Source: [ggml-rpc.h#L13](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-rpc.h#L13)

### ggml_backend_is_rpc

Checks if the active backend is RPC.

```c
int ggml_backend_is_rpc(ggml_backend_t backend)
```

Source: [ggml-rpc.h#L14](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-rpc.h#L14)

### ggml_backend_rpc_buffer_type

Returns the buffer type for the RPC backend.

```c
ggml_backend_buffer_type_t ggml_backend_rpc_buffer_type(const char * endpoint)
```

Source: [ggml-rpc.h#L16](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-rpc.h#L16)

### ggml_backend_rpc_get_device_memory

Returns the device memory for an RPC backend device.

```c
void ggml_backend_rpc_get_device_memory(const char * endpoint, int * free, int * total)
```

Source: [ggml-rpc.h#L18](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-rpc.h#L18)

### ggml_backend_rpc_start_server

Starts an RPC server for backend communication.

```c
void ggml_backend_rpc_start_server(ggml_backend_t backend, const char * endpoint, int free_mem, int total_mem)
```

Source: [ggml-rpc.h#L20](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-rpc.h#L20)

### ggml_backend_rpc_reg

Registers the RPC backend.

```c
ggml_backend_reg_t ggml_backend_rpc_reg()
```

Source: [ggml-rpc.h#L22](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-rpc.h#L22)

### ggml_backend_rpc_add_device

Adds a device to the RPC backend.

```c
ggml_backend_dev_t ggml_backend_rpc_add_device(const char * endpoint)
```

Source: [ggml-rpc.h#L24](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-rpc.h#L24)

### ggml_backend_sycl_init

Initializes the SYCL backend.

```c
ggml_backend_t ggml_backend_sycl_init(int device)
```

Source: [ggml-sycl.h#L20](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-sycl.h#L20)

### ggml_backend_is_sycl

Checks if the active backend is SYCL.

```c
int ggml_backend_is_sycl(ggml_backend_t backend)
```

Source: [ggml-sycl.h#L22](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-sycl.h#L22)

### ggml_backend_sycl_buffer_type

Returns the buffer type for the SYCL backend.

```c
ggml_backend_buffer_type_t ggml_backend_sycl_buffer_type(int device)
```

Source: [ggml-sycl.h#L25](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-sycl.h#L25)

### ggml_backend_sycl_split_buffer_type

Returns the split buffer type for SYCL operations.

```c
ggml_backend_buffer_type_t ggml_backend_sycl_split_buffer_type(const float * tensor_split)
```

Source: [ggml-sycl.h#L28](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-sycl.h#L28)

### ggml_backend_sycl_host_buffer_type

Returns the host buffer type for the SYCL backend.

```c
ggml_backend_buffer_type_t ggml_backend_sycl_host_buffer_type()
```

Source: [ggml-sycl.h#L31](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-sycl.h#L31)

### ggml_backend_sycl_print_sycl_devices

Prints available SYCL devices.

```c
void ggml_backend_sycl_print_sycl_devices()
```

Source: [ggml-sycl.h#L33](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-sycl.h#L33)

### ggml_backend_sycl_get_gpu_list

Retrieves a list of SYCL GPU devices.

```c
void ggml_backend_sycl_get_gpu_list(int * id_list, int max_len)
```

Source: [ggml-sycl.h#L34](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-sycl.h#L34)

### ggml_backend_sycl_get_device_description

Gets the description of a SYCL device.

```c
void ggml_backend_sycl_get_device_description(int device, char * description, int description_size)
```

Source: [ggml-sycl.h#L35](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-sycl.h#L35)

### ggml_backend_sycl_get_device_count

Returns the number of available SYCL devices.

```c
int ggml_backend_sycl_get_device_count()
```

Source: [ggml-sycl.h#L38](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-sycl.h#L38)

### ggml_backend_sycl_get_device_memory

Returns the memory capacity of a SYCL device.

```c
void ggml_backend_sycl_get_device_memory(int device, int * free, int * total)
```

Source: [ggml-sycl.h#L39](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-sycl.h#L39)

### ggml_backend_sycl_reg

Registers the SYCL backend.

```c
ggml_backend_reg_t ggml_backend_sycl_reg()
```

Source: [ggml-sycl.h#L45](https://github.com/ggml-org/ggml/blob/9a4acb3/include/ggml-sycl.h#L45)
