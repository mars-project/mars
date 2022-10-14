import ray

original_remote_function_options = ray.remote_function.RemoteFunction.options


def _wrap_original_remote_function_options(*args, **kwargs):
    assert kwargs["num_cpus"] == 5, "expect num_cpus==5"
    return original_remote_function_options(*args, **kwargs)


ray.remote_function.RemoteFunction.options = _wrap_original_remote_function_options
