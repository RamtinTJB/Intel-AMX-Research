import time
import functools

def timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        print(f'\nFunction {func.__name__} took {end_time - start_time:.6f} seconds to execute')
        return result
    return wrapper
