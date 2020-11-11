import time


def time_calc(func):
    def wrapper(*args, **kargs):
        start_time = time.time()
        f = func(*args, **kargs)
        print('{}: {:.2f} s'.format(func.__name__, time.time() - start_time))
        return f
    return wrapper
