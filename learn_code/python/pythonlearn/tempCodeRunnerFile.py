#coding=utf-8

def fib(max):
    n,a,b = 0,0,1
    while True:
        if n < max:
            yield b
            a,b = a,a+b
            n += 1
        else :
            raise StopIteration 
# g = fib(6)
# g = iter(fib(6))
iter_fib=iter(fib(10))

try:
    while True:
        # x = next(iter_fib)
        print('generater:' ,next(iter_fib))
except StopIteration:
    pass