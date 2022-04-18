#!/usr/bin/python
# -*- coding: UTF-8 -*-
 
 #####1
 #斐波那契数列方法
 #1
def fab(max): 
    n, a, b = 0, 0, 1 
    while n < max: 
        print(b) 
        a, b = b, a + b 
        n = n + 1

fab(5)
 #2
#返回一个List
def fab(max): 
    n, a, b = 0, 0, 1 
    L = [] 
    while n < max: 
        L.append(b) 
        a, b = b, a + b 
        n = n + 1 
    return L
 
for n in fab(5): 
    print (n)

#3
#返回一个 iterable 对象
class Fab(object): 
 
    def __init__(self, max): 
        self.max = max 
        self.n, self.a, self.b = 0, 0, 1 
 
    def __iter__(self): 
        return self 
 
    def next(self): 
        if self.n < self.max: 
            r = self.b 
            self.a, self.b = self.b, self.a + self.b 
            self.n = self.n + 1 
            return r 
        raise StopIteration()
 
for n in Fab(5): 
    print(n)

#4
#保持第一版 fab 函数的简洁性，同时又要获得 iterable 的效果,
#采用yield
def fab(max): 
    n, a, b = 0, 0, 1 
    while n < max: 
        yield b      # 使用 yield
        # print b 
        a, b = b, a + b 
        n = n + 1
 
for n in fab(5): 
    print (n)

    """
    yield 的作用就是把一个函数变成一个 generator，带有 yield 的函数不再是一个普通函数，Python 解释器会将其视为一个 generator，调用 fab(5) 不会执行 fab 函数，
    而是返回一个 iterable 对象！在 for 循环执行时，每次循环都会执行 fab 函数内部的代码，执行到 yield b 时，fab 函数就返回一个迭代值，下次迭代时，
    代码从 yield b 的下一条语句继续执行，而函数的本地变量看起来和上次中断执行前是完全一样的，于是函数继续执行，直到再次遇到 yield。
    """


####2
#看列表 [0，1，2，3，4，5，6，7，8，9]，要求你把列表里面的每个值加1
#1
info = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
b = []
# for index,i in enumerate(info):
#     print(i+1)
#     b.append(i+1)
# print(b)
for index,i in enumerate(info):
    info[index] +=1
print(info)

#2
info = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
a = map(lambda x:x+1,info)
print(a)
for i in a:
    print(i)

#3
info = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
a = [i+1 for i in range(10)]
print(a)

#4
#这种一边循环一边计算的机制，称为生成器：generator
#成器一次只能产生一个值，这样消耗的内存数量将大大减小，而且允许调用函数可以很快的处理前几个返回值，因此生成器看起来像是一个函数，但是表现得却像是迭代器
#创建一个generator，有很多种方法，第一种方法很简单，只有把一个列表生成式的[]中括号改为（）小括号，就创建一个generator，
#每次打印一个，可通过调用next（）函数获得生成器下一个运算结果，一般用for循环

#eg
generator_ex = (x*x for x in range(10))
for i in generator_ex:
    print(i+1)

#fib
#py3生成器return不可返回带参数值
def fib(max):
    n,a,b =0,0,1
    while n < max:
        yield b
        a,b =b,a+b
        n = n+1
    return                 
for i in fib(6): 
    print(i)





####
#coding=utf-8
import random

from torch._C import preserve_format
base_set = list(range(9))
print(base_set)
random_set = random.sample(base_set, k=2)     #base_set集随机排序后，随机选取k个值
ps_set = random_set[0]
pu_set = random_set[1:]
a = ps_set.size()
b = pu_set.size()
print('-----------------------------------------')
print(ps_set)
print(a)
print(pu_set)  
print(b)



#####
"""
单星号（*）：*agrs,将所有参数以元组(tuple)的形式导入
双星号（**）：**kwargs,将参数以字典的形式导入
"""

def foo(a, b=10, *args, **kwargs):
    print (a)
    print (b)
    print (args)
    print (kwargs)
foo(1, 2, 3, 4, e=5, f=6, g=7)

# 1
# 2
# (3, 4)
# {'e': 5, 'g': 7, 'f': 6}


######

# 函数有了yield之后，函数名+（）就变成了生成器
# return在生成器中代表生成器的中止，直接报错
# next的作用是唤醒并继续执行
# send的作用是唤醒并继续执行，发送一个信息到生成器内部
'''生成器'''
#coding=utf-8
######
def create_counter(n):
    print("cream_counter")
    while True:
        yield n
        print("increment n")
        n += 1

gen = create_counter(2)
print(gen)
print(next(gen))
print(next(gen))


#####防止报错
#coding=utf-8
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


def gen(n):
    i=0
    while True:
        if i<=n:
            yield i
            i+=1
        else:
            raise StopIteration
 
iter_gen=iter(gen(10))
try:
    while True:
        print(next(iter_gen))
except StopIteration:
    pass

