

list = []
x = 1.2312
y = 12.23141
c = 213.213

list.append(x,y,c)
print(list)


from math import sqrt, pow, acos
 
def angle_of_vector(v1, v2):
    pi = 3.1415
    vector_prod = v1[0] * v2[0] + v1[1] * v2[1]
    length_prod = sqrt(pow(v1[0], 2) + pow(v1[1], 2)) * sqrt(pow(v2[0], 2) + pow(v2[1], 2))
    cos = vector_prod * 1.0 / (length_prod * 1.0 + 1e-6)
    return (acos(cos) / pi) * 180
 
 
if __name__ == '__main__':
    a = [1, 0]
    b = [5, 8]
    print(angle_of_vector(a, b))

import numpy as np

x = [21.32131,321.1231]
a = np.array(x)
print(a)


import math
import numpy as np

#hypot() 返回欧几里德范数 sqrt(x*x + y*y)。



def getlen(x1,x2):
	p1 = np.array(x1)
	p2 = np.array(x2)
	p3 = p2-p1
	len = math.hypot(p3[0],p3[1])
	return len

x = [3,0]
y = [0,4]
print(getlen(x,y))



import math
import numpy as np

#计算两点的模长
def getlen(x1,x2):
	p1 = np.array(x1)
	p2 = np.array(x2)
	p3 = p2-p1
	len = math.hypot(p3[0],p3[1])	#hypot() 返回欧几里德范数 sqrt(x*x + y*y)
	return len
#求前后角度
#一直一个点坐标和两个线长，目的是重新赋值后一帧端点座标
def angle(x1,x2,x3,x4):	#输入多点,顺序帧数据
			#(x1-x2);(x3-x4)

	len1 = getlen(x1,x2)
	len2 = getlen(x3,x4)
	poor = len2/len1
	x3[0] = x1[0]*poor
    x3[1] = x1[1]*poor
    #获取同平面的斜边边长
    len3 = getlen(x3,x2)
    len4 = sqrt(pow(len1,2) - pow(len3,2))
    #计算cos
    cos = (pow(len3,2) + pow(len4,2) - pow(len1,2)) / (2 * len3 * len4)
    return (acos(cos) / math.pi) * 180

x1 =[10,10]
x2 = [0,0]
x3 = [5,5]
x3 = [0,0]
angle = angle(x1,x2,x3,x4)
print(angle)