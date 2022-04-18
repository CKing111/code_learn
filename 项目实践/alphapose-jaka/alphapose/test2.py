import math
import numpy as np

#计算两点的模长
def getlen(p1,p2):
    # p1 = np.array(x1)
    # p2 = np.array(x2)
    # p3 = p1-p2
    # print(p3)
    # len = math.hypot(p3[0],p3[1])	#hypot() 返回欧几里德范数 sqrt(x*x + y*y)
    # return len
    return math.sqrt(math.pow((p2[0] - p1[0]), 2) + math.pow((p2[1] - p1[1]), 2))
#求前后角度
#一直一个点坐标和两个线长，目的是重新赋值后一帧端点座标
def angle(x1,x2,x3,x4):	#输入多点,顺序帧数据
			#(x1-x2);(x3-x4)
    len1 = getlen(x2,x1)
    print(len1)
    len2 = getlen(x4,x3)
    poor = len2/len1
    print(poor)
    x3[0] = x1[0]*poor
    x3[1] = x1[1]*poor
    print(x3)
    #获取同平面的斜边边长
    len3 = getlen(x2,x3)
    print(len3)
    len4 = math.sqrt(pow(len1,2) - pow(len3,2))
    print(len4)
    #计算cos
    cos = (pow(len3,2) + pow(len1,2) - pow(len4,2)) / (2 * len3 * len1)
    print(cos)
    return (math.acos(cos) / math.pi) * 180

x1 =[10,10]
x2 = [0,0]
x3 = [5,5]
x4 = [0,0]
angle = angle(x1,x2,x3,x4)
print(angle)