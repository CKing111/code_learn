#! /bin/bash

#读取结果文件
slope1=$(cat /home/cxking/AlphaPose/examples/res/test/programing_es.txt)
slope2=$(cat /home/cxking/AlphaPose/examples/res/test/programing_we.txt)
echo $slope1
echo $slope2
echo 'ok'

rosservice call /l_arm_controller/robot_driver/move_joint "pose: [2.346557885806249, $slope1, $slope2, 1.6377367646215473, 3.29039844590268, -0.7121641635894775]
has_ref: false
ref_joint: [0]
mvvelo: 0.5
mvacc: 0.3
mvtime: 0.0
mvradii: 0.0
coord_mode: 0
index: 0"



# rosservice call /l_arm_controller/robot_driver/move_joint "pose: [2.346557885806249, 1.6406094968746698, 2.775073510670984, 1.6377367646215473, 3.29039844590268, -0.7121641635894775]
# has_ref: false
# ref_joint: [0]
# mvvelo: 0.5
# mvacc: 0.3
# mvtime: 0.0
# mvradii: 0.0
# coord_mode: 0
# index: 0"


# rosservice call /l_arm_controller/robot_driver/move_joint "pose: [2.0610390161318306,1.78493706377358,-2.4962230193585553,2.286636530852438,-1.570357907722175,0.9398120695412225]
# has_ref: false
# ref_joint: [0]
# mvvelo: 2.0
# mvacc: 0.5
# mvtime: 0.0
# mvradii: 0.0
# coord_mode: 0
# index: 0"

#1.
#pose:[134.44786553327276, 88.68442699324197, -1.829272371720994, 93.83540616688443, 188.52594708253463, -40.80400159018262]
#value:[2.346557885806249, 1.5478352198733847, -0.031926825257684026, 1.6377367646215473, 3.29039844590268, -0.7121641635894775]

#(x*3.1415926/180)

# a=[]
# # x = [-79.873,-0.196,91.797,-3.188,89.424,134.826]
# x = [136.826,89.424,76.819,25.549,82.699,-40.804]
# for i in range(len(x)):
# 	y = x[i]*3.1415926/180
# 	a.append(y)
# print(a)

# a = []
# y = [2.346557885806249, 1.5478352198733847, -0.031926825257684026, 1.6377367646215473, 3.29039844590268, -0.7121641635894775]
# for i in range(len(y)):
# 	x = y[i]*180/3.1415926
# 	a.append(x)
# print(a)




