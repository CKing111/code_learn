#! /bin/bash


# rosservice call /robot_driver/move_joint "pose: [2.346557885806249, 1.5478352198733847, -0.031926825257684026, 1.6377367646215473, 3.29039844590268, -0.7121641635894775]
# has_ref: false
# ref_joint: [0]
# mvvelo: 0.1
# mvacc: 0.3
# mvtime: 0.0
# mvradii: 0.0
# coord_mode: 0
# index: 0"



rosservice call /l_arm_controller/robot_driver/move_joint "pose: [2.346557885806249, 1.5478352198733847, -0.031926825257684026, 1.6377367646215473, 3.29039844590268, -0.7121641635894775]
has_ref: false
ref_joint: [0]
mvvelo: 0.1
mvacc: 0.3
mvtime: 0.0
mvradii: 0.0
coord_mode: 0
index: 0"

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

























# rosservice call /robot_driver/move_line "pose: [-0.138197,-0.707232,0.087840,0.05101597,-0.08091346,1.16439639]                    
# has_ref: false
# ref_joint: [0]
# mvvelo: 0.1
# mvacc: 0.03
# mvtime: 0.0
# mvradii: 0.0
# coord_mode: 0
# index: 0"


# lev1:
# (x/1000)
# -138.197      /-0.138197
# -707.232      /-0.707232
# 87.840        /0.087840
# (x*3.1415926/180)
# 2.923        /0.05101597
# -4.636       /-0.08091346
# 66.715       /1.16439639

# lev2:
# (x/1000)
# -4.19           -0.004190
# -760.288        -0.760288
# -41.504         -0.041504
# (x*3.1415926/180)
# -84.082
# 13.863
# -88.564