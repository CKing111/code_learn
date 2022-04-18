#-- coding:UTF-8 --

import json
import os
import sys

import subprocess       #系统函数调用模块,可以帮助我们生成子进程,并可以通过管道来连接他们的输入输出,并获取返回值,进而达到通信
import  os
import time
import signal

import math

import numpy as np

# Computer Vision
import cv2

from openni import openni2  # , nite2
from openni import _openni2 as c_api


#全局变量

#接受shell脚本中的全局变量
arm = os.getenv('arm')
# print(arm,'--------1',file=sys.stderr)
root_path = '/home/cxking/AlphaPose/examples/res'
# print("work path :","\n",root_path)

file_path = os.path.join(root_path, 'alphapose-results.json')

# cmd = ['python', '/home/cxking/AlphaPose/scripts/demo_inference_test2.py','--cfg /home/cxking/AlphaPose/configs/coco/resnet/256x192_res50_lr1e-3_2x-dcn.yaml','--checkpoint /home/cxking/AlphaPose/pretrained_models/fast_dcn_res50_256x192.pth','--outdir /home/cxking/AlphaPose/examples/res','--vis','--webcam 0']
# cmd = ['python','/home/cxking/AlphaPose/scripts/demo_inference_test2.py',' --cfg ','/home/cxking/AlphaPose/configs/coco/resnet/256x192_res50_lr1e-3_2x-dcn.yaml ','--checkpoint ','/home/cxking/AlphaPose/pretrained_models/fast_dcn_res50_256x192.pth ','--outdir',' /home/cxking/AlphaPose/examples/res ','--vis ','--webcam',' 0']
alphapose_cmd = ['python /home/cxking/AlphaPose/scripts/demo_inference_test2.py --cfg /home/cxking/AlphaPose/configs/coco/resnet/256x192_res50_lr1e-3_2x-dcn.yaml --checkpoint /home/cxking/AlphaPose/pretrained_models/fast_dcn_res50_256x192.pth --outdir /home/cxking/AlphaPose/examples/res --vis --webcam 7']
ros_cmd=['roscore']
ros_jaka_cmd=['roslaunch jaka_ros_driver start.launch']
test_sh_cmd = ['/home/cxking/jaka_ros_ws/test_sh/test_joint2.sh']

ros_service = 'rosservice call'
ros_node = '/robot_driver/move_joint'
pose=[1.9785875504410695,0.9541067204678558,-1.654319252462968,2.261601485528601,1.60328049944293,0.8329127855351778]
x = pose[0]
y = pose[1]


ros_pose=['rosservice call /robot_driver/move_joint "pose: [%f,%f,-1.654319252462968,2.261601485528601,1.60328049944293,0.8329127855351778] has_ref: false ref_joint: [0] mvvelo: 2.0 mvacc: 0.5 mvtime: 0.0 mvradii: 0.0 coord_mode: 0 index: 0"'%(x,y)]

#txt写入的文件名
filename1 = '/home/cxking/AlphaPose/examples/res/test/programing_es.txt'
filename2 = '/home/cxking/AlphaPose/examples/res/test/programing_we.txt'


'''
{0,  "Nose"},
    {1,  "L Eye"},  左眼
    {2,  "R Eye"},  右眼
    {3,  "L Ear"},
    {4,  "R Ear"},
    {5,  "L Shoulder"},左肩
    {6,  "R Shoulder"},
    {7,  "L Elbow"},
    {8,  "R Elbow"},
    {9,  "L Wrist"},   左手腕
    {10, "R Wrist"},

    {11, "L Hip"},
    {12, "R Hip"},
    {13, "L Knee"},
    {14, "R knee"},
    {15, "L Ankle"},  左脚踝
    {16, "R Ankle"},  右脚踝
'''


#https://cloud.tencent.com/developer/article/1590030
#执行AlphaPose-wocam
def alphapose1_open(cmd):

    p = subprocess.Popen(cmd,shell=True, close_fds=True, preexec_fn = os.setsid)   
                        #执行cmd指令，
                        # close_fds=True：此时除了文件描述符为0 , 1 and 2，其他子进程都要被杀掉。( Linux中所有的进程都是进程0的子进程。

    print('子线程PID：',p.pid,file=sys.stderr) 

    # pid = p.pid 
    # time.sleep(5)       #函数推迟调用线程的运行，可通过参数secs指秒数，表示进程挂起的时间。



    # return pid


# alphapose1(alphapose_cmd)

def alphapose1_end(pid):

    os.killpg( pid,signal.SIGUSR1)
    time.sleep(3)


# def alphapose2(cmd, timeout=10, skip=False):
#     """
#     执行linux命令,返回list
#     :param cmd: linux命令
#     :param timeout: 超时时间,生产环境, 特别卡, 因此要3秒
#     :param skip: 是否跳过超时限制
#     :return: list
#     """
#     p = subprocess.Popen(cmd, stderr=subprocess.STDOUT, stdout=subprocess.PIPE,shell=True,close_fds=True,preexec_fn=os.setsid)

#     t_beginning = time.time()  # 开始时间
#     while True:
#         if p.poll() is not None:
#             break
#         seconds_passed = time.time() - t_beginning
#         if not skip:
#             if seconds_passed > timeout:
#                 # p.terminate()
#                 # p.kill()
#                 # raise TimeoutError(cmd, timeout)
#                 print('错误, 命令: {},本地执行超时!'.format(cmd))
#                 # 当shell=True时，只有os.killpg才能kill子进程
#                 try:
#                     # time.sleep(1)
#                     os.killpg(p.pid, signal.SIGUSR1)
#                 except Exception as e:
#                     pass
#                 return False

#     result = p.stdout.readlines()  # 结果输出列表
#     return result

# #执行ros指令
# def ros_work(cmd):
#         p = subprocess.Popen(cmd,shell=True, close_fds=True, preexec_fn = os.setsid)   
#         print('ROS子线程PID：',p.pid,file=sys.stderr)

# #将点位列表当做参数传递 
# def ros_call(x,y):
#         # x = map(int, x)
#         # y = map(int, y)
#         call =   subprocess.Popen(['rosservice call /robot_driver/move_joint "pose: [%*.*f,%*.*f,-1.654319252462968,2.261601485528601,1.60328049944293,0.8329127855351778] has_ref: false ref_joint: [0] mvvelo: 2.0 mvacc: 0.5 mvtime: 0.0 mvradii: 0.0 coord_mode: 0 index: 0" '% (x,y)],shell=True, close_fds=True, preexec_fn = os.setsid)   
#         print('ROS_call子线程PID：',call.pid,file=sys.stderr)


def end_ros(x):
    p = subprocess.getoutput("pgrep %s "% x)
    subprocess.getoutput("killall -9 %s " % p)


#返回模型的json文件
def jsonwork(file_path):
    # path = file_path
    with open(file_path) as f:
        img_data = json.load(f)     #json转为python数据类型

        # 读取json文件中图像关节点数据
        for img_dict in img_data:   #json中各图片数据
            image_name = img_dict['image_id']   #读取imgID
            keypoints = img_dict['keypoints']   #读取关节点值，list

            #关节点分组
            keypoints_group=[]
            for i in range(0, len(keypoints), 2):
                keypoints_group.append(keypoints[i:i+2])

            #读取左手腕的位姿
            LWrist=keypoints_group[9]
            LShoulder=keypoints_group[5]
            RShoulder=keypoints_group[6]
            LElbow=keypoints_group[7]

    return LWrist,LElbow,LShoulder,RShoulder

#关节数据判断（Up, middle, down）
def arm_index_es(le,ls):

    diff = le[1]-ls[1]
    if diff > 15:
        print('Arm hem！',file=sys.stderr)
        arm = -1
    elif diff < -15:
        print('Arm up！',file=sys.stderr)
        arm = 1
    else:
        print('Arm level！',file=sys.stderr)
        arm = 0
    return arm

def arm_index(x1,x2):

    diff = x1[1]-x2[1]
    if diff > 10:
        print('(1)Arm hem！',file=sys.stderr)
        arm = -1
    elif diff < -10:
        print('(1)Arm up！',file=sys.stderr)
        arm = 1
    else:
        print('(1)Arm level！',file=sys.stderr)
        arm = 0
    return arm

#通过计算关节夹角，输出机械臂实际角度
#https://www.jb51.net/article/164697.htm
def angle(x1,x2,x3,arm):  #输入两个向量

    arm_value = arm
    #拼接向量list
    v1 = x2 + x1
    v2 = x3 + x2
    dx1 = v1[2] - v1[0]
    dy1 = v1[3] - v1[1]
    dx2 = v2[2] - v2[0]
    dy2 = v2[3] - v2[1]
    #计算向量v1的斜率
    #atan2( y2-y1, x2-x1 );
    angle1 = math.atan2(dy1, dx1)  #v1斜率 
    angle1 = int(angle1 * 180/math.pi)    #转换为角度输出
    # print('v1 angle： ',angle1)
    #计算向量v2的斜率
    angle2 = math.atan2(dy2, dx2) #v2斜率
    angle2 = int(angle2 * 180/math.pi)    
    # print('v2 angle: ',angle2)
    #得到偏移角度
    if angle1*angle2 >= 0:        #两斜率角度均大于0，（PI）
        included_angle = abs(angle1-angle2)
    else:
        included_angle = abs(angle1) + abs(angle2)
        if included_angle > 180:
            included_angle = 360 - included_angle
    print('关节夹角：',included_angle,file = sys.stderr)
    #映射到机械臂的二维角度中
    #增加转角限制
    if arm_value > 0 and arm_value < 10:    #arm up
        try:
            if included_angle < 85:
                included_angle = 90 - included_angle
        except IOError:
            print ("Error: 没有找到文件或写入文件失败",file=sys.stderr)
    elif arm_value >= 10:
        included_angle = included_angle
    else:   #arm_value<= 0
        try:
            if included_angle < 85:
                included_angle = 90 + included_angle
        except IOError:
            print ("Error: 没有找到文件或写入文件失败",file=sys.stderr)

    print('机械臂关节偏转角度： ',included_angle,file = sys.stderr)
    return included_angle

def angle_we(x1,x2,x3):  #输入两个向量

    # arm_value = arm
    #拼接向量list
    v1 = x2 + x1
    v2 = x3 + x2
    dx1 = v1[2] - v1[0]
    dy1 = v1[3] - v1[1]
    dx2 = v2[2] - v2[0]
    dy2 = v2[3] - v2[1]
    #计算向量v1的斜率
    #atan2( y2-y1, x2-x1 );
    angle1 = math.atan2(dy1, dx1)  #v1斜率 
    angle1 = int(angle1 * 180/math.pi)    #转换为角度输出
    # print('v1 angle： ',angle1)
    #计算向量v2的斜率
    angle2 = math.atan2(dy2, dx2) #v2斜率
    angle2 = int(angle2 * 180/math.pi)    
    # print('v2 angle: ',angle2)
    #得到偏移角度
    if angle1*angle2 >= 0:        #两斜率角度均大于0，（PI）
        included_angle = abs(angle1-angle2)
    else:
        included_angle = abs(angle1) + abs(angle2)
        if included_angle > 180:
            included_angle = 360 - included_angle
    print('关节夹角：',included_angle,file = sys.stderr)
    #映射到机械臂的二维角度中
    #增加转角限制

    try:
        if included_angle < 125 and included_angle > 0 :
            included_angle =  included_angle * -1
        elif included_angle >= 125 and included_angle < 170 :
            included_angle = ( included_angle - 90 ) * -1
        else :
            included_angle = 0 
        # else:   #arm_value<= 0
            # included_angle = 0
    except IOError:
        print ("Error: 没有找到文件或写入文件失败",file=sys.stderr)



    print('机械臂关节偏转角度： ',included_angle,file = sys.stderr)
    return included_angle


#计算角度斜率
def Slope(angle):
    slope = angle * math.pi / 180
    print('斜率： ',slope,file=sys.stderr)
    return slope





#关节斜率判断（Up, middle, down）
def slope_index(slope):
    #输入斜率
    diff = slope
    if diff >= 100 and diff < 180:
        print('Arm hem！',file=sys.stderr)
    elif diff < 80 and diff >= 0:
        print('Arm up！',file=sys.stderr)
    elif diff < 100 and diff >= 80:
        print('Arm level！',file=sys.stderr)
    else:
        print('数据获取有误，超出机械臂运动区间，请重试！！！！')


#写入文件
def writer(x,filename):
    #错误检测
    try:
        with open (filename,'w') as file_object:
            file_object.write(str(x))  
    except IOError:
        print ("Error: 没有找到文件或写入文件失败",file=sys.stderr)
    else:
        print ("内容写入文件成功",file=sys.stderr)
        file_object.close()

#主程序，最后得到机械臂关节移动斜率
def main():
    # openni2.initialize()        #初始化
    openni2.initialize()  #
    if (openni2.is_initialized()):
        print("openNI2 initialized")
    else:
        print("openNI2 not initialized")

    dev = openni2.Device.open_any()     #读取可用设备
    print(dev.get_device_info())        #打印

    depth_stream = dev.create_depth_stream()    #创建深度视频流
    depth_stream.start()    #执行

    
    p = subprocess.Popen(alphapose_cmd,shell=True, close_fds=True, preexec_fn = os.setsid)
    print('alphapose子线程PID：',p.pid,file=sys.stderr)

    time.sleep(5)
    var = 1
    while var <= 3:

        #记录时间
        # t = time.clock()
        t = time.time()

        # frame = depth_stream.read_frame('/home/cxking/OpenNI-Linux-x64-2.3/Redist')   #读取视频帧
        frame = depth_stream.read_frame()   #读取视频帧

        dframe_data = np.array(frame.get_buffer_as_triplet()).reshape([480, 640, 2])    #深度帧的数据重塑为三维
        dpt1 = np.asarray(dframe_data[:, :, 0], dtype='float32')
        dpt2 = np.asarray(dframe_data[:, :, 1], dtype='float32')
        
        dpt2 *= 255
        dpt = dpt1 + dpt2

        cv2.imshow('depth', dpt)



        #滞后读取结果文件
        time.sleep(3)
        #处理程序输出数据
        try:
            lw,le,ls,rs = jsonwork(file_path)   #左手腕和左肩坐标
            print('LW = ',lw,'\n','LE = ',le,'\n','LS = ',ls,'\n','RS = ',rs,'\n',file=sys.stderr)    #打印
        except IOError:
            print ("Error: 没有找到文件或写入文件失败",file=sys.stderr)


        # lw_z = dpt_z_coor(frame,lw[0],lw[1])
        # le_z = dpt_z_coor(frame,le[0],le[1])
        # ls_z = dpt_z_coor(frame,ls[0],ls[1])
        # rs_z = dpt_z_coor(frame,rs[0],rs[1])

        x = lw[0]
        y = lw[1]

        lw_z = dpt[y,x]
        Lw_xyz = lw.append(lw_z)
        print("lw空间坐标",Lw_xyz)
        #得到关节状态索引值
        arm_value_es = arm_index(le,ls)  #(手肘，肩膀)
        # arm_value_we = arm_index(lw,le)  #(手腕，手肘)
        # print('arm_value = ',arm_value,file=sys.stderr)

        print('----------------------------------------------------',file=sys.stderr)
        #计算机械臂关节二偏移角度
        print('机械臂关节二：',file=sys.stderr)
        arm_angle_es = angle(le,ls,rs,arm_value_es)
        slope_index(arm_angle_es)
        #计算机械臂偏移斜率
        arm_slope_es = Slope(arm_angle_es)

        print('----------------------------------------------------',file=sys.stderr)
        #计算机械臂关节三偏移角度
        print('机械臂关节三：',file=sys.stderr)
        arm_angle_we = angle_we(lw,le,ls)
        arm_slope_we = Slope(arm_angle_we)
        
        #将arm写入文件
        # print('机械臂关节二旋转斜率： ',es,file=sys.stderr)
        writer(arm_slope_es,filename1)
        # print('机械臂关节三旋转斜率： ',we,file=sys.stderr)
        writer(arm_slope_we,filename2)

        time.sleep(2)

        var += 1
        print('单次循环时间： ',time.time()-t,' 秒。')
        # return arm_slope_es,arm_slope_we
    os.killpg(p.pid,signal.SIGUSR1)
    depth_stream.stop()
    dev.close()
    time.sleep(3)

def depth():

            dev = openni2.Device.open_any()     #读取可用设备
            print(dev.get_device_info())        #打印

            depth_stream = dev.create_depth_stream()    #创建深度视频流
            depth_stream.start()    #执行
            # assert depth_stream.isOpened(), 'Cannot capture source(depth)'
            return depth_stream

def dpt_z_coor(frame,x,y):
            dframe_data = np.array(frame.get_buffer_as_triplet()).reshape([480, 640, 2])    #深度帧的数据重塑为三维
            dpt1 = np.asarray(dframe_data[:, :, 0], dtype='float32')
            dpt2 = np.asarray(dframe_data[:, :, 1], dtype='float32')
            
            dpt2 *= 255
            dpt = dpt1 + dpt2
            x = dpt[y,x]
            return x
if __name__=='__main__':

    # openni2.initialize()  #
    # if (openni2.is_initialized()):
    #     print("openNI2 initialized")
    # else:
    #     print("openNI2 not initialized")

    # depth_stream = depth()


    #关节姿态判断
    arm = main()

    sys.exit(arm)

