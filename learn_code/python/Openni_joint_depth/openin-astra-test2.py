from openni import openni2
import numpy as np
import cv2

#鼠标触碰反馈函数
def mousecallback(event,x,y,flags,param):
     if event==cv2.EVENT_LBUTTONDBLCLK:
         print(y, x, dpt[y,x])


if __name__ == "__main__": 

    openni2.initialize()        #初始化

    dev = openni2.Device.open_any()     #读取可用设备
    print(dev.get_device_info())        #打印

    depth_stream = dev.create_depth_stream()    #创建深度视频流
    depth_stream.start()    #执行

    cap = cv2.VideoCapture(4)       #opencv捕获RGB视频镜头
    cv2.namedWindow('depth')
    cv2.setMouseCallback('depth',mousecallback) #实例化鼠标反馈

    while True: #非ctrl + c 或 ‘q’一直循环

        frame = depth_stream.read_frame()   #读取视频帧
        dframe_data = np.array(frame.get_buffer_as_triplet()).reshape([480, 640, 2])    #深度帧的数据重塑为三维
        dpt1 = np.asarray(dframe_data[:, :, 0], dtype='float32')
        dpt2 = np.asarray(dframe_data[:, :, 1], dtype='float32')
        
        dpt2 *= 255
        dpt = dpt1 + dpt2
        
        cv2.imshow('depth', dpt)

        ret,frame = cap.read()
        cv2.imshow('color', frame)

        key = cv2.waitKey(1)
        if int(key) == ord('q'):
            break

    depth_stream.stop()
    dev.close()

