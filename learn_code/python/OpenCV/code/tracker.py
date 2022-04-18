# -*- coding: utf-8 -*-
 
import sys
import cv2
import click


@click.command()
@click.option('--video', help = 'input video')
@click.option('--algorithm', help = 'tracker algorithm, BOOSTING、MIL、KCF、TLD、MEDIANFLOW、GOTURN、CSRT、MOSSE')
def main(video, algorithm):
    '''

    :param video: 待处理的视频文件
    :param algorithm: 指定OpenCV中的跟踪算法
    :return:
    '''

    major_ver, minor_ver, subminor_ver = (cv2.__version__).split('.')

    # 根据opencv的不同版本，创建跟踪器
    if int(minor_ver) < 3:
        tracker = cv2.Tracker_create(algorithm)
    else:
        if algorithm == 'BOOSTING':
            tracker = cv2.TrackerBoosting_create()
        if algorithm == 'MIL':
            tracker = cv2.TrackerMIL_create()
        if algorithm == 'KCF':
            tracker = cv2.TrackerKCF_create()
        if algorithm == 'TLD':
            tracker = cv2.TrackerTLD_create()
        if algorithm == 'MEDIANFLOW':
            tracker = cv2.TrackerMedianFlow_create()
        if algorithm == 'GOTURN':
            tracker = cv2.TrackerGOTURN_create()
        if algorithm == "CSRT":
            tracker = cv2.TrackerCSRT_create()
        if algorithm == 'MOSSE':
            tracker = cv2.TrackerMOSSE_create()

    # 读取视频文件
    video_cap = cv2.VideoCapture(video)

    # 检查视频文件是否被正确打开
    if not video_cap.isOpened():
        print("Open video failed.")
        sys.exit()

    # 读取第一帧数据
    ok, frame = video_cap.read()
    if not ok:
        print('Read video file failed.')
        sys.exit()

    # 手动选择关注的区域
    bbox = cv2.selectROI(frame, False)

    #
    ok = tracker.init(frame, bbox)

    while True:
        # 读取下一帧数据
        ok, frame = video_cap.read()
        if not ok:
            break

        # 开始计时器
        timer = cv2.getTickCount()

        # 更新跟踪器
        ok, bbox = tracker.update(frame)

        # 计算fps
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

        # 画出 bounding box
        if ok:

            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
        else:
            # Tracking failure
            cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        # 显示
        cv2.putText(frame, algorithm + " Tracker", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

        cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

        cv2.imshow("Tracking", frame)

        # 接收到q键，退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == '__main__':
    main()

