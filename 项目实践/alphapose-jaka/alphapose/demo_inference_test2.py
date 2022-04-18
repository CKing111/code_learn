"""Script for single-gpu/multi-gpu demo."""



import argparse
    #命令行选项、参数和子命令解析器
    #https://docs.python.org/zh-cn/3/library/argparse.html
import os
import platform
import sys
import time

import numpy as np
import torch
from tqdm import tqdm
import natsort

from detector.apis import get_detector
from test_json_to_coor import write_to_file
from trackers.tracker_api import Tracker
from trackers.tracker_cfg import cfg as tcfg
from trackers import track
from alphapose.models import builder
from alphapose.utils.config import update_config
from alphapose.utils.detector import DetectionLoader
from alphapose.utils.transforms import flip, flip_heatmap
from alphapose.utils.vis import getTime
from alphapose.utils.webcam_detector import WebCamDetectionLoader
from alphapose.utils.writer import DataWriter

#####################ROS#######################################
start = time.time()
# from numpy.core.numeric import True_
# import rospy

# from geometry_msgs.msg import Twist

# class rplidarDetector_insert_value:
#     def __init__(self):
#         rospy.init_node('rplidar_detection', anonymous=True)
#         self.pub = rospy.Publisher('/sssssss', Twist, queue_size=1)
#         print(" '----------------/car_original is OK-----------------'")

#     ## 回调函数
#     # def callback(self):
#     #     twist = Twist()
#     #     twist.linear.x = 12
#     #     twist.angular.z = 14
#     #     print ("is ok! ")
#     #     self.pub.publish(twist)

# if __name__ == "__main__":

#     detector = rplidarDetector_insert_value()
    
#     while True:
#         detector.callback()
"""----------------------------- Demo options -----------------------------"""
parser = argparse.ArgumentParser(description='AlphaPose Demo')
                    #ArgumentParser 对象包含将命令行解析成 Python 数据类型所需的全部信息。
                    #description:简要描述这个程序做什么以及怎么做.
parser.add_argument('--cfg', type=str, required=True,
                    help='experiment configure file name')
        #定义单个的命令行参数应当如何解析。
parser.add_argument('--checkpoint', type=str, required=True,
                    help='checkpoint file name')
parser.add_argument('--sp', default=False, action='store_true',
                    help='Use single process for pytorch')  #单一pytorch处理
parser.add_argument('--detector', dest='detector',          #人物框检测器
                    help='detector name', default="yolo")
parser.add_argument('--detfile', dest='detfile',
                    help='detection result file', default="")
parser.add_argument('--indir', dest='inputpath',
                    help='image-directory', default="")
parser.add_argument('--list', dest='inputlist',
                    help='image-list', default="")
parser.add_argument('--image', dest='inputimg',
                    help='image-name', default="")
parser.add_argument('--outdir', dest='outputpath',
                    help='output-directory', default="examples/res/")
parser.add_argument('--save_img', default=False, action='store_true',   #action:对象将命令行参数与动作相关联
                    help='save result as image')
parser.add_argument('--vis', default=False, action='store_true',
                    help='visualize image')
parser.add_argument('--showbox', default=False, action='store_true',
                    help='visualize human bbox')
parser.add_argument('--profile', default=False, action='store_true',
                    help='add speed profiling at screen output')
parser.add_argument('--format', type=str,
                    help='save in the format of cmu or coco or openpose, option: coco/cmu/open')
parser.add_argument('--min_box_area', type=int, default=0,
                    help='min box area to filter out')
parser.add_argument('--detbatch', type=int, default=5,
                    help='detection batch size PER GPU')
parser.add_argument('--posebatch', type=int, default=80,
                    help='pose estimation maximum batch size PER GPU')
parser.add_argument('--eval', dest='eval', default=False, action='store_true',  #输出结果保存为coco数据结果形式
                    help='save the result json as coco format, using image index(int) instead of image name(str)')
parser.add_argument('--gpus', type=str, dest='gpus', default="0",
                    help='choose which cuda device to use by index and input comma to use multi gpus, e.g. 0,1,2,3. (input -1 for cpu only)')
parser.add_argument('--qsize', type=int, dest='qsize', default=1024,
                    help='the length of result buffer, where reducing it will lower requirement of cpu memory')
parser.add_argument('--flip', default=False, action='store_true',
                    help='enable flip testing')
parser.add_argument('--debug', default=False, action='store_true',
                    help='print detail information')
"""----------------------------- Video options -----------------------------"""
parser.add_argument('--video', dest='video',
                    help='video-name', default="")
parser.add_argument('--webcam', dest='webcam', type=int,
                    help='webcam number', default=-1)
parser.add_argument('--save_video', dest='save_video',
                    help='whether to save rendered video', default=False, action='store_true')
parser.add_argument('--vis_fast', dest='vis_fast',
                    help='use fast rendering', action='store_true', default=False)
"""----------------------------- Tracking options -----------------------------"""
parser.add_argument('--pose_flow', dest='pose_flow',
                    help='track humans in video with PoseFlow', action='store_true', default=False)
parser.add_argument('--pose_track', dest='pose_track',
                    help='track humans in video with reid', action='store_true', default=False)

args = parser.parse_args()  
                #ArgumentParser 通过 parse_args() 方法解析参数。它将检查命令行，把每个参数转换为适当的类型然后调用相应的操作。
                # parse_args() 会被不带参数调用，而 ArgumentParser 将自动从 sys.argv 中确定命令行参数。
cfg = update_config(args.cfg)   #加载配置文件

# if platform.system() == 'Windows':      #操作系统类型
#     args.sp = True

args.gpus = [int(i) for i in args.gpus.split(',')] if torch.cuda.device_count() >= 1 else [-1]
args.device = torch.device("cuda:" + str(args.gpus[0]) if args.gpus[0] >= 0 else "cpu")
args.detbatch = args.detbatch * len(args.gpus)
args.posebatch = args.posebatch * len(args.gpus)
args.tracking = args.pose_track or args.pose_flow or args.detector=='tracker'

#cuda处理
if not args.sp:
    torch.multiprocessing.set_start_method('forkserver', force=True)    #使用forkserver模式启动子程序cuda
    torch.multiprocessing.set_sharing_strategy('file_system')       #设置共享CPU张量的策略

#检测数据源
def check_input():
    # for wecam
    if args.webcam != -1:
        args.detbatch = 1
        return 'webcam', int(args.webcam)       #返回摄像头参数值，既返回摄像头的系统代号

#     # for video
#     if len(args.video):
#         if os.path.isfile(args.video):
#             videofile = args.video
#             return 'video', videofile
#         else:
#             raise IOError('Error: --video must refer to a video file, not directory.')

# ##*****
#     # for detection results
#     if len(args.detfile):
#         if os.path.isfile(args.detfile):
#             detfile = args.detfile
#             return 'detfile', detfile
#         else:
#             raise IOError('Error: --detfile must refer to a detection json file, not directory.')

#     # for images
#     if len(args.inputpath) or len(args.inputlist) or len(args.inputimg):
#         inputpath = args.inputpath
#         inputlist = args.inputlist
#         inputimg = args.inputimg

#         if len(inputlist):
#             im_names = open(inputlist, 'r').readlines()
#         elif len(inputpath) and inputpath != '/':
#             for root, dirs, files in os.walk(inputpath):
#                 im_names = files
#             im_names = natsort.natsorted(im_names)
#         elif len(inputimg):
#             args.inputpath = os.path.split(inputimg)[0]
#             im_names = [os.path.split(inputimg)[1]]

#         return 'image', im_names

#     else:
#         raise NotImplementedError


def print_finish_info():
    print('===========================> Finish Model Running.',file=sys.stderr)
    if (args.save_img or args.save_video) and not args.vis_fast:
        print('===========================> Rendering remaining images in the queue...',file=sys.stderr)
        print('===========================> If this step takes too long, you can enable the --vis_fast flag to use fast rendering (real-time).',file=sys.stderr)


# def loop():
#     n = 0
#     while True:
#         yield n
#         n += 1
def loop():
    times = 1
    while times>0:
        yield times
        times+=1
        time.sleep(0.5)

if __name__ == "__main__":
    #输入源赋值
    mode, input_source = check_input()#wecam
    
    #结果保存文件路径
    if not os.path.exists(args.outputpath):
        os.makedirs(args.outputpath)

    # Load detection loader
    # 内服程序检测执行
    if mode == 'webcam':
        det_loader = WebCamDetectionLoader(input_source, get_detector(args), cfg, args)
        det_worker = det_loader.start()
    elif mode == 'detfile':
        det_loader = FileDetectionLoader(input_source, cfg, args)
        det_worker = det_loader.start()
    else:
        det_loader = DetectionLoader(input_source, get_detector(args), cfg, args, batchSize=args.detbatch, mode=mode, queueSize=args.qsize)
        det_worker = det_loader.start()

    # Load pose model
    # 加载检测模型
    pose_model = builder.build_sppe(cfg.MODEL, preset_cfg=cfg.DATA_PRESET)

    print('Loading pose model from %s...' % (args.checkpoint,),file=sys.stderr)
    pose_model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
    pose_dataset = builder.retrieve_dataset(cfg.DATASET.TRAIN)
    if args.pose_track:
        tracker = Tracker(tcfg, args)
    if len(args.gpus) > 1:
        pose_model = torch.nn.DataParallel(pose_model, device_ids=args.gpus).to(args.device)    #使用多个GPU来并行训练。
    else:
        pose_model.to(args.device)
    pose_model.eval()       #声明测试，return self.train(False)

    runtime_profile = {
        'dt': [],
        'pt': [],
        'pn': []
    }

    # Init data writer
    queueSize = 2 if mode == 'webcam' else args.qsize   #缓存区长度，wecam时使用2倍qsize

    # #数据源为固定video或者image
    # if args.save_video and mode != 'image':
    #     from alphapose.utils.writer import DEFAULT_VIDEO_SAVE_OPT as video_save_opt
    #     if mode == 'video': #视频预测输出
    #         video_save_opt['savepath'] = os.path.join(args.outputpath, 'AlphaPose_' + os.path.basename(input_source))
    #     else:   #摄像头预测输出
    #         video_save_opt['savepath'] = os.path.join(args.outputpath, 'AlphaPose_webcam' + str(input_source) + '.mp4')
    #     video_save_opt.update(det_loader.videoinfo)
    #     writer = DataWriter(cfg, args, save_video=False, video_save_opt=video_save_opt, queueSize=queueSize).start()
    # else:      #图像数据输出
    writer = DataWriter(cfg, args, save_video=False, queueSize=queueSize).start()
    # print("write type: ",type(writer))
    # #读取输出dict结果中的keypoint
    # keypoints = writer['keypoints']
    # #按xy进行分组
    # keypoint_group=[]
    # for i in range(0, len(keypoints), 2):
    #     keypoint_group.append(keypoints[i:i+2])
    # print(type(keypoint_group))





    if mode == 'webcam':
        print('Starting webcam demo, press Ctrl + C to terminate...',file=sys.stderr)
        sys.stdout.flush()
        im_names_desc = tqdm(loop())        #tqdm基于迭代对象运行
    else:
        data_len = det_loader.length
        im_names_desc = tqdm(range(data_len), dynamic_ncols=True)

    batchSize = args.posebatch
    if args.flip:
        batchSize = int(batchSize / 2)
    try:
        for i in im_names_desc:
            start_time = getTime()
            with torch.no_grad():
                (inps, orig_img, im_name, boxes, scores, ids, cropped_boxes) = det_loader.read()
                if orig_img is None:
                    break
                if boxes is None or boxes.nelement() == 0:
                    writer.save(None, None, None, None, None, orig_img, im_name)
                    continue
                if args.profile:
                    ckpt_time, det_time = getTime(start_time)
                    runtime_profile['dt'].append(det_time)
                # Pose Estimation
                inps = inps.to(args.device)
                datalen = inps.size(0)
                leftover = 0
                if (datalen) % batchSize:
                    leftover = 1
                num_batches = datalen // batchSize + leftover
                hm = []
                for j in range(num_batches):
                    inps_j = inps[j * batchSize:min((j + 1) * batchSize, datalen)]
                    if args.flip:
                        inps_j = torch.cat((inps_j, flip(inps_j)))
                    hm_j = pose_model(inps_j)
                    if args.flip:
                        hm_j_flip = flip_heatmap(hm_j[int(len(hm_j) / 2):], pose_dataset.joint_pairs, shift=True)
                        hm_j = (hm_j[0:int(len(hm_j) / 2)] + hm_j_flip) / 2
                    hm.append(hm_j)
                hm = torch.cat(hm)
                if args.profile:
                    ckpt_time, pose_time = getTime(ckpt_time)
                    runtime_profile['pt'].append(pose_time)
                if args.pose_track:
                    boxes,scores,ids,hm,cropped_boxes = track(tracker,args,orig_img,inps,boxes,hm,cropped_boxes,im_name,scores)
                hm = hm.cpu()
                writer.save(boxes, scores, ids, hm, cropped_boxes, orig_img, im_name)
                if args.profile:
                    ckpt_time, post_time = getTime(ckpt_time)
                    runtime_profile['pn'].append(post_time)

            if args.profile:
                # TQDM
                im_names_desc.set_description(
                    'det time: {dt:.4f} | pose time: {pt:.4f} | post processing: {pn:.4f}'.format(
                        dt=np.mean(runtime_profile['dt']), pt=np.mean(runtime_profile['pt']), pn=np.mean(runtime_profile['pn']))
                )
        print_finish_info()
        while(writer.running()):
            time.sleep(1)
            print('===========================> Rendering remaining ' + str(writer.count()) + ' images in the queue...',file=sys.stderr)
        writer.stop()
        det_loader.stop()
    except Exception as e:
        print(repr(e),file=sys.stderr)
        print('An error as above occurs when processing the images, please check it',file=sys.stderr)
        pass
    except KeyboardInterrupt:
        print_finish_info()
        # Thread won't be killed when press Ctrl+C
        if args.sp:
            det_loader.terminate()
            while(writer.running()):
                time.sleep(1)
                print('===========================> Rendering remaining ' + str(writer.count()) + ' images in the queue...',file=sys.stderr)
            writer.stop()
        else:
            # subprocesses are killed, manually clear queues

            det_loader.terminate()
            writer.terminate()
            writer.clear_queues()
            det_loader.clear_queues()

    end = time.time()
    print(end-start,file=sys.stderr)