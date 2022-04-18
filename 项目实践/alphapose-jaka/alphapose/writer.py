import os
import time
from threading import Thread
from queue import Queue

import cv2
import numpy as np
import torch
import torch.multiprocessing as mp

from alphapose.utils.transforms import get_func_heatmap_to_coord
from alphapose.utils.pPose_nms import pose_nms, write_json

import json

DEFAULT_VIDEO_SAVE_OPT = {
    'savepath': 'examples/res/1.mp4',
    'fourcc': cv2.VideoWriter_fourcc(*'mp4v'),
    'fps': 25,
    'frameSize': (640, 480)
}

EVAL_JOINTS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

# 预测数据输出
class DataWriter():
    def __init__(self, cfg, opt, save_video=False,
                 video_save_opt=DEFAULT_VIDEO_SAVE_OPT,
                 queueSize=1024):
        self.cfg = cfg
        self.opt = opt
        self.video_save_opt = video_save_opt

        self.eval_joints = EVAL_JOINTS
        self.save_video = save_video
        self.heatmap_to_coord = get_func_heatmap_to_coord(cfg)
        # initialize the queue used to store frames read from
        # the video file
        if opt.sp:
            self.result_queue = Queue(maxsize=queueSize)
        else:
            self.result_queue = mp.Queue(maxsize=queueSize)

        if opt.save_img:
            if not os.path.exists(opt.outputpath + '/vis'):
                os.mkdir(opt.outputpath + '/vis')

        if opt.pose_flow:
            from trackers.PoseFlow.poseflow_infer import PoseFlowWrapper
            self.pose_flow_wrapper = PoseFlowWrapper(save_path=os.path.join(opt.outputpath, 'poseflow'))

    def start_worker(self, target):
        if self.opt.sp:
            p = Thread(target=target, args=())
        else:
            p = mp.Process(target=target, args=())
        # p.daemon = True
        p.start()
        return p

    def start(self):
        # start a thread to read pose estimation results per frame
        self.result_worker = self.start_worker(self.update)
        return self



    def update(self):
        final_result = []
        norm_type = self.cfg.LOSS.get('NORM_TYPE', None)
        hm_size = self.cfg.DATA_PRESET.HEATMAP_SIZE
        if self.save_video:
            # initialize the file video stream, adapt ouput video resolution to original video
            # 初始化文件视频流，使输出视频分辨率适应原始视频

            #读取写入视频
            stream = cv2.VideoWriter(*[self.video_save_opt[k] for k in ['savepath', 'fourcc', 'fps', 'frameSize']])
            if not stream.isOpened():   
                print("Try to use other video encoders...")
                ext = self.video_save_opt['savepath'].split('.')[-1]
                fourcc, _ext = self.recognize_video_ext(ext)
                self.video_save_opt['fourcc'] = fourcc
                self.video_save_opt['savepath'] = self.video_save_opt['savepath'][:-4] + _ext
                stream = cv2.VideoWriter(*[self.video_save_opt[k] for k in ['savepath', 'fourcc', 'fps', 'frameSize']])
            assert stream.isOpened(), 'Cannot open video for writing'
        # keep looping infinitelyd
        while True:
            # ensure the queue is not empty and get item
            (boxes, scores, ids, hm_data, cropped_boxes, orig_img, im_name) = self.wait_and_get(self.result_queue)
            if orig_img is None:
                # if the thread indicator variable is set (img is None), stop the thread
                # 如果线程指示器变量被设置（img为 None），则停止线程。
                if self.save_video:
                    stream.release()
                write_json(final_result, self.opt.outputpath, form=self.opt.format, for_eval=self.opt.eval)
                print("Results have been written to json.")
                return
            # image channel RGB->BGR
            orig_img = np.array(orig_img, dtype=np.uint8)[:, :, ::-1]
            if boxes is None or len(boxes) == 0:
                if self.opt.save_img or self.save_video or self.opt.vis:
                    self.write_image(orig_img, im_name, stream=stream if self.save_video else None)
            
            # 正常数据条件下
            else:     
                # location prediction (n, kp, 2) | score prediction (n, kp, 1)
                # 位置预测（n，kp，2）|得分预测（n，kp，1）
                assert hm_data.dim() == 4
                #pred = hm_data.cpu().data.numpy()

                # 不同人体关节点数对应不同检测方式
                if hm_data.size()[1] == 136:
                    self.eval_joints = [*range(0,136)]
                elif hm_data.size()[1] == 26:
                    self.eval_joints = [*range(0,26)]

                #声明位姿点和预测得分容器
                pose_coords = []
                pose_scores = []
                for i in range(hm_data.shape[0]):       #遍历关节数
                    bbox = cropped_boxes[i].tolist()        #更新裁剪后的框
                    
                    #输出热图得分和坐标
                    pose_coord, pose_score = self.heatmap_to_coord(hm_data[i][self.eval_joints], bbox, hm_shape=hm_size, norm_type=norm_type)
                    pose_coords.append(torch.from_numpy(pose_coord).unsqueeze(0))
                    pose_scores.append(torch.from_numpy(pose_score).unsqueeze(0))
                # 用cat将当前帧坐标点和得分拼接起来得到当前图像的预测数据
                preds_img = torch.cat(pose_coords)
                preds_scores = torch.cat(pose_scores)
                # nms位姿筛选
                if not self.opt.pose_track: 
                    boxes, scores, ids, preds_img, preds_scores, pick_ids = \
                        pose_nms(boxes, scores, ids, preds_img, preds_scores, self.opt.min_box_area)

                # # 输出关节key值字典
                # print(type(preds_img))     #list
                # #对单张图片单人关节点输出点分组，两个一组
                # keypoint_group=[]
                # for i in range(0, len(preds_img), 2):
                #     keypoint_group.append(preds_img[i:i+2])
                # # 转换为字典
                # keypoint_dict=dict(enumerate(keypoint_group))
                # #preds_img=keypoint_group
                # print("keypoint_dict type :",type(keypoint_dict))
                


                # 打印结果
                # 结果容器
                _result = []
                xy = []
                # xy_list=[]
                for k in range(len(scores)):
                    _result.append(
                        {
                            'keypoints':preds_img[k],
                            'kp_score':preds_scores[k],
                            'proposal_score': torch.mean(preds_scores[k]) + scores[k] + 1.25 * max(preds_scores[k]),
                            #'idx':ids[k],
                            #'box':[boxes[k][0], boxes[k][1], boxes[k][2]-boxes[k][0],boxes[k][3]-boxes[k][1]] 
                        }
                    )
                #     # xy_list[k]=preds_img[k].numpy().tolist()  #tensor 转list
                #     np.append(xy,preds_img[k].numpy().tolist())
                #     xy = np.array(xy)
                # #print("preds_img type ",type(preds_img))
                # print("xy_shape:",xy.shape)


                result = {
                    'imgname': im_name,
                    'result': _result
                }
                # result2 = {
                #     "imgname":im_name,
                #     'result':xy
                # }


                if self.opt.pose_flow:
                    poseflow_result = self.pose_flow_wrapper.step(orig_img, result)
                    for i in range(len(poseflow_result)):
                        result['result'][i]['idx'] = poseflow_result[i]['idx']

                # 最终结果
                final_result.append(result)
                # print("final_result type :",type(final_result))


                #输出json
                # list 转成Json格式数据
                # def listToJson(lst):
                #     keys = [str(x) for x in np.arange(len(lst))]
                #     list_json = dict(zip(keys, lst))
                #     str_json = json.dumps(list_json, indent=2, ensure_ascii=False)  # json转为string
                #     return str_json
                # result_json = listToJson(final_result)

                # def change_type(byte):    
                #     if isinstance(byte,bytes):
                #         return str(byte,encoding="utf-8")  
                #     return json.JSONEncoder.default(byte)
                # result_json = json.dumps(final_result,cls=change_type,indent=4)

                # result_json = json.dumps(final_result)
                # with open("./record1.json","w") as f:        #打开用于读写的文件
                #     json.dump(result_json,f)
                #     print("当前帧加载入文件完成...")

                #输出摄像头预测结果到json
                write_json(final_result, self.opt.outputpath, form=self.opt.format, for_eval=self.opt.eval)
                print("Results have been written to json.")
                # 对应的图像视频保存操作
                if self.opt.save_img or self.save_video or self.opt.vis:
                    if hm_data.size()[1] == 49:
                        from alphapose.utils.vis import vis_frame_dense as vis_frame
                    elif self.opt.vis_fast:
                        from alphapose.utils.vis import vis_frame_fast as vis_frame
                    else:
                        from alphapose.utils.vis import vis_frame
                    img = vis_frame(orig_img, result, self.opt) # 包含原始图像和图片预测结果
                    self.write_image(img, im_name, stream=stream if self.save_video else None)
        
            #return final_result
            print('end final_result: ',type(final_result))
            # print('xy type: ',type(xy))
            f=open("/home/cxking/AlphaPose/k4_xy.txt","w")
            for line in final_result:
                f.write(str(line))

            # jsObj = json.dumps(result2)
            # f = open('/home/cxking/AlphaPose/k3.json', 'w')  
            # # fileObject.write(jsObj) 
            # for line in jsObj:
            #     f.write(str(line)) 
            # f.close()  

    def write_image(self, img, im_name, stream=None):
        if self.opt.vis:
            cv2.imshow("AlphaPose Demo", img)
            cv2.waitKey(30)
        if self.opt.save_img:
            cv2.imwrite(os.path.join(self.opt.outputpath, 'vis', im_name), img)
        if self.save_video:
            stream.write(img)
            

    def wait_and_put(self, queue, item):
        queue.put(item)

    def wait_and_get(self, queue):
        return queue.get()

    def save(self, boxes, scores, ids, hm_data, cropped_boxes, orig_img, im_name):
        # save next frame in the queue
        self.wait_and_put(self.result_queue, (boxes, scores, ids, hm_data, cropped_boxes, orig_img, im_name))

    def running(self):
        # indicate that the thread is still running
        return not self.result_queue.empty()

    def count(self):
        # indicate the remaining images
        return self.result_queue.qsize()

    def stop(self):
        # indicate that the thread should be stopped
        self.save(None, None, None, None, None, None, None)
        self.result_worker.join()

    def terminate(self):
        # directly terminate
        self.result_worker.terminate()

    def clear_queues(self):
        self.clear(self.result_queue)
        
    def clear(self, queue):
        while not queue.empty():
            queue.get()

    def results(self):
        # return final result
        print(self.final_result)
        return self.final_result

    def recognize_video_ext(self, ext=''):
        if ext == 'mp4':
            return cv2.VideoWriter_fourcc(*'mp4v'), '.' + ext
        elif ext == 'avi':
            return cv2.VideoWriter_fourcc(*'XVID'), '.' + ext
        elif ext == 'mov':
            return cv2.VideoWriter_fourcc(*'XVID'), '.' + ext
        else:
            print("Unknow video format {}, will use .mp4 instead of it".format(ext))
            return cv2.VideoWriter_fourcc(*'mp4v'), '.mp4'
