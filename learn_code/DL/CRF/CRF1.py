#coding=gbk


#https://blog.csdn.net/weixin_42181588/article/details/89341563


import numpy as np
import pydensecrf.densecrf as dcrf
# try:
#     from cv2 import imread, imwrite
# except ImportError:
#     # ���û�а�װOpenCV��������skimage
    # from skimage.io import imread, imsave
    # imwrite = imsave
# from cv2 import imread, imwrite
from skimage.io import imread, imsave       #pip install scikit-image
imwrite = imsave
from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral, create_pairwise_gaussian

"""
original_image_path  ԭʼͼ��·��
predicted_image_path  ֮ǰ���Լ���ģ��Ԥ���ͼ��·��
CRF_image_path  ��������CRF����õ��Ľ��ͼ�񱣴�·��
"""
def CRFs(original_image_path,predicted_image_path,CRF_image_path):

    img = imread(original_image_path)

    # ��predicted_image��RGB��ɫת��Ϊuint32��ɫ 0xbbggrr
    anno_rgb = imread(predicted_image_path).astype(np.uint32)
    anno_lbl = anno_rgb[:,:,0] + (anno_rgb[:,:,1] << 8) + (anno_rgb[:,:,2] << 16)

    # ��uint32��ɫת��Ϊ1,2,...
    colors, labels = np.unique(anno_lbl, return_inverse=True)

    # ������predicted_image��ĺ�ɫ��0ֵ�����Ǵ�������𣬱�ʾ��ȷ�����򣬼�����Ϊ�������
    # ��ô��ȡ��ע�����´���
    #HAS_UNK = 0 in colors
    #if HAS_UNK:
    #colors = colors[1:]

    # ������predicted_image��32λ������ɫ��ӳ�䡣
    colorize = np.empty((len(colors), 3), np.uint8)
    colorize[:,0] = (colors & 0x0000FF)
    colorize[:,1] = (colors & 0x00FF00) >> 8
    colorize[:,2] = (colors & 0xFF0000) >> 16

    # ����predicted_image�е�������
    n_labels = len(set(labels.flat))
    #n_labels = len(set(labels.flat)) - int(HAS_UNK) ##����в�ȷ����������һ�д����滻��һ��

    ###########################
    ###     ����CRFģ��     ###
    ###########################
    use_2d = False               
    #use_2d = True   
    ###########################################################   
    ##���Ǻ����ʲô�����2D        
    ##����˵������ͼ��ʹ�ô˿����򵥷�����ʹ��DenseCRF2D�ࡱ
    ##���߻�˵��DenseCRF�������ͨ�ã��Ƕ�ά���ܼ�CRF��
    ##���Ǹ����ҵĲ��Խ��һ�������DenseCRF�Ƚ϶�
    #########################################################33
    if use_2d:                   
        # ʹ��densecrf2d��
        d = dcrf.DenseCRF2D(img.shape[1], img.shape[0], n_labels)

        # �õ�һԪ�ƣ����������ʣ�
        U = unary_from_labels(labels, n_labels, gt_prob=0.2, zero_unsure=None)
        #U = unary_from_labels(labels, n_labels, gt_prob=0.2, zero_unsure=HAS_UNK)## ����в�ȷ����������һ�д����滻��һ��
        d.setUnaryEnergy(U)

        # ����������ɫ�޹ص��������ֻ��λ�ö���
        d.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=dcrf.DIAG_KERNEL,
                              normalization=dcrf.NORMALIZE_SYMMETRIC)

        # ��������ɫ��������������(x,y,r,g,b)
        d.addPairwiseBilateral(sxy=(80, 80), srgb=(13, 13, 13), rgbim=img,compat=10,
                               kernel=dcrf.DIAG_KERNEL,
                               normalization=dcrf.NORMALIZE_SYMMETRIC)
    else:
        # ʹ��densecrf��
        d = dcrf.DenseCRF(img.shape[1] * img.shape[0], n_labels)

        # �õ�һԪ�ƣ����������ʣ�
        U = unary_from_labels(labels, n_labels, gt_prob=0.7, zero_unsure=None)  
        #U = unary_from_labels(labels, n_labels, gt_prob=0.7, zero_unsure=HAS_UNK)## ����в�ȷ����������һ�д����滻��һ��
        d.setUnaryEnergy(U)

        # �⽫��������ɫ�޹صĹ��ܣ�Ȼ��������ӵ�CRF��
        feats = create_pairwise_gaussian(sdims=(3, 3), shape=img.shape[:2])
        d.addPairwiseEnergy(feats, compat=3,kernel=dcrf.DIAG_KERNEL,
                            normalization=dcrf.NORMALIZE_SYMMETRIC)

        # �⽫��������ɫ��صĹ��ܣ�Ȼ��������ӵ�CRF��
        feats = create_pairwise_bilateral(sdims=(80, 80), schan=(13, 13, 13),
                                          img=img, chdim=2)
        d.addPairwiseEnergy(feats, compat=10,
                            kernel=dcrf.DIAG_KERNEL,
                            normalization=dcrf.NORMALIZE_SYMMETRIC)

    ####################################
    ###         ������ͼ���         ###
    ####################################

    # ����5������
    Q = d.inference(5)

    # �ҳ�ÿ����������ܵ���
    MAP = np.argmax(Q, axis=0)

    # ��predicted_imageת������Ӧ����ɫ������ͼ��
    MAP = colorize[MAP,:]
    imwrite(CRF_image_path, MAP.reshape(img.shape))
    print("CRFͼ�񱣴���",CRF_image_path,"!")
if __name__ == "__main__":
    CRFs("original.png","predict.png","predict_CRFs.png")
