import numpy as np

# Computer Vision
import cv2

from openni import openni2  # , nite2
from openni import _openni2 as c_api

# dist = '..\\astra_program\\OpenNI_2.3.0.63\\Windows\\Astra OpenNI2 Development Instruction(x64)_V1.3\\OpenNI2\\OpenNI-Windows-x64-2.3.0.63\Redist'
## Initialize openni and check
openni2.initialize()  #
if (openni2.is_initialized()):
    print("openNI2 initialized")
else:
    print("openNI2 not initialized")


## Register the device
dev = openni2.Device.open_any()

## Create the streams stream
rgb_stream = dev.create_color_stream()
depth_stream = dev.create_depth_stream()

## Configure the depth_stream -- changes automatically based on bus speed
# print 'Depth video mode info', depth_stream.get_video_mode() # Checks depth video configuration
depth_stream.set_video_mode(c_api.OniVideoMode(pixelFormat=c_api.OniPixelFormat.ONI_PIXEL_FORMAT_DEPTH_1_MM, resolutionX=640, resolutionY=480, fps=30))

## Check and configure the mirroring -- default is True
## Note: I disable mirroring
# print 'Mirroring info1', depth_stream.get_mirroring_enabled()
#设置关闭镜像
depth_stream.set_mirroring_enabled(False)
rgb_stream.set_mirroring_enabled(False)

## More infor on streams depth_ and rgb_
help(depth_stream)
print(depth_stream.get_max_pixel_value())
print(depth_stream.get_sensor_info())
## Start the streams
rgb_stream.start()
depth_stream.start()

## Synchronize the streams
#同步数据流
dev.set_depth_color_sync_enabled(True)  # synchronize the streams

## IMPORTANT: ALIGN DEPTH2RGB (depth wrapped to match rgb stream)
dev.set_image_registration_mode(openni2.IMAGE_REGISTRATION_DEPTH_TO_COLOR)

def get_rgb():
    """
  Returns numpy 3L ndarray to represent the rgb image.
  """
    # bgr = np.fromstring(rgb_stream.read_frame().get_buffer_as_uint8(), dtype=np.uint8).reshape(480, 640, 3)
    # rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    bgr = np.fromstring(rgb_stream.read_frame().get_buffer_as_uint8(), dtype=np.uint8).reshape(480, 640, 3)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


    # cv2.imshow('test', rgb)
    # cv2.waitKey()
    return rgb
# get_rgb()
def get_depth():
    """
  Returns numpy ndarrays representing the raw and ranged depth images.
  Outputs:
      dmap:= distancemap in mm, 1L ndarray, dtype=uint16, min=0, max=2**12-1
      d4d := depth for dislay, 3L ndarray, dtype=uint8, min=0, max=255
  Note1:
      fromstring is faster than asarray or frombuffer
  Note2:
      .reshape(120,160) #smaller image for faster response
              OMAP/ARM default video configuration
      .reshape(240,320) # Used to MATCH RGB Image (OMAP/ARM)
              Requires .set_video_mode
  """
    # dmap = np.fromstring(depth_stream.read_frame().get_buffer_as_uint16(), dtype=np.uint16).reshape(480, 640)  # Works & It's FAST original
    #距离，单位毫米
    dmap = np.fromstring(depth_stream.read_frame().get_buffer_as_uint8(), dtype=np.uint16).reshape(480, 640)  # Works & It's FAST
    #深度信息，三维
    d4d = np.uint8(dmap.astype(float) * 255 / 2 ** 12 - 1)  # Correct the range. Depth images are 12bits
    d4d = 255 - cv2.cvtColor(d4d, cv2.COLOR_GRAY2RGB)
    return dmap, d4d
while True:
    frame = get_rgb()
    dmap, depth = get_depth()
    frame_depth = frame.copy()
    for i in range(1, 6):
        x_seed = 128
        duration = 64
        frame_depth[:, x_seed*i-(x_seed - duration):x_seed*i - 1, :] = depth[:, x_seed*i-(x_seed - duration):x_seed*i - 1, :]

    numpy_horizontal = np.hstack((frame, depth, frame_depth))   #hstack：沿着水平方向将数组堆叠起来。
    cv2.imshow('test', numpy_horizontal)

    if cv2.waitKey(1)>0: break


openni2.unload()