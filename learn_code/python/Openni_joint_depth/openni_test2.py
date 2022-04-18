import numpy as np
import cv2
from openni import openni2
from openni import _openni2 as c_api

openni2.initialize()     # can also accept the path of the OpenNI redistribution

dev = openni2.Device.open_any()
print (dev.get_device_info())

depth_stream = dev.create_depth_stream()
depth_stream.start()
frame = depth_stream.read_frame()
frame_data = frame.get_buffer_as_uint16()
depth_stream.stop()

openni2.unload()


#output
# Warning: USB events thread - failed to set priority. This might cause loss of data...
#OniDeviceInfo(uri = b'2bc5/0403@1/8', vendor = b'Orbbec', name = b'Astra', usbVendorId = 11205, usbProductId = 1027)
