from __future__ import absolute_import, division, print_function
import os
import argparse
import time
import numpy as np
import cv2
import sys
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
import errno
import matplotlib.pyplot as plt
from tqdm import tqdm

# Custom imports
# from bts_dataloader import *

# ROS imports
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

# Argument parser configuration
def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield arg

parser = argparse.ArgumentParser(description='BTS PyTorch implementation.', fromfile_prefix_chars='@')
parser.convert_arg_line_to_args = convert_arg_line_to_args

parser.add_argument('--model_name', type=str, help='model name', default='bts_nyu_v2')
parser.add_argument('--encoder', type=str, help='type of encoder, vgg or desenet121_bts or densenet161_bts',
                    default='densenet161_bts')
parser.add_argument('--data_path', type=str, help='path to the data', required=True)
parser.add_argument('--filenames_file', type=str, help='path to the filenames text file', required=True)
parser.add_argument('--input_height', type=int, help='input height', default=480)
parser.add_argument('--input_width', type=int, help='input width', default=640)
parser.add_argument('--max_depth', type=float, help='maximum depth in estimation', default=80)
parser.add_argument('--checkpoint_path', type=str, help='path to a specific checkpoint to load', default='')
parser.add_argument('--dataset', type=str, help='dataset to train on, make3d or nyudepthv2', default='nyu')
parser.add_argument('--do_kb_crop', help='if set, crop input images as kitti benchmark images', action='store_true')
parser.add_argument('--save_lpg', help='if set, save outputs from lpg layers', action='store_true')
parser.add_argument('--bts_size', type=int,   help='initial num_filters in bts', default=512)

if sys.argv.__len__() == 2:
    arg_filename_with_prefix = '@' + sys.argv[1]
    args = parser.parse_args([arg_filename_with_prefix])
else:
    args = parser.parse_args()

model_dir = os.path.dirname(args.checkpoint_path)
sys.path.append(model_dir)
for key, val in vars(__import__(args.model_name)).items():
    if key.startswith('__') and key.endswith('__'):
        continue
    vars()[key] = val

def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})

def get_num_lines(file_path):
    f = open(file_path, 'r')
    lines = f.readlines()
    f.close()
    return len(lines)

class ToTensor(object):
    def __init__(self):
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    def __call__(self, image):
        image = self.to_tensor(image)
        image = self.normalize(image)
        return image
    
    def to_tensor(self, pic):
        if not _is_numpy_image(pic):
            raise TypeError('pic should be ndarray. Got {}'.format(type(pic)))
        
        img = torch.from_numpy(pic.transpose((2, 0, 1)))
        return img

class ImagePreprocessor:
    def __init__(self, args):
        self.args = args
    
    def preprocess_for_test(self, image):
        image = np.asarray(image, dtype=np.float32) / 255.0

        if self.args.do_kb_crop:
            height = image.shape[0]
            width = image.shape[1]
            top_margin = int(height - 512) # 28
            left_margin = int((width - 928) / 2) # 16
            image = image[top_margin:top_margin + 512, left_margin:left_margin + 928, :]
            
        return image
    
class DepthEstimator:
    def __init__(self, params):

        self.params = params
        self.depth_pub = rospy.Publisher('/estimate_depth_image', Image, queue_size=10)
        self.colored_depth_pub = rospy.Publisher('/colored_depth_image', Image, queue_size=10) # 새로운 컬러맵 토픽
        self.bridge = CvBridge()
        self.model = self.initialize_model()
        self.image_sub = rospy.Subscriber("/front_cam/image_raw", Image, self.image_callback)
                                          
    def initialize_model(self):
        model = BtsModel(params=self.params)
        model = torch.nn.DataParallel(model)
        checkpoint = torch.load(self.params.checkpoint_path)
        model.load_state_dict(checkpoint['model'])
        model.eval()
        model.cuda()
        return model
    
    def preprocess_image(self, cv_image):
        # Convert BGR to RGB
        image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        preprocessor = ImagePreprocessor(self.params)
        
        # Apply preprocessing for test mode
        preprocessed_image = preprocessor.preprocess_for_test(image)
        
        # Convert the preprocessed numpy image to tensor
        tensor_converter = ToTensor()
        preprocessed_tensor = tensor_converter(preprocessed_image)
        
        # Add the batch dimension
        preprocessed_tensor = preprocessed_tensor.unsqueeze(0)

        return preprocessed_tensor
    
    def undo_kb_crop(self, cropped_image):

        # Initialize an empty image of the original size
        orig_height, orig_width = 540, 960
        restored_image = np.zeros((orig_height, orig_width), dtype=cropped_image.dtype)
        top_margin = 28
        left_margin = 16
        # Place the cropped image at the right position
        restored_image[top_margin:top_margin + 512, left_margin:left_margin + 928] = cropped_image
        
        return restored_image

    def image_callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            preprocessed_image = self.preprocess_image(cv_image)
            depth_image_np = self.process_image(preprocessed_image)
            
            # Undo crop
            depth_image_np = self.undo_kb_crop(depth_image_np)
            
            # Publish depth image
            self.publish_depth_image(depth_image_np, data.header)
        except CvBridgeError as e:
            print(e)


    def process_image(self, image):
        # Convert the BGR image to RGB
        focal_value = 298.3504
        focal = torch.tensor([focal_value]).float().cuda()

        # Process the image using the loaded depth estimation model
        with torch.no_grad():
            _, _, _, _, depth_est = self.model(image, focal)
            pred_depth = depth_est.cpu().numpy().squeeze()

        return pred_depth

    def publish_depth_image(self, depth_image, header):
        try:
            depth_ros_msg = self.bridge.cv2_to_imgmsg(depth_image, encoding="32FC1")
            depth_ros_msg.header = header
            self.depth_pub.publish(depth_ros_msg)
        except CvBridgeError as e:
            print(e)

if __name__ == '__main__':
    rospy.init_node('depth_estimator', anonymous=False)
    depth_estimator = DepthEstimator(args)
    rospy.spin()