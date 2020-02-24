# --------------------------------------------------------
# Jetvengers HQ AI & IOT Integration
# ~ Sample gstream integration code reference taken from JK Jung for ~  
# Camera sample code for Tegra X2/X1
# (We tested 
# This program could capture and display video from
# IP CAM, USB webcam, or the Tegra onboard camera.
# Refer to the following blog post for how to set up
# and run the code:
#   https://jkjung-avt.github.io/tx2-camera-with-python/
#
# Tegra Cam written by JK Jung <jkjung13@gmail.com>
# Code customized by Amlan from Jetvengers HQ team
# Modules added: preprocess_frame, displaytext, play_audio,
# gpio_initialize, validate, door_activate
# --------------------------------------------------------


import sys
import argparse
import subprocess
import numpy as np
import cv2
from playsound import playsound
import torch
import torchvision.transforms as transforms
from torch2trt import TRTModule
import time
import os
from os import listdir
from os.path import isfile, join
import Jetson.GPIO as GPIO
import time
WINDOW_NAME = 'Jetvengers House'
device = torch.device('cuda')
counter = 0
label_ctr = 0
model_trt = TRTModule()
model_trt.load_state_dict(torch.load('resnet18_torch2rt_1602.pth'))
labels = ['antman', 'batman', 'captain-america', 'hulk', 'ironman', 'reverse-flash', 'spiderman', 'thor']
mean = np.array([0.485, 0.456, 0.406])
stdev = np.array([0.229, 0.224, 0.225])
normalize = transforms.Normalize(mean, stdev)

label = None
prev_audio = None
prev_label = None
value = None

# for 1st Motor on ENA
ENA = 33
IN1 = 35
IN2 = 37

"Example python3 tegra-cam.py --usb --vid 1 --width 1280 --height 720"
def parse_args():
    # Parse input arguments
    desc = 'Capture and display live camera video on Jetson'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--rtsp', dest='use_rtsp',
                        help='use IP CAM (remember to also set --uri)',
                        action='store_true')
    parser.add_argument('--uri', dest='rtsp_uri',
                        help='RTSP URI, e.g. rtsp://192.168.1.64:554',
                        default=None, type=str)
    parser.add_argument('--latency', dest='rtsp_latency',
                        help='latency in ms for RTSP [200]',
                        default=200, type=int)
    parser.add_argument('--usb', dest='use_usb',
                        help='use USB webcam (remember to also set --vid)',
                        action='store_true')
    parser.add_argument('--vid', dest='video_dev',
                        help='device # of USB webcam (/dev/video?) [1]',
                        default=1, type=int)
    parser.add_argument('--width', dest='image_width',
                        help='image width [1920]',
                        default=1920, type=int)
    parser.add_argument('--height', dest='image_height',
                        help='image height [1080]',
                        default=1080, type=int)
    args = parser.parse_args()
    return args


def open_cam_rtsp(uri, width, height, latency):
    gst_str = ('rtspsrc location={} latency={} ! '
               'rtph264depay ! h264parse ! omxh264dec ! '
               'nvvidconv ! '
               'video/x-raw, width=(int){}, height=(int){}, '
               'format=(string)BGRx ! '
               'videoconvert ! appsink').format(uri, latency, width, height)
    return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)


def open_cam_usb(dev, width, height):
    # We want to set width and height here, otherwise we could just do:
    #     return cv2.VideoCapture(dev)
    gst_str = ('v4l2src device=/dev/video{} ! '
               'video/x-raw, width=(int){}, height=(int){} ! '
               'videoconvert ! appsink').format(dev, width, height)
    return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)


def open_cam_onboard(width, height):
    gst_elements = str(subprocess.check_output('gst-inspect-1.0'))
    if 'nvcamerasrc' in gst_elements:
        # On versions of L4T prior to 28.1, add 'flip-method=2' into gst_str
        gst_str = ('nvcamerasrc ! '
                   'video/x-raw(memory:NVMM), '
                   'width=(int)2592, height=(int)1458, '
                   'format=(string)I420, framerate=(fraction)30/1 ! '
                   'nvvidconv ! '
                   'video/x-raw, width=(int){}, height=(int){}, '
                   'format=(string)BGRx ! '
                   'videoconvert ! appsink').format(width, height)
    elif 'nvarguscamerasrc' in gst_elements:
        gst_str = ('nvarguscamerasrc ! '
                   'video/x-raw(memory:NVMM), '
                   'width=(int)1920, height=(int)1080, '
                   'format=(string)NV12, framerate=(fraction)30/1 ! '
                   'nvvidconv flip-method=2 ! '
                   'video/x-raw, width=(int){}, height=(int){}, '
                   'format=(string)BGRx ! '
                   'videoconvert ! appsink').format(width, height)
    else:
        raise RuntimeError('onboard camera source not found!')
    return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)


def open_window(width, height):
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, width, height)
    cv2.moveWindow(WINDOW_NAME, 0, 0)
    # cv2.setWindowTitle(WINDOW_NAME, 'Standby')

# To convert the frame to a tensor
def preprocess_frame(frame):
    x = frame
    x = cv2.cvtColor(np.float32(x), cv2.COLOR_BGR2RGB)
    x = x.transpose((2, 0, 1))
    x = torch.from_numpy(x).float()
    x = x/255
    # x = resize(x)
    # x = crop(x)
    x = normalize(x)
    x = x[:3,:,:].unsqueeze_(0)	
    x = x.to(device)
    x = x[None, ...]
    return x
# To predict the input frame tensor
def predict_image(tensor):
    output = model_trt(tensor)
    # output = F.log_softmax(output, dim=1)
    prob = torch.exp(output)
    sm = torch.nn.Softmax()
    proba = sm(output)
    top_prob, top_class = prob.topk(1, dim=1)	
    return (proba, top_class)

def displaytxt(label, value, img, font):
    # label = str(label)
    # value = str(value)
    text = [label, "Probability: ", value]
    # text = "Label: " + label + "," + "Prob: " + value
    # cv2.putText(img, str(text), (11, 20), font, 1.0, (32, 32, 32), 4, cv2.LINE_AA)
    img = cv2.putText(img, "Hello", (10, 10), cv2.FONT_HERSHEY_SIMPLEX,  2, (245,10,255), 4, cv2.LINE_AA)
    
    return img

# Plays the voice message
def play_audio(label):
    global prev_audio
    files = [f for f in listdir('audio/') if isfile(join('audio/', f))]
    audiolabel = label + '.mp3'
    #if (prev_audio == audiolabel): 
        #return
    if (audiolabel) in files:
        playsound('audio/' + audiolabel)
    # prev_audio = audiolabel
    # time.sleep(1)

# For sliding the door
def door_activate():
    # initialize EnA, In1 and In2
    GPIO.setup(ENA, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(IN1, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(IN2, GPIO.OUT, initial=GPIO.LOW)

    # Stop
    GPIO.output(ENA, GPIO.HIGH)
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.LOW)
    time.sleep(1)

    # Forward
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    time.sleep(0.38)

    # Stop
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.LOW)
    time.sleep(4)

    # Backward
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)
    time.sleep(0.39)

    # Stop
    GPIO.output(ENA, GPIO.LOW)
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.LOW)
    time.sleep(1)

# Validate input image for necessary action    
def validate(label, value):
    global prev_label
    if (prev_label == label):
       return
    
    elif label == 'batman':    
       play_audio(label)
    elif (value >= 60):
       play_audio(label)
       door_activate()
       prev_label = label
               
def read_cam(cap):
    global counter, label, value
    # show_pred = True
    # full_scrn = False
    # help_text = '"Esc" to Quit, "H" for Help, "F" to Toggle Fullscreen'
    while True:
        _, img = cap.read() # grab the next image frame from camera
        
        cv2.imshow(WINDOW_NAME, img)
            
        if (counter%48) == 0:
            iframe = img	  
            tensor = preprocess_frame(iframe)
            pred = predict_image(tensor)
            label = labels[pred[1]]
            value = round((float(torch.max(pred[0]))*100),4)
            # img = displaytxt(label,str(value),img, font)
            print(pred, "Class: ", label, "Value: ", str(value))
            validate(label, value)
        # cv2.imshow(WINDOW_NAME, img)       
        counter += 1
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
# To initialize the Jetson GPIO pins
def gpio_initialize():
    # set pin numbers to the board's
    GPIO.setmode(GPIO.BOARD)

    # initialize EnA, In1 and In2
    GPIO.setup(ENA, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(IN1, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(IN2, GPIO.OUT, initial=GPIO.LOW) 

def main():
    args = parse_args()
    print('Called with args:')
    print(args)
    print('OpenCV version: {}'.format(cv2.__version__))
    if args.use_rtsp:
        cap = open_cam_rtsp(args.rtsp_uri,
                            args.image_width,
                            args.image_height,
                            args.rtsp_latency)
    elif args.use_usb:
        cap = open_cam_usb(args.video_dev,
                           args.image_width,
                           args.image_height)
    else: # by default, use the Jetson onboard camera
        cap = open_cam_onboard(args.image_width,
                               args.image_height)

    if not cap.isOpened():
        sys.exit('Failed to open camera!')
    gpio_initialize()
    open_window(args.image_width, args.image_height)
    read_cam(cap)
    cap.release()
    cv2.destroyAllWindows()
    GPIO.cleanup()

if __name__ == '__main__':
    main()
