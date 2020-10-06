import argparse
#import logging
import time

import cv2
import numpy as np

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

##logger = logging.getLogger('TfPoseEstimator-WebCam')
#logger.setLevel(logging.DEBUG)
#ch = logging.StreamHandler()
#ch.setLevel(logging.DEBUG)
#formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
#ch.setFormatter(formatter)
#logger.addHandler(ch)

fps_time = 0

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation realtime webcam')
    parser.add_argument('--camera', type=bool, default=False)

    parser.add_argument('--model', type=str, default='mobilenet_v2_large', help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    parser.add_argument('--tensorrt', type=str, default="False",
                        help='for tensorrt process.')
    args = parser.parse_args()
    
    if args.camera:
        cam = cv2.VideoCapture(0)
    else:
        cam = cv2.VideoCapture(input("Enter image relative path(try giving 1.webm): "))
    
    numf = int(cam.get(5))
    fps = 15
    val = numf//fps
    
    out = cv2.VideoWriter("out.avi", cv2.VideoWriter_fourcc(*"MJPG"), int(cam.get(5)), (int(cam.get(3)), int(cam.get(4))))
    #logger.info('cam image=%dx%d' % (image.shape[1], image.shape[0]))

    # print(cam.get(cv2.CAP_PROP_FRAME_COUNT))
    # print(cam.get(3), cam.get(4))
    
    w, h = int(cam.get(3)), int(cam.get(4))

    if w > 0 and h > 0:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h), trt_bool=str2bool(args.tensorrt))
    else:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368), trt_bool=str2bool(args.tensorrt))
    
    
    num = 0
    while cam.isOpened():
        for _ in range(val):
            try:
                ret_val, image = cam.read()
            except:
                break
        num += 1
        if ret_val == False:
            break
        #logger.debug('image process+')
        humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=3)
        image_out = image.copy()
        image_out[:,:,:] = 0 
        #logger.debug('postprocess+')
        image_out = TfPoseEstimator.draw_humans(image_out, humans, imgcopy=False)
        
        #logger.debug('show+')
        # cv2.putText(image,
        #             "FPS: %f" % (1.0 / (time.time() - fps_time)),
        #             (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
        #             (0, 255, 0), 2)
        cv2.imshow('original', image)
        cv2.imshow('out', image_out)
        for _ in range(val):
            out.write(image_out)
        
        if cv2.waitKey(40) == 27:
            break
        #logger.debug('finished+')
    print("check out.avi in current directory")
    out.release()
    cv2.destroyAllWindows()
