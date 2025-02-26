import cv2
import depthai as dai
import numpy as np
import time
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../ORB_SLAM3/lib"))

from ORB_SLAM3 import System

# WLS parameters
WLS_LAMBDA = 8000
WLS_SIGMA = 1.0

class Pose:
    def __init__(self):
        self.position = None
        self.rotation = None

def create_pipeline():
    pipeline = dai.Pipeline()

    mono_left = pipeline.create(dai.node.MonoCamera)
    mono_right = pipeline.create(dai.node.MonoCamera)
    stereo = pipeline.create(dai.node.StereoDepth)
    xout_rectif_left = pipeline.create(dai.node.XLinkOut)
    xout_rectif_right = pipeline.create(dai.node.XLinkOut)
    xout_disp = pipeline.create(dai.node.XLinkOut)

    xout_rectif_left.setStreamName("rectified_left")
    xout_rectif_right.setStreamName("rectified_right")
    xout_disp.setStreamName("disparity")

    mono_left.setBoardSocket(dai.CameraBoardSocket.LEFT)
    mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
    mono_left.setFps(20.0)
    mono_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
    mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
    mono_right.setFps(20.0)

    stereo.setOutputRectified(True)
    stereo.setOutputDepth(False)
    stereo.setRectifyEdgeFillColor(0)
    stereo.setRectifyMirrorFrame(False)
    stereo.setExtendedDisparity(True)

    mono_left.out.link(stereo.left)
    mono_right.out.link(stereo.right)
    stereo.rectifiedLeft.link(xout_rectif_left.input)
    stereo.rectifiedRight.link(xout_rectif_right.input)
    stereo.disparity.link(xout_disp.input)

    return pipeline

def main():
    print("OAK-D/ORB_SLAM3 Experiment")
    
    pipeline = create_pipeline()
    device = dai.Device(pipeline)
    device.startPipeline()
    
    rectif_left_queue = device.getOutputQueue("rectified_left", 8, False)
    rectif_right_queue = device.getOutputQueue("rectified_right", 8, False)
    disp_queue = device.getOutputQueue("disparity", 8, False)
    
    wls_filter = cv2.ximgproc.createDisparityWLSFilterGeneric(False)
    wls_filter.setLambda(WLS_LAMBDA)
    wls_filter.setSigmaColor(WLS_SIGMA)
    
    SLAM = System("ORB_SLAM3/Vocabulary/ORBvoc.txt", "oak_d_orbslam_settings.yaml", System.STEREO, True)
    
    pose = Pose()
    slam_epoch = time.time()
    
    while True:
        rectif_left_frame = rectif_left_queue.get()
        rectif_right_frame = rectif_right_queue.get()
        disp_map_frame = disp_queue.get()
        
        rectif_left = rectif_left_frame.getCvFrame()
        rectif_right = rectif_right_frame.getCvFrame()
        disp_map = disp_map_frame.getCvFrame()
        
        frame_timestamp_s = time.time() - slam_epoch
        print(f"{frame_timestamp_s:.4f}: ", end="")
        
        raw_pose = SLAM.TrackStereo(rectif_left, rectif_right, frame_timestamp_s)
        
        if raw_pose is not None and not raw_pose.empty():
            pose.rotation = raw_pose[:3, :3]
            T = raw_pose[:3, 3]
            pose.position = -np.dot(pose.rotation.T, T)
            print("position:", pose.position.T)
        else:
            print("no pose update")
        
        disp_map = cv2.flip(disp_map, 1)
        filtered_disp_map = wls_filter.filter(disp_map, rectif_right)
        colour_disp = cv2.applyColorMap(filtered_disp_map, cv2.COLORMAP_JET)
        cv2.imshow("disparity", colour_disp)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    SLAM.Shutdown()

if __name__ == "__main__":
    main()
