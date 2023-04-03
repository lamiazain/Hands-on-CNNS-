#!/usr/bin/env python
import sys
import message_filters ### ADD THIS
import rospy
import numpy as np
import math
from std_msgs.msg import String

from sensor_msgs.msg import Image as ImageMsg
from cv_bridge import CvBridge, CvBridgeError
import cv2
from geometry_msgs.msg import Twist
import os
import PIL.Image as Image
from NetsArch import loma_data,load_data,Net1,Net2,Net3,generate_csv

          

#------Start of Class
class image_converter:
    def __init__(self):
        self.bridge = CvBridge()
        self.depth = message_filters.Subscriber('/camera/depth/image_rect_raw', ImageMsg)
        self.color = message_filters.Subscriber('/camera/color/image_raw', ImageMsg)

    def callback(self,rgb,depth):
        #give paths to color and depth images
        path_rgb ='Documents/Images/RGB'
        path_d ='Documents/Images/Depth'
        time_d = depth.header.stamp
        time_c = rgb.header.stamp                                      #Get time stamp 


        cv_image_rgb = self.bridge.imgmsg_to_cv2(rgb, "bgr8")          #convert ros message to CV image bgr8 format
        #save images in their specified direectory with their time stamp and corressponding velocities
        cv2.imwrite(os.path.join(path_rgb,''+str(time_c)+'LV{}'.format(LV)+'AV{}'.format(AV)+'.jpeg'),cv_image_rgb)  


        cv_image_depth = self.bridge.imgmsg_to_cv2(depth, "passthrough")
        #depth_array = np.array(cv_image_depth, dtype=np.float32) #convert depth to array since 
        #im=Image.fromarray(depth_array,mode="I")


        rospy.loginfo(cv_image_rgb,"#############",cv_image_depth)
        #rospy.loginfo(depth_array)
        rospy.loginfo(cv_image_rgb.dtype,"#############",cv_image_depth.dtype)
        #rospy.loginfo(depth_array.max())


        #im.save(os.path.join(path_d,''+str(time_d)+'LV{}'.format(LV)+'AV{}'.format(AV)),'png')

        cv2.imwrite(os.path.join(path_d,''+str(time_d)+'LV{}'.format(LV)+'AV{}'.format(AV)+'.png'),cv_image_depth)
  

        #rospy.sleep(0.1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            rospy.signal_shutdown('Quit')
            cv2.destroyAllWindows()



def main():
    img_cvt = image_converter()
    rospy.init_node('image_converter', anonymous=True)
    clear_dir('Documents/Images/RGB')
    clear_dir('Documents/Images/Depth')
    ##upload model#simply convert the initialized model to a CUDA optimized model using model.to(torch.device('cuda')).
    the_model = Net1()
    the_model.load_state_dict(torch.load('TestNet1_0.pt'))
    the_model.to(torch.device("cuda"))

    try:
        vel_sub = message_filters.Subscriber("/mobile_base/commands/velocity", Twist)
        ts = message_filters.ApproximateTimeSynchronizer([img_cvt.color,img_cvt.depth],queue_size=10,0.1)
        #queue size parameter specifies how many sets of messages it should store from each input filter (by timestamp) while waiting for messages to arrive and complete their “set”.
        #time in secs
        ts.registerCallback(img_cvt.callback)   
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")

if __name__ == '__main__':
    main()
