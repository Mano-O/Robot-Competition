#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import Int16, Int16MultiArray

forward_distance = right_distance = left_distance = yaw = None

def ultrasonics_callback(msg):
    global forward_distance, right_distance, left_distance
    rospy.loginfo('got ultrasonic data')

    forward_distance = msg.data[0]
    right_distance = msg.data[1]
    left_distance = msg.data[2]
    rospy.loginfo("Forward: %d, Right: %d, Left: %d",
                      forward_distance, right_distance, left_distance)

def yaw_callback(msg):
    global yaw
    rospy.loginfo('got yaw data')
    yaw = msg.data
    rospy.loginfo("Yaw: %d", yaw)

def listener():
    rospy.init_node('robot')
    rospy.Subscriber('ultrasonics', Int16MultiArray, ultrasonics_callback)
    rospy.Subscriber('yaw', Int16, yaw_callback)
    rospy.spin()

def movement():
    cmd = Twist()
    if forward_distance == None:
        cmd.linear.x = 0
        cmd.angular.z = 0
    elif forward_distance > 5:
        # velocity directly proportional to distance, the second term makes the rbot stop at 5cm
        velocity = forward_distance * 0.3 - 5 * 0.3 # to stop at 5cm
        # velocity = max(min(velocity, 5), 5)  # clamp speed to 5
        cmd.linear.x = velocity
    elif left_distance > 11:
        while(yaw > -90):
            cmd.angular.z = -0.3
    else:
        while(yaw < 90):
            cmd.angular.z = 0.3
    pub.publish(cmd)


if __name__ == "__main__":
    pub = rospy.Publisher('cmd_vel', Twist, queue_size=10)
    listener()

    while not rospy.is_shutdown():
        pub.publish