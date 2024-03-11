# motor_control.py
import rospy
from std_msgs.msg import String
from geometry_msgs.msg import Twist

def eye_position_callback(data):
    move = Twist()
    last_position = data.data
    if last_position == "left":
        move.linear.x = 0.1
        move.angular.z = 0.1
    elif last_position == "right":
        move.linear.x = 0.1
        move.angular.z = -0.1
    elif last_position == "center":
        move.linear.x = 0.22
        move.angular.z = 0.0
    elif last_position == "lower-center":
        move.linear.x = 0.1
        move.angular.z = 0.0
    elif last_position == "upper-down":
        move.linear.x = 0.05
        move.angular.z = 0.0
    elif last_position == "down":
        move.linear.x = 0.0
        move.angular.z = 0.0
    else:
        move.linear.x = 0.0
        move.angular.z = 0.0

    pub.publish(move)

def listener():
    rospy.init_node('motor_control', anonymous=True)
    rospy.Subscriber("eye_position", String, eye_position_callback)
    global pub
    pub = rospy.Publisher('cmd_vel', Twist, queue_size=10)
    rospy.spin()

if __name__ == '__main__':
    listener()
