# obstacle_avoidance.py
import rospy
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist

def callback(data):
    regions = {
        'right':  min(min(data.ranges[0:143]), 10),
        'front':  min(min(data.ranges[144:287]), 10),
        'left':   min(min(data.ranges[288:431]), 10),
    }
    take_action(regions)

def take_action(regions):
    msg = Twist()
    linear_x = 0
    angular_z = 0

    if regions['front'] < 0.5:
        linear_x = 0.
        angular_z = 0.1

    msg.linear.x = linear_x
    msg.angular.z = angular_z
    pub.publish(msg)

def main():
    global pub
    rospy.init_node('obstacle_avoidance', anonymous=True)
    pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
    rospy.Subscriber('/scan', LaserScan, callback)
    rospy.spin()

if __name__ == '__main__':
    main()
