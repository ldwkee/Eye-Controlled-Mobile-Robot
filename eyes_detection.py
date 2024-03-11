#eyes_detection.py
import cv2 as cv
import numpy as np
import mediapipe as mp
import math
import time
import matplotlib.pyplot as plt
from collections import Counter
import rospy
from std_msgs.msg import String

mp_face_mesh = mp.solutions.face_mesh

mp_face_mesh = mp.solutions.face_mesh
RIGHT_EYE =[362,382,380,374,373,390,249,263,466,388,387,386,385,384,398]
RIGHT_IRIS = [474,475,476,477]
R_H_LEFT = [362]
R_H_RIGHT = [263]
R_V_UP = [386]
R_V_DOWN = [374]

blink_timer = None

def euclidean_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance

def iris_position_H(iris_center, right_point, left_point):
    center_to_right_dist = euclidean_distance(iris_center, right_point)
    total_distance = euclidean_distance(right_point, left_point)
    ratio = center_to_right_dist/total_distance
    mapped = (ratio - 0.5) * 2
    position = int(mapped * 255)
    #print(position)
    iris_position = ""
    if position > 80:
        iris_position = "left"
    elif -90 < position < 80:
        iris_position = "center"
    else:
        iris_position = "right"
    return iris_position, position

def iris_position_V(iris_center, top_point, bottom_point):
    vertical_distance = euclidean_distance(top_point, bottom_point)
    iris_position = ""
    threshold = 2.0 * vertical_distance
    #print(threshold)
    scaled_threshold = min(255, max(0, int(255 - (threshold / 30) * 255)))
    print(scaled_threshold)

    if scaled_threshold < 40:
        iris_position = "center" #มองไปข้างหน้าเลย
    elif scaled_threshold < 110:
        iris_position = "lower-center" #มองกลางจอ
    elif scaled_threshold < 145:
        iris_position = "upper-down" #มองปุุ่ม F8
    elif scaled_threshold < 220:
        iris_position = "down" #มองเม้าส์แผด
    else:
        iris_position = "closed"
    return iris_position, scaled_threshold

def main():

    rospy.init_node('eyes_detection', anonymous=True)
    pub = rospy.Publisher('eye_position', String, queue_size=10)
    rate = rospy.Rate(10) 

    blink_timer = None
    global vertical_distance
    cap = cv.VideoCapture(0)

    time_list = []
    iris_position_list = []
    filtered_position_list = []
    filter_window_size = 10

    frame_count = 0
    start_time = time.time()
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv.flip(frame, 1)
            rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            img_h, img_w = frame.shape[:2]
            results = face_mesh.process(rgb_frame)
            if results.multi_face_landmarks:
                mesh_points = np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in results.multi_face_landmarks[0].landmark])

                vertical_distance = euclidean_distance(mesh_points[R_V_UP[0]], mesh_points[R_V_DOWN[0]])

                (r_cx, r_cy), r_radius = cv.minEnclosingCircle(mesh_points[RIGHT_IRIS])
                center_right = np.array([r_cx, r_cy], dtype=np.int32)
                cv.circle(frame, center_right, 2, (255, 0, 255), 1, cv.LINE_AA)
                cv.circle(frame, tuple(mesh_points[R_H_LEFT][0]), 1, (255, 0, 0), -1)
                cv.circle(frame, tuple(mesh_points[R_H_RIGHT][0]), 1, (255, 0, 0), -1)
                cv.circle(frame, tuple(mesh_points[R_V_UP][0]), 1, (255, 0, 0), 1)
                cv.circle(frame, tuple(mesh_points[R_V_DOWN][0]), 1, (255, 0, 0), 1)
                #cv.line(frame, tuple(mesh_points[R_V_DOWN][0]), tuple(mesh_points[R_V_UP][0]), (0, 255, 0), 1)
                cv.line(frame, tuple(mesh_points[R_H_RIGHT][0]), tuple(mesh_points[R_H_LEFT][0]), (0, 255, 0), 1)
                cv.line(frame, tuple(mesh_points[R_H_RIGHT][0]), tuple(center_right), (0, 255, 0), 1)
                cv.line(frame, tuple(mesh_points[R_H_LEFT][0]), tuple(center_right), (0, 255, 0), 1)

                iris_pos_h, ratio_h = iris_position_H(center_right, mesh_points[R_H_RIGHT][0], mesh_points[R_H_LEFT][0])
                iris_pos_v, ratio_v = iris_position_V(center_right, mesh_points[R_V_UP][0], mesh_points[R_V_DOWN][0])

                combined_position = ""
                if iris_pos_v == "center" and iris_pos_h == "center":
                    combined_position = "center"
                elif iris_pos_v == "lower-center" and iris_pos_h == "center":
                    combined_position = "lower-center"
                elif iris_pos_v == "upper-down" and iris_pos_h == "center":
                    combined_position = "upper-down"
                elif iris_pos_v == "down" and iris_pos_h == "center":
                    combined_position = "down"
                elif iris_pos_v == "closed":
                    combined_position = "closed"
                elif iris_pos_v == "center" and iris_pos_h == "left":
                    combined_position = "left"
                elif iris_pos_v == "center" and iris_pos_h == "right":
                    combined_position = "right"
                else:
                    combined_position = "center"

                frame_count += 1

                cv.putText(
                frame,
                f"A-RE10 KMUTNB",
                (30, 90),
                cv.FONT_HERSHEY_PLAIN,
                1.2,
                (0, 255, 0),
                1,
                cv.LINE_AA
            )

            # Append the current time and iris position to the lists
            current_time = time.time()
            time_list.append(current_time)
            iris_position_list.append(combined_position)

            # Apply a simple moving average filter to smooth the signal
            if len(iris_position_list) >= filter_window_size:
                window_positions = iris_position_list[-filter_window_size:]
                position_counts = Counter(window_positions)
                most_common_position = position_counts.most_common(1)[0][0]
                filtered_position_list.append(most_common_position)
            else:
                filtered_position_list.append(combined_position)

            pub.publish(filtered_position_list[-1])

            # Plot the filtered iris position in real-time
            plt.clf()
            if combined_position == "closed":
                plt.plot(time_list[-len(filtered_position_list):], [1] * len(filtered_position_list), label='Eyes Closed')
            else:
                plt.plot(time_list[-len(filtered_position_list):], filtered_position_list, label='Filtered Iris Position')

            plt.xlabel('Time')
            plt.ylabel('Iris Position')
            plt.title('Filtered Iris Position')
            plt.legend()
            plt.pause(0.01)

            cv.imshow('Iris Tracking', frame)
            key = cv.waitKey(1)
            if key == ord('q'):
                break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
