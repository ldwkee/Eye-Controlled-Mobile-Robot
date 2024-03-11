#parallel.py
import subprocess

# Paths to your scripts
script_paths = ["/home/chakri/catkin_ws/src/term2/src/pala/eyes_detection_1.py", "/home/chakri/catkin_ws/src/term2/src/pala/motor_control_1.py", "/home/chakri/catkin_ws/src/term2/src/pala/obstacle_avoidance_1.py"]

processes = []

for script in script_paths:
    # Start each script as a subprocess
    processes.append(subprocess.Popen(['python3', script]))

# Wait for all processes to finish
for proc in processes:
    proc.wait()
