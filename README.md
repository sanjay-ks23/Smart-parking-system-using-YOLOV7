# Smart-parking-system-using-YOLOV7

YOLOv7 Smart Car Parking System
This project uses YOLOv7 object detection algorithm to detect and recognize vehicles in a parking lot, and keep track of the available parking spaces.

# Problem at hand
As a part of Smart Parking system, the problem is to identify available parking spots in the parking areas given bird’s eye view images of the parking lots captured by the cameras using Computer Vision and Image Processing.

Prerequisites
Python 3.x
PyTorch 1.8 or higher
OpenCV 4.x or higher

Installation

# Clone this repository:

git clone https://github.com/sanjay-ks23/yolov7_smart_car_parking_system.git

# Implementation

Download the pre-trained YOLOv7 weights file from the official repository:

wget https://github.com/WongKinYiu/yolov7/raw/master/assets/yolov7.pt

# Run the detect.py script to start the parking system:

python detect.py --weights yolov7.pt --source images/run1.jpg --classes 

# Run the detect1.py script to start the parking system:

The system will start capturing the video feed from the source (either a camera or a pre-recorded video file), and detecting the vehicles in the parking lot. The available parking spaces will be marked with green bounding boxes, while the occupied spaces will be marked with red bounding boxes. The number of available and occupied spaces will be displayed on the screen.

To exit the system, press q on the keyboard.
