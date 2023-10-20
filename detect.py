# Import the necessary library for computer vision
import cv2

# Access the default camera (0) and set the resolution
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# Read the class names from a file
classNames = []
classFile = 'label.txt'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

# Set the configuration and weight paths for the pre-trained model
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightPath = 'frozen_inference_graph.pb'

# Initialize the deep neural network model
net = cv2.dnn_DetectionModel(weightPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

#Instruction on the terminal on how to stop the program.
print("To close the Webcam, please press 'Q' twice.")

# Infinite loop for capturing and processing frames
while True:
  # Read a frame from the camera
  success, img = cap.read()

  # Detect objects in the frame
  classIds, confs, bbox = net.detect(img, confThreshold=0.5)

  # Check if any objects are detected
  if len(classIds) != 0:
    # Loop through the detected objects and draw bounding boxes
    for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
      cv2.rectangle(img, box, color=(255, 0, 0), thickness=2)
      cv2.putText(img, classNames[classId - 1].upper(), (box[0] + 10, box[1] + 30),cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
      cv2.putText(img, str(round(confidence * 100, 2)), (box[0] + 400, box[1] + 30),cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

    # Display the frame with detected objects
    cv2.imshow("Output", img)
    cv2.waitKey(1)

    # Check for user input to exit the loop
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
      break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
