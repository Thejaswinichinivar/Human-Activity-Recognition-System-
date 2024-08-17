from collections import deque
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt


# Parameters class include important paths and constants
class Parameters:
    def __init__(self):
        self.CLASSES = open("model/action_recognition_kinetics.txt"

                            ).read().strip().split("\n")
        self.ACTION_RESNET = 'model/resnet-34_kinetics.onnx'
        self.VIDEO_PATH = None
        # self.VIDEO_PATH = "test/example1.mp4"
        # SAMPLE_DURATION is maximum deque size
        self.SAMPLE_DURATION = 16
        self.SAMPLE_SIZE = 112


# Initialise instance of Class Parameter
param = Parameters()

# Check if model file exists
if not os.path.isfile(param.ACTION_RESNET):
    raise FileNotFoundError(f"Model file {param.ACTION_RESNET} not found.")

# A Double ended queue to store our frames captured and with time
# old frames will pop out of the deque
captures = deque(maxlen=param.SAMPLE_DURATION)

# Load the human activity recognition model
print("[INFO] loading human activity recognition model...")
try:
    net = cv2.dnn.readNet(model=param.ACTION_RESNET)
except cv2.error as e:
    print(f"[ERROR] Could not load model. {e}")
    exit(1)

print("[INFO] accessing video stream...")
# Take video file as input if given else turn on web-cam
# So, the input should be mp4 file or live web-cam video
vs = cv2.VideoCapture(param.VIDEO_PATH if param.VIDEO_PATH else 0)

# Initialize matplotlib for display
plt.ion()
fig, ax = plt.subplots()


def check_for_quit():
    # Check if 'q' key is pressed to quit
    fig.canvas.flush_events()
    key = plt.waitforbuttonpress(timeout=0.001)
    if key is not None:
        return key
    return False


while True:
    # Loop over and read capture from the given video input
    (grabbed, capture) = vs.read()

    # break when no frame is grabbed (or end if the video)
    if not grabbed:
        print("[INFO] no capture read from stream - exiting")
        break

    # Resize frame and append it to our deque
    capture = cv2.resize(capture, dsize=(550, 400))
    captures.append(capture)

    # Process further only when the deque is filled
    if len(captures) < param.SAMPLE_DURATION:
        continue

    # Now that our captures array is filled we can construct our image blob
    imageBlob = cv2.dnn.blobFromImages(captures, 1.0,
                                       (param.SAMPLE_SIZE, param.SAMPLE_SIZE),
                                       (114.7748, 107.7354, 99.4750),
                                       swapRB=True, crop=True)

    # Manipulate the image blob to make it fit as an input
    imageBlob = np.transpose(imageBlob, (1, 0, 2, 3))
    imageBlob = np.expand_dims(imageBlob, axis=0)

    # Forward pass through model to make prediction
    net.setInput(imageBlob)
    outputs = net.forward()
    # Index the maximum probability
    label = param.CLASSES[np.argmax(outputs)]

    # Show the predicted activity
    cv2.rectangle(capture, (0, 0), (300, 40), (255, 255, 255), -1)
    cv2.putText(capture, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 0, 0), 2)

    # Display it using matplotlib
    ax.clear()
    ax.imshow(cv2.cvtColor(capture, cv2.COLOR_BGR2RGB))
    ax.set_title("Human Activity Recognition")
    plt.draw()
    plt.pause(0.001)

    # Check for 'q' key to break the loop
    if check_for_quit():
        break

vs.release()
plt.ioff()
plt.show()
