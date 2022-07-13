import cv2
import pafy
import numpy as np
import time

from csflow import CSFlow

# Initialize model
model_path='models/csflow_things_iter20_480x640.onnx'
flow_estimator = CSFlow(model_path)

FLOW_FRAME_OFFSET = 10 # Number of frame difference to estimate the optical flow

# Initialize video
# cap = cv2.VideoCapture("input.mp4")

videoUrl = 'https://youtu.be/3wdsE1UgP6k'
videoPafy = pafy.new(videoUrl)
print(videoPafy.streams)
cap = cv2.VideoCapture(videoPafy.streams[-1].url)

# Skip first {start_time} seconds
start_time = 5
cap.set(cv2.CAP_PROP_POS_FRAMES, start_time*30)

cv2.namedWindow("Estimated CSFlow", cv2.WINDOW_NORMAL)
frame_list = []
frame_num = 0

# rec = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc(*"MJPG"), 30.0,
#                       (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)*0.50), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)*0.50)))
# print(rec.isOpened())

while cap.isOpened():

    try:
        # Read frame from the video
        ret, prev_frame = cap.read()
        frame_list.append(prev_frame)
        if not ret:
            break
    except:
        continue

    # Skip the first frames to be able to
    frame_num += 1
    if frame_num <= FLOW_FRAME_OFFSET:
        continue

    t = time.time()
    flow_map = flow_estimator(frame_list[0], frame_list[-1])
    dt = time.time() - t
    print(f"Inference: {1 / dt:.3f} FPS")
    flow_img = flow_estimator.draw_flow()

    alpha = 0.5
    combined_img = cv2.addWeighted(frame_list[0], alpha, flow_img, (1-alpha),0)
    # combined_img = np.hstack((frame_list[-1], flow_img))

    cv2.imshow("Estimated CSFlow", combined_img)
    # rec.write(cv2.resize(combined_img, None, fx=0.50, fy=0.50))

    # Remove the oldest frame
    frame_list.pop(0)

    # Press key q to stop
    if cv2.waitKey(1) == ord('q'):
        break

# rec.release()
cap.release()
cv2.destroyAllWindows()
