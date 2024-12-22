from ultralytics import YOLO
import cv2

model = YOLO("yolo11n-pose.pt")


results = model(source = 0, show = True, save = True)

cv2.waitKey(0)
cv2.destroyAllWindows()