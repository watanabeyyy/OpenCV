import cv2
import pyocr.builders
from PIL import Image
import numpy as np

tools = pyocr.get_available_tools()
tool = tools[0]

capture = cv2.VideoCapture(1)
capture.set(3, 128)  # Width
capture.set(4, 64)  # Heigh
capture.set(5, 15)   # FPS

cv2.namedWindow("Capture", cv2.WINDOW_AUTOSIZE)
ret, image = capture.read()

while True:

    ret, image = capture.read()

    if ret == False:
        continue

    cv2.imshow("Capture", image)

    image = Image.fromarray(np.uint8(image))
    res = tool.image_to_string(image)
    print(res)

    if cv2.waitKey(33) >= 0:
        break

capture.release()
cv2.destroyAllWindows()