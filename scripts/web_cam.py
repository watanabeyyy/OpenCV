import cv2

capture = cv2.VideoCapture(1)
capture.set(3, 640)  # Width
capture.set(4, 480)  # Heigh
capture.set(5, 15)   # FPS

if capture.isOpened() is False:
    raise("IO Error")

cv2.namedWindow("Capture", cv2.WINDOW_AUTOSIZE)

while True:

    ret, image = capture.read()

    if ret == False:
        continue

    cv2.imshow("Capture", image)

    if cv2.waitKey(33) >= 0:
        cv2.imwrite("image.png", image)
        break

capture.release()
cv2.destroyAllWindows()