import numpy as np
import cv2

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    cv2.imshow('frame', frame)
               
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

// Function to detect shapes
def shapeDetection(cap):
    gray_image = cv2.cvtColor(cap, cv2.COLOR_BGR2GRAY)

    threshold = cv2.threshold(gray_image, 220, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for i, contour in enumerate(contours):
        if i == 0:
            continue

        epsilon = 0.01*cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        cv2.drawContours(cap, contour, 0, (0, 0, 255), 5)