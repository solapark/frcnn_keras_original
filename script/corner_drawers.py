import numpy as np
import cv2
 
def draw_corners(imageA, imageB, src, dst):
    src = src.astype('int')
    dst = dst.astype('int')
    for i, (a, b) in enumerate(zip(src, dst)):
        dot_color = np.random.randint(100, 256, size=(3,)).tolist()

        imageA = cv2.circle(imageA, a, 4, dot_color, -1)
        imageB = cv2.circle(imageB, b, 4, dot_color, -1)

        imageA = cv2.putText(imageA, str(i), a, cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
        imageB = cv2.putText(imageB, str(i), b, cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)

draw_corners(imageA, imageB, src.squeeze(), dst.squeeze())
