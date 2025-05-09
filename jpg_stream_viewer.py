import cv2

# 假设 v.jpg 是由 ffmpeg 持续生成的
while True:
    img = cv2.imread("outputs/tmp/v.jpg")
    if img is not None:
        cv2.imshow("Stream", img)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()