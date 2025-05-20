import cv2
# 视频逐帧分解为图片

cap = cv2.VideoCapture("Highway.mp4")
cnt = 0
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret:
        filename = f'frame/{cnt}.jpg'
        cv2.imwrite(filename,frame)
        cnt += 1
    else:
        print(f'cnt = {cnt}')
        break
