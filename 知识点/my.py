# 读取图片
import cv2 as cv
import numpy as np
from matplotlib import pyplot as pl
# 同目录下可以用相对路径
img = cv.imread("Photos/cat.jpg")
cv.imshow("cat",img)
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow("gray",gray)
# 图像分割
# threshold是阈值的意思,type(切割类型)是THRESH_BINARY为二值类型，返回阈值和阈值图，下划线的意思是不读取阈值（黑色展示）
_,threshold=cv.threshold(gray,thresh=150,maxval=255,type=cv.THRESH_BINARY)
cv.imshow("threshold",threshold)
# 与上面相反色调展示（白色）,表示使用反向二值化方法进行阈值处理
_,threshold1=cv.threshold(gray,thresh=150,maxval=255,type=cv.THRESH_BINARY_INV)
cv.imshow("threshold",threshold1)

# 自适应分割（亮度对自适应分割的影响较小，主要是轮廓）,adapyiveMethod（表示使用高斯加权方法计算自适应阈值），bockSize（高斯核大小）表示用于计算阈值的区域大小为 11x11 像素
# ，c（对比度阈值：局部区域的最小值和最大值之间的差值（通常为正数））c 的值在这里表示常数 C，用于调整高斯加权方法的权重，自适应分割可用在文字检测上
threshold2=cv.adaptiveThreshold(gray,maxValue=255,
                     adaptiveMethod=cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                     thresholdType=cv.THRESH_BINARY_INV,
                     blockSize=11,
                     C=9
                     )
cv.imshow("threshold2",threshold2)

# 平移 dimensions(是一个包含图像宽度和高度的元组)为图像的维度,x,y为平移的量
img = cv.imread("Photos\park.jpg")
cv.imshow("park",img)
def translate(img,x,y):
    #平移矩阵
    tmat = np.array([[1,0,x],
                     [0,1,y]],dtype=np.float32)
    # 0为行（高），1为列（宽），这个代码将宽设置为第一维度，宽设置为第二维度,
    # 这样做的目的是在进行某些操作时，需要将图像的形状调整为特定的顺序，例如在卷积神经网络中，通常要求输入数据的通道数在前，高度在后，宽度在中间。
    dimensions = (img.shape[1],img.shape[0])
    # warpAffine包裹映射,使用cv.warpAffine函数将图像按照平移矩阵进行平移，并返回平移后的图像。
    return cv.warpAffine(img,tmat,dimensions)

timg=translate(img,100,100)
cv.imshow("park_translate",timg)






# cv.waitKey(0)


# # 读取视频
# dog = cv.VideoCapture("Resources/Videos/dog.mp4")
# # 读取视频帧
# while True:
#     isTRUE,frame=dog.read()
#     if isTRUE:
#         cv.imshow("dog",frame)
#         # 进入if语句视频帧等待20ms进入下一帧，如果按下d退出循环（视频）（&是与的意思，两者为真才会退出循环break）0xFF为十六进制
#         if cv.waitKey(20) & 0xFF==ord('d'):
#             break
#     else:
#         break
# #释放内存和窗口
# dog.release()
# cv.destroyALLWindows()

# import numpy as np
# # 500×500，三通道（彩色图,矩阵内每个元素为三个值（b,g,r）），dtype设置矩阵内数据格式
# blank = np.zeros((500,500,3),dtype="uint8")
# cv.imshow('blank',blank)
# # 切片0到100行，0到200列=bgr(blue,green,red)
# blank[0:100,0:200] = 255,0,0
# blank[0:100,200:400]=0,255,0
# cv.imshow('rgb',blank)
# # rectangle为矩形，矩形图要确定左上角的点和右下角的点,线条的颜色（彩色图，所以要（b,g,r））线条的粗细
# cv.rectangle(blank,(0,0),(400,400),(0,0,255),thickness=2)
# cv.imshow('rectangle',blank)
#
# # 圆，需要中心和半径，圆的颜色，线条粗细
# cv.circle(blank,(400,400),40,(0,255,0),thickness=4)
# cv.imshow('circle',blank)
#
# # 线段，要起始点和终止点，颜色，粗细
# cv.line(blank,(200,200),(500,500),(255,255,255),thickness=4)
# cv.imshow('line',blank)
#
# # 写文字，要内容，文字框左上角坐标(列，行)，文字字体
# cv.putText(blank,"hello",(0,200),cv.FONT_HERSHEY_TRIPLEX,1.0,(0,255,0),2)
# cv.imshow('putText',blank)
#
# cv.waitKey(0)

# img = cv.imread("Resources/Photos/park.jpg")
# cv.imshow('park',img)
# # # cv.COLOR_BGR2GRAY可以写成6
# # gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
# # gray=cv.cvtColor(img,6)
# # cv.imshow('gray',gray)
# #高斯模糊(去噪点)，7×7矩阵，矩阵中间值大，呈现山峰状
# blur=cv.GaussianBlur(img,(7,7),cv.BORDER_DEFAULT)
# cv.imshow('blur', blur)
# #边缘检测，需要设置两个阈值
# 小于125的不是边缘，大于125小于175（大于175的为边缘，根据连通性得到）和大于175的为边缘，得到一个二值图
# canny = cv.Canny(blur,125,175)
# cv.imshow('canny',canny)
# # canny为得到的二值图，可以直接膨胀，图像的膨胀和腐蚀,需要结构块，下列为7×7的结构块，iterations为膨胀或腐蚀次数
# dilate = cv.dilate(canny,(7,7),iterations=3)
# cv.imshow('dilate',dilate)
#
# erode= cv.erode(canny,(7,7),iterations=3)
# cv.imshow('erode',erode)
#
# # 重新定义尺寸，差值
# resize=cv.resize(img,(500,500))
# cv.imshow('resize',resize)
#
# # 剪切，用数组切片来处理
# crop=img[50,200,200:400]
# cv.show('crop',crop)



cv.waitKey(0)





























