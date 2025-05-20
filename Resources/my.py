# 读取图片
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
# 代码可以放在python web中的视图中，结合

# # 同目录下可以用相对路径
# img = cv.imread("Resources\Photos\cat.jpg")
# cv.imshow("cat",img)
# gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
# cv.imshow("gray",gray)
# # 图像分割
# # threshold是阈值的意思,type(切割类型)是THRESH_BINARY为二值类型，返回阈值和阈值图，下划线的意思是不读取阈值（黑色展示）
# _,threshold=cv.threshold(gray,thresh=150,maxval=255,type=cv.THRESH_BINARY)
# cv.imshow("threshold",threshold)
# # 与上面相反色调展示（白色）
# _,threshold1=cv.threshold(gray,thresh=150,maxval=255,type=cv.THRESH_BINARY_INV)
# cv.imshow("threshold",threshold1)
#
# # 自适应分割（亮度对自适应分割的影响较小，主要是轮廓）,adapyiveMethod（高斯方法），bockSize（高斯核大小），c（），用在文字检测上
# threshold2=cv.adaptiveThreshold(gray,maxValue=255,
#                      adapyiveMethod=cv.ADAPTIVE_THRESH_GAUSSIAN_C,
#                      thresholdType=cv.THRESH_BINARY_INV,
#                      bockSize=11,
#                      C=9
#                      )
# cv.imshow("threshold2",threshold2)
#
# # 平移 dimensions为图像的维度,x,y为平移的量
img = cv.imread("Photos\park.jpg")
# # print(img.shape)
# cv.imshow("park",img)
# def translate(img,x,y):
#     #平移矩阵
#     tmat = np.array([[1,0,x],
#                      [0,1,y]],dtype=np.float32)
#     # 0为行，1为列，原图的行为第一维度，列为第二维度，此操作将图片的列为第一维度，行为第二维度
#     dimensions = (img.shape[1],img.shape[0])
#     # wrapAffine包裹映射
#     return cv.wrapAffine(img,tmat,dimensions)
#
# timg=translate(img,100,100)
# cv.imshow("park",timg)
#
# # 旋转
# # 角度，旋转中心
# def rotMat(image,ang,point):
#     # 创建旋转矩阵，平面为2D,旋转中心，角度，缩放尺寸
#     mat = cv.getRotationMatrix2D(ang,point)
#     # 0为行，1为列
#     dim= (img.shape[1], img.shape[0])
#     # wrapAffine包裹映射
#     return cv.wrapAffine(img, mat, dim)
# # 逆时针30度
# rimg = rotMat(img,30,(200,200))
# cv.imshow("rimg",rimg)
# # 再次旋转(最好用原图旋转，要不然图片会缺失)
# rrimg = rotMat(rimg,30,(200,200))
# cv.imshow("rrimg",rrimg)

# # 轮廓检测
# # 先对灰度图进行阈值分割(二值图)
# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# _, th=cv.threshold(gray,120,255,cv.THRESH_BINARY_INV)
# cv.imshow('th',th)
# # #先寻找轮廓,返回轮廓和一个不需要的值
# # 参数为图像，模式，方法
# contours,_ = cv.findContours(th,mode=0,method=cv.CHAIN_APPROX_SIMPLE)
# # 再画出轮廓图,轮廓图是在img上画的，参数为原图，轮廓图，-1为（查），颜色（BGR），线宽
# cv.drawContours(img, contours , -1, (0,0,255),2)
# cv.imshow("img1",img)


# 各类颜色空间
# HSV（透明度）,色调，饱和度，亮度
# hsv = cv.cvtColor(img,cv.COLOR_BGR2HSV)
# cv.imshow("hsv",hsv)
# # LAB，不常用
# lab = cv.cvtColor(img,cv.COLOR_BGR2LAB)
# cv.imshow("lab",lab)
# # RGB,三通道彩色图,与BGR不同,排列通道不同，展现的也 不同
# rgb = cv.cvtColor(img,cv.COLOR_BGR2HSV)
# cv.imshow("rgb",rgb)

# # 通道颜色提取
# # 单通道提取
# # 方式1
# # img0=img[:,:,0]
# # 方式2
# b,g,r=cv.split(img)
# # 分别显示为蓝色，绿色，红色
# # 蓝
# img[:, :, 1] = 0
# img[:, :, 2] = 0
# # 绿
# img[:, :, 0] = 0
# img[:, :, 2] = 0
# # 红
# img[:, :, 0] = 0
# img[:, :, 1] = 0
# # 三色结合(就是原图)，中括号中bgr的顺序不同，展现的颜色也不同
# mer=cv.merge([b,g,r])
# cv.imshow('mer',mer)

# # 图像模糊(平滑噪声，锐利点)
# # 平均模糊,点的值为周围的值平均得到
# # 3×3的滤波核
# blur = cv.blur(img, (3,3))
# cv.imshow('blur',blur)
# # 高斯模糊
# # 3×3的滤波核,2为高斯滤波核的均值
# gblur = cv.GaussianBlur(img,(3,3),3)
# cv.imshow('gblur',gblur)
# # 中值模糊,3为中值滤波选3个数字
# mblur = cv.medianBlur(img,3)
# cv.imshow('mblur',mblur)
#
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

# # 剪切，用数组切片来处理
# crop=img[50,200,200:400]
# cv.imshow('crop',crop)

# 人脸检测（级联分类器（文件形式），因为要检测边缘轮廓信息和梯度（用sobel(梯度算子)实现）信息，所以用灰度图）,(这个方法需要机器学习训练，要不缺点太多)
# 检测的下一步就是识别，在数据库中匹配相似图片
img = cv.imread('Photos/lady.jpg')
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
# 将gray图数据格式从uint8转换为float64,dx是是否要对x轴方向检测（微分），1为是，0为否，dy为对y轴
sobelx = cv.Sobel(gray, cv.CV_64F, dx=1, dy=0)
sobely = cv.Sobel(gray, cv.CV_64F, dx=0, dy=1)
cv.imshow("sobelx", sobelx)
cv.imshow("sobely", sobely)

img = cv.imread('Photos/group 1.jpg')
cv.imshow("img", img)
# 将图片缩放，看人脸数量是否增加
# 参数为图片，长宽,也可选择按比例缩放
# img = cv.resize(img,(1200,1200))
# 按比例缩放,img.shape返回宽长两个数字，resize接收的是整数，如果乘的是小数要转换
img = cv.resize(img,(img.shape[1]*2,img.shape[0]*2))
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
# 算梯度图
sobel = cv.Sobel(gray, cv.CV_64F, dx=1, dy=1)
cv.imshow("sobel",sobel)
# 传入级联分类器（分类器（cv2自带）加特征（将模型（特征）扔进箱子（分类器）））(特折由训练得到)
classifier = cv.CascadeClassifier('haar_face.xml')
# 多尺度检测,将符合标准的特征进行检测后返回检测结果
rects = classifier.detectMultiScale(gray)
# 遍历得到二维数组的各行，rect会先变成rects[0],循环成rects[1]，再循环成.........
for rect in rects:
    rect[0]
    cv.rectangle(img, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 0, 255))
# rects得到多行的二维数组，每行表示一个人脸检测框,每行的元素分别代表框的左上角坐标（前两个元素）和长宽（后两个元素）
print(rects[0])
# 画一个矩形框,参数分别代表原图，左上角坐标，右下角坐标，颜色，线宽(rects[0][0]代表第1行第一个数字)
# 可以替换成上面的for循环语句
# cv.rectangle(img,(rects[0][0],rects[0][1]),(rects[0][0]+rects[0][2],rects[0][1]+rects[0][3]),(0,0,255))
cv.imshow('img',img)
# 计算检测到的人脸数量
print(len(rects))

# 检测的下一步就是识别，在数据库中匹配相似图片
# 图像识别



cv.waitKey(0)



# 图像特征
# 明显的才叫图像特征
# 颜色特征（通过直方图描述hist（）），mask代表原图大小相同的黑色矩阵（np.zeros生成0矩阵）中的方框(人脸位置制成白色)（如照片中识别人脸的框）
# 先生成0矩阵
img = cv.imread('Photos/lady.jpg')
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow('lady',gray)
# 创建一个与原图大小相同的空白图像，并将结果存储在变量blank中。
# img.shape[:2]表示提取彩色图片的长，宽，img.shape[:3]表示提取彩色图片的长，宽，通道
# blank = np.zeros((img.shape[0],img.shape[1]),dtype=np.uint8)
blank = np.zeros(img.shape[:2],dtype='uint8')
cv.imshow('blank',blank)
# 生成方框，参数代表在blank上画，左上角，右下角，白色（255），1是线宽为1（-1是填充方框成白色）
mask = cv.rectangle(blank,(233,162),(381,403),100,255,-1)
cv.imshow('blank',blank)
# 用mask生成直方图
# 如果是彩色图要用for显示出三个颜色通道一起对应的直方图，灰度图不用这个操作

# 彩色图示例
# plt.figure()
# plt.title('Color Histogram')
# plt.xlabel('bins')
# plt.ylabel('# of pixels')
# colors=('b','g','r')
# enumerate返回一个元组对象，其中包含两个值，索引和对应元素
# for i,col in enumerate(colors):
# 使用cv.calcHist()函数计算直方图，参数分别为原图、颜色通道、掩码、直方块数量和亮度范围，并将结果存储在变量hist中。
#     # 通过直方图描述hist（），参数为图，颜色通道为i，mask，256个直方块，亮度
#     hist =cv.calcHist([img],[i],mask,[256],[0,256])
# 使用plt.plot()函数绘制直方图，颜色分别为蓝色、绿色和红色，并将结果显示出来
#     plt.plot(hist,color=col)
#     plt.xlim([0,256])
# plt.show()

# 灰度图示例
# 生成白色区域的直方图统计，参数代表灰度图，灰度图通道为0，mask，128个直方块，亮度
# 返回的是直方图的特征向量(128维)（hist）
hist=cv.calvHist([gray],[0],blank,[128],[0,256])
print(hist)
# 特征向量生成直方图
plt.plot(hist)
plt.show()


# 纹理特征也是全局特征，只是一个表面特征

# 形状特征（轮廓＋区域）（也叫hog特征）（结合svm分类器）（形状用梯度表示）
# 灰度化（上面已经实现）
# 归一化（灰度图颜色均衡，这一步不用）
# 计算每个像素点的梯度

# 将图像划分成小cells（块）

#统计每个cell的梯度直方图（通过角度统计个数），得到cell描述子（向量）（sobel算子算x轴，y轴的梯度）
# 将人脸分成四个小方格（cell）

# 将每几个cell组成block，得到block的描述子
# （得到四十维的向量或画四个cell的直方图）












# 图像距离
# 图片与图片相减相当于像素相减，也就是矩阵(每个元素都是一个矩阵)相减，
# 也就是向量（矩阵拉开为向量）相减（欧式距离，各个元素相减平方之后相加再开平方，np.sqrt(np.sum(np.square(gray-gray2)))）
# 维度相同的矩阵不能相减,要把其化成相同维度 gray2=cv.resize(gray2,(gray.shape[1],gray.shape[0]))

# hog距离




# 特征相减一般是在分类器中完成，而不是特征直接相减
# jupter产生文件夹!mkdir 邓子方  mkdir 邓子方/positive mkdir 邓子方/negative
# 分类器 随机数生成random.randint()生成左上角坐标
# jupter代码
# 生成文件夹
# !mkdir 邓子方
# !mkdir 邓子方/positive
# !mkdir 邓子方/negative
# 三维彩图(在原图上截取矩形)，逗号前是长，后面是宽
temp=im[ret[0]:ret[0]+ret[2],ret[1]:ret[1]+ret[3]]
# 存储临时变量temp
imwrite('',temp)
# 生成正样本,保存方框图 查看图片大小（print(temp.shape)）
img = cv.imread('邓子方/group 2.jpg')
gray = cv.cvtColor(img,cv.COLOR_BGR2RGB)
classifier = cv.CascadeClassifier('hear_face.xml')
rexts = classifier.detectMultiScale(gray)
i=0
for rect in rexts:
    i += 1
    temp = img[rect[1]:rect[1]+rect[3],rect[0]:rect[0]+rect[2]]
    print(temp.shape)
    cv.imread(f'邓子方/positive/{i}.jpg',temp)
    plt.imshow(temp)
cv.waitKey(0)
# 图片大小统一

# 生成负样本
import random
for i in range(250):
    x0 = random.randint(0,gray.shape[0]-64)
    y0 = random.randint(0,gray.shape[1]-64)
    temp = img[x0:x0+64,y0:y0+64]
    cv.imwrite(f'邓子方/negative/{i}.jpg',temp)

# 获取单个图片的hog特征
# 定义函数
def hotimg(gray):
    hog = cv.HOGDescriptor()
    feat = hog.computer(gray,winStride=(8,8),padding=(0,0))
    return feat
# 获取单个图片的hog特征


# 获取整个文件夹的hog特征
# 0为正样本，1为负样本
import os
def hotdata(pdir,label):
    # 获取文件夹下所有文件路径（以列表呈现，冒号为隐藏文件）
    paths=os.listdir(pdir)
    print(paths)
hotdata('',1)
















