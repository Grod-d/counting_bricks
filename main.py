import cv2
import numpy as np

# чтение пикчи
input_image = cv2.imread("wall.jpg")
input_image = cv2.resize(input_image,(400,400))

# подбор порога
gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
gray_image = cv2.medianBlur(gray_image, 7)
edges = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                              cv2.THRESH_BINARY, 9, 9)

# уничтожение неровностей и текстур
color = cv2.bilateralFilter(input_image, 9, 250, 250)
cartoon = cv2.bitwise_and(color, color, mask=edges)

img_grey = cv2.cvtColor(cartoon,cv2.COLOR_BGR2GRAY)
canny = cv2.Canny(img_grey, 30, 150, 3)
cv2.imshow('canny',canny) # детектор краев

dilated = cv2.dilate(canny, (1, 1), iterations=0)
cv2.imshow('dilate',dilated) # дилатация для повышения чувствительности

(cnt, hierarchy) = cv2.findContours(
    dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
rgb = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB) #построение контура по краевым точкам

cv2.drawContours(rgb, cnt, -1, (0, 255, 0), 2) #set a thresh
cv2.imshow('rgb',rgb)
print("coins in the image : ", len(cnt))
# thresh = 180
#
# #get threshold image
# ret,thresh_img = cv2.threshold(img_grey, thresh, 255, cv2.THRESH_BINARY)
#
# #find contours
# contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#
# #create an empty image for contours
# img_contours = np.zeros(input_image.shape)
#
# # draw the contours on the empty image
# cv2.drawContours(img_contours, contours, -1, (255,255,255), 1)
#
# cv2.imshow('contours', img_contours)

#cv2.imshow("Image", input_image)
#cv2.imshow("Image", gray_image)
cv2.imshow("edges", edges)
#cv2.imshow("Cartoon", cartoon)
cv2.waitKey(0)
cv2.destroyAllWindows()