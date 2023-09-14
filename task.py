import cv2
import numpy as np
from random import randrange
from math import pi, fabs
import matplotlib as plt


im_rgb = cv2.imread("wall.jpg", cv2.IMREAD_COLOR)
im_rgb=cv2.resize(im_rgb,(400,400))

# Преобразование входного изображения в оттенки серого для работы MSER с одноканальным изображением
im_gray = cv2.cvtColor(im_rgb, cv2.COLOR_BGR2GRAY)
im_gray = cv2.GaussianBlur(im_gray,(5,5),0)

mser = cv2.MSER_create()

# Выполняя вычисление MSER, он вернет список точек, принадлежащих каждому стабильному региону
regions, _ = mser.detectRegions(im_gray)

# Фильтрация результатов и выбор наилучшего из них
best_center = (-1,-1)
best_radius = int(0)
best_area_coeff = 0
# Проверка всех найденных регионов
for region in regions:
    # Выбор яркого цвета
    color = (randrange(128,255), randrange(128,255), randrange(128,255))
    # Вычисляя выпуклую оболочку вокруг этой области, она будет внешней границей области
    hull = np.int32([cv2.convexHull(region.reshape(-1, 1, 2))])
    # Рисуем каждую выпуклую оболочку синей линией
    cv2.polylines(im_rgb, hull, True, (255,0,0), 2)
    # Заполняем каждую область MSER линиями выбранного случайным образом цвета
    cv2.polylines(im_rgb, np.int32([region]), True, color, 1)
    # Вычисление наименьшей окружности выпуклой оболочки с помощью встроенной функции
    (x,y),radius = cv2.minEnclosingCircle(hull[0])
    center = (int(x),int(y))
   # Вычисление площади области по выпуклой оболочке
    area_hull = cv2.contourArea(hull[0])
    area_circle = pi*radius*radius
    area_coeff = area_hull/area_circle
    # Выбор региона с наилучшим соотношением площадей
    if area_coeff > best_area_coeff:
        best_area_coeff = area_coeff
        best_center = center
        best_radius = int(radius)

if best_area_coeff > 0.9:

    cv2.circle(im_rgb, best_center, best_radius, (0,0,255),5)


cv2.imshow('213',im_rgb)



cv2.waitKey(0)
cv2.destroyAllWindows()
# import cv2
# import matplotlib.pyplot as plt
# from skimage.measure import label, regionprops
#
# image = cv2.imread("wall.jpg")
# rgb = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
#
# gray = cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
#
# # split the image into its BGR components
# (R, G, B) = cv2.split(rgb)
#
# mean = gray+gray*0.1
#
# R[R > mean] = 0
# R[R > 0] = 255
# G = R
# B = R
#
# bit = cv2.merge([R, G, B])
#
# gray = cv2.cvtColor(bit,cv2.COLOR_RGB2GRAY)
# the, bw = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
#
# dilated = cv2.dilate(bw,(15,15),iterations = 2)
#
# eroded = cv2.erode(dilated, (5,5), iterations=2)
#
# plt.imshow(eroded,cmap='gray')
# plt.show()
#
# # # label image regions
# object_labels = label(eroded, connectivity=2, background=1)
#
# font = cv2.FONT_HERSHEY_COMPLEX
# bricks_no = 0
#
# for region in regionprops(object_labels):
#     # take regions with large enough areas
#     print(region.label, region.area)
#     if region.area > 10:
#         bricks_no += 1
#
#         row, col = region.centroid
#         cv2.putText(rgb, str(bricks_no), (int(col), int(row)), font, 0.5, (0, 0, 255))
#
# plt.imshow(rgb)
# plt.show()
#
# print("Number of bricks ", len(regionprops(object_labels)))
# print('Real Number of brick ',bricks_no)