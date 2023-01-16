import cv2
import numpy as np


img_array = cv2.imread('test1.jpg', cv2.IMREAD_GRAYSCALE)

out_array_1 = np.zeros((img_array.shape[0],img_array.shape[1]), dtype=np.uint8)

h = np.histogram(img_array, bins=256)
hist = h[0]

cdf = np.cumsum(hist)
y = np.zeros((256))

y = np.round((cdf/(256*256))*255)

for i in range(256):
    for j in range(256):
        out_array_1[i][j] = y[img_array[i][j]]


cv2.imshow('output_1', out_array_1)


out_array_2 = cv2.equalizeHist(img_array)
cv2.imshow('output_2', out_array_2)

sub = (abs(out_array_1 - out_array_2))
cv2.imshow('abs_diff_1_2', sub)

total = 0
for i in range(sub.shape[0]):
    for j in range(sub.shape[1]):
        total += sub[i][j]

print("Total absolute difference of 1 and 2:" , total)
cv2.waitKey(0)
cv2.destroyAllWindows()

################Q2################

img_array = cv2.imread('test1.jpg', cv2.IMREAD_GRAYSCALE)

out_array_3 = np.zeros((img_array.shape[0],img_array.shape[1]), dtype=np.uint8)

H = np.zeros((256))

for i in range(img_array.shape[0]):
    for j in range(img_array.shape[1]):
        k = img_array[i][j]
        H[k] = H[k]+1

g_min = 0
for i in range(H.size):
    if H[i] > 0:
        if g_min > H[i]:
            g_min = H[i]

cdf = np.zeros((256))
cdf[0] = H[0]
H_min = 256
for i in range(1,256):
    cdf[i] = cdf[i-1] + H[i]
    if H_min > cdf[i]:
        H_min = cdf[i]

H_min = cdf[g_min]

y = np.zeros((256))

y = np.round(((cdf-H_min)/((256*256)-H_min))*255)

for i in range(256):
    for j in range(256):
        out_array_3[i][j] = y[img_array[i][j]]


cv2.imshow('output_3', out_array_3)

out_array_2 = cv2.equalizeHist(img_array)
cv2.imshow('output_2', out_array_2)
sub = (abs(out_array_2 - out_array_3))
cv2.imshow('abs_diff_2_3', sub)
cv2.waitKey(0)
cv2.destroyAllWindows()

total = 0
for i in range(sub.shape[0]):
    for j in range(sub.shape[1]):
        total += sub[i][j]

print("Total absolute difference of 2 and 3:" , total)