import cv2
import numpy as np


def box_filter(img_arr, n):
    padding = n // 2
    arr = np.zeros((img_arr.shape[0]+padding*2, img_arr.shape[1]+padding*2))
    out = np.zeros((img_arr.shape[0], img_arr.shape[1]), dtype=np.uint8)

    for i in range(img_arr.shape[0]):
        for j in range(img_arr.shape[1]):
            arr[i+padding][j+padding] = img_arr[i][j]
    
    for i in range(padding, arr.shape[0]-padding):
        for j in range(padding, arr.shape[1]-padding):
            sum = 0
            for x in range(0,n):
                for y in range(0,n):
                    sum += arr[i-padding+x][j-padding+y]

            out[i-padding][j-padding] = np.round(sum/(n*n))

    return out


img_array = cv2.imread('lena_grayscale_hq.jpg', cv2.IMREAD_GRAYSCALE)
out_array_1_1 = box_filter(img_array, 3)
out_array_1_2 = box_filter(img_array, 11)
out_array_1_3 = box_filter(img_array, 21)

cv2.imshow("output_1_1", out_array_1_1)
cv2.imshow("output_1_2", out_array_1_2)
cv2.imshow("output_1_3", out_array_1_3)

cv2.waitKey(0)
cv2.destroyAllWindows()


out_array_2_1 = cv2.boxFilter(img_array, 8, (3,3), normalize=True, borderType=cv2.BORDER_CONSTANT)
out_array_2_2 = cv2.boxFilter(img_array, 8, (11,11), normalize=True, borderType=cv2.BORDER_CONSTANT)
out_array_2_3 = cv2.boxFilter(img_array, 8, (21,21), normalize=True, borderType=cv2.BORDER_CONSTANT)

cv2.imshow("output_2_1", out_array_2_1)
cv2.imshow("output_2_2", out_array_2_2)
cv2.imshow("output_2_3", out_array_2_3)

cv2.waitKey(0)
cv2.destroyAllWindows()

abs = []
abs.append(np.abs(out_array_1_1-out_array_2_1))
abs.append(np.abs(out_array_1_2-out_array_2_2))
abs.append(np.abs(out_array_1_3-out_array_2_3))
max = abs[0].sum()
ind = 1
for i in range(1,3):
    if max < abs[i].sum():
        max = abs[i].sum()
        ind = i + 1

cv2.imshow("Difference 1", abs[0])
cv2.imshow("Difference 2", abs[1])
cv2.imshow("Difference 3", abs[2])

cv2.waitKey(0)
cv2.destroyAllWindows()

print("Max absolute difference is", max, "between image number", ind, "s")

###################Q2###################

def separable_box_filter(img_arr, n):
    padding = n // 2
    arr = np.zeros((img_arr.shape[0]+padding*2, img_arr.shape[1]+padding*2))
    arr2 = np.zeros((img_arr.shape[0]+padding*2, img_arr.shape[1]+padding*2))
    out = np.zeros((img_arr.shape[0], img_arr.shape[1]), dtype=np.uint8)

    for i in range(img_arr.shape[0]):
        for j in range(img_arr.shape[1]):
            arr[i+padding][j+padding] = img_arr[i][j]

    for i in range(padding, arr.shape[0]-padding):
        for j in range(padding, arr.shape[1]-padding):
            sum1 = 0
            for x in range(0,n):
                sum1 += arr[i-padding+x][j]
            arr2[i][j] = sum1/n

    for i in range(padding, arr2.shape[0]-padding):
        for j in range(padding, arr2.shape[1]-padding):
            sum2 = 0
            for y in range(0,n):
                sum2 += arr2[i][j-padding+y]

            out[i-padding][j-padding] = np.round((sum2/n))

    return out

out_array_3_1 = separable_box_filter(img_array, 3)
out_array_3_2 = separable_box_filter(img_array, 11)
out_array_3_3 = separable_box_filter(img_array, 21)

cv2.imshow("output_3_1", out_array_3_1)
cv2.imshow("output_3_2", out_array_3_2)
cv2.imshow("output_3_3", out_array_3_3)

cv2.waitKey(0)
cv2.destroyAllWindows()


abs = []
abs.append(np.abs(out_array_3_1-out_array_2_1))
abs.append(np.abs(out_array_3_2-out_array_2_2))
abs.append(np.abs(out_array_3_3-out_array_2_3))
max = abs[0].sum()
ind = 1
for i in range(1,3):
    if max < abs[i].sum():
        max = abs[i].sum()
        ind = i + 1

cv2.imshow("Difference 1", abs[0])
cv2.imshow("Difference 2", abs[1])
cv2.imshow("Difference 3", abs[2])

cv2.waitKey(0)
cv2.destroyAllWindows()

print("Max absolute difference is", max, "between image number", ind, "s")
