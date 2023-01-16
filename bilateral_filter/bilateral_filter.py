import cv2
import numpy as np


###################Q1###################

print("###################Q1###################")


def adaptive_mean_filter(img_arr, v):
    img8 = np.uint8(img_arr)
    normalized = cv2.normalize(img8, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    padding = 5//2#Sxy=5x5
    out = np.zeros((img_arr.shape[0], img_arr.shape[1]))
    
    arr = cv2.copyMakeBorder(normalized, padding, padding, padding, padding, cv2.BORDER_REPLICATE)

    for i in range(padding, arr.shape[0]-padding):
        for j in range(padding, arr.shape[1]-padding):
            a = []
            for x in range(0,5):
                for y in range(0,5):
                    a.append(arr[i-padding+x][j-padding+y])
            var = np.var(a)
            avr = np.average(a)

            out[i-padding][j-padding] = arr[i][j]-((v/var)*(arr[i][j]-avr))
    
    k = cv2.normalize(out, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return np.uint8(k)

img_array = cv2.imread("noisyImage_Gaussian.jpg", cv2.IMREAD_GRAYSCALE)

normalized = cv2.normalize(np.uint8(img_array), None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    

output_1 = cv2.blur(normalized, (3,3), borderType=cv2.BORDER_REPLICATE)
output_2 = cv2.blur(normalized, (5,5), borderType=cv2.BORDER_REPLICATE)
output_3 = cv2.GaussianBlur(normalized,(3,3),0)
output_4 = cv2.GaussianBlur(normalized,(5,5),0)
output_5 = adaptive_mean_filter(img_array, 0.0042)## normalization is done inside the method then converted back
output_6 = cv2.bilateralFilter(normalized, 5, 3, 0.9, borderType = cv2.BORDER_REPLICATE)

output_1 = np.uint8(cv2.normalize(output_1, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F))
output_2 = np.uint8(cv2.normalize(output_2, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F))
output_3 = np.uint8(cv2.normalize(output_3, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F))
output_4 = np.uint8(cv2.normalize(output_4, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F))
output_6 = np.uint8(cv2.normalize(output_6, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F))

clean = cv2.imread("lena_grayscale_hq.jpg", cv2.IMREAD_GRAYSCALE)

psnr_1 = cv2.PSNR(output_1, clean)
psnr_2 = cv2.PSNR(output_2, clean)
psnr_3 = cv2.PSNR(output_3, clean)
psnr_4 = cv2.PSNR(output_4, clean)
psnr_5 = cv2.PSNR(output_5, clean)
psnr_6 = cv2.PSNR(output_6, clean)

print("PSNR of opencv box filter(3x3):", psnr_1)
print("PSNR of opencv box filter(5x5):", psnr_2)
print("PSNR of opencv Gaussian filter(3x3):", psnr_3)
print("PSNR of opencv Gaussian filter(5x5):", psnr_4)
print("PSNR of adaptive mean filter:", psnr_5)
print("PSNR of opencv bilateral filter:", psnr_6)

cv2.imshow("output_1 psnr "+str(psnr_1), output_1)
cv2.imshow("output_2 psnr "+str(psnr_2), output_2)
cv2.imshow("output_3 psnr "+str(psnr_3), output_3)
cv2.imshow("output_4 psnr "+str(psnr_4), output_4)
cv2.imshow("output_5 psnr "+str(psnr_5), output_5)
cv2.imshow("output_6 psnr "+str(psnr_6), output_6)
cv2.waitKey(0)
cv2.destroyAllWindows


###################Q2###################

print("###################Q2###################")

img_array = cv2.imread("noisyImage_Gaussian_01.jpg", cv2.IMREAD_GRAYSCALE)

normalized = cv2.normalize(np.uint8(img_array), None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    

output_7 = cv2.blur(normalized, (3,3), borderType=cv2.BORDER_REPLICATE)
output_8 = cv2.blur(normalized, (5,5), borderType=cv2.BORDER_REPLICATE)
output_9 = cv2.GaussianBlur(normalized,(3,3),0)
output_10 = cv2.GaussianBlur(normalized,(5,5),0)
output_11 = adaptive_mean_filter(img_array, 0.0009)## normalization is done inside the method then converted back
output_12 = cv2.bilateralFilter(normalized, 3, 0.1, 1, borderType = cv2.BORDER_REPLICATE)

output_7 = np.uint8(cv2.normalize(output_7, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F))
output_8 = np.uint8(cv2.normalize(output_8, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F))
output_9 = np.uint8(cv2.normalize(output_9, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F))
output_10 = np.uint8(cv2.normalize(output_10, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F))
output_12 = np.uint8(cv2.normalize(output_12, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F))

clean = cv2.imread("lena_grayscale_hq.jpg", cv2.IMREAD_GRAYSCALE)

psnr_7 = cv2.PSNR(output_7, clean)
psnr_8 = cv2.PSNR(output_8, clean)
psnr_9 = cv2.PSNR(output_9, clean)
psnr_10 = cv2.PSNR(output_10, clean)
psnr_11 = cv2.PSNR(output_11, clean)
psnr_12 = cv2.PSNR(output_12, clean)

print("PSNR of opencv box filter(3x3):", psnr_7)
print("PSNR of opencv box filter(5x5):", psnr_8)
print("PSNR of opencv Gaussian filter(3x3):", psnr_9)
print("PSNR of opencv Gaussian filter(5x5):", psnr_10)
print("PSNR of adaptive mean filter:", psnr_11)
print("PSNR of opencv bilateral filter:", psnr_12)

cv2.imshow("output_1 psnr "+str(psnr_7), output_7)
cv2.imshow("output_2 psnr "+str(psnr_8), output_8)
cv2.imshow("output_3 psnr "+str(psnr_9), output_9)
cv2.imshow("output_4 psnr "+str(psnr_10), output_10)
cv2.imshow("output_5 psnr "+str(psnr_11), output_11)
cv2.imshow("output_6 psnr "+str(psnr_12), output_12)
cv2.waitKey(0)
cv2.destroyAllWindows

###################Q3###################

print("###################Q3###################")

def bilateral_filter(img_arr, n, sigmaColor, sigmaSpace):
    normalized = cv2.normalize(np.uint8(img_arr), None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    padding = n//2
    out = np.zeros((img_arr.shape[0], img_arr.shape[1]))
    
    arr = cv2.copyMakeBorder(normalized, padding, padding, padding, padding, cv2.BORDER_REPLICATE)

    for i in range(padding, arr.shape[0]-padding):
        for j in range(padding, arr.shape[1]-padding):
            a = 0
            wp_total = 0
            for x in range(0,n):
                for y in range(0,n):
                    euc = (np.sqrt(np.abs((i-(i-padding+x))**2+(j-(j-padding+y))**2)))
                    G_s = (1.0/(sigmaSpace*np.sqrt(2*np.pi)))*np.exp(-((euc*euc)/(2*sigmaSpace*sigmaSpace)))
                    G_r = (1.0/(sigmaColor*np.sqrt(2*np.pi))) * np.exp(-(((arr[i-padding+x][j-padding+y]-arr[i][j])*(arr[i-padding+x][j-padding+y]-arr[i][j]))/(2*sigmaColor*sigmaColor)))
                    wp = G_s*G_r
                    a += arr[i-padding+x][j-padding+y]*wp
                    wp_total += wp
            out[i-padding][j-padding] = a/wp_total
    
    k = cv2.normalize(out, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return np.uint8(k)


img_array = cv2.imread("noisyImage_Gaussian.jpg", cv2.IMREAD_GRAYSCALE)
normalized = cv2.normalize(img_array, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
output_3_1_1 = bilateral_filter(img_array, 5, 3, 0.9)## normalization is done inside the method and converted back
output_3_1_2 = cv2.bilateralFilter(normalized, 5, 3, 0.9, borderType = cv2.BORDER_REPLICATE)
output_3_1_2 = np.uint8(cv2.normalize(output_3_1_2, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F))

img_array2 = cv2.imread("noisyImage_Gaussian_01.jpg", cv2.IMREAD_GRAYSCALE)
normalized2 = cv2.normalize(img_array2, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
output_3_2_1 = bilateral_filter(img_array2, 3, 0.1, 1)## normalization is done inside the method and converted back
output_3_2_2 = cv2.bilateralFilter(normalized2, 3, 0.1, 1, borderType = cv2.BORDER_REPLICATE)
output_3_2_2 = np.uint8(cv2.normalize(output_3_2_2, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F))

clean = cv2.imread("lena_grayscale_hq.jpg", cv2.IMREAD_GRAYSCALE)

psnr_3_1_1 = cv2.PSNR(output_3_1_1, clean)
psnr_3_1_2 = cv2.PSNR(output_3_1_2, clean)
psnr_3_2_1 = cv2.PSNR(output_3_2_1, clean)
psnr_3_2_2 = cv2.PSNR(output_3_2_2, clean)

print("PSNR of my bilateral filter:", psnr_3_1_1)
print("PSNR of opencv bilateral filter:", psnr_3_1_2)
print("PSNR of my bilateral filter(01):", psnr_3_2_1)
print("PSNR of opencv bilateral filter(01):", psnr_3_2_2)
print()
print("Max absolute difference between first bilateral filters on noisyImage_Gaussian:", np.max(np.abs(output_3_1_2.astype(float)-output_3_1_1.astype(float))))
print("Max absolute difference between second bilateral filters on noisyImage_Gaussian_01:", np.max(np.abs(output_3_2_2.astype(float)-output_3_2_1.astype(float))))


cv2.imshow("output_3_1_1 psnr "+str(psnr_3_1_1), output_3_1_1)
cv2.imshow("output_3_1_2 psnr "+str(psnr_3_1_2), output_3_1_2)
cv2.imshow("output_3_2_1 psnr "+str(psnr_3_2_1), output_3_2_1)
cv2.imshow("output_3_2_2 psnr "+str(psnr_3_2_2), output_3_2_2)
cv2.waitKey(0)
cv2.destroyAllWindows
