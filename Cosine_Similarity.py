import cv2
import numpy as np


def cosine_similarity(x, y):
    numerator = np.dot(x, y)
    denominator = np.linalg.norm(x)*np.linalg.norm(y)

    return numerator/denominator


def wwindow_based_matching(left_img, right_img, disparity_range, kernel_size=5, save_result=True):
    left = cv2.imread(left_img, 0)
    right = cv2.imread(right_img, 0)

    left = left.astype(np.float32)
    right = right.astype(np.float32)

    height, width = left.shape[:2]

    depth = np.zeros((height, width), np.uint8)
    kernel_half = int((kernel_size-1)/2)
    scale = 3

    for y in range(kernel_half, height-kernel_half):
        for x in range(kernel_half, width-kernel_half):
            disparity = 0
            cost_optimal = -1

            for j in range(disparity_range):
                d = x-j
                cost = -1
                if (d-kernel_half) > 0:
                    wp = left[(y-kernel_half):(y+kernel_half)+1,
                              (x-kernel_half):(x+kernel_half)+1]
                    wqd = right[(y-kernel_half):(y+kernel_half)+1,
                                (d-kernel_half):(d+kernel_half)+1]

                    wp_flattened = wp.flatten()
                    wqd_flattend = wqd.flatten()

                    cost = cosine_similarity(wp_flattened, wqd_flattend)

                if cost > cost_optimal:
                    cost_optimal = cost
                    disparity = j

            depth[y, x] = disparity*scale

    if save_result == True:
        print('Saving result...')

        cv2.imwrite(f'window_based_cosine_similarity.png', depth)
        cv2.imwrite(f'window_based_cosine_similarity.png',
                    cv2.applyColorMap(depth, cv2.COLORMAP_JET))

    print('Done.')

    return depth


def gaussian(x, std, muy):
    return 1/np.sqrt(2*3.14*(std**2))+np.exp(-(x-muy)**2/(2*(std)**2))


x = [805, 780, 795, 820, 810, 785, 815, 800, 610, 590, 620, 580, 600, 595, 605, 585]
y = [51000, 49500, 52000, 50500, 48000, 53000, 49000, 50000, 31000, 28000, 33000, 29000, 32000, 27000, 34000, 31500]
x = np.array(x)
y = np.array(y)
muy_x = np.mean(x[0:9])
muy_not_x = np.mean(x[9:])
muy_y = np.mean(y[0:9])
muy_not_y = np.mean(y[9:])
std_x = np.std(x[0:9])
std_not_x = np.std(x[9:])
std_y = np.std(y[0:9])
std_not_y = np.std(x[9:])


p_1_l = gaussian(5.5, 0.4, 6.2)
p_1_w = gaussian(3, 0.3, 2.9)
p_0_l = gaussian(5.5, 0.2, 4.8)
p_0_w = gaussian(3, 0.3, 3.3)
p_l_w = p_0_l*p_0_w*0.5 + p_1_l*p_1_w*0.5
print((p_1_l*p_1_w*0.5)/p_l_w)
