import cv2
import numpy as np
import sys

def convert_to_grayscale(color_img):
    if len(color_img.shape) == 3:
        R_channel,G_channel,B_channel = color_img[:, :, 0],color_img[:, :, 1],color_img[:, :, 2]
        grayscale_image = ((0.2989 * R_channel) + (0.5870 * G_channel) + (0.1140 * B_channel))
        return grayscale_image.astype(np.uint8) 
    else:
        return color_img

def gaussian_kernel(size, sigma):
    kernel = np.fromfunction(lambda x, y: (1 / (2 * np.pi * sigma**2)) * np.exp(-((x - (size - 1) / 2)**2 + (y - (size - 1) / 2)**2) / (2 * sigma**2)), (size, size))
    return kernel / np.sum(kernel)

def apply_kernel(image, kernel):
    """Applies convolutional kernal"""
    kernel_size = kernel.shape[0]
    pad_size = kernel_size // 2
    padded_image = np.pad(image, pad_size, mode='constant')
    output = np.zeros_like(image)
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded_image[i:i + kernel_size, j:j + kernel_size]
            output[i, j] = np.sum(region * kernel)
    return output

def sobel_filter(gray_image, axis):
    """Sobel kernals according to lecture slides"""
    sobel_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
        ])

    sobel_y = np.array([
        [ 1,  2,  1],
        [ 0,  0,  0],
        [-1, -2, -1]
        ])

    if axis == 'x':
        kernel = sobel_x
    elif axis == 'y':
        kernel = sobel_y

    rows, cols = gray_image.shape
    sobel_image = np.zeros_like(gray_image)
    kernel_size = 3
    pad = kernel_size // 2

    padded_image = np.pad(gray_image, ((pad, pad), (pad, pad)), mode='constant', constant_values=0)

    # Convolution
    for i in range(pad, rows + pad):
        for j in range(pad, cols + pad):
            region = padded_image[i - pad:i + pad + 1, j - pad:j + pad + 1]
            sobel_value = np.sum(region * kernel)
            sobel_image[i - pad, j - pad] = np.abs(sobel_value)

    # Normalize
    sobel_image = np.clip(sobel_image, 0, 255).astype(np.uint8)

    return sobel_image

def non_max_suppression(gradient_strength, gradient_direction):
    suppressed_image = np.zeros_like(gradient_strength)
    angle = gradient_direction * 180. / np.pi
    angle[angle < 0] += 180

    for i in range(1, gradient_strength.shape[0] - 1):
        for j in range(1, gradient_strength.shape[1] - 1):
            q = 255
            r = 255

            if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                q = gradient_strength[i, j + 1]
                r = gradient_strength[i, j - 1]
            elif 22.5 <= angle[i, j] < 67.5:
                q = gradient_strength[i + 1, j - 1]
                r = gradient_strength[i - 1, j + 1]
            elif 67.5 <= angle[i, j] < 112.5:
                q = gradient_strength[i + 1, j]
                r = gradient_strength[i - 1, j]
            elif 112.5 <= angle[i, j] < 157.5:
                q = gradient_strength[i - 1, j - 1]
                r = gradient_strength[i + 1, j + 1]

            if gradient_strength[i, j] >= q and gradient_strength[i, j] >= r:
                suppressed_image[i, j] = gradient_strength[i, j]
            else:
                suppressed_image[i, j] = 0

    return suppressed_image

def edge_tracking(img, weak, strong):
    for i in range(1, img.shape[0] - 1):
        for j in range(1, img.shape[1] - 1):
            if img[i, j] == weak:
                if strong in [img[i + 1, j - 1], img[i + 1, j], img[i + 1, j + 1], 
                              img[i, j - 1],     img[i, j + 1], 
                              img[i - 1, j - 1], img[i - 1, j], img[i - 1, j + 1]]:
                    img[i, j] = strong
                else:
                    img[i, j] = 0
    return img

def canny_edge_detect(img_path):
    print("Canny edge detection started...")
    output_path = img_path.rsplit(".", 1)[0] + "_edge.png"
    # Load the image
    image = cv2.imread(img_path)

    # Step 1: Convert to grayscale
    gray_image = convert_to_grayscale(image)

    # Step 2: Apply Gaussian filter
    gaussian_k = gaussian_kernel(5, sigma=1.2)
    gaussian_blured_img = apply_kernel(gray_image, gaussian_k)

    # Step 3: Gradient estimation
    gradient_x = sobel_filter(gaussian_blured_img, 'x')
    gradient_y = sobel_filter(gaussian_blured_img, 'y')

    # Step 4: Calculate gradient strength and direction
    gradient_strength = np.hypot(gradient_x, gradient_y)
    gradient_direction = np.arctan2(gradient_y, gradient_x)
    # cv2.imwrite('img/gradient_strength.png', gradient_strength.astype(np.uint8))

    # Step 5: Non maxima suppression
    nms_image = non_max_suppression(gradient_strength, gradient_direction)
    # cv2.imwrite('img/non_maxima_suppressed.png', nms_image.astype(np.uint8))

    # Step 6: Double threshold
    high_threshold = nms_image.max() * 0.2
    low_threshold = high_threshold * 0.5

    double_threshold_img = np.zeros_like(nms_image)
    strong_pixel = 255
    weak_pixel = 75

    strong_i, strong_j = np.where(nms_image >= high_threshold)
    weak_i, weak_j = np.where((nms_image <= high_threshold) & (nms_image >= low_threshold))

    double_threshold_img[strong_i, strong_j] = strong_pixel
    double_threshold_img[weak_i, weak_j] = weak_pixel
    # cv2.imwrite('img/double_threshold.png', double_threshold_img.astype(np.uint8))

    # Step 7: Edge tracking 

    tracked_edges = edge_tracking(double_threshold_img, weak_pixel,strong_pixel)
    cv2.imwrite(output_path, tracked_edges.astype(np.uint8))

    print("Canny edge detection executed successfully.")
    print("Final image saved into -> ",output_path)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Please provide image path")
        print("Example Usage: python <fiel_name.py> <image_path/name>")
        sys.exit()
    
    canny_edge_detect(sys.argv[1])