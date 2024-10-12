
# Canny Edge Detection from Scratch

This project implements the Canny Edge Detection algorithm from scratch using Python and NumPy, without using advanced OpenCV functions like `cv2.Canny`. The algorithm consists of several steps including grayscale conversion, Gaussian smoothing, Sobel filter application, non-maximum suppression, double thresholding, and edge tracking by hysteresis.

## Features

- Manual Sobel filter implementation without using `cv2.Sobel` or `cv2.filter2D`.
- Custom Gaussian smoothing for noise reduction.
- Full manual convolution for image processing.
- Complete Canny Edge Detection implementation with edge tracking and suppression.

## Installation

1. Clone this repository or download the script.
2. Install the necessary dependencies using pip:

    ```bash
    pip install opencv-python numpy
    ```

## Usage

To run the script, provide the image path as an argument:

```bash
python canny_edge_detection.py <image_path>
```

Example:

```bash
python canny_edge_detection.py example.jpg
```

The script will output an image with `_edge.png` appended to the original filename (e.g., `example_edge.png`).

## Canny Edge Detection Steps

### 1. Grayscale Conversion

The function `convert_to_grayscale()` converts a colored image into a grayscale image by applying a weighted sum to the R, G, and B channels of the image.

### 2. Gaussian Smoothing

The Gaussian filter, implemented with `gaussian_kernel()`, reduces image noise by applying a 5x5 Gaussian kernel to the grayscale image.

### 3. Sobel Filter for Gradient Calculation

The gradients in the x and y directions are calculated manually using the Sobel operator. This is done by convolving the image with two 3x3 Sobel kernels. The function `sobel_filter()` handles this step and computes the gradient magnitudes and directions.

### 4. Gradient Strength and Direction

After calculating the gradients, the function computes the magnitude and direction of the gradients for each pixel using Euclidean distance and arctangent operations.

### 5. Non-Maximum Suppression

The function `non_max_suppression()` removes pixels that are not considered local maxima in the gradient direction, thinning out the edges.

### 6. Double Threshold

The `double_threshold()` function classifies pixels as strong, weak, or non-relevant edges by comparing them to high and low thresholds. Strong edges are kept, weak edges are further processed, and non-relevant edges are discarded.

### 7. Edge Tracking by Hysteresis

The final edge detection is performed in `edge_tracking()`. It preserves weak edges that are connected to strong edges, removing isolated weak edges.

## License

This project is licensed under the MIT License. Feel free to modify and use the code as per your requirements.
