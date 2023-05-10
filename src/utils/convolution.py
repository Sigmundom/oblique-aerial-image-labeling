from timeit import default_timer
import numpy as np


np.set_printoptions(precision=2, suppress=True)

def create_kernel(kernel_size, sigma):
    # Create an array with values centered at the middle pixel of the kernel
    x, y = np.meshgrid(np.arange(kernel_size), np.arange(kernel_size))
    x -= kernel_size // 2
    y -= kernel_size // 2

    # Calculate the Gaussian kernel using the gaussian_filter function
    kernel = np.exp(-(x**2 + y**2) / (2*sigma**2))
    kernel /= kernel.max()
    return kernel

def smooth(data, data_count, kernel_size=7, sigma=2.5):
    kernel = create_kernel(kernel_size, sigma)
    radius = kernel_size//2
    data = np.pad(data, (radius, radius))
    data_count = np.pad(data_count, (radius, radius))
    # Create an empty output raster
    result = np.zeros_like(data)

    t = default_timer()
    # Loop over each pixel in the input raster
    for i in range(radius, data.shape[0]-radius):
        for j in range(radius, data.shape[1]-radius):
            # Calculate the weights for the current pixel based on the neighboring observations
            weights = data_count[i-radius:i+radius+1, j-radius:j+radius+1]
            # Calculate weighted kernel
            weighted_kernel = kernel * weights
            # Normalize the kernel so that it sums to 1
            weighted_kernel /= weighted_kernel.sum()
            # Apply the kernel to the current pixel
            result[i, j] = np.sum(data[i-radius:i+radius+1, j-radius:j+radius+1] * weighted_kernel)
    result = result[radius:-radius]
    print('Process time:', default_timer()-t)
    return result

if __name__ == '__main__':
    print(create_kernel(9, 3))