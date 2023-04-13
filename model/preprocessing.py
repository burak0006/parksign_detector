import cv2
import numpy as np
from skimage import measure

blur_th = 20
glare_ratio = 0.1


def read_image(path):
    '''Read the image'''
    img = cv2.imread(path)
    return img


def convert_image(image):
    '''Convert the image from BGR to RGB'''
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def check_image(func):
    '''Sanity check that original image has no glare or blurry'''

    def preprocess(image):
        glare = glare_identify(image)
        if glare:
            raise Exception("Glare found! Please remove any illumination")
        final_image = func(image)
        return final_image

    return preprocess


def blur_identify(image):
    '''Identify blurry image'''
    if cv2.Laplacian(image, cv2.CV_64F).var() < blur_th:
        return True
    else:
        return False


def remove_noise(image):
    '''Denoising image'''
    return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)


def glare_identify(image):
    '''Glare identification. Area is limited 10% of the original image size'''
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    _, thresh_img = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)
    thresh_img = cv2.erode(thresh_img, None, iterations=2)
    thresh_img = cv2.dilate(thresh_img, None, iterations=4)

    # perform a connected component analysis on the thresholded image,
    # then initialize a mask to store only the "large" components
    labels = measure.label(thresh_img, connectivity=2, background=0)
    mask = np.zeros(thresh_img.shape, dtype="uint8")
    # loop over the unique components
    glare = False

    for label in np.unique(labels):
        # if this is the background label, ignore it
        if label == 0:
            continue
        # otherwise, construct the label mask and count the
        # number of pixels
        labelMask = np.zeros(thresh_img.shape, dtype="uint8")
        labelMask[labels == label] = 255
        numPixels = cv2.countNonZero(labelMask)
        # if the number of pixels in the component is sufficiently
        # large, then add it to our mask of "large blobs"
        height, width, channels = image.shape
        area = height * width * glare_ratio
        if numPixels > area:
            mask = cv2.add(mask, labelMask)
            glare = True
    return glare


def automatic_brightness_and_contrast(image, clip_hist_percent=1):
    '''# Automatic brightness and contrast optimization with optional histogram clipping
         Calculate new histogram with desired range and show histogram'''
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate grayscale histogram
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist_size = len(hist)

    # Calculate cumulative distribution from the histogram
    accumulator = []
    accumulator.append(float(hist[0]))
    for index in range(1, hist_size):
        accumulator.append(accumulator[index - 1] + float(hist[index]))

    # Locate points to clip
    maximum = accumulator[-1]
    clip_hist_percent *= (maximum / 100.0)
    clip_hist_percent /= 2.0

    # Locate left cut
    minimum_gray = 0
    while accumulator[minimum_gray] < clip_hist_percent:
        minimum_gray += 1

    # Locate right cut
    maximum_gray = hist_size - 1
    while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1

    # Calculate alpha and beta values
    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha

    auto_result = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return (auto_result, alpha, beta)


# TODO resolution needs to be enhanced
def convert_to_300dpi(image):
    '''Conversion to higher resolution for better OCR'''
    return image


@check_image
def enhance_image(image):
    dpi_image = convert_to_300dpi(image)
    noise_removed = remove_noise(dpi_image)
    contrast_improved = automatic_brightness_and_contrast(noise_removed, clip_hist_percent=1)
    return contrast_improved[0]
