import numpy as np
import cv2
from model.description import get_descriptors
from model.ocr import extract_information
from model.preprocessing import enhance_image
from model.explain import _explain


descriptor = get_descriptors()['descriptor']


def _read(contents):
    '''Read the file after clicking button'''
    nparr = np.fromstring(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return image


def condition_io(func):
    def process(image):
        return func(image)
    return process


@condition_io
def extract(image):
    enhanced = enhance_image(image)
    information = extract_information(enhanced)
    detail = _explain(" ".join(information), descriptor)
    return detail
