import easyocr

reader = easyocr.Reader(['en'], gpu=True)


def extract_information(image):
    result = reader.readtext(image, detail=0)
    return result
