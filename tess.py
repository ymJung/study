from PIL import Image
import numpy as np
import pytesseract

filename = 'sc2.png'
img1 = np.array(Image.open(filename))
text = pytesseract.image_to_string(img1, config=('-l kor+eng'))
print(text)