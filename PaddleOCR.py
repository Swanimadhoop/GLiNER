import numpy as np
from PIL import Image
from paddleocr import PaddleOCR

# Load the image
img_path = '/content/1.jpg'  # Update with your actual image path
image = Image.open(img_path).convert('RGB')

# Initialize PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='en')

# Run OCR
result = ocr.ocr(np.array(image))

# Extract texts from the OCR result
texts = [line[1][0] for line in result[0]]

# Print the extracted texts with a comma at the end of each line
for text in texts:
    print(f"{text},")
