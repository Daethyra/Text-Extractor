import os
import cv2
from dotenv import load_dotenv
import numpy as np
import pytesseract

# Load environment variables from .env file
load_dotenv()

pytesseract.pytesseract.tesseract_cmd = os.getenv("PYTESSERACT_PATH")

class BoundingBoxTextExtractor:
    """
    A module for extracting and reading text from within a red bounding box in an image.
    """

    def __init__(self, tesseract_cmd=None):
        """
        Initializes the BoundingBoxTextExtractor.

        Args:
            tesseract_cmd (str): Path to the Tesseract OCR executable. If None, the system default is used.
        """
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    def extract_text_from_bounding_box(self, image_path, preprocess="thresh"):
        """
        Extracts text from within the red bounding box in the given image.
        Preprocesses the image for better OCR performance.

        Args:
            image_path (str): Path to the image file.
            preprocess (str): Preprocessing method: "thresh" (default), "blur", or "none".

        Returns:
            str: The extracted text from the red bounding box, or None if no bounding box is found.
        """
        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image file not found: {image_path}")

        # Convert the image to HSV color space to detect the red bounding box
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array([0, 70, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 70, 50])
        upper_red2 = np.array([180, 255, 255])
        
        # Create masks for red color
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = mask1 | mask2

        # Find contours to locate the bounding box
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # If no contours are found, return None
        if not contours:
            print("No red bounding box found in the image.")
            return None

        # Assume the largest red contour is the bounding box
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Extract the region of interest (ROI)
        roi = image[y:y+h, x:x+w]

        # Preprocessing for OCR
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        if preprocess == "thresh":
            gray_roi = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        elif preprocess == "blur":
            gray_roi = cv2.medianBlur(gray_roi, 3)

        # Scale up for better OCR performance
        scale_factor = 2
        resized_roi = cv2.resize(gray_roi, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)

        # Perform OCR | Remove `-c` param to recognize all characters
        custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_'
        text = pytesseract.image_to_string(resized_roi, config=custom_config)

        return text.strip()

if __name__ == "__main__":
    # Example usage of BoundingBoxTextExtractor
    import argparse

    parser = argparse.ArgumentParser(description="Extract and read text from a red bounding box in an image.")
    parser.add_argument("image_path", help="Path to the image file.")
    parser.add_argument("--tesseract_cmd", help="Path to the Tesseract OCR executable.", default=None)
    args = parser.parse_args()

    extractor = BoundingBoxTextExtractor(tesseract_cmd=args.tesseract_cmd)
    extracted_text = extractor.extract_text_from_bounding_box(args.image_path)

    if extracted_text:
        print("Extracted Text:")
        print(extracted_text)
    else:
        print("No text could be extracted from the bounding box.")
