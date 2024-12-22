import cv2
import numpy as np
import argparse
import os

def draw_rectangle(event, x, y, flags, param):
    """
    Handles mouse events to draw a rectangle on an image.

    This function is used as a callback for mouse events in OpenCV. It updates
    global variables to track the drawing state and coordinates, and modifies
    the global image to display the drawn rectangle while the mouse is being
    dragged.

    Args:
        event (int): The type of mouse event (e.g., button down, move, button up).
        x (int): The x-coordinate of the mouse event.
        y (int): The y-coordinate of the mouse event.
        flags (int): Any relevant flags passed by OpenCV.
        param (any): Additional parameters (not used in this function).
    """
    # Global variables
    global ix, iy, drawing, img, img_copy

    # Handle mouse events
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            img = img_copy.copy()
            cv2.rectangle(img, (ix, iy), (x, y), (0, 0, 255), 2)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.rectangle(img, (ix, iy), (x, y), (0, 0, 255), 2)
        img_copy = img.copy()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Draw a red bounding box on an image and save it.")
    parser.add_argument("image_path", help="Path to the image file.")
    args = parser.parse_args()

    # Check if the image file exists
    if not os.path.exists(args.image_path):
        print("Error: Image file not found.")
        exit()

    # Load the image
    img = cv2.imread(args.image_path)
    if img is None:
        print("Error: Could not load the image file.")
        exit()

    # Initialize variables
    img_copy = img.copy()
    ix, iy = -1, -1
    drawing = False

    # Create a window and set the mouse callback
    cv2.namedWindow("Image")
    cv2.setMouseCallback("Image", draw_rectangle)

    while True:
        
        # Display the image
        cv2.imshow("Image", img)
        key = cv2.waitKey(1) & 0xFF

        # Save the image if 's' is pressed
        if key == ord('s'):
            # Extract file name and extension
            file_name, file_ext = os.path.splitext(os.path.basename(args.image_path))
            save_name = f"{file_name}_bounding_boxed{file_ext}"
            save_path = os.path.join(os.getcwd(), save_name)
            
            # Save the image
            cv2.imwrite(save_path, img_copy)
            print(f"Image saved as {save_path}")
            break

        # Exit if 'q' is pressed
        elif key == ord('q'):
            print("Exiting without saving.")
            break

    # Close the window
    cv2.destroyAllWindows()
