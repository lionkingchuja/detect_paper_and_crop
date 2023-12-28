import cv2
import numpy as np
from scipy.spatial import distance as dist

def crop(image_path):
    # Read the image
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: Unable to read the image at {image_path}")
        return None

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection
    edged = cv2.Canny(blurred, 75, 200)

    # Find contours and sort them by size
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    for contour in contours:
        # Approximate the contour
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        # If our approximated contour has four points, we can assume we have found the paper
        if len(approx) == 4:
            screenCnt = approx
            break

    # Apply the perspective transform
    warped = point_tranform(image, screenCnt.reshape(4, 2))

    return warped


def point_tranform(image, pts):
    # Obtain a consistent order of the points and unpack them individually
    rect = point_order(pts)
    (tl, tr, br, bl) = rect

    # Compute the width of the new image
    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = max(int(width_a), int(width_b))

    # Compute the height of the new image
    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = max(int(height_a), int(height_b))

    # Construct the set of destination points
    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]], dtype="float32")

    # Compute the perspective transform matrix and apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (max_width, max_height))
    # cv2.namedWindow('final',cv2.WINDOW_NORMAL)
    # //cv2.imshow('Sobel x', warped)

    return warped

if __name__ == "__main__":
    input_image_path = '2.jpg'
    result = crop(input_image_path)

    if result is not None:
        cv2.imshow("Cropped Paper", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
