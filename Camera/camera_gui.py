import cv2
import numpy as np


def get_rectangle_coordinates(image_path):
    """ the user can draw a rectangle on an image and have the rect params returned """
    start_x, start_y = -1, -1
    end_x, end_y = -1, -1

    def draw_rectangle(event, x, y, flags, param):
        nonlocal start_x, start_y, end_x, end_y
        if event == cv2.EVENT_LBUTTONDOWN:
            start_x, start_y = x, y
        elif event == cv2.EVENT_LBUTTONUP:
            end_x, end_y = x, y
        elif event == cv2.EVENT_MOUSEMOVE:
            end_x, end_y = x, y
        # Draw the rectangle on a copy of the image (to keep the original image unmodified)
        image_with_rectangle = image.copy()
        cv2.rectangle(image_with_rectangle, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
        cv2.imshow("Image", image_with_rectangle)

    image = cv2.imread(image_path)
    cv2.imshow("Image", image)
    cv2.setMouseCallback("Image", draw_rectangle)
    # Wait until the user presses any key
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # y, x, h , w
    return min(start_y, end_y), min(start_x, end_x), abs(start_y - end_y), abs(start_x - end_x)


def select_points(img):
    """ selects points on a given image using GUI interface """
    points = []
    zoom_factor = 1.0  # Initial zoom factor
    zoom_step = 0.1  # Amount to increase/decrease zoom by

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))

    def get_zoomed_image(image, factor):
        height, width = image.shape[:2]
        new_height, new_width = int(height * factor), int(width * factor)
        return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    img = cv2.imread(img) if isinstance(img, str) else img

    # Create a window for displaying the image
    cv2.namedWindow('Select Points')
    cv2.setMouseCallback('Select Points', on_click)

    # Display the image and allow the user to select points
    while True:
        display_img = img.copy()

        # Zoom the image for display
        zoomed_img = get_zoomed_image(display_img, zoom_factor)

        for point in points:
            # Draw the points on the original image using the current zoom factor
            cv2.circle(display_img, (int(point[0] * zoom_factor), int(point[1] * zoom_factor)),
                       3, (0, 255, 0), -1)

        cv2.imshow('Select Points', zoomed_img)

        # Wait for key press
        key = cv2.waitKey(1)

        if key == ord('q'):
            break
        elif key == ord('+'):
            zoom_factor += zoom_step  # Zoom in
        elif key == ord('-'):
            zoom_factor = max(zoom_factor - zoom_step, 1.0)  # Zoom out but not beyond original size

    cv2.destroyAllWindows()

    # Scale the selected points back to the original image size
    scaled_points = np.array([(point[0] / zoom_factor, point[1] / zoom_factor) for point in points],
                             dtype=np.float32)

    return np.expand_dims(scaled_points, axis=1)
