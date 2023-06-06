import os
import cv2
import numpy as np
from sklearn.cluster import KMeans


def load_images_from_folder(folder):
    # print(folder)
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        if os.path.isfile(img_path):
            img = cv2.imread(img_path)
            if img is not None:
                images.append((img_path, img))
    return images


def predict_temperature(image, n_clusters):
    # Reshape the image to a 2D array
    height, width, _ = image.shape
    image_2d = image.reshape(-1, 3)

    X = np.array([[1, 2], [1, 4], [1, 0],
                  [10, 2], [10, 4], [10, 0]])
    # Apply K-means clustering
    kMeans = KMeans(n_clusters=n_clusters,
                    random_state=0, n_init="auto").fit(X)
    kMeans.fit(image_2d)

    # Get the cluster labels for each pixel
    cluster_labels = kMeans.labels_

    # Get the cluster centers (representative temperatures)
    cluster_centers = kMeans.cluster_centers_

    # Predict the temperature for each pixel
    predicted_temperatures = cluster_centers[cluster_labels]

    # Reshape the predicted temperatures back to the original image shape
    predicted_temperatures_image = predicted_temperatures.reshape(
        (height, width, 3))

    return predicted_temperatures_image


# Load the images from the "train_data" folder
folder_path = "./TEMP_PRED-MASTER"
images = load_images_from_folder(folder_path)

# Set the number of clusters (temperature categories)
n_clusters = 5

# Process each image and make predictions
for img_path, img in images:
    # Find the red region in the image
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0, 70, 50])
    upper_red = np.array([10, 255, 255])
    mask = cv2.inRange(hsv_img, lower_red, upper_red)

    # Find the contours of the red region
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour (assumed to be the red region)
    if len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Set the size of the center box
        center_box_size = max(w, h)

        # Calculate the coordinates of the center box
        center_x1 = x
        center_y1 = y
        center_x2 = x + center_box_size
        center_y2 = y + center_box_size

        # Set the size of the outer box
        outer_box_size = int(center_box_size * 1.5)

        # Calculate the coordinates of the outer box
        outer_x1 = max(x - (outer_box_size - center_box_size) // 2, 0)
        outer_y1 = max(y - (outer_box_size - center_box_size) // 2, 0)
        outer_x2 = min(outer_x1 + outer_box_size, img.shape[1])
        outer_y2 = min(outer_y1 + outer_box_size, img.shape[0])

        # Extract the regions for the center and outer boxes
        center_region = img[center_y1:center_y2, center_x1:center_x2, :]
        outer_region = img[outer_y1:outer_y2, outer_x1:outer_x2, :]

        # Predict the temperature for the center region
        center_predicted = predict_temperature(center_region, n_clusters)

        # Predict the temperature for the outer region
        outer_predicted = predict_temperature(outer_region, n_clusters)

        # Find the maximum temperature in the center region
        center_max_temp = np.max(center_predicted[:, :, 2])

        # Find the maximum temperature in the outer region
        outer_max_temp = np.max(outer_predicted[:, :, 2])

        # Draw a red square box around the center region
        cv2.rectangle(img, (center_x1, center_y1),
                      (center_x2, center_y2), (0, 0, 255), 2)

        # Label the temperature prediction inside the center box
        cv2.putText(img, f"{center_max_temp:.2f}°C", (center_x1, center_y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Draw a green square box around the outer region
        cv2.rectangle(img, (outer_x1, outer_y1),
                      (outer_x2, outer_y2), (0, 255, 0), 2)

        # Label the temperature prediction outside the outer box
        cv2.putText(img, f"{outer_max_temp:.2f}°C", (outer_x1, outer_y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Display the image with the boxes and labels
        cv2.imshow("Image with Predictions", img)
        cv2.waitKey(0)

        # Check if the temperature outside the center region is higher
        if outer_max_temp > center_max_temp:
            print(f"Warning: Temperature outside the center region ({outer_max_temp:.2f}°C) is higher "
                  f"than the temperature inside the center region ({center_max_temp:.2f}°C)")

        # Print the image file name and the predicted temperatures
        print(f"Image: {img_path}")
        print(f"Center Temperature: {center_max_temp:.2f}°C")
        print(f"Outer Temperature: {outer_max_temp:.2f}°C")
        

    else:
        print(f"No red region found in {img_path}")
