import numpy as np
import random
from pathlib import Path
import torch
import matplotlib.pyplot as plt


# Initial prediction before association algorithm
def get_initial_predictions(X, threshold):
    """
    Generates initial predictions for each video in the dataset.
    Args:
        X (list): List of videos (each video is a list of images).
        threshold (float): Threshold for object detection model.

    Returns:
        initial_predictions (list): List of initial predictions for each video.
        predicted_hex (list): List of predicted hex values for each video.
    """
    threshold = threshold
    initial_predictions = []
    predicted_hex = []

    for video in X:
        tmp = []
        for i, img in enumerate(video):
            # Processing each image using the given model
            encoding = processor(images=img, annotations=None, return_tensors="pt")
            pixel_values = encoding["pixel_values"].squeeze()  # remove batch dimension

            device = torch.device("cuda")
            pixel_values = pixel_values.unsqueeze(0).to(device)
            model.to(device)

            with torch.no_grad():
                # Forward pass to get class logits and bounding boxes
                outputs = model(pixel_values=pixel_values, pixel_mask=None)

            # Postprocess model outputs
            width, height = img.shape[1], img.shape[0]
            postprocessed_outputs = processor.post_process_object_detection(outputs,
                                                                            target_sizes=[(height, width)],
                                                                            threshold=threshold)
            results = postprocessed_outputs[0]

            for ids, (score, label, (xmin, ymin, xmax, ymax)) in enumerate(
                    zip(results['scores'].tolist(), results['labels'].tolist(), results['boxes'].tolist())):
                center = (xmin + 32.5, ymin + 32.5)
                array = [int(i), int(ids), xmin + 32.5, ymin + 32.5]
                np.set_printoptions(suppress=True)
                farray = np.array(array)

                frame, ids, x, y = farray

                if int(ids + 1) <= 4:
                    # Creating a dictionary for each region of interest (ROI)
                    e = {
                        't': frame,
                        'hexbug': ids,
                        'x': y,
                        'y': x
                    }
                    tmp.append(e)

        total_frames = int(i) + 1
        n = round(len(tmp) / total_frames)
        predicted_hex.append(n)

        initial_predictions.append(tmp)

    return initial_predictions, predicted_hex

def calculate_distance(x1, y1, x2, y2):
    """
    Calculates Euclidean distance between two points.
    Args:
        x1, y1, x2, y2 (float): Coordinates of the two points.

    Returns:
        float: Euclidean distance.
    """
    return ((x2 - x1)**2 + (y2 - y1)**2)


# Final predictions after applying associative alogorithm
def get_final_predictions(X, threshold, predicted_hex):
    """
    Generates final predictions after applying association algorithm.
    Args:
        X (list): List of videos (each video is a list of images).
        threshold (float): Threshold for object detection model.
        predicted_hex (list): Predicted number of hexbugs for each video.

    Returns:
        final_predictions (list): List of final predictions for each video.
    """
    threshold = threshold
    final_predictions = []
    n_hex = predicted_hex

    for m, video in enumerate(X):

        tmp = []
        tmp_array = []

        for i, img in enumerate(video):

            # Similar processing as in get_initial_predictions
            encoding = processor(images=img, annotations=None, return_tensors="pt")
            pixel_values = encoding["pixel_values"].squeeze()  # remove batch dimension

            device = torch.device("cuda")
            pixel_values = pixel_values.unsqueeze(0).to(device)
            model.to(device)

            with torch.no_grad():
                # Forward pass to get class logits and bounding boxes
                outputs = model(pixel_values=pixel_values, pixel_mask=None)

            # Postprocess model outputs
            width, height = img.shape[1], img.shape[0]
            postprocessed_outputs = processor.post_process_object_detection(outputs,
                                                                            target_sizes=[(height, width)],
                                                                            threshold=threshold)
            results = postprocessed_outputs[0]

            image_pred_history = []
            for ids, (score, label, (xmin, ymin, xmax, ymax)) in enumerate(
                    zip(results['scores'].tolist(), results['labels'].tolist(), results['boxes'].tolist())):
                pred_n_hexs = len(results['scores'].tolist())
                center = (xmin + 32.5, ymin + 32.5)
                array = [int(i), int(ids), xmin + 32.5, ymin + 32.5]
                farray = np.array(array)
                frame, ids, x, y = farray

                if int(ids + 1) <= n_hex[m]:
                    if int(ids + 1) <= 4:
                        # we create a dictionary for each roi in the correct format
                        e = {
                            't': frame,
                            'hexbug': ids,
                            'x': y,
                            'y': x
                        }
                        image_pred_history.append(farray)
                        tmp.append(e)

            tmp_array.append(image_pred_history)

            if (int(ids + 1) < n_hex[m]):
                difference = n_hex[m] - int(ids + 1)
                for n in range(len(tmp_array) - 1, -1, -1):
                    if len(tmp_array[n]) == n_hex[m]:

                        associative_list = []
                        a_list = tmp_array[n]
                        # Iterate over elements of list B
                        for b_element in image_pred_history:
                            min_distance = float('inf')  # Initialize minimum distance as infinity
                            associated_a_element = None  # Initialize associated A element as None

                            # Iterate over elements of list A
                            for index, a_element in enumerate(a_list):
                                if not index in associative_list:
                                    # Extract frame ID, hexbug ID, x, and y coordinates from list A element
                                    a_frame_id, a_hexbug_id, a_x, a_y = a_element

                                    # Calculate distance between B element and A element
                                    distance = calculate_distance(a_x, a_y, b_element[2], b_element[3])

                                    # Check if the distance is smaller than the minimum distance
                                    if distance < min_distance:
                                        min_distance = distance
                                        associated_a_element = a_element
                                        associated_a_element_index = index
                            associative_list.append(associated_a_element_index)

                        expected_size = n_hex[m]
                        expected_range = range(0, expected_size)
                        missing_elements = list(set(expected_range) - set(associative_list))

                        for missed_number, missed in enumerate(missing_elements):
                            x = tmp_array[n][missed][2]
                            y = tmp_array[n][missed][3]
                            e = {
                                't': int(frame),
                                'hexbug': int(ids + missed_number + 1),
                                'x': y,
                                'y': x}
                            tmp.append(e)
                        break

                    else:
                        pass

        final_predictions.append(tmp)

    return final_predictions
