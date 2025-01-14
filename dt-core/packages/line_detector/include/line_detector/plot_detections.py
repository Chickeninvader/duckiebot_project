import cv2
import numpy as np


def plotSegments(image, detections):
    """

    Draws a set of line segment detections on an image.

    Args:
        image (:obj:`numpy array`): an image
        detections (`dict`): a dictionary that has keys :py:class:`ColorRange` and values :py:class:`Detection`

    Returns:
        :obj:`numpy array`: the image with the line segments drawn on top of it.

    """

    im = np.copy(image)

    for color_range, det in list(detections.items()):

        # convert HSV color to BGR
        c = color_range.representative
        c = np.uint8([[[c[0], c[1], c[2]]]])
        color = cv2.cvtColor(c, cv2.COLOR_HSV2BGR).squeeze().astype(int)
        # plot all detected line segments and their normals
        for i in range(len(det.normals)):
            center = det.centers[i]
            normal = det.normals[i]
            im = cv2.line(
                im,
                tuple(center.astype(int)),
                tuple((center + 10 * normal).astype(int)),
                color=(0, 0, 0),
                thickness=2,
            )
            # im = cv2.circle(im, (center[0], center[1]), radius=3, color=color, thickness=-1)
        for line in det.lines:
            im = cv2.line(im, (line[0], line[1]), (line[2], line[3]), color=(0, 0, 0), thickness=5)
            im = cv2.line(
                im, (line[0], line[1]), (line[2], line[3]), color=tuple([int(x) for x in color]), thickness=2
            )
    return im

def plotSegments_sl(image, detections, st_x_max, st_y_max, st_x_min, st_y_min, xymaxmin_red_normalized):
    im = np.copy(image)
    for color_range, det in list(detections.items()):

        # convert HSV color to BGR
        c = color_range.representative
        c = np.uint8([[[c[0], c[1], c[2]]]])
        color = cv2.cvtColor(c, cv2.COLOR_HSV2BGR).squeeze().astype(int)
        # plot all detected line segments and their normals
        for i in range(len(det.normals)):
            center = det.centers[i]
            normal = det.normals[i]
            im = cv2.line(
                im,
                tuple(center.astype(int)),
                tuple((center + 10 * normal).astype(int)),
                color=(0, 0, 0),
                thickness=2,
            )
            # im = cv2.circle(im, (center[0], center[1]), radius=3, color=color, thickness=-1)
        for line in det.lines:
            im = cv2.line(im, (line[0], line[1]), (line[2], line[3]), color=(0, 0, 0), thickness=5)
            im = cv2.line(
                im, (line[0], line[1]), (line[2], line[3]), color=tuple([int(x) for x in color]), thickness=2
            )

    st_x_max = int(st_x_max)
    st_y_max = int(st_y_max)
    st_x_min = int(st_x_min)
    st_y_min = int(st_y_min)

    im = cv2.line(im, (st_x_max, st_y_max), (st_x_min, st_y_max), color=(255, 255, 255), thickness=1)
    im = cv2.line(im, (st_x_max, st_y_max), (st_x_max, st_y_min), color=(255, 255, 255), thickness=1)
    im = cv2.line(im, (st_x_min, st_y_min), (st_x_min, st_y_max), color=(255, 255, 255), thickness=1)
    im = cv2.line(im, (st_x_min, st_y_min), (st_x_min, st_y_min), color=(255, 255, 255), thickness=1)
    st_x_max_n, st_y_max_n, st_x_min_n, st_y_min_n = list(xymaxmin_red_normalized)
    area = (st_x_max - st_x_min) * (st_y_max - st_y_min)
    area_normalized = (st_x_max_n - st_x_min_n) * (st_y_max_n - st_y_min_n)

    # im = cv2.line(im, (0, 0), (60, 60), color=(0, 255, 0), thickness=1)

    text_max_xy, text_min_xy = '['+str(st_x_max)+','+str(st_y_max)+'], ', '['+str(st_x_min)+','+str(st_y_min)+']'
    text_max_xy_normalized, text_min_xy_normalized = 'normalized: ['+str(st_x_max_n)+','+str(st_y_max_n)+'], ', '['+str(st_x_min_n)+','+str(st_y_min_n)+']'
    text_area = 'area: ' + str(area)
    text_area_normalized = 'normalized area: ' + str(area_normalized)

    # put texts
    font = cv2.FONT_HERSHEY_SIMPLEX
    # im = cv2.putText(im, "test", (st_x_max, st_y_max), font, fontScale=0.3, color=(0, 255, 0), thickness=1)
    im = cv2.putText(im, text_max_xy, (60, 80), font, fontScale=0.3, color=(0, 255, 0), thickness=1)
    im = cv2.putText(im, text_min_xy, (20, 40), font, fontScale=0.3, color=(0, 255, 0), thickness=1)
    im = cv2.putText(im, text_max_xy_normalized, (60, 85), font, fontScale=0.3, color=(0, 255, 0), thickness=1)
    im = cv2.putText(im, text_min_xy_normalized, (20, 45), font, fontScale=0.3, color=(0, 255, 0), thickness=1)
    im = cv2.putText(im, text_area, (40, 60), font, fontScale=0.3, color=(0, 255, 0), thickness=1)
    im = cv2.putText(im, text_area_normalized, (40, 65), font, fontScale=0.3, color=(0, 255, 0), thickness=1)

    # put dots
    im = cv2.circle(im, (st_x_max, st_y_max), 1, color=(255, 0, 0), thickness=1)
    im = cv2.circle(im, (st_x_min, st_y_min), 1, color=(0, 255, 0), thickness=1)
    im = cv2.circle(im, (st_y_max, st_x_max), 1, color=(255, 0, 0), thickness=1)
    im = cv2.circle(im, (st_y_min, st_x_min), 1, color=(0, 255, 0), thickness=1)

    return im


def plotMaps(image, detections):
    """

    Draws a set of color filter maps (the part of the images falling in a given color range) on an image.

    Args:
        image (:obj:`numpy array`): an image
        detections (`dict`): a dictionary that has keys :py:class:`ColorRange` and values :py:class:`Detection`

    Returns:
        :obj:`numpy array`: the image with the line segments drawn on top of it.

    """

    im = np.copy(image)
    im = cv2.cvtColor(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)

    color_map = np.zeros_like(im)

    for color_range, det in list(detections.items()):

        # convert HSV color to BGR
        c = color_range.representative
        c = np.uint8([[[c[0], c[1], c[2]]]])
        color = cv2.cvtColor(c, cv2.COLOR_HSV2BGR).squeeze().astype(int)
        color_map[np.where(det.map)] = color

    im = cv2.addWeighted(im, 0.3, color_map, 1 - 0.3, 0.0)

    return im
