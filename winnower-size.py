#based on
#https://www.pyimagesearch.com/2016/03/28/measuring-size-of-objects-in-an-image-with-opencv/

# requires numpy, cv2, and imutils
import os
import math
import argparse
from itertools import chain
import numpy as np
import cv2
import imutils
from imutils import contours


# colors
REJECT = (130, 130, 225)
ACCEPT = (80, 220, 50)
LENGTH = (255, 150, 100)
WIDTH = (255, 180, 50)
ANGLE = (80, 160, 225)


def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)


def rotate_contour(contour, angle):
    m = cv2.getRotationMatrix2D((0, 0), angle, 1)  # rotate about origin, don't scale
    rotated = np.zeros_like(contour)
    for i, a, in enumerate(contour):
        for j, b in enumerate(a):
            rot_b = [m[0,0]*b[0] + m[0,1]*b[1],
                        m[1,0]*b[0] + m[1,1]*b[1]]  # dot product Mx
            rotated[i, j] = rot_b
    return rotated


def min_max_bbox(hull):
    # stupid but easy way to get min and max bounding rectangle dimensions
    widths, heights, boxes = [], [], []
    for i in range(args['rotate_steps']):  # rotate image over 1/4 circle
        angle = i * 90 / args['rotate_steps']  # degrees
        rotated = rotate_contour(hull, angle)
        x, y, w, h = cv2.boundingRect(rotated)
        box = np.array([[[x, y], [x + w, y], [x + w, y + h], [x, y + h]]], dtype="int")
        box = rotate_contour(box, -angle)  # rotate bounding rect back to original position
        # store dims
        widths.append(w)
        heights.append(h)
        boxes.append(box)

    # find min and max height/width, retrieve appropriate boxes
    dims = widths + heights
    maxbox = boxes[np.argmax(dims) % args['rotate_steps']]
    minbox = boxes[np.argmin(dims) % args['rotate_steps']]
    return np.min(dims), np.max(dims), minbox, maxbox


def calculate_angle(pt_1, pt_mid, pt_2):
    a = pt_1 - pt_mid
    b = pt_2 - pt_mid
    return np.arccos((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def contour_angles(contour, args):
    # take cumulative sum of angle deviation (from straight) over segments
    # each segment is longer than angle_dist
    points = np.array(contour).squeeze()
    last = points[0,:]
    angle_sum, dist_sum = 0, 0
    angle_pts, angles = [], []
    # visit first point twice to close loop
    i = 0
    for prev, pt, next in zip(points,
            chain(points[1:], points[:1]),
            chain(points[2:], points[:2])):
        angle = math.pi - calculate_angle(prev, pt, next)
        angle_sum += angle
        dist_sum += np.linalg.norm(pt - prev)
        if format_num(dist_sum, args['scale']) > args['angle_dist']:
            angle_pts.append(pt)
            angles.append(angle_sum)
            angle_sum, dist_sum = 0, 0
        i += 1
    angles[0] += angle_sum  # add remaining unaccounted angle onto first segment
    return np.stack(angle_pts, axis=0), np.array(angles)


def draw_contour(image, contour, color, size):
    cv2.drawContours(image, [contour.astype("int")], -1, color, thickness=size)
    # draw points for all nodes
    sm_size = int(size * 0.5)
    darker = (int(color[0] * 0.5), int(color[1] * 0.5), int(color[2] * 0.5))
    for segment in contour:
        for pt in segment:
            cv2.circle(image, tuple(pt), size, darker, thickness=sm_size)


def draw_text(image, texts, center, colors, size=1.6):
    n_rows = len(texts)
    for row, (text, color) in enumerate(zip(texts, colors)):
        pos = (center[0] - len(text) * 16, center[1] + int(row * 60 - n_rows / 2 * 60))
        # draw white border
        cv2.putText(image, text, pos, cv2.FONT_HERSHEY_SIMPLEX, size,
                    (255, 255, 255), int(size * 15))
        cv2.putText(image, text, pos, cv2.FONT_HERSHEY_SIMPLEX, size, color, int(size * 2.5))


def format_num(num, scale=(180 / math.pi)):
    return int(round(num * scale))


def process_image(classdir, file, args, start_idx):
    # load the image, convert it to grayscale, and blur it slightly
    image = cv2.imread(os.path.join(args['input'], classdir, file))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    # perform edge detection, then perform a dilation + erosion to
    # close gaps in between object edges
    edged = cv2.Canny(gray, 50, 100)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)
    # find contours in the edge map
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    # sort the contours from left-to-right
    (cnts, _) = contours.sort_contours(cnts)

    hulls = []
    for c in cnts:  # loop over the contours individually, cull by size
        hull = cv2.convexHull(c)
        area = format_num(cv2.contourArea(hull), args['scale']**2)
        if area < args['min_area'] or area > args['max_area']:
            draw_contour(image, hull, REJECT, 2)
        else:  # save for calculation
            hulls.append(hull)

    # draw and save measurements
    with open(os.path.join(args['output'], 'measurements.csv'), 'a') as f:
        for i, hull in enumerate(hulls):
            # get measurements
            min_len, max_len, minbox, maxbox = min_max_bbox(hull)
            angle_points, angles = contour_angles(hull, args)
            sorted = np.sort(angles)

            # save measurements
            center = (np.mean(minbox + maxbox, axis=1) / 2)[0].astype('int')
            id = i + start_idx
            w = format_num(min_len, args['scale'])
            l = format_num(max_len, args['scale'])
            p = format_num(cv2.arcLength(hull, True), args['scale'])
            area = format_num(cv2.contourArea(hull), args['scale']**2)
            a1 = format_num(sorted[-1])
            a2 = format_num(sorted[-2])
            avg = format_num(np.mean(sorted))
            f.write('{},{},{},{},{},{},{},{}\n'.format
                (id, classdir, w, l, p, area, a1, a2))

            #draw everything
            draw_contour(image, hull, ACCEPT, 8)
            draw_contour(image, maxbox, LENGTH, 8)
            draw_contour(image, minbox, WIDTH, 4)
            for i, (pt, angle) in enumerate(zip(angle_points, angles)):
                cv2.circle(image, tuple(pt), 12, ANGLE, thickness=4)
                draw_text(image, ['{}|{}*'.format(i, format_num(angle))], pt, ANGLE, size=0.8)
            draw_text(image, ["{} {:d}".format(classdir, id), "w={:d}".format(w),
                    "l={:d}".format(l), "p={:d}".format(p), "a={:d}".format(area)],
                    center, [ANGLE, WIDTH, LENGTH, ACCEPT, ACCEPT])

    # draw legend
    draw_text(image,
        ["Rejected hull", "Accepted hull", "Length box", "Width box", "Hull angles"],
        (250, 200), [REJECT, ACCEPT, LENGTH, WIDTH, ANGLE])

    cv2.imwrite(os.path.join(args['output'], classdir,  # save image
                os.path.splitext(file)[0] + '-labelled.png'), image)
    print('identified {} objects'.format(len(hulls)))

    if args['display']:  # show window with image, hulls, measurements
        window_name = 'image'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 600,600)
        cv2.imshow(window_name, image)
        while True:  # this loop closes the window when ESC pressed
            k = cv2.waitKey(100)
            if k == 27:
                print('ESC')
                cv2.destroyAllWindows()
                break
            if cv2.getWindowProperty('image',cv2.WND_PROP_VISIBLE) < 1:
                break
        cv2.destroyAllWindows()

    return start_idx + len(hulls)


if __name__ == '__main__':
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True,
        help="path to the input dir")
    ap.add_argument("-o", "--output", required=True,
        help="path to output dir")
    ap.add_argument("-s", "--scale", type=float, default=1./300*25.4,
        help="pixels to unit (default 300 DPI to mm)")
    ap.add_argument("-m", "--min_area", type=float, default=1500,
        help="smallest area to consider valid object")
    ap.add_argument("-M", "--max_area", type=float, default=7000,
        help="largest area to consider valid object")
    ap.add_argument("-r", "--rotate_steps", type=int, default=90,
        help="number of bounding box rotations to test")
    ap.add_argument("-a", "--angle_dist", type=float, default=5.,
        help="min distance to travel along perimeter before logging angle")
    ap.add_argument("-d", "--display", type=bool, default=False,
        help="show images as they are processed")
    args = vars(ap.parse_args())

    start_idx = 1
    # write output header
    if not os.path.exists(args['output']):
        os.makedirs(args['output'])
    with open(os.path.join(args['output'], 'measurements.csv'), 'w') as f:
        f.write('ID,CLASS,WIDTH,LENGTH,PERIMETER,AREA,ANGLE1,ANGLE2\n')
    # process each image file classifying by subdirectory
    for classdir in os.listdir(args['input']):
        out_path = os.path.join(args['output'], classdir)
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        sub_path = os.path.join(args['input'], classdir)
        if os.path.isdir(sub_path):
            for file in os.listdir(sub_path):
                if file.endswith('.png'):
                    print('Processing', file, end=' ... ')
                    start_idx = process_image(classdir, file, args, start_idx)
