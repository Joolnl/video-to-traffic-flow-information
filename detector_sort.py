#!/usr/bin/env python
import copy
import math
import os
import os.path as osp
import pickle as pkl
import sys
from datetime import datetime

import cv2
import numpy
import pandas as pd
import torch
from shapely.geometry import Polygon, LineString
from torch.autograd import Variable

from darknet import Darknet
from sort import *
from util import process_result, cv_image2tensor, transform_result

bt = list()  # bottom top
tb = list()  # top bottom
count_vehicles = list()

tb_line_points = [(235, 674), (840, 641)]
bt_line_points = [(880, 638), (1350, 600)]

left_lane = [
    [(235, 673), (480, 660)],
    [(480, 660), (638, 652)],
    [(638, 652), (840, 641)],
]

right_lane = [
    [(876, 638), (978, 629)],
    [(978, 629), (1080, 622)],
    [(1080, 622), (1349, 600)]
]

left_lane_count = [list(), list(), list()]
right_lane_count = [list(), list(), list()]

line_a = [
    [(387, 581), (735, 716)], #IN
    [(270, 402), (387, 581)] #OUT
]

line_b = [
    [(1595, 521), (1603, 372)], #IN
    [(1131, 725), (1529, 600)] #OUT
]

line_c = [
    [(373, 287), (528, 199)], #IN
    [(779, 128), (626, 163)]  #OUT
]

line_d = [
    [(1069, 126), (1230, 151)], #IN
    [(1317, 173), (1479, 245)] #OUT
]

a_count = [list(), list()]
b_count = [list(), list()]
c_count = [list(), list()]
d_count = [list(), list()]

bbox_history = {}


def load_classes(namesfile):
    fp = open(namesfile, "r")
    names = fp.read().split("\n")[:-1]
    return names


def parse_args():
    parser = argparse.ArgumentParser(description='Object detection/tracking with YOLOv3 and SORT')
    parser.add_argument('-i', '--input', required=True, help='input directory or video')
    parser.add_argument('-t', '--obj-thresh', type=float, default=0.5, help='objectness threshold, DEFAULT: 0.5')
    parser.add_argument('-n', '--nms-thresh', type=float, default=0.4,
                        help='non max suppression threshold, DEFAULT: 0.4')
    parser.add_argument('--min-hits', default=3, help='A tracker needs to match a bounding box for at least this many '
                                                      'frames before it is registered. Prevents false positives')
    parser.add_argument('--max-age', default=10, help='The number of frames a tracker is kept alive without matching '
                                                      'bounding boxes. Useful for tracker while an object is '
                                                      'temporarily blocked')
    parser.add_argument('-o', '--outdir', default='output', help='output directory, DEFAULT: output/')
    parser.add_argument('-w', '--webcam', action='store_true', default=False,
                        help='flag for detecting from webcam. Specify webcam ID in the input. usually 0 for a single '
                             'webcam connected')
    parser.add_argument('--debug-trackers', action='store_true', default=False,
                        help="Show the kalman trackers instead of the YOLO bounding boxes. Useful for debugging "
                             "and setting parameters. No output is saved.")
    parser.add_argument('--cuda', action='store_true', default=False, help='flag for running on GPU')
    parser.add_argument('--no-show', action='store_true', default=False,
                        help='do not show the detected video in real time')

    args = parser.parse_args()

    return args


def create_batches(imgs, batch_size):
    num_batches = math.ceil(len(imgs) // batch_size)
    batches = [imgs[i * batch_size: (i + 1) * batch_size] for i in range(num_batches)]

    return batches


def draw_simple_bbox(img, bbox, text):
    p1 = tuple(bbox[:2].astype(int))
    p2 = tuple(bbox[2:].astype(int))

    color = [80, 80, 80]

    cv2.rectangle(img, p1, p2, color, 4)
    label = text
    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)[0]
    p3 = (p1[0], p1[1] - text_size[1] - 4)
    p4 = (p1[0] + text_size[0] + 4, p1[1])
    cv2.rectangle(img, p3, p4, color, -1)

    cv2.putText(img, label, p1, cv2.FONT_HERSHEY_SIMPLEX, 1, [225, 255, 255], 1)


def draw_bbox(imgs, bbox, uid, cls_ind, colors, classes):
    img = imgs[0]
    label = classes[cls_ind]
    p1 = tuple(bbox[0:2].astype(int))
    p2 = tuple(bbox[2:4].astype(int))

    color = colors[cls_ind]
    cv2.rectangle(img, p1, p2, color, 4)
    label = label + '_' + str(int(uid))
    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)[0]
    p3 = (p1[0], p1[1] - text_size[1] - 4)
    p4 = (p1[0] + text_size[0] + 4, p1[1])
    cv2.rectangle(img, p3, p4, color, -1)

    cv2.putText(img, label, p1, cv2.FONT_HERSHEY_SIMPLEX, 1, [225, 255, 255], 1)


def format_output(bbox, uid, cls_ind, classes, read_frames, output_path, fps):
    label = classes[cls_ind]
    p1 = tuple(bbox[0:2].astype(int))
    p2 = tuple(bbox[2:4].astype(int))

    label = label + '_' + str(int(uid))
    start_time_video = output_path.split('det_')[1].split('.')[0]

    return {
        'uid': uid,
        'label': str(label),
        'type': label.split('_')[0],
        'frame_number': str(read_frames),
        'coord_X_0': int(p1[0]),
        'coord_Y_0': int(p1[1]),
        'coord_X_1': int(p2[0]),
        'coord_Y_1': int(p2[1]),
        'process_time': str(datetime.now()),
        'filename': output_path.replace('.avi', '.csv'),
        'start_time_video': start_time_video,
        'fps': fps
    }


def detect_video(model, args):
    input_size = [int(model.net_info['height']), int(model.net_info['width'])]
    mot_tracker = Sort(min_hits=int(args.min_hits), max_age=int(args.max_age))
    colors = pkl.load(open("cfg/pallete", "rb"))
    classes = load_classes("cfg/coco.names")

    # TODO: Turn this into an external config file (relevant classes and mapping)
    relevant_classes = [
        "car",
        "truck"
    ]
    relevant_classes_indices = [classes.index(cls) for cls in relevant_classes]

    # If you want to merge classes together
    class_mapping = {
        classes.index("car"): [classes.index(cls) for cls in ['truck']]
    }

    if not osp.isdir(args.outdir):
        os.mkdir(args.outdir)

    if args.webcam:
        cap = cv2.VideoCapture(int(args.input))
        output_path = osp.join(args.outdir, 'det_webcam.avi')
    else:
        cap = cv2.VideoCapture(args.input)
        output_path = osp.join(args.outdir, 'det_' + osp.basename(args.input).rsplit('.')[0] + '.avi')

    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = output_path.replace('.avi', '.mp4')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    read_frames = 0

    start_time = datetime.now()
    print('Detecting...')

    lod = []
    while cap.isOpened():
        retflag, frame = cap.read()
        frame2 = frame
        # frame2 = copy.deepcopy(frame)
        draw_area_mask(frame)

        read_frames += 1

        if retflag:
            frame_tensor = cv_image2tensor(frame, input_size).unsqueeze(0)
            frame_tensor = Variable(frame_tensor)

            if args.cuda:
                frame_tensor = frame_tensor.cuda()

            detections = model(frame_tensor, args.cuda).cpu()
            detections = process_result(detections, args.obj_thresh, args.nms_thresh,
                                        relevant_classes_indices, class_mapping)

            if len(detections) != 0:
                detections = transform_result(detections, [frame], input_size)

            if len(detections) == 0:
                tracked_objects = mot_tracker.update(detections.cpu().detach().numpy())
            else:
                tracked_objects = mot_tracker.update(detections[:, 1:].cpu().detach().numpy())

            if args.debug_trackers:
                for n, tracker in enumerate(mot_tracker.trackers):
                    tracker_bb = tracker.predict()[0]
                    tracker_id = tracker.id
                    draw_simple_bbox(frame, tracker_bb, f"{tracker_id}")
            else:
                if len(tracked_objects) != 0:
                    for obj in tracked_objects:
                        bbox = obj[:4]
                        uid = int(obj[4])
                        cls_ind = int(obj[5])
                        draw_collision_lines(frame2)
                        draw_bbox([frame2], bbox, uid, cls_ind, colors, classes)

                        # count_object(bbox, uid)
                        count_lane(bbox, uid)

                        draw_count(frame2, f'A op: 1 af: 2 | B op: 2 af: 3 | C op: 2 af: 1 | D op: 7 af: 3')
                        draw_direction_letter(frame2)

                        add_bbox_history(uid, bbox)

                        lod.append(format_output(bbox, uid, cls_ind, classes, read_frames, output_path, fps))

            if not args.no_show:
                cv2.imshow('frame', frame2)
            out.write(frame2)
            if read_frames % 30 == 0:
                print(f'Frames processed: {read_frames / total_frames * 100:0.2f}%')
            if not args.no_show and cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    end_time = datetime.now()
    print(f'Detection finished in {end_time - start_time}')
    print('Total frames:', read_frames)
    cap.release()
    out.release()
    if not args.no_show:
        cv2.destroyAllWindows()

    print('Detected video saved to ' + output_path)

    name = output_path.replace('.mp4', '.csv')
    pd.DataFrame(lod).to_csv(name, index=False)
    print('Detected meta data saved as ' + name)


def count_object(bbox, uid):
    x0, y0, x1, y1 = bbox.tolist()
    box = Polygon([(x0, y0), (x1, y0), (x0, y1), (x1, y1)])

    tb_line = LineString(tb_line_points)
    bt_line = LineString(bt_line_points)

    if box.intersects(tb_line) and uid not in tb and uid not in bt:
        tb.append(uid)
        add_to_csv('left', bbox, uid)

    if box.intersects(bt_line) and uid not in bt and uid not in tb:
        bt.append(uid)
        add_to_csv('right', bbox, uid)


def count_lane(bbox, uid):
    if uid not in bbox_history.keys():
        return

    point = get_center_bottom_point(bbox)
    history_point = bbox_history[uid]

    line = LineString([history_point, point])

    merged_lanes = left_lane[0] + left_lane[1] + left_lane[2] + right_lane[0] + right_lane[1] + right_lane[2]

    left0, left1, left2 = LineString(left_lane[0]), LineString(left_lane[1]), LineString(left_lane[2])
    right0, right1, right2 = LineString(right_lane[0]), LineString(right_lane[1]), LineString(right_lane[2])

    update_line_on_intersect(line, left0, left_lane_count[0], merged_lanes, uid, 0)
    update_line_on_intersect(line, left1, left_lane_count[1], merged_lanes, uid, 1)
    update_line_on_intersect(line, left2, left_lane_count[2], merged_lanes, uid, 2)

    update_line_on_intersect(line, right0, right_lane_count[0], merged_lanes, uid, 0)
    update_line_on_intersect(line, right1, right_lane_count[1], merged_lanes, uid, 1)
    update_line_on_intersect(line, right2, right_lane_count[2], merged_lanes, uid, 2)


def update_line_on_intersect(line, lane, lane_count, merged_lanes, uid, lane_number):
    if line.intersects(lane) and uid not in merged_lanes:
        lane_count.append(uid)
        update_lane_in_csv(uid, lane_number)


def get_center_bottom_point(bbox):
    x0, y0, x1, y1 = bbox.tolist()
    x = ((x0 + x1) / 2)

    return x, y1


def add_bbox_history(uid, bbox):
    bbox_history[uid] = get_center_bottom_point(bbox)


def draw_count(f, value):
    color = (255, 0, 0)
    thickness = 2
    org = (50, 50)
    fontScale = 1
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(f, value, org, font, fontScale, color, thickness, cv2.LINE_AA)


def draw_direction_letter(f):
    color = (0, 255, 0)
    thickness = 2
    fontScale = 1
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(f, 'A', (307, 606), font, fontScale, color, thickness, cv2.LINE_AA)
    cv2.putText(f, 'B', (1586, 572), font, fontScale, color, thickness, cv2.LINE_AA)
    cv2.putText(f, 'C', (531, 166), font, fontScale, color, thickness, cv2.LINE_AA)
    cv2.putText(f, 'D', (1304, 145), font, fontScale, color, thickness, cv2.LINE_AA)


def draw_area_mask(f):
    center_coordinates = (940, 480)
    axesLength = (750 + 500, 450 + 500)
    angle = 0
    startAngle = 0
    endAngle = 360

    color = (0, 0, 0)
    thickness = 1000

    cv2.ellipse(f, center_coordinates, axesLength, angle, startAngle, endAngle, color, thickness)

    points = [[883, 95], [987, 113], [987, 130], [883, 110]]
    pts = numpy.array(points, numpy.int32)
    cv2.fillPoly(f, [pts], color)


def draw_collision_lines(f):
    color_green = (0, 255, 0)
    color_red = (0, 0, 255)
    color_blue = (255, 0, 0)

    cv2.line(f, line_a[0][0], line_a[0][1], color_red, 5)
    cv2.line(f, line_a[1][0], line_a[1][1], color_blue, 5)

    cv2.line(f, line_b[0][0], line_b[0][1], color_red, 5)
    cv2.line(f, line_b[1][0], line_b[1][1], color_blue, 5)

    cv2.line(f, line_c[0][0], line_c[0][1], color_red, 5)
    cv2.line(f, line_c[1][0], line_c[1][1], color_blue, 5)

    cv2.line(f, line_d[0][0], line_d[0][1], color_red, 5)
    cv2.line(f, line_d[1][0], line_d[1][1], color_blue, 5)


def add_to_csv(lane, bbox, uid):
    count_vehicles.append({
        'lane': lane,
        'box': bbox,
        'uid': uid
    })

    save_lanes_csv()


def save_lanes_csv():
    pd.DataFrame(count_vehicles).to_csv("output/lanes.csv")


def update_lane_in_csv(uid, lane_index):
    vehicles = filter(lambda x: x['uid'] == uid, count_vehicles)

    for vehicle in vehicles:
        vehicle['lane_number'] = int(lane_index)

    save_lanes_csv()


def main():
    args = parse_args()

    if args.cuda and not torch.cuda.is_available():
        print("ERROR: cuda is not available, try running on CPU")
        sys.exit(1)

    print('Loading network...')
    model = Darknet("cfg/yolov3.cfg")
    model.load_weights('yolov3.weights')
    if args.cuda:
        model.cuda()

    model.eval()
    print('Network loaded')

    detect_video(model, args)


if __name__ == '__main__':
    main()

