import csv
import json
from collections import OrderedDict
from pprint import pprint
from shapely.geometry import Point, Polygon


def read_csv_to_dict():
    objects = list()
    with open("results/results_det_15s.csv") as result:
        stream = csv.DictReader(result)
        for item in stream:
            objects.append(item)
    return objects


def get_collision_areas():
    return {
        'a': Polygon([(0, 0), (320, 0), (320, 98), (0, 98)]),
        'b': Polygon([(320, 0), (640, 0), (640, 98), (320, 98)]),
        'c': Polygon([(0, 98), (320, 98), (320, 400), (0, 400)]),
        'd': Polygon([(320, 98), (640, 98), (640, 400), (320, 400)])
    }


def determine_object_path(o: OrderedDict) -> dict:
    collision_areas = get_collision_areas()
    start_coord = json.loads(o['start_coord'])
    end_coord = json.loads(o['end_coord'])

    return {
        'uid': o['uid'],
        'start': determine_object_pip(start_coord, collision_areas),
        'exit': determine_object_pip(end_coord, collision_areas)
    }


def determine_object_pip(coord, collision_areas):
    for entry, collision_area in collision_areas.items():
        if Point(coord[0] / 3, coord[1] / 3).within(collision_area):
            return entry
    return 'unknown'


def filter_same_points(path):
    if path['start'] == path['exit']:
        return False
    else:
        return True


def main():
    paths = list()
    objects = read_csv_to_dict()
    for o in objects:
        paths.append(determine_object_path(o))

    paths = list(filter(filter_same_points, paths))

    pprint(paths)


if __name__ == '__main__':
    main()
