import csv
import pandas as pd

data = list()
frame_threshold = 40  # matchen van twee items mag maximaal zoveel frames tussen zitten


def read_csv_to_dict():
    with open("output/roundabout.csv") as result:
        stream = csv.DictReader(result)
        for item in stream:
            data.append(item)


def save_csv():
    pd.DataFrame(data).to_csv("output/roundabout_post.csv", index=False)


def remove_waste():
    remove = []

    for item in data:
        if item['enter'] is '' and item['leave'] is '':
            remove.append(item['uid'])  # add to remove queue

    for i in remove:  # for all ids in the queue
        items = list(filter(lambda x: x['uid'] == i, data))
        for item in items:  # all items matching the uid
            data.remove(item)


def match_paths():
    vehicles_without_leave = list(filter(lambda x: x['leave'] == '', data))

    for vehicle in vehicles_without_leave:
        frame_number = int(vehicle['ending_frame'])
        vehicles_without_enter = list(filter(lambda x: x['enter'] == '', data))

        possible_matches = list(filter(lambda car: 0 < (int(car['starting_frame']) - frame_number) < frame_threshold,
                                       vehicles_without_enter))

        if len(possible_matches) > 0:
            match = possible_matches[0]

            remove = list(filter(lambda m: m['uid'] == match['uid'], data))

            vehicle['ending_frame'] = match['ending_frame']
            vehicle['ending_duration'] = match['ending_duration']
            vehicle['ending_duration'] = match['ending_duration']
            vehicle['leave'] = match['leave']

            vehicles_without_enter.remove(match)

            for r in remove:
                data.remove(r)


def clean_after_match():
    remove = list(filter(lambda x: x['enter'] == '' or x['leave'] == '', data))

    if len(remove) > 0:
        for r in remove:
            data.remove(r)


def main():
    # note, enter/leave contains a letter where the car enters/leaves the roundabout
    # read all csv records
    read_csv_to_dict()
    # remove all cars that did not enter or leave the roundabout
    remove_waste()
    # find out what cars should be te same
    match_paths()
    # remove all entries without enter or leave
    clean_after_match()

    save_csv()


if __name__ == '__main__':
    main()
