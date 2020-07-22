"""
Created on Mon July 07 20:01:56 2020

@author: ji758507
"""
import pandas as pd
import numpy as np
import os
import time
import datetime
from scipy.spatial import distance
from Trajectory_processing import get_trajectory_slope, ped_reclassification
from numpy.linalg import inv
import cv2
from bokeh.io import output_file, show
from bokeh.models import ColumnDataSource, GMapOptions
from bokeh.plotting import gmap
import geopy.distance


def trajectory_cleaning(trajectory_data):
    """
    :param trajectory_data: p1x, p1y represent the upper left corner; p2x, p2y represent the bottom right corner;
            bx, by represent center point; cx, cy represent bottom middle point;
            type: https://gist.github.com/AruniRC/7b3dadd004da04c80198557db5da4bda, id-1
            lostCounter: number of frames
    :param threshold: threshold for identify conflict
    :return: identified conflicts
    """
    # exclude the detection id with insufficient trajectory (frames)
    number_of_frames = trajectory_data.groupby(by='objectID', as_index=False).size().reset_index(name='counts')
    list_of_id = number_of_frames.loc[number_of_frames['counts'] >= 30]['objectID'].tolist()
    trajectory_data = trajectory_data.loc[trajectory_data['objectID'].isin(list_of_id)].reset_index(drop=True)

    # exclude the detection id with insufficient trajectory (length)
    # trajectory_length = get_trajectory_slope(trajectory_data, [1, 2, 3, 4, 5, 6, 7, 8, 9])
    # stopped_object = trajectory_length.loc[trajectory_length['trajectory_length'] <= 25]['objectID'].tolist()
    # trajectory_data = trajectory_data.loc[~trajectory_data['objectID'].isin(stopped_object)].reset_index(drop=True)

    return trajectory_data


def identify_actual_ped_num(pedestrian_trajectory):
    # identify the actual number of pedestrian
    ped_trajectory_start_end = get_trajectory_slope(pedestrian_trajectory, [1, 2, 3, 4, 5, 6, 7, 8, 9])

    ped_trajectory_start_end['actual_ped_num'] = 0
    for i in range(len(ped_trajectory_start_end) - 1):
        first_x = ped_trajectory_start_end.loc[i, 'cx_first']
        last_x = ped_trajectory_start_end.loc[i, 'cx_last']
        last_y = ped_trajectory_start_end.loc[i, 'cy_last']
        last_frame = ped_trajectory_start_end.loc[i, 'frameNUM_last']
        ped_num = ped_trajectory_start_end.loc[i, 'actual_ped_num']
        angle = ped_trajectory_start_end.loc[i, 'trajectory_angle']

        slope = ped_trajectory_start_end.loc[i, 'trajectory_slope']
        angle_r = (np.arctan(slope) / np.pi) * 180 + 90

        for j in range(i + 1, len(ped_trajectory_start_end)):
            first_x_2 = ped_trajectory_start_end.loc[j, 'cx_first']
            first_y_2 = ped_trajectory_start_end.loc[j, 'cy_first']
            first_frame_2 = ped_trajectory_start_end.loc[j, 'frameNUM_first']
            ped_num_2 = ped_trajectory_start_end.loc[j, 'actual_ped_num']
            angle_2 = ped_trajectory_start_end.loc[j, 'trajectory_angle']

            dist = distance.euclidean((last_x, last_y), (first_x_2, first_y_2))
            connect_slope = (first_y_2 - last_y)/(first_x_2 - last_x)
            connect_angle = (np.arctan(connect_slope) / np.pi) * 180 + 90

            # select the threshold of 25 to identify whether the two trajectory belong to one ped
            if (dist < 100 and abs(angle_2 - angle) <= 15 and 0 < (first_frame_2 - last_frame) <= 100) \
                    or (dist < 100 and abs(connect_angle - angle_r) <= 15 and np.sign(last_x-first_x) == np.sign(first_x_2-last_x) and 0 < (first_frame_2 - last_frame) <= 100):
                ped_trajectory_start_end.loc[j, 'actual_ped_num'] = ped_num
            elif ped_num_2 >= ped_num:
                ped_trajectory_start_end.loc[j, 'actual_ped_num'] = ped_num + 1
            else:
                continue

    # agg_ped = ped_trajectory_start_end.groupby(by='actual_ped_num', as_index=False).agg({'objectID': lambda x: x.unique().tolist()})
    pedestrian_trajectory = pedestrian_trajectory.join(ped_trajectory_start_end[['objectID', 'actual_ped_num']].set_index('objectID'), on='objectID')
    pedestrian_trajectory = pedestrian_trajectory.sort_values(by=['actual_ped_num', 'frameNUM']).reset_index(drop=True)

    return pedestrian_trajectory, ped_trajectory_start_end[['objectID', 'actual_ped_num']]
# i = 0
# j = 1
# np.sign(-3)


def del_duplicate_trajectory(pedestrian_trajectory):
    # list_actual_ped = pedestrian_trajectory['actual_ped_num'].unique().tolist()
    tem_dict = pedestrian_trajectory.groupby(by='actual_ped_num').size().reset_index(name='count')
    tem_dict = tem_dict.sort_values(by='count', ascending=False).reset_index(drop=True)

    overlapped_id = []
    if len(tem_dict) >= 2:
        i = 0
        while i < len(tem_dict) - 1:
            actual_id = tem_dict.loc[i, 'actual_ped_num']
            ped_1 = pedestrian_trajectory[pedestrian_trajectory['actual_ped_num'] == actual_id].reset_index(drop=True)
            frame_range_1 = range(ped_1['frameNUM'].min(), ped_1['frameNUM'].max())
            # print(i)
            for j in range(i + 1, len(tem_dict)):
                actual_id_2 = tem_dict.loc[j, 'actual_ped_num']
                ped_2 = pedestrian_trajectory[pedestrian_trajectory['actual_ped_num'] == actual_id_2].reset_index(drop=True)
                frame_range_2 = range(ped_2['frameNUM'].min(), ped_2['frameNUM'].max())

                # check frame number overlapping
                xs = set(frame_range_1)
                overlap = xs.intersection(frame_range_2)

                if len(frame_range_2) > 0:
                    if len(overlap)/len(frame_range_2) > 0.95:
                        ped_1 = ped_1.join(ped_2.set_index('frameNUM'), on='frameNUM', lsuffix='_left', rsuffix='_right')
                        ped_1['relative_dist'] = np.linalg.norm(ped_1[['cx_left', 'cy_left']].values - ped_1[['cx_right', 'cy_right']].values, axis=1)
                        coefficient_of_variance = ped_1['relative_dist'].std()/ped_1['relative_dist'].mean()

                        # identify the overlapping based on the coefficient of variance in relative distance
                        if coefficient_of_variance < 0.15 and ped_1['relative_dist'].mean() < 50:
                            overlapped_id.append(actual_id_2)
                            tem_dict = tem_dict.drop(j, axis=0)

                        else:
                            # print(i, j)
                            continue
                    else:
                        # print(i, j)
                        continue
                else:
                    overlapped_id.append(actual_id_2)
                    tem_dict = tem_dict.drop(j, axis=0)
                    continue
                # print(i, j)

            tem_dict = tem_dict.reset_index(drop=True)
            i += 1

    else:
        return pedestrian_trajectory

    pedestrian_trajectory = pedestrian_trajectory[~pedestrian_trajectory['actual_ped_num'].isin(overlapped_id)].reset_index(drop=True)

    return pedestrian_trajectory


# j = 2
# i = 0
# len(ped_1)
def get_centroid(points):
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    centroid = (sum(x) / len(points), sum(y) / len(points))
    return centroid


def get_detection_zone_info(intersectionApproachDict, intersection, sub_video_start_time):

    if intersection == 'SR46_@_PARK' and sub_video_start_time <= datetime.datetime.strptime('2019-10-30 19:00:00', '%Y-%m-%d %H:%M:%S'):
        p2 = intersectionApproachDict.loc[(intersectionApproachDict['intersectionName'] == intersection) & (intersectionApproachDict['Sub'] == 1), 'P2'].values[0]
        p4 = intersectionApproachDict.loc[(intersectionApproachDict['intersectionName'] == intersection) & (intersectionApproachDict['Sub'] == 1), 'P4'].values[0]

    elif intersection == 'SR46_@_PARK':
        p2 = intersectionApproachDict.loc[(intersectionApproachDict['intersectionName'] == intersection) & (intersectionApproachDict['Sub'] == 2), 'P2'].values[0]
        p4 = intersectionApproachDict.loc[(intersectionApproachDict['intersectionName'] == intersection) & (intersectionApproachDict['Sub'] == 2), 'P4'].values[0]

    else:
        p2 = intersectionApproachDict.loc[intersectionApproachDict['intersectionName'] == intersection, 'P2'].values[0]
        p4 = intersectionApproachDict.loc[intersectionApproachDict['intersectionName'] == intersection, 'P4'].values[0]

    p2_array = np.array(eval(p2))
    p2_array.sort(axis=0)

    p4_array = np.array(eval(p4))
    p4_array.sort(axis=0)

    # calculate the distance based on the leftmost two points in zone 2 and rightmost two points in zone 4
    crosswalk_length = distance.euclidean(get_centroid(p2_array[0:2, :]), get_centroid(p4_array[-2:, :]))

    slope = (get_centroid(p4_array)[1] - get_centroid(p2_array)[1])/(get_centroid(p4_array)[0] - get_centroid(p2_array)[0])
    crosswalk_angle = (np.arctan(slope) / np.pi) * 180 + 90 # [-90, 90] to [0, 180], original 0 is the base

    return crosswalk_length, crosswalk_angle


def get_gps_from_pixel(x, y, M):
    pixel = np.array([x, y]).reshape(1, 2)
    pixel = np.concatenate([pixel, np.ones((pixel.shape[0], 1))], axis=1)
    lonlat = np.dot(M, pixel.T)

    lat = (lonlat[:2, :] / lonlat[2, :]).T[0, 0]
    lon = (lonlat[:2, :] / lonlat[2, :]).T[0, 1]

    return lat, lon


def identify_crossing_ped(trajectory_data, crosswalk_length, crosswalk_angle):
    pedestrian_trajectory = trajectory_data.loc[(trajectory_data['type'].isin([0, 1])) & (trajectory_data['zone'].isin([1, 2, 3, 4, 5, 6, 7, 8, 9])), :].sort_values(by=['objectID', 'frameNUM']).reset_index(drop=True)

    # Exclude tiny detections and pedestrian in vehicle (y axis smaller than 50)
    # pedestrian_trajectory = pedestrian_trajectory.loc[pedestrian_trajectory['p2y'] >= (pedestrian_trajectory['p1y'] + 50)].reset_index(drop=True)

    ped_list = pedestrian_trajectory['objectID'].unique()

    # identify the actual number of pedestrian
    pedestrian_trajectory, ped_id_dict = identify_actual_ped_num(pedestrian_trajectory)

    # delete duplicated pedestrian trajectory detection
    pedestrian_trajectory = del_duplicate_trajectory(pedestrian_trajectory)
    zone_list = pedestrian_trajectory.groupby(by='actual_ped_num', as_index=False).agg({'zone': lambda x: "_".join(map(str, x.unique().tolist()))})

    # identify the crossing ped (zone include 2,3,4,5,6; trajectory length > half of road width; direction is within a threshold of 2-3-4)
    tem_ped_trajectory = pedestrian_trajectory.copy()
    tem_ped_trajectory['objectID'] = tem_ped_trajectory['actual_ped_num']
    tem_ped_trajectory = tem_ped_trajectory.drop(['actual_ped_num'], axis=1)
    tem_agg_ped_trajectory = get_trajectory_slope(tem_ped_trajectory, [2, 3, 4, 5, 6])

    if len(tem_agg_ped_trajectory) > 0:
        tem_agg_ped_trajectory = tem_agg_ped_trajectory.join(zone_list.set_index('actual_ped_num'), on='objectID')

        # convert the trajectory_angle from [90, 180] and [0, 90] to [0, 90] and [90, 180]
        tem_agg_ped_trajectory['trajectory_angle'] = (np.arctan(tem_agg_ped_trajectory['trajectory_slope'])/np.pi)*180 + 90

        # identify the crossing pedestrian based on two types of conditions
        # tem_agg_ped_trajectory['crossing_ped'] = np.where(((tem_agg_ped_trajectory['zone'].str.contains('|'.join(['2', '3', '4']))) & (tem_agg_ped_trajectory['trajectory_length'] > 0.7 * crosswalk_length))
        #                                                   | ((tem_agg_ped_trajectory['trajectory_angle'].between(crosswalk_angle - 22.5, crosswalk_angle + 22.5)) & (tem_agg_ped_trajectory['trajectory_length'] > 0.7 * crosswalk_length)), 1, 0)
        tem_agg_ped_trajectory['crossing_ped'] = np.where((tem_agg_ped_trajectory['trajectory_angle'].between(crosswalk_angle - 22.5, crosswalk_angle + 22.5)) & (tem_agg_ped_trajectory['trajectory_length'] > 0.5 * crosswalk_length), 1, 0)

        crossing_ped = tem_agg_ped_trajectory.loc[tem_agg_ped_trajectory['crossing_ped'] == 1, 'objectID'].values.tolist()

        if len(crossing_ped) > 0:
            crossing_ped_trajectory = pedestrian_trajectory[pedestrian_trajectory['actual_ped_num'].isin(crossing_ped)].reset_index(drop=True)
        else:
            return None
    else:
        return None

    return crossing_ped_trajectory


def get_gps_for_crossing_ped(intersectionApproachDict, intersection, crossing_ped_trajectory, sub_video_start_time):

    if intersection == 'SR46_@_PARK' and sub_video_start_time <= datetime.datetime.strptime('2019-10-30 19:00:00', '%Y-%m-%d %H:%M:%S'):
        lonlat = intersectionApproachDict.loc[(intersectionApproachDict['intersectionName'] == intersection) & (intersectionApproachDict['Sub'] == 1), 'validate_longlat'].values[0]
        pixel = intersectionApproachDict.loc[(intersectionApproachDict['intersectionName'] == intersection) & (intersectionApproachDict['Sub'] == 1), 'validate_pixel'].values[0]

    elif intersection == 'SR46_@_PARK':
        lonlat = intersectionApproachDict.loc[(intersectionApproachDict['intersectionName'] == intersection) & (intersectionApproachDict['Sub'] == 2), 'validate_longlat'].values[0]
        pixel = intersectionApproachDict.loc[(intersectionApproachDict['intersectionName'] == intersection) & (intersectionApproachDict['Sub'] == 2), 'validate_pixel'].values[0]

    else:
        lonlat = intersectionApproachDict.loc[intersectionApproachDict['intersectionName'] == intersection, 'validate_longlat'].values[0]
        pixel = intersectionApproachDict.loc[intersectionApproachDict['intersectionName'] == intersection, 'validate_pixel'].values[0]

    lonlat = np.array(eval(lonlat))
    pixel = np.array(eval(pixel))

    M = cv2.getPerspectiveTransform(np.float32(pixel),np.float32(lonlat))

    crossing_ped_trajectory['lat'] = [get_gps_from_pixel(x, y, M)[0] for x, y in zip(crossing_ped_trajectory['cx'], crossing_ped_trajectory['cy'])]
    crossing_ped_trajectory['lon'] = [get_gps_from_pixel(x, y, M)[1] for x, y in zip(crossing_ped_trajectory['cx'], crossing_ped_trajectory['cy'])]

    return crossing_ped_trajectory


def dist_to_crossing(test, a, b):
    dist = []
    for i in range(len(test)):
        a_long = test.loc[i, a]
        b_lat = test.loc[i, b]

        c_long = test[a].values[-1]
        d_lat = test[b].values[-1]

        dist.append(geopy.distance.geodesic((b_lat, a_long), (d_lat, c_long)).meters)

    return dist


def total_dist(test, a, b, c, d):
    a_long = test.loc[0, a]
    b_lat = test.loc[0, b]

    c_long = test[c].values[-1]
    d_lat = test[d].values[-1]

    dist = geopy.distance.geodesic((b_lat, a_long), (d_lat, c_long)).meters

    return dist


def adjacent_distance_r(test, a, b):
    dist = []
    for i in range(len(test)):

        if i < len(test) - 1:
            a_long = test.loc[i, a]
            b_lat = test.loc[i, b]
            c_long = test.loc[i+1, a]
            d_lat = test.loc[i+1, b]
        else:
            a_long = test.loc[i, a]
            b_lat = test.loc[i, b]
            c_long = test.loc[i, a]
            d_lat = test.loc[i, b]

        dist.append(geopy.distance.geodesic((b_lat, a_long), (d_lat, c_long)).meters)

    return dist


def adjacent_distance(test, a, b, c, d):
    dist = []
    for i in range(len(test)):
        a_long = test.loc[i, a]
        b_lat = test.loc[i, b]

        c_long = test.loc[i, c]
        d_lat = test.loc[i, d]

        dist.append(geopy.distance.geodesic((b_lat, a_long), (d_lat, c_long)).meters)

    return dist


def get_interval_frame_speed(test, a, b, interval):
    speed = []
    for i in range(len(test)):

        if i >= interval:
            a_long = test.loc[i, a]
            b_lat = test.loc[i, b]
            c_long = test.loc[i-interval, a]
            d_lat = test.loc[i-interval, b]

            time = (test.loc[i, 'frameNUM'] - test.loc[i-interval, 'frameNUM'])/30
        else:
            a_long = test.loc[i, a]
            b_lat = test.loc[i, b]
            c_long = test.loc[0, a]
            d_lat = test.loc[0, b]

            time = (test.loc[i, 'frameNUM'] - test.loc[0, 'frameNUM'])/30

        speed.append(geopy.distance.geodesic((b_lat, a_long), (d_lat, c_long)).meters/time)

    return speed


def get_waiting_crossing_time(tem_crossing_trajectory, waiting_zone, df_frame_time, sub_video_start_frame):
    ped_arrive_trajectory = tem_crossing_trajectory[tem_crossing_trajectory['zone'] == waiting_zone].reset_index(drop=True)
    ped_arrive_trajectory['dist_to_crossing'] = dist_to_crossing(ped_arrive_trajectory, 'lon', 'lat')
    ped_arrive_trajectory['adjacent_dist'] = adjacent_distance_r(ped_arrive_trajectory, 'lon', 'lat')
    ped_arrive_trajectory['frame_speed'] = get_interval_frame_speed(ped_arrive_trajectory, 'lon', 'lat', 15)

    # ped_arrive_trajectory['frame_speed'] = ped_arrive_trajectory['adjacent_dist'] / ((ped_arrive_trajectory['frameNUM'].shift(-1) - ped_arrive_trajectory['frameNUM']) / 30)
    ped_arrive_trajectory['frame_speed'] = ped_arrive_trajectory['frame_speed'].rolling(15).mean()
    ped_arrive_trajectory['frame_speed'] = ped_arrive_trajectory['frame_speed'].fillna(method='backfill', axis=0)

    # select 2 meters as waiting area, waiting time is determined by speed less than 0.5 m/s
    waiting_time = (ped_arrive_trajectory[(ped_arrive_trajectory['dist_to_crossing'] <= 3.5) & (ped_arrive_trajectory['frame_speed'] <= 0.5)]['frameNUM'].max()
                    - ped_arrive_trajectory[(ped_arrive_trajectory['dist_to_crossing'] <= 3.5) & (ped_arrive_trajectory['frame_speed'] <= 0.5)]['frameNUM'].min()) / 30

    # ped arriving time
    # ped_arriving_time = sub_video_start_time + datetime.timedelta(seconds=ped_arrive_trajectory[ped_arrive_trajectory['dist_to_crossing'] <= 2]['frameNUM'].min() / 30)
    ped_arriving_time = df_frame_time.loc[df_frame_time['frameNum'] == (sub_video_start_frame + ped_arrive_trajectory[ped_arrive_trajectory['dist_to_crossing'] <= 3.5]['frameNUM'].min()), 'time'].values[0]
    # ped crossing time
    # start_crossing_time = sub_video_start_time + datetime.timedelta(seconds=ped_arrive_trajectory['frameNUM'].values[-1] / 30)
    start_crossing_time = df_frame_time.loc[df_frame_time['frameNum'] == (sub_video_start_frame + ped_arrive_trajectory['frameNUM'].values[-1]), 'time'].values[0]
    end_crossing_time = df_frame_time.loc[df_frame_time['frameNum'] == (sub_video_start_frame + tem_crossing_trajectory.loc[tem_crossing_trajectory['zone'].isin([2, 3, 4, 5, 6]), 'frameNUM'].values[-1]), 'time'].values[0]

    return waiting_time, ped_arriving_time, start_crossing_time, end_crossing_time


# def get_ped_crossing_traffic_signal(video_name, ped_arriving_time, start_crossing_time, atspm_path):

def get_ped_crossing_behavior(crossing_ped_trajectory, df_frame_time, sub_video_start_frame, video_name, sub_id):

    # identify the direction of travel (2->4 (0) or 4->2 (1))
    first = crossing_ped_trajectory.groupby('actual_ped_num', as_index=False).first()
    last = crossing_ped_trajectory.groupby('actual_ped_num', as_index=False).last()
    first = first.merge(last, how='left', left_on='actual_ped_num', right_on='actual_ped_num', suffixes=('_first', '_last')).reset_index(drop=True)

    first['crossing_direction'] = np.where(first['cx_first'] > first['cx_last'], 1, 0)

    crossing_behavior = []
    # waiting time on zone 8 (4->2) or zone 1 (2->4)
    for i in range(len(first)):
        crossing_direction = first.loc[i, 'crossing_direction']
        actual_ped_id = first.loc[i, 'actual_ped_num']
        tem_crossing_trajectory = crossing_ped_trajectory[crossing_ped_trajectory['actual_ped_num'] == actual_ped_id].reset_index(drop=True)
        original_ped_ids = tem_crossing_trajectory['objectID'].unique().tolist()
        zone_list = tem_crossing_trajectory['zone'].unique().tolist()

        # select only the crossing region
        tem = tem_crossing_trajectory[tem_crossing_trajectory['zone'].isin([2, 3, 4, 5, 6])].reset_index(drop=True)

        # check the spatial violation (walking not on crosswalk)
        spatial_violation = [1 if len(tem[tem['zone'].isin([5, 6])]) >= 30 else 0][0]

        # to reduce the instability, calculate speed at every second (30 frames)
        tem['frame_group'] = tem['frameNUM']//30

        tem_tail = tem.groupby(by='frame_group', as_index=False).last().reset_index(drop=True)
        tem_head = tem.groupby(by='frame_group', as_index=False).first().reset_index(drop=True)

        tem = tem_tail.join(tem_head.set_index('frame_group'), on='frame_group', lsuffix='_tail', rsuffix='_head').reset_index(drop=True)
        tem['crossing_distance'] = adjacent_distance(tem, 'lon_tail', 'lat_tail', 'lon_head', 'lat_head')
        tem['crossing_time'] = (tem['frameNUM_tail'] - tem['frameNUM_head'])/30
        tem['crossing_speed'] = tem['crossing_distance']/tem['crossing_time']

        total_crossing_distance = total_dist(tem, 'lon_head', 'lat_head', 'lon_tail', 'lat_tail')

        # summarize the avg and std crossing speed
        if total_crossing_distance > tem['crossing_distance'].sum():
            avg_crossing_speed = total_crossing_distance/((tem['frameNUM_tail'].values[-1] - tem['frameNUM_head'].values[0])/30)
        else:
            avg_crossing_speed = tem['crossing_distance'].sum()/((tem['frameNUM_tail'].values[-1] - tem['frameNUM_head'].values[0])/30)

        std_crossing_speed = tem['crossing_speed'].std()

        # calculate the waiting time and start crossing time
        if crossing_direction == 1 and 8 in zone_list:
            waiting_time, ped_arriving_time, start_crossing_time, end_crossing_time = get_waiting_crossing_time(tem_crossing_trajectory, 8, df_frame_time, sub_video_start_frame)

        elif crossing_direction == 1 and 8 not in zone_list:
            waiting_time = 0
            ped_arriving_time = df_frame_time.loc[df_frame_time['frameNum'] == (sub_video_start_frame + tem_crossing_trajectory['frameNUM'].values[0]), 'time'].values[0]
            start_crossing_time = df_frame_time.loc[df_frame_time['frameNum'] == (sub_video_start_frame + tem_crossing_trajectory['frameNUM'].values[0]), 'time'].values[0]
            end_crossing_time = df_frame_time.loc[df_frame_time['frameNum'] == (sub_video_start_frame + tem_crossing_trajectory.loc[tem_crossing_trajectory['zone'].isin([2, 3, 4, 5, 6]), 'frameNUM'].values[-1]), 'time'].values[0]

        elif crossing_direction == 0 and 1 in zone_list:
            waiting_time, ped_arriving_time, start_crossing_time, end_crossing_time = get_waiting_crossing_time(tem_crossing_trajectory, 1, df_frame_time, sub_video_start_frame)

        elif crossing_direction == 0 and 1 not in zone_list:
            waiting_time = 0
            ped_arriving_time = df_frame_time.loc[df_frame_time['frameNum'] == (sub_video_start_frame + tem_crossing_trajectory['frameNUM'].values[0]), 'time'].values[0]
            start_crossing_time = df_frame_time.loc[df_frame_time['frameNum'] == (sub_video_start_frame + tem_crossing_trajectory['frameNUM'].values[0]), 'time'].values[0]
            end_crossing_time = df_frame_time.loc[df_frame_time['frameNum'] == (sub_video_start_frame + tem_crossing_trajectory.loc[tem_crossing_trajectory['zone'].isin([2, 3, 4, 5, 6]), 'frameNUM'].values[-1]), 'time'].values[0]

        crossing_behavior.append({'original_ped_ids': original_ped_ids, 'actual_ped_id': actual_ped_id, 'crossing_direction': crossing_direction,
                                  'avg_crossing_speed': avg_crossing_speed, 'std_crossing_speed': std_crossing_speed, 'ped_arriving_time': ped_arriving_time,
                                  'waiting_time': waiting_time, 'start_crossing_time': start_crossing_time, 'end_crossing_time': end_crossing_time, 'spatial_violation': spatial_violation, 'video_name': video_name, 'sub_id': sub_id})

    crossing_behavior = pd.DataFrame(crossing_behavior)
    return crossing_behavior

# i = 0
# # transformation US17-92_@_13TH_ST
# a = np.array([[28.801304, -81.273222],[28.801257, -81.273002],[28.800994, -81.273059],[28.800965, -81.273201]])
# b = np.array([[384,287],[854,277],[985,380],[481,444]])
#
# # transformation US17-92_@_25TH_ST
# a = np.array([[28.787582, -81.273173], [28.787454, -81.273068], [28.786817, -81.272919],[28.786809, -81.273165]])
# b = np.array([[628,246],[717,261],[830,503],[49,465]])
#
# # transformation US17-92_@_3RD_ST
# a = np.array([[28.809253, -81.273116],[28.809447, -81.273364],[28.809728, -81.273190],[28.809730, -81.273077]])
# b = np.array([[311,419],[830,445],[822,584],[354,620]])
#
# # transformation SR46_@_PARK (first 20)
# a = np.array([[28.786625, -81.268236],[28.786726, -81.268386],[28.786764, -81.267950],[28.786525, -81.267965]])
# b = np.array([[763,416],[1061,393],[1018,696],[115,522]])
#
# # transformation SR46_@_PARK (last 80)
# a = np.array([[28.786625, -81.268236],[28.786726, -81.268386],[28.786726, -81.268136],[28.786626, -81.268086]])
# b = np.array([[539,459],[862,419],[776,525],[390,536]])
#
# # transformation SR434_@_WINDING_HOLLOW
# a = np.array([[28.704726, -81.281816],[28.704869, -81.281737],[28.704924, -81.281524],[28.704661, -81.281398]])
# b = np.array([[368,230],[692,251],[1015,315],[139,408]])
#
# # transformation HOWELL_BR_@_LK_HOWELL
# a = np.array([[28.624552, -81.323967],[28.624818, -81.324084],[28.624961, -81.324087],[28.624982, -81.323920]])
# b = np.array([[868,445],[897,544],[677,648],[95,569]])
#
# M = cv2.getPerspectiveTransform(np.float32(b),np.float32(a))

# M = np.load('US17-92_@_13TH_ST_10022019_144715_1.npy')
# M = np.load('US17-92_@_25TH_ST_10032019_094715_7.npy')
# M = np.load('US17-92_@_3RD_ST_10022019_144715_3.npy')
#
# pixel = np.array(trajectory_data.loc[0, ['cx', 'cy']]).reshape(1, 2)
# pixel = np.array([353,493]).reshape(1, 2)
# pixel.shape
# pixel = np.concatenate([pixel, np.ones((pixel.shape[0], 1))], axis=1)
# lonlat = np.dot(M, pixel.T)
#
# (lonlat[:2, :] / lonlat[2, :]).T

# get_gps_from_pixel(353, 493, M)

# coordinates transformation
# trajectory_data['lat'] = [get_gps_from_pixel(x, y, M)[0] for x, y in zip(trajectory_data['cx'], trajectory_data['cy'])]
# trajectory_data['lon'] = [get_gps_from_pixel(x, y, M)[1] for x, y in zip(trajectory_data['cx'], trajectory_data['cy'])]

# output_file("gmap.html")
# # map_options = GMapOptions(lat=28.801257, lng=-81.273002, map_type="satellite", zoom=23)
# # map_options = GMapOptions(lat=28.787454, lng=-81.273068, map_type="satellite", zoom=23)
# # map_options = GMapOptions(lat=28.809447, lng=-81.273364, map_type="satellite", zoom=23)
# # map_options = GMapOptions(lat=28.786727, lng=-81.268391, map_type="satellite", zoom=23)
# # map_options = GMapOptions(lat=28.704869, lng=-81.281737, map_type="satellite", zoom=23)
# map_options = GMapOptions(lat=28.624818, lng=-81.324084, map_type="satellite", zoom=23)
# p = gmap("AIzaSyD1NilerEg8s-qVmUQEnHIOffxLxItIZeA&libraries=visualization&callback=initMap", map_options, title="Austin")
# p.circle(x="lon", y="lat", size=5, fill_alpha=0.8, source=trajectory_data)
# show(p)


def check_frame_num_file():
    """
    :param CCTV_Ped:
    :return:
    """
    # get the root path
    root_path = os.getcwd()
    intersection_path = os.path.join(root_path, 'Processed Video Data')
    intersections_folder = os.listdir(intersection_path)

    video_list = []
    for intersection in intersections_folder:
        # get the path of sub_intersection folders
        sub_intersections_path = os.path.join(intersection_path, intersection)

        # for select intersection, filter all the sub videos with pedestrians
        intersection_pedestrain_subvideos = CCTV_Ped.loc[(CCTV_Ped['Video Name'].str.contains(intersection))
                                                         & ((CCTV_Ped['Pedestrian'] == '1') | (CCTV_Ped['Cyclist'] == 1))].reset_index(drop=True)

        for i in range(len(intersection_pedestrain_subvideos)):
            video_name = intersection_pedestrain_subvideos.loc[i, 'Video Name']
            sub_id = intersection_pedestrain_subvideos.loc[i, 'Sub_ID']
            multi_ped = intersection_pedestrain_subvideos.loc[i, 'Multi_Ped']

            # get the time frame information for this 1-hour video
            path_frame = os.path.join(root_path, 'Filtered Video', intersection, video_name, video_name + '.csv')
            try:
                df_frame = pd.read_csv(path_frame, index_col=False)
            except:
                print(video_name)
                video_list.append(video_name)
                continue

    return video_list


# video_list = check_frame_file()
# video_list.sort()
# myset = set(video_list)


def get_ped_veh_conflict(trajectory_data, crossing_ped_trajectory, threshold, intersect_buffer_x, intersect_buffer_y):
    """
    :param trajectory_data: p1x, p1y represent the upper left corner; p2x, p2y represent the bottom right corner;
            bx, by represent center point; cx, cy represent bottom middle point;
            type: https://gist.github.com/AruniRC/7b3dadd004da04c80198557db5da4bda, id-1
            lostCounter: number of frames
    :param threshold: threshold for identify conflict
    :return: identified conflicts
    """
    threshold_frame = threshold * 30
    # exclude the detection id with insufficient trajectory (frames)
    number_of_frames = trajectory_data.groupby(by='objectID', as_index=False).size().reset_index(name='counts')
    list_of_id = number_of_frames.loc[number_of_frames['counts'] >= 30]['objectID'].tolist()
    trajectory_data = trajectory_data.loc[trajectory_data['objectID'].isin(list_of_id)]

    # exclude the detection id with insufficient trajectory (length)
    trajectory_length = get_trajectory_slope(trajectory_data, [1, 2, 3, 4, 5, 6, 7, 8, 9])
    stoped_object = trajectory_length.loc[trajectory_length['trajectory_length'] <= 25]['objectID'].tolist()
    trajectory_data = trajectory_data.loc[~trajectory_data['objectID'].isin(stoped_object)].reset_index(drop=True)

    pedestrian_trajectory = crossing_ped_trajectory.loc[crossing_ped_trajectory['zone'].isin([2, 3, 4, 5, 6, 7, 9]), :].sort_values(by=['objectID', 'frameNUM']).reset_index(drop=True)
    # Exclude tiny detections and pedestrian in vehicle (y axis smaller than 50)
    # pedestrian_trajectory = pedestrian_trajectory.loc[pedestrian_trajectory['p2y'] >= (pedestrian_trajectory['p1y'] + 50)].reset_index(drop=True)

    veh_trajectory = trajectory_data.loc[trajectory_data['type'].isin([2, 5, 7])].sort_values(by=['objectID', 'frameNUM']).reset_index(drop=True)

    ped_list = pedestrian_trajectory['actual_ped_num'].unique()

    ped_veh_conflicts = []
    for ped in ped_list:
        ped_trajectory = pedestrian_trajectory.loc[pedestrian_trajectory['actual_ped_num'] == ped].reset_index(drop=True)

        # filter out all the non-essential vehicle trajectories for PET calculation
        tem_veh = veh_trajectory.loc[(veh_trajectory['p1x'] <= ped_trajectory['cx'].max()) & (veh_trajectory['p2x'] >= ped_trajectory['cx'].min())
                                     & (veh_trajectory['p1y'] <= ped_trajectory['cy'].max()) & (veh_trajectory['p2y'] >= ped_trajectory['cy'].min())
                                     & (veh_trajectory['frameNUM'] <= (ped_trajectory['frameNUM'].max() + threshold_frame))
                                     & (veh_trajectory['frameNUM'] >= (ped_trajectory['frameNUM'].min() - threshold_frame))].reset_index(drop=True)

        # merge temporary vehicle trajectory data and ped trajectory
        tem_veh['key'] = 0
        ped_trajectory['key'] = 0
        conflict = tem_veh.merge(ped_trajectory, how='outer', left_on='key', right_on='key', suffixes=('_veh', '_ped')).reset_index(drop=True)
        # conflict = conflict.loc[conflict['frameNUM_veh'] >= conflict['frameNUM_ped']].reset_index(drop=True)

        # Euclidean distance between ped and veh
        conflict['distance'] = np.linalg.norm(conflict[['cx_veh', 'cy_veh']].values - conflict[['cx_ped', 'cy_ped']].values, axis=1)
        conflict['ped_veh_overlap'] = np.where((conflict['cx_ped'].between(conflict['p1x_veh'], conflict['p2x_veh'])) & (conflict['cy_ped'].between(conflict['p1y_veh'], conflict['p2y_veh'])), 1, 0)

        # Condition of buffer intersect
        conflict['ped_veh_buffer_intersect'] = np.where((conflict['cx_ped'].between(conflict['cx_veh'] - intersect_buffer_x, conflict['cx_veh'] + intersect_buffer_x))
                                                        & (conflict['cy_ped'].between(conflict['cy_veh'] - intersect_buffer_y, conflict['cy_veh'] + intersect_buffer_y)), 1, 0)
        conflict = conflict.loc[(conflict['ped_veh_overlap'] == 1) & (conflict['ped_veh_buffer_intersect'] == 1)].sort_values(by=['objectID_veh', 'frameNUM_veh']).reset_index(drop=True)

        if len(conflict) > 0:
            conflict = conflict.loc[conflict.groupby(by=['objectID_veh'], as_index=False)["distance"].idxmin()]
            conflict['pet'] = (conflict['frameNUM_veh'] - conflict['frameNUM_ped'])/30
            conflict = conflict.loc[(abs(conflict['pet']) <= threshold) & (conflict['lostCounter_ped'] < 30) & (conflict['objectID_veh'] != conflict['objectID_ped'])].reset_index(drop=True)
            ped_veh_conflicts.append(conflict)

        else:
            # print('no conflict')
            continue

    return ped_veh_conflicts


def extract_ped_trajectory(CCTV_Ped, intersectionApproachDict, threshold, intersect_buffer_x, intersect_buffer_y):
    """
    :param CCTV_Ped:
    :return:
    """
    # get the root path
    root_path = os.getcwd()
    intersection_path = os.path.join(root_path, 'Processed Video Data')
    intersections_folder = os.listdir(intersection_path)

    agg_crossing_peds = []
    trajectory_crossing_peds = []
    total_conflicts = []

    for intersection in intersections_folder:
        # get the path of sub_intersection folders
        sub_intersections_path = os.path.join(intersection_path, intersection)

        # for select intersection, filter all the sub videos with pedestrians
        intersection_pedestrain_subvideos = CCTV_Ped.loc[(CCTV_Ped['Video Name'].str.contains(intersection))
                                                         & ((CCTV_Ped['Pedestrian'] == '1') | (CCTV_Ped['Cyclist'] == 1))].reset_index(drop=True)

        for i in range(len(intersection_pedestrain_subvideos)):
            video_name = intersection_pedestrain_subvideos.loc[i, 'Video Name']
            sub_id = intersection_pedestrain_subvideos.loc[i, 'Sub_ID']
            multi_ped = intersection_pedestrain_subvideos.loc[i, 'Multi_Ped']

            # get the time frame information for this 1-hour video
            path_frame = os.path.join(root_path, 'Filtered Video', intersection, video_name, video_name + '.csv')
            df_frame = pd.read_csv(path_frame, index_col=False)
            df_frame['sub_id'] = pd.Series([int(i[-1]) for i in df_frame['subID'].str.split('_')])

            # get the time of the video based on the video name
            # year, month, day, hour, minute, second = video_name[-11:-7], video_name[-15:-13], video_name[-13:-11], video_name[-6:-4], video_name[-4:-2], video_name[-2:]
            # hour_video_start_time = datetime.datetime.strptime('{}-{}-{} {}:{}:{}'.format(year, month, day, hour, minute, second), '%Y-%m-%d %H:%M:%S')

            try:
                # read the csv of frame to time
                path_frame_time = os.path.join(root_path, 'Filtered Video', intersection, 'csv', video_name + '.csv')
                df_frame_time = pd.read_csv(path_frame_time, index_col=0)
                df_frame_time['time'] = pd.to_datetime(df_frame_time['time'], format="%m/%d/%Y, %H:%M:%S:%f")
            except FileNotFoundError:
                print('{}_{} frame to time file not found'.format(video_name, sub_id))
                continue
            # only the end frame is correct. the start frame is wrong
            # sub_video_end_time = hour_video_start_time + datetime.timedelta(seconds=df_frame[df_frame['sub_id'] == sub_id]['end'].values[0]/30)
            sub_video_end_frame = df_frame[df_frame['sub_id'] == sub_id]['end'].values[0]
            # get the name of the corresponding trajectory csv file
            trajectory_data_filename = video_name + '_' + str(sub_id) + '.csv'

            try:
                trajectory_data = pd.read_csv(os.path.join(sub_intersections_path, video_name, trajectory_data_filename), skiprows=1)
                # sub_video_start_time = sub_video_end_time - datetime.timedelta(seconds=(trajectory_data['frameNUM'].max()/30))

                trajectory_video_path = video_name + '_' + str(sub_id) + '.mov'
                cap = cv2.VideoCapture(os.path.join(sub_intersections_path, video_name, trajectory_video_path))
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                sub_video_start_frame = sub_video_end_frame - total_frames + 1

                sub_video_start_time = df_frame_time.loc[df_frame_time['frameNum'] == sub_video_start_frame, 'time'].values[0]
                sub_video_start_time = pd.Timestamp(sub_video_start_time).to_pydatetime()

                # get the crosswalk zone information
                crosswalk_length, crosswalk_angle = get_detection_zone_info(intersectionApproachDict, intersection, sub_video_start_time)

                # rewrite the type to be the most frequent type for every object
                object_type = trajectory_data.groupby('objectID', as_index=False).agg({'type': lambda x: list(x.mode())[0]})
                trajectory_data = trajectory_data.join(object_type.set_index('objectID'), on='objectID', rsuffix='_new')
                trajectory_data = trajectory_data.drop(columns='type')
                trajectory_data.rename(columns={'type_new': 'type'}, inplace=True)

                # trajectory reclassification (motocycle) and cleaning
                trajectory_data = ped_reclassification(trajectory_data)
                trajectory_data = trajectory_cleaning(trajectory_data)

                # identify the crossing pedestrians and their corresponding variables
                crossing_ped_trajectory = identify_crossing_ped(trajectory_data, crosswalk_length, crosswalk_angle)

                # calculate variables for crossing pedestrian
                if crossing_ped_trajectory is not None:
                    crossing_ped_trajectory = get_gps_for_crossing_ped(intersectionApproachDict, intersection, crossing_ped_trajectory, sub_video_start_time)

                    # append all the crossing ped trajectories
                    crossing_ped_trajectory['video_name'] = video_name
                    crossing_ped_trajectory['sub_id'] = sub_id
                    trajectory_crossing_peds.append(crossing_ped_trajectory)

                    # aggregate crossing behavior variables
                    crossing_behavior = get_ped_crossing_behavior(crossing_ped_trajectory, df_frame_time, sub_video_start_frame, video_name, sub_id)

                    # conflict calculation
                    conflicts = get_ped_veh_conflict(trajectory_data, crossing_ped_trajectory, threshold, intersect_buffer_x, intersect_buffer_y)

                    if len(conflicts) > 0:
                        conflicts = pd.concat(conflicts, ignore_index=True)
                        conflicts['intersection'] = intersection
                        conflicts['video_name'] = video_name
                        conflicts['sub_id'] = sub_id
                        print(intersection, video_name, sub_id)
                        total_conflicts.append(conflicts)

                        # join the crossing behavior variables with conflict data
                        agg_conflict = conflicts.groupby(by='actual_ped_num', as_index=False).agg({'frameNUM_veh': lambda x: x.unique().tolist(),
                                                                                                   'objectID_veh': lambda x: x.unique().tolist(),
                                                                                                   'objectID_ped': lambda x: x.unique().tolist(),
                                                                                                   'zone_ped': lambda x: x.unique().tolist(),
                                                                                                   'pet': lambda x: abs(x).min(),
                                                                                                   'sub_id': 'count'})
                        agg_conflict.rename(columns={'sub_id': 'num_conflict'}, inplace=True)
                        crossing_behavior = crossing_behavior.join(agg_conflict.set_index('actual_ped_num'), on='actual_ped_id')
                        crossing_behavior['conflict_label'] = np.where(crossing_behavior['num_conflict'] >= 1, 1, 0)
                    else:
                        print('{}_{} no conflict'.format(video_name, sub_id))
                        crossing_behavior['conflict_label'] = 0
                        pass

                    agg_crossing_peds.append(crossing_behavior)

                else:
                    print('{}_{} no crossing pedestrian'.format(video_name, sub_id))
                    continue

            except FileNotFoundError:
                print('{}_{} file not found'.format(video_name, sub_id))
                continue

            except:
                print('{}_{} trajectory data other error'.format(video_name, sub_id))
                continue

            print(trajectory_data_filename[0:-4])

    agg_data = pd.concat(agg_crossing_peds, ignore_index=True)
    ped_trajectory_data = pd.concat(trajectory_crossing_peds, ignore_index=True)
    conflict_data = pd.concat(total_conflicts, ignore_index=True)

    return agg_data, ped_trajectory_data, conflict_data


# i = 8
# i = 0
# i = 55
# intersection = 'SR434_@_WINDING_HOLLOW'

if __name__ == '__main__':
    # read the google sheet
    CCTV_Ped = pd.read_excel('CCTV_Ped.xlsx', sheet_name='Sheet1')
    CCTV_Ped = CCTV_Ped.fillna(9999)
    CCTV_Ped['Sub_ID'] = CCTV_Ped['Sub_ID'].astype(int)
    CCTV_Ped['Video Name'] = CCTV_Ped['Video Name'].astype(str)
    CCTV_Ped['Pedestrian'] = CCTV_Ped['Pedestrian'].astype(str)

    intersectionApproachDict = pd.read_excel('intersectionApproachDict.xlsx', sheet_name='Sheet1')

    threshold = 5
    intersect_buffer_x = 20
    intersect_buffer_y = 20
    # # call the function
    start = time.process_time()
    agg_data, ped_trajectory_data, conflict_data = extract_ped_trajectory(CCTV_Ped, intersectionApproachDict, threshold, intersect_buffer_x, intersect_buffer_y)
    # conflict_data = get_ped_conflict_data(CCTV_Ped, threshold, intersect_buffer_x, intersect_buffer_y)
    print(time.process_time() - start)

    agg_data.to_csv('agg_data.csv', sep=',')
    ped_trajectory_data.to_csv('trajectory_data.csv', sep=',')
    conflict_data.to_csv('conflict_data.csv', sep=',')

