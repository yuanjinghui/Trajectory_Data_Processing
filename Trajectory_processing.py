"""
Created on Mon March 23 20:01:56 2020

@author: ji758507
"""
import pandas as pd
import numpy as np
import os
import time


def get_trajectory_slope(trajectory_data, zone_list):
    """

    :param trajectory_data:
    :param zone_list:
    :return:
    """
    zone_5 = trajectory_data.loc[trajectory_data['zone'].isin(zone_list)].reset_index(drop=True)
    zone_5 = zone_5.sort_values(by=['objectID', 'frameNUM']).reset_index(drop=True)

    # select the first and last trajectory points and calculate the trajectory slope
    first = zone_5.groupby('objectID', as_index=False).first()
    last = zone_5.groupby('objectID', as_index=False).last()
    count = zone_5.groupby('objectID', as_index=False).size().reset_index(name='counts')
    first = first.merge(last, how='left', left_on='objectID', right_on='objectID', suffixes=('_first', '_last')).reset_index(drop=True)
    first = first.merge(count, how='left', left_on='objectID', right_on='objectID').reset_index(drop=True)
    first['trajectory_slope'] = (first['cy_last'] - first['cy_first'])/(first['cx_last'] - first['cx_first'])

    # convert slope to angle [-90, 90]
    first['trajectory_angle'] = (np.arctan(first['trajectory_slope'])/np.pi)*180

    # convert angle from [-90, 90] to [0, 180]
    first.loc[first['trajectory_slope'] <= 0, 'trajectory_angle'] = \
        first.loc[first['trajectory_slope'] <= 0, 'trajectory_angle'] + 180

    # convert angle from [-90, 90] to [0, 360]
    # first.loc[(first['trajectory_slope'] <= 0) & (first['cy_first'] >= first['cy_last']), 'trajectory_angle'] = \
    #     first.loc[(first['trajectory_slope'] <= 0) & (first['cy_first'] >= first['cy_last']), 'trajectory_angle'] + 90
    # first.loc[(first['trajectory_slope'] <= 0) & (first['cy_first'] < first['cy_last']), 'trajectory_angle'] = \
    #     first.loc[(first['trajectory_slope'] <= 0) & (first['cy_first'] < first['cy_last']), 'trajectory_angle'] + 270
    # first.loc[(first['trajectory_slope'] > 0) & (first['cy_first'] <= first['cy_last']), 'trajectory_angle'] = \
    #     first.loc[(first['trajectory_slope'] <= 0) & (first['cy_first'] <= first['cy_last']), 'trajectory_angle'] + 90
    # first.loc[(first['trajectory_slope'] > 0) & (first['cy_first'] > first['cy_last']), 'trajectory_angle'] = \
    #     first.loc[(first['trajectory_slope'] > 0) & (first['cy_first'] > first['cy_last']), 'trajectory_angle'] + 270

    first['trajectory_length'] = np.linalg.norm(first[['cx_last', 'cy_last']].values - first[['cx_first', 'cy_first']].values, axis=1)
    return first


def ped_reclassification(trajectory_data):
    """
    Exclude the conflicts caused by on road motorcycles
    :param trajectory_data:
    :return:
    """
    first = get_trajectory_slope(trajectory_data, [5])

    general_angle = first.loc[(first['counts'] >= 30) & (first['type_last'].isin([2, 5, 7])) & (first['cy_first'] < first['cy_last']) & (first['trajectory_length'] >= 50)]['trajectory_angle'].quantile(0.5)
    min_angle = first.loc[(first['counts'] >= 30) & (first['type_last'].isin([2, 5, 7])) & (first['cy_first'] < first['cy_last']) & (first['trajectory_length'] >= 50)]['trajectory_angle'].min()
    max_angle = first.loc[(first['counts'] >= 30) & (first['type_last'].isin([2, 5, 7])) & (first['cy_first'] < first['cy_last']) & (first['trajectory_length'] >= 50)]['trajectory_angle'].max()

    # modify the pedestrian in zone 5 and 6 traveling in line with vehicles to be motorcycle
    if general_angle <= 90:
        try:
            ped_zone_5 = get_trajectory_slope(trajectory_data, [5])
            ped_motorcyce_5 = ped_zone_5.loc[(ped_zone_5['type_last'].isin([0, 1])) & ((ped_zone_5['counts'] >= 30) | (ped_zone_5['trajectory_length'] >= 100)) &
                                             (ped_zone_5['trajectory_angle'].between(max(min_angle - 10, general_angle - 22.5), 180 - general_angle))]['objectID'].tolist()

            ped_zone_6 = get_trajectory_slope(trajectory_data, [6])
            ped_motorcyce_6 = ped_zone_6.loc[(ped_zone_6['type_last'].isin([0, 1])) & ((ped_zone_6['counts'] >= 30) | (ped_zone_6['trajectory_length'] >= 100)) &
                                             (ped_zone_6['trajectory_angle'].between(max(min_angle - 10, general_angle - 22.5), 180 - general_angle))]['objectID'].tolist()

            trajectory_data.loc[(trajectory_data['objectID'].isin(ped_motorcyce_5)) | (trajectory_data['objectID'].isin(ped_motorcyce_6)), 'type'] = 3
        except:
            print('no changes')
            pass

    elif general_angle > 90:
        try:
            ped_zone_5 = get_trajectory_slope(trajectory_data, [5])
            ped_motorcyce_5 = ped_zone_5.loc[(ped_zone_5['type_last'].isin([0, 1])) & ((ped_zone_5['counts'] >= 30) | (ped_zone_5['trajectory_length'] >= 100)) &
                                             (ped_zone_5['trajectory_angle'].between(180 - general_angle, min(max_angle + 10, general_angle + 22.5)))]['objectID'].tolist()

            ped_zone_6 = get_trajectory_slope(trajectory_data, [6])
            ped_motorcyce_6 = ped_zone_6.loc[(ped_zone_6['type_last'].isin([0, 1])) & ((ped_zone_6['counts'] >= 30) | (ped_zone_6['trajectory_length'] >= 100)) &
                                             (ped_zone_6['trajectory_angle'].between(180 - general_angle, min(max_angle + 10, general_angle + 22.5)))]['objectID'].tolist()

            trajectory_data.loc[(trajectory_data['objectID'].isin(ped_motorcyce_5)) | (trajectory_data['objectID'].isin(ped_motorcyce_6)), 'type'] = 3
        except:
            print('no changes')
            pass

    return trajectory_data


def get_ped_veh_conflict(trajectory_data, threshold, intersect_buffer_x, intersect_buffer_y):
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

    pedestrian_trajectory = trajectory_data.loc[(trajectory_data['type'].isin([0, 1])) & (trajectory_data['zone'].isin([2, 3, 4, 5, 6, 7, 9])), :].sort_values(by=['objectID', 'frameNUM']).reset_index(drop=True)
    # Exclude tiny detections and pedestrian in vehicle (y axis smaller than 50)
    pedestrian_trajectory = pedestrian_trajectory.loc[pedestrian_trajectory['p2y'] >= (pedestrian_trajectory['p1y'] + 50)].reset_index(drop=True)

    veh_trajectory = trajectory_data.loc[trajectory_data['type'].isin([2, 5, 7])].sort_values(by=['objectID', 'frameNUM']).reset_index(drop=True)

    ped_list = pedestrian_trajectory['objectID'].unique()

    ped_veh_conflicts = []
    for ped in ped_list:
        ped_trajectory = pedestrian_trajectory.loc[pedestrian_trajectory['objectID'] == ped].reset_index(drop=True)

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
            print('no conflict')

    return ped_veh_conflicts


def get_ped_conflict_data(CCTV_Ped, threshold, intersect_buffer_x, intersect_buffer_y):
    """

    :param CCTV_Ped:
    :return:
    """
    # get the root path
    root_path = os.getcwd()
    intersection_path = os.path.join(root_path, 'Data')
    intersections_folder = os.listdir(intersection_path)

    total_conflicts = []

    for intersection in intersections_folder:
        # get the path of sub_intersection folders
        sub_intersections_path = os.path.join(intersection_path, intersection)
        # get the list of sub_intersections
        sub_intersections = os.listdir(sub_intersections_path)

        # for select intersection, filter all the sub videos with pedestrians
        intersection_pedestrain_subvideos = CCTV_Ped.loc[(CCTV_Ped['Video Name'].str.contains(intersection)) & ((CCTV_Ped['Pedestrian'] == '1') | (CCTV_Ped['Cyclist'] == 1))].reset_index(drop=True)

        for i in range(len(intersection_pedestrain_subvideos)):
            video_name = intersection_pedestrain_subvideos.loc[i, 'Video Name']
            sub_id = intersection_pedestrain_subvideos.loc[i, 'Sub_ID']

            # get the name of the corresponding trajectory csv file
            trajectory_data_filename = video_name + '_' + str(sub_id) + '.csv'

            try:
                trajectory_data = pd.read_csv(os.path.join(sub_intersections_path, video_name, trajectory_data_filename), skiprows=1)
                # rewrite the type to be the most frequent type for every object
                object_type = trajectory_data.groupby('objectID', as_index=False).agg({'type': np.median})
                trajectory_data = trajectory_data.join(object_type.set_index('objectID'), on='objectID', rsuffix='_new')
                trajectory_data = trajectory_data.drop(columns='type')
                trajectory_data.rename(columns={'type_new': 'type'}, inplace=True)

                # trajectory_data = trajectory_data.loc[trajectory_data['lostCounter'] <= 30].reset_index(drop=True)

                trajectory_data = ped_reclassification(trajectory_data)
                conflicts = get_ped_veh_conflict(trajectory_data, threshold, intersect_buffer_x, intersect_buffer_y)

                if len(conflicts) > 0:
                    conflicts = pd.concat(conflicts, ignore_index=True)
                    conflicts['intersection'] = intersection
                    conflicts['video_name'] = video_name
                    conflicts['sub_id'] = sub_id
                    print(intersection, video_name, sub_id)
                    total_conflicts.append(conflicts)
                else:
                    print('no conflict')
                    pass
            except:
                print('file not found')
                pass

    data = pd.concat(total_conflicts, ignore_index=True)

    return data


if __name__ == '__main__':
    # read the google sheet
    CCTV_Ped = pd.read_csv('https://docs.google.com/spreadsheets/d/1rVAFS2oCya8shF2PeREq00lXzUcyVFCmmbTQQr0PHx8/export?gid=0&format=csv')
    CCTV_Ped = CCTV_Ped.fillna(9999)
    CCTV_Ped['Sub_ID'] = CCTV_Ped['Sub_ID'].astype(int)
    CCTV_Ped['Video Name'] = CCTV_Ped['Video Name'].astype(str)
    CCTV_Ped['Pedestrian'] = CCTV_Ped['Pedestrian'].astype(str)

    threshold = 5
    # intersect_buffer_x = 60
    # intersect_buffer_y = 30

    intersect_buffer_x = 20
    intersect_buffer_y = 20
    # call the function
    start = time.process_time()
    conflict_data = get_ped_conflict_data(CCTV_Ped, threshold, intersect_buffer_x, intersect_buffer_y)
    print(time.process_time() - start)

    conflict_data.to_csv('conflict_data_r13.csv', sep=',')


# intersection = 'US17-92_@_25TH_ST'
# i = 71
# ped = 529