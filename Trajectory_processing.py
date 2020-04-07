"""
Created on Mon March 23 20:01:56 2020

@author: ji758507
"""
import pandas as pd
import numpy as np
import os
import time


def ped_reclassification(trajectory_data):
    """
    Exclude the conflicts caused by on road motorcycles
    :param trajectory_data:
    :return:
    """
    zone_5 = trajectory_data.loc[trajectory_data['zone'] == 5].reset_index(drop=True)
    # veh_data_zone_5 = trajectory_data.loc[(trajectory_data['zone'] == 5) & (trajectory_data['type'].isin([2, 5, 7]))].reset_index(drop=True)
    zone_5 = zone_5.sort_values(by=['objectID', 'frameNUM']).reset_index(drop=True)

    # select the first and last trajectory points and calculate the trajectory slope
    first = zone_5.groupby('objectID', as_index=False).first()
    last = zone_5.groupby('objectID', as_index=False).last()
    count = zone_5.groupby('objectID', as_index=False).size().reset_index(name='counts')
    first = first.merge(last, how='left', left_on='objectID', right_on='objectID', suffixes=('_first', '_last')).reset_index(drop=True)
    first = first.merge(count, how='left', left_on='objectID', right_on='objectID').reset_index(drop=True)
    first['trajectory_slope'] = abs((first['by_last'] - first['by_first'])/(first['bx_last'] - first['bx_first']))

    general_slope = first.loc[(first['counts'] >= 30) & (first['type_last'].isin([2, 5, 7]))]['trajectory_slope'].quantile(0.5)

    # modify the pedestrian in zone 5 traveling in line with vehicles to be motorcycle
    try:
        ped_motorcyce = first.loc[(first['type_last'].isin([0, 1])) & (first['trajectory_slope'].between(general_slope - 0.4, general_slope + 0.4))]['objectID'].tolist()
        trajectory_data.loc[trajectory_data['objectID'].isin(ped_motorcyce), 'type'] = 3
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
    # exclude the detection id with insufficient trajectory
    number_of_frames = trajectory_data.groupby(by='objectID', as_index=False).size().reset_index(name='counts')
    list_of_id = number_of_frames.loc[number_of_frames['counts'] >= 15]['objectID'].tolist()
    trajectory_data = trajectory_data.loc[trajectory_data['objectID'].isin(list_of_id)]

    pedestrian_trajectory = trajectory_data.loc[(trajectory_data['type'].isin([0, 1])) & (trajectory_data['zone'].isin([2, 3, 4, 5, 6, 7])), :].sort_values(by=['objectID', 'frameNUM']).reset_index(drop=True)
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
            conflict = conflict.loc[abs(conflict['pet']) <= threshold].reset_index(drop=True)

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
        intersection_pedestrain_subvideos = CCTV_Ped.loc[(CCTV_Ped['Video Name'].str.contains(intersection)) & (CCTV_Ped['Pedestrian'] == '1')].reset_index(drop=True)

        for i in range(len(intersection_pedestrain_subvideos)):
            video_name = intersection_pedestrain_subvideos.loc[i, 'Video Name']
            sub_id = intersection_pedestrain_subvideos.loc[i, 'Sub_ID']

            # get the name of the corresponding trajectory csv file
            trajectory_data_filename = video_name + '_' + str(sub_id) + '.csv'

            try:
                trajectory_data = pd.read_csv(os.path.join(sub_intersections_path, video_name, trajectory_data_filename), skiprows=1)
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
    intersect_buffer_x = 40
    intersect_buffer_y = 20
    # call the function
    start = time.process_time()
    conflict_data = get_ped_conflict_data(CCTV_Ped, threshold, intersect_buffer_x, intersect_buffer_y)
    print(time.process_time() - start)

    conflict_data.to_csv('conflict_data_r1.csv', sep=',')


# intersection = 'US17-92_@_13TH_ST'
# # i = 30
