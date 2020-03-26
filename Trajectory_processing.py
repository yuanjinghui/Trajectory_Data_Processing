"""
Created on Mon March 23 20:01:56 2020

@author: ji758507
"""
import pandas as pd
import numpy as np
import os
import time


def get_ped_veh_conflict(trajectory_data, threshold):
    """
    :param trajectory_data: p1x, p1y represent the upper left corner; p2x, p2y represent the bottom right corner;
    bx, by represent center point; cx, cy represent bottom middle point;
    type: https://gist.github.com/AruniRC/7b3dadd004da04c80198557db5da4bda, id-1
    :param threshold: threshold for identify conflict
    :return: identified conflicts
    """
    threshold_frame = threshold * 30
    pedestrian_trajectory = trajectory_data.loc[(trajectory_data['type'].isin([0, 1])) & (trajectory_data['zone'].isin([2, 3, 4, 5, 6, 7])), :].sort_values(by=['objectID', 'frameNUM']).reset_index(drop=True)
    veh_trajectory = trajectory_data.loc[~(trajectory_data['type'].isin([0, 1]))].sort_values(by=['objectID', 'frameNUM']).reset_index(drop=True)

    ped_list = pedestrian_trajectory['objectID'].unique()

    ped_veh_conflicts = []
    for ped in ped_list:
        ped_trajectory = pedestrian_trajectory.loc[pedestrian_trajectory['objectID'] == ped].reset_index(drop=True)
        # ped_trajectory = pd.merge(ped_trajectory, veh_trajectory, how='left', left_on=['cx', 'cy'], right_on=['cx', 'cy'], suffixes=('_ped', '_veh'))

        # filter out all the
        tem_veh = veh_trajectory.loc[(veh_trajectory['p1x'] <= ped_trajectory['cx'].max()) & (veh_trajectory['p2x'] >= ped_trajectory['cx'].min())
                                     & (veh_trajectory['p1y'] <= ped_trajectory['cy'].max()) & (veh_trajectory['p2y'] >= ped_trajectory['cy'].min())
                                     & (veh_trajectory['frameNUM'] <= (ped_trajectory['frameNUM'].max() + threshold_frame))
                                     & (veh_trajectory['frameNUM'] >= (ped_trajectory['frameNUM'].min() - threshold_frame))].reset_index(drop=True)

        # merge temporary vehicle trajectory data and ped trajectory
        tem_veh['key'] = 0
        ped_trajectory['key'] = 0
        conflict = tem_veh.merge(ped_trajectory, how='outer', left_on='key', right_on='key', suffixes=('_veh', '_ped'))
        conflict = conflict.loc[conflict['frameNUM_veh'] >= conflict['frameNUM_ped']].reset_index(drop=True)

        # Euclidean distance between ped and veh
        conflict['distance'] = np.linalg.norm(conflict[['cx_veh', 'cy_veh']].values - conflict[['cx_ped', 'cy_ped']].values, axis=1)
        conflict['ped_veh_overlap'] = np.where((conflict['cx_ped'].between(conflict['p1x_veh'], conflict['p2x_veh'])) & (conflict['cy_ped'].between(conflict['p1y_veh'], conflict['p2y_veh'])), 1, 0)
        conflict = conflict.loc[conflict['ped_veh_overlap'] == 1].sort_values(by=['objectID_veh', 'frameNUM_veh']).reset_index(drop=True)

        if len(conflict) > 0:
            conflict = conflict.loc[conflict.groupby(by=['objectID_veh'], as_index=False)["distance"].idxmin()]
            conflict['pet'] = (conflict['frameNUM_veh'] - conflict['frameNUM_ped'])/30
            conflict = conflict.loc[conflict['pet'] <= threshold].reset_index(drop=True)

            ped_veh_conflicts.append(conflict)

        else:
            print('no conflict')

    return ped_veh_conflicts


def get_ped_conflict_data(CCTV_Ped, threshold):
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
            except:
                print('file not found')
                pass

            conflicts = get_ped_veh_conflict(trajectory_data, threshold)

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
    data = pd.concat(total_conflicts, ignore_index=True)

    return data


if __name__ == '__main__':
    # read the google sheet
    CCTV_Ped = pd.read_csv('https://docs.google.com/spreadsheets/d/1rVAFS2oCya8shF2PeREq00lXzUcyVFCmmbTQQr0PHx8/export?gid=0&format=csv')
    CCTV_Ped = CCTV_Ped.fillna(9999)
    CCTV_Ped['Sub_ID'] = CCTV_Ped['Sub_ID'].astype(int)
    CCTV_Ped['Video Name'] = CCTV_Ped['Video Name'].astype(str)
    CCTV_Ped['Pedestrian'] = CCTV_Ped['Pedestrian'].astype(str)

    threshold = 3
    # call the function
    start = time.process_time()
    conflict_data = get_ped_conflict_data(CCTV_Ped, threshold)
    print(time.process_time() - start)

    conflict_data.to_csv('conflict_data.csv', sep=',')


