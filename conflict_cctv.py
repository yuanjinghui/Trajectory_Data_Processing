# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import torch 
import pandas as pd
import numpy as np
import os
import matplotlib.path as mplPath
import numpy as np
import matplotlib.pyplot as plt
import glob
import traceback
#n=0
i=0
j=0
#directory = os.chdir('C:/Users/yi683992/Desktop/paper/With Jinghui/data/data/US17-92_@_13TH_ST/US17-92_@_13TH_ST_10022019_174715')
           #raw = pd.read_csv("US17-92_@_13TH_ST_10022019_174715_12.csv", skiprows=[0])
raw=[]

###############test one file###########################
#os.chdir('C:/Users/yi683992/Desktop/paper/With Jinghui/data/data/US17-92_@_13TH_ST/US17-92_@_13TH_ST_10022019_174715')
#raw = pd.read_csv("US17-92_@_13TH_ST_10022019_174715_12.csv", skiprows=[0])
#filename=1
#######################################################
conflict_sum=pd.DataFrame()
for file in glob.glob('C:/Users/yi683992/Desktop/paper/With Jinghui/data/data/US17-92_@_13TH_ST/US17-92_@_13TH_ST_10022019_144715/*.csv'):
#for file in glob.glob('C:/Users/yi683992/Desktop/paper/test/*.csv'):
    try:
    
        raw = pd.read_csv(file,skiprows=[0])
        filename=file[-6:-4]
        print("file name " + str(file))
        raw['p1y']=-raw['p1y']
        raw['p2y']=-raw['p2y']
        raw['cy']=-raw['cy']
        raw['by']=-raw['by']
        #raw=raw.drop(['p2x','p2y','p1x','p1y','p1y'], axis=1)
        ped=raw.loc[raw['type'] == 0]
        car=raw.loc[raw['type'] != 0]
        #the method that if the the center point of the vehicle pass the pedestrian has many missing conflict;
        ####join car with ped trajectory;car use bottom middle point; pedestrian use bottom middle point;
        #merge_PedCar =car.merge(ped, how='left', left_on=['bx','by'], right_on=['bx','by'])
        len_car=car.shape[0]
        len_ped=ped.shape[0]
        Inside_sum=pd.DataFrame()
      
        for j in range (0,len_ped):
            try:
            #print("ped "+str(j)+" - "+ str(j)+"/"+str(len_ped))
            #print("ped row number " +str(j) + "out of " + str(len_ped))
            #j=354
            #bx=1180
            #by=-560
    
                for i in range (0,len_car):
                    #print("car"+str(index)+" - "+ str(i)+"/"+str(len_car)
                    
                    #i=1939
                    try:
                        if ped.iloc[j,11]==car.iloc[i,11] and ped.iloc[j,11]>0:
                            a=ped.iloc[j,11]
                            #print("zone"+str(a))
                            
                            p1x=car.iloc[i,2]
                            p1y=car.iloc[i,3]
                            p2x=car.iloc[i,4]
                            p2y=car.iloc[i,5]
                            cx=car.iloc[i,6]
                            cy=car.iloc[i,7]
                            bx=car.iloc[i,8]
                            by=car.iloc[i,9]
                            p3x=2*bx-p2x
                            p3y=2*by-p2y
                            p4x=2*cx-p3x
                            p4y=2*cy-p3y
                            poly = [p1x, p1y, p2x, p2y,p3x,p3y,p4x,p4y]
                            
                            Ped_bx=ped.iloc[j,8]
                            Ped_by=ped.iloc[j,9]
                            #plt.plot(p1x,p1y,p2x, p2y,p3x,p3y,p4x,p4y, marker=11)
                            bbPath = mplPath.Path(np.array([[poly[0], poly[1]],
                                                 [poly[2], poly[3]],
                                                 [poly[4], poly[5]],
                                                 [poly[7], poly[7]]]))
                            Inside=bbPath.contains_point((Ped_bx, Ped_by))
                            #print(Inside)
                            #Inside_temp=[]
                            Inside_temp=np.array([0])
                            Inside_temp=pd.DataFrame(Inside_temp)
                            #pd.DataFrame(columns=['PedID','CarID','Ped_frame','Car_frame','Ped_bx','Ped_by','PET'])            
                            #pd.DataFrame(columns=['PedID','CarID','Ped_frame','Car_frame','Ped_bx','Ped_by','PET'])
                            if str(Inside)=='True':
                                #print("car row number " + str(i))
                                b=ped.iloc[j,1]
                                c=car.iloc[i,1]
                                #print("Ped ID" + str(b) +"CarID" + str(c) + "at zone " + str(a))
                                Inside_temp['PedID']=ped.iloc[j,1]
                                Inside_temp['CarID']=car.iloc[i,1]
                                Inside_temp['Ped_frame']=ped.iloc[j,0]
                                Inside_temp['Car_frame']=car.iloc[i,0]
                                Inside_temp['Ped_bx']=ped.iloc[j,8]
                                Inside_temp['Ped_by']=ped.iloc[j,9]
                                #Inside_temp=np.array([ped.iloc[j,1],car.iloc[i,1]])
                                #if PET<threhold then dangerous;
                                Inside_temp['PET']=abs(Inside_temp['Car_frame']-Inside_temp['Ped_frame'])
                                Inside_temp['file']=filename
                                Inside_temp['zone']=ped.iloc[j,11]
                                d=ped.iloc[j,11]
                                e=car.iloc[i,11]
                                Inside_sum=Inside_sum.append(Inside_temp)
                                #print("ped zone" + str(d)+ " car zone" + str(e))
                                #print("finish finding overlapping users")
                #Inside_sum.to_csv(('export_dataframe.csv'))
                #plt.plot(p1x,p1y,p2x, p2y,p3x,p3y,p4x,p4y,bx,by, marker=11)
                ###############################test code#####################
                #ped_frame=ped.iloc[j,1]
                #car_frame=car.iloc[i,1]
                #car_id=car.iloc[i,1]
                #PET=abs(Inside_temp['Car_frame']-Inside_temp['Ped_frame'])
                    except:
                        print ("ERROR @ j " +str(j) +" i" +str(i))
            except:
                print ("ERROR @ j " +str(j) +" i" +str(i))
        e=Inside_sum.shape[0]
        if e !=0:
            threshold=3*30
            Inside_sum=Inside_sum[abs(Inside_sum['PET'])< threshold]
            Inside_sum['file']=filename
            Inside_sum3=Inside_sum.sort_values(['PedID','CarID','Ped_frame'])
            Inside_sum3=Inside_sum3.drop_duplicates(['file','PedID','CarID'],keep="first")
            #Inside_sum2=Inside_sum.groupby(['file','PedID','CarID']).size().reset_index(name='counts')
            #Inside_sum2=Inside_sum2[Inside_sum2['counts']>10]
            Inside_sum4=Inside_sum.sort_values(['file','PedID','Ped_bx','Ped_by','Ped_frame'])
            Inside_sum4=Inside_sum4.drop_duplicates(['PedID','Ped_bx','Ped_by','file'],keep="first")
            Inside_sum2=Inside_sum4.groupby(['file','PedID','CarID']).size().reset_index(name='counts')
            Inside_sum2=Inside_sum2[Inside_sum2['counts']>10]
            result = pd.concat([Inside_sum3, Inside_sum2], axis=1, join='inner')
            #result2 = pd.concat([Inside_sum4, result], axis=1, join='inner')
            d=result.shape[0]
            #print outputs;
            print("detect " + str(d) + " conflicts in file " +str(filename) )
            #result3=result[['PedID','CarID','PET','Ped_frame']]
            print(result)
            print("finish file" + str(filename))
            conflict_sum=conflict_sum.append(result)
        elif e==0:
            print ("detection no conflict in file " + str(filename))
    except:
        #print ("ERROR @ file " +str(filename) +" !!!!!!!!!!!!!!") 
        traceback.print_exc()
conflict_sum.to_csv(r'C:/Users/yi683992/Desktop/paper/With Jinghui/data/data/US17-92_@_13TH_ST/US17-92_@_13TH_ST_10022019_144715/output.csv',index=False)
        