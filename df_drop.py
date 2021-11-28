import pandas as pd


####  Drop ######
def drop_Mag(df_list):
    for df in range(len(df_list)):
        df_list[df].drop(['MagXR','MagYR','MagZR','MagXL','MagYL','MagZL'], axis=1, inplace=True)
def drop_Gyr(df_list):
    for df in range(len(df_list)):
        df_list[df].drop(['GyrXR','GyrYR','GyrZR','GyrXL','GyrYL','GyrZL'], axis=1, inplace=True)
def drop_Acc(df_list):
    for df in range(len(df_list)):
        df_list[df].drop(['AccXR','AccYR','AccZR','AccXL','AccYL','AccZL'], axis=1, inplace=True)
def drop_Roll_Yaw_Pitch(df_list):
    for df in range(len(df_list)):
        df_list[df].drop(['RollR','PitchR','YawR','RollL','PitchL','YawL'], axis=1, inplace=True)
def drop_wxyz(df_list):
    for df in range(len(df_list)):
        df_list[df].drop(['WR','XR','YR','ZR','WL','XL','YL','ZL'], axis=1, inplace=True)
####  Drop ######