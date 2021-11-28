#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
1.使用前先用 Uranus.exe 確認連接器的Port
2.去改config.json的port ex:   "port": "COM7",
3.執行camera.py要加輸出的svo檔名以及csv檔名，Ex: python camera.py svo_name.svo csv_name.csv
4.要改成avi要去zed那包找export_svo.py
左邊是L2右邊是R1 燈在下面兩腳踝側
'''

from hipnuc_module import *
import time
import csv
import cv2
import pyzed.sl as sl
from signal import signal, SIGINT
import tkinter.filedialog
import tkinter as tk
from os import path
import serial.tools.list_ports
import json

def read_port():
    ports = sorted(serial.tools.list_ports.comports())
    displays = []
    selections = []
    for port, desc, hwid in ports:
            displays.append("{}: {} ".format(port, desc))
            selections.append(port)
    return displays, selections

def write_port(port_index,port_list):
    a_file = open("config.json", "r")
    json_object = json.load(a_file)
    a_file.close()
    json_object["port"] = port_list[port_index]
    a_file = open("config.json","w")
    json.dump(json_object, a_file)
    a_file.close()


def vid_display(cam, runtime, mat):
    err = cam.grab(runtime)
    if err == sl.ERROR_CODE.SUCCESS:
        cam.retrieve_image(mat)
        cv2.imshow("ZED", mat.get_data())

def init_csv(filename):
    csv_output = (filename + ".csv")
    csvfile = open(csv_output, 'w', newline='')
    # 定義欄位
    fieldnames = ['RollR', 'PitchR', 'YawR','WR','XR','YR','ZR','AccXR', 'AccYR', 'AccZR','GyrXR','GyrYR','GyrZR','MagXR','MagYR','MagZR',
                  'RollL', 'PitchL', 'YawL','WL','XL','YL','ZL', 'AccXL', 'AccYL', 'AccZL','GyrXL','GyrYL','GyrZL','MagXL','MagYL','MagZL']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    return writer, csvfile


def imu_record(serial, writer):
    data = serial.get_module_data(1)
    try:
        writer.writerow({  # R
            'RollR': data['euler'][0]['Roll'],
            'PitchR': data['euler'][0]['Pitch'],
            'YawR': data['euler'][0]['Yaw'],
            'WR': data['quat'][0]['W'],
            'XR': data['quat'][0]['X'],
            'YR': data['quat'][0]['Y'],
            'ZR': data['quat'][0]['Z'],
            'AccXR': data['acc'][0]['X'],
            'AccYR': data['acc'][0]['Y'],
            'AccZR': data['acc'][0]['Z'],
            'GyrXR': data['gyr'][0]['X'],
            'GyrYR': data['gyr'][0]['Y'],
            'GyrZR': data['gyr'][0]['Z'],
            'MagXR': data['mag'][0]['X'],
            'MagYR': data['mag'][0]['Y'],
            'MagZR': data['mag'][0]['Z'],
            # L
            'RollL': data['euler'][1]['Roll'],
            'PitchL': data['euler'][1]['Pitch'],
            'YawL': data['euler'][1]['Yaw'],
            'WL': data['quat'][1]['W'],
            'XL': data['quat'][1]['X'],
            'YL': data['quat'][1]['Y'],
            'ZL': data['quat'][1]['Z'],
            'AccXL': data['acc'][1]['X'],
            'AccYL': data['acc'][1]['Y'],
            'AccZL': data['acc'][1]['Z'],
            'GyrXL': data['gyr'][1]['X'],
            'GyrYL': data['gyr'][1]['Y'],
            'GyrZL': data['gyr'][1]['Z'],
            'MagXL': data['mag'][1]['X'],
            'MagYL': data['mag'][1]['Y'],
            'MagZL': data['mag'][1]['Z']})
    except:
        print('Print error.')
        serial.close()


def checkName(filename):
    while path.exists(filename+'.csv'):
        print('found: '+ filename[:])
        filename = filename[:-1] + str(int(filename[-1])+1)
    print('creat: '+ filename)
    return filename


def record_button(cam, runtime, mat, json, filename):
    filename = checkName(filename)
    writer , csvfile = init_csv(filename)
    serial = hipnuc_module(json)
    record_parm = sl.RecordingParameters(filename + ".svo")
    vid = cam.enable_recording(record_parm)

    if vid == sl.ERROR_CODE.SUCCESS:
        print("Recording started...")
        print("Hit spacebar to stop recording: ")
        key = ''
        while key != 32:  # for spacebar
            key = vid_display(cam, runtime, mat) 
            imu_record(serial, writer)
            key = cv2.waitKey(5)
    else:
        print("Recording not started.")
    csvfile.close()
    cam.disable_recording()
    serial.close()

def start_serial(json):
    serial = hipnuc_module(json)
    return serial