import os
import numpy as np
import tkinter as tk
from tkinter import filedialog
from gui_unit import add_text, add_button, add_combobox, add_listbox
import record
from time import sleep
import time
import threading
import cv2
import pyzed.sl as sl
from signal import signal, SIGINT
from hipnuc_module import *
from functools import partial


class GUI(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master)
        self.root = master
        self.grid()
        self.startime=0
        self.now=0
        self.running = False
        self.create_widgets()
        self.json = './config.json'
        init = sl.InitParameters()
        init.camera_resolution = sl.RESOLUTION.HD720
        init.camera_fps = 60
        init.depth_mode = sl.DEPTH_MODE.NONE
        self.cam = sl.Camera()
        if not self.cam.is_opened():
            print("Opening ZED Camera...")
        status = self.cam.open(init)
        if status != sl.ERROR_CODE.SUCCESS:
                print(repr(status))
        self.runtime = sl.RuntimeParameters()
        self.mat = sl.Mat()

        
    
    def create_widgets(self):
      
        self.winfo_toplevel().title("ComeOn! Run")
        self.master.geometry('800x650')
        
        
        
        Com_Group=tk.LabelFrame(self.master, font=('Times New Roman', 12),text='Choose COM',padx=10,pady=10)
        Com_Group.grid(column=0, row=0,pady=10,padx=10)
        self.com_box, self.ports = record.read_port()
        self.fl = add_combobox(Com_Group,1,self.com_box,'')
        self.serial = None
        self.fl.bind("<<ComboboxSelected>>", lambda _:[record.write_port(self.fl.current(), self.ports), self.serial_button()])
        # 感測器檢測是否連上
        Feature_Group=tk.LabelFrame(self.master, font=('Times New Roman', 12),text='Feature check',padx=10,pady=10)
        Feature_Group.grid(column=1, row=0,pady=10,padx=10)

        lf_Group=tk.LabelFrame(Feature_Group, font=('Times New Roman', 12),text='Left',padx=10,pady=10)
        lf_Group.grid(column=0, row=0,pady=10,padx=10)
        _, self.lf_x=add_text(lf_Group, 0, "左腳狀態: ","")


        rf_Group=tk.LabelFrame(Feature_Group, font=('Times New Roman', 12),text='Right',padx=10,pady=10)
        rf_Group.grid(column=1, row=0,pady=10,padx=10)

        _, self.rf_x=add_text(rf_Group, 0, "右腳狀態: ","")
 


        #受測者選擇
        subject_Group=tk.LabelFrame(self.master, font=('Times New Roman', 12),text='Choose Subject',padx=10,pady=10)
        subject_Group.grid(column=0, row=1,pady=10,padx=10)

        _, self.subject = add_listbox(subject_Group,'請選擇受測者',555)
        self.subject.bind("<<ListboxSelect>>", lambda _:[self.select_listbox()])
        self.filename = 'b41_1'
        self.se_button=add_button(subject_Group,2,'Start',lambda :[record.record_button(self.cam, self.runtime, self.mat, self.json, self.filename)])


        # 計時
        clock_Group=tk.LabelFrame(self.master, font=('Times New Roman', 12),text='Timing',padx=10,pady=10)
        clock_Group.grid(column=1, row=1,pady=10,padx=10)

        self.clock = tk.Label(clock_Group,text="", font=('Helvetica', 32), fg='green')
        self.clock.grid(column=0, row=0,pady=10,padx=10)
        self.clock.configure(text='0')

    def select_listbox(self):
        self.filename = str(self.subject.get(self.subject.curselection()))+"_1"

    def serial_button(self):
        self.serial = record.start_serial(self.json)
        data = self.serial.get_module_data(1)
        try :
            data['acc'][0]['X']
            self.rf_x.config(text='OK') 
        except:
            self.rf_x.config(text='重新確認連接')

        try:
            data['acc'][1]['X']
            self.lf_x.config(text='OK') 
        except:
            self.lf_x.config(text='請重新確認連接') 
        self.serial.close()
    

    def start_end(self):
        if self.running is False:
            self.running = True
            print(self.running)
        else:
            self.running = False
            print(self.running)
        self.startime = time.time()
        self.update_clock()

    def update_clock(self):
        if self.running is True:
            self.root.after(100, self.update_clock)
            self.now=('{:.2f}'.format(time.time()-self.startime))
            self.clock.configure(text=self.now)
        print(self.running)



if __name__ == "__main__":
    root = tk.Tk()
    app = GUI(root)
    root.mainloop()