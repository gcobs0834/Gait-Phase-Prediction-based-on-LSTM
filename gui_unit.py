import tkinter as tk
from tkinter import ttk

def add_text(frame, row, label, text):

    l = tk.Label(frame)
    l["text"] = label
    l.grid(row=row, column=0, sticky=tk.N+tk.W)
    t = tk.Label(frame)
    t["text"] = text
    t.grid(row=row, column=1, sticky=tk.N+tk.W)
    return l,t



def add_button(frame, row, text, func):
    b = tk.Button(frame)
    b["text"] = text
    b.grid(row=row, column=0, sticky=tk.N+tk.W,pady=10)
    b["command"] = func
    return b

def add_combobox(frame, row, text, func):
    b = ttk.Combobox(frame)
    #b.value=text
    b["value"]=text
    b.grid(row=0, column=0, sticky=tk.N+tk.W)
    #b["command"] = func
    return b



def add_listbox(frame,label,to):
    ll= tk.Listbox(frame,height=20)
    l = tk.Label(frame)
    l["text"] = label
    
    for item in [*range(1,50)]:
        ll.insert('end',item)
    l.grid(row=0, column=0, sticky=tk.N+tk.W)
    ll.grid(row=1, column=0, sticky=tk.N+tk.W)
    return l,ll
 



    