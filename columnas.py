
import tkinter as tk
from tkinter import messagebox as mb

    
    
class Tooltip:
    def __init__(self, widget, texto):
        self.widget = widget
        self.texto = texto
        self.tooltip = None

        widget.bind("<Enter>", self.mostrar)
        widget.bind("<Leave>", self.ocultar)

    def mostrar(self, event=None):
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + 20
        self.tooltip = tk.Toplevel(self.widget)
        self.tooltip.wm_overrideredirect(True)
        self.tooltip.geometry(f"+{x}+{y}")
        label = tk.Label(
            self.tooltip, 
            text=self.texto, 
            bg="#ffffe0", 
            relief="solid", 
            borderwidth=1,
            anchor="w",         # Alinea el texto al oeste (izquierda)
            justify="left"      # Justifica el texto a la izquierda
            )
        label.pack()

    def ocultar(self, event=None):
        if self.tooltip:
            self.tooltip.destroy()
            self.tooltip = None


class BtnToolTip:
    def __init__(self, widget, text='Widget info', bg="#ffffe0", font=("tahoma", "8", "normal")):
        self.widget = widget
        self.text = text
        self.bg = bg
        self.font = font
        self.tipwindow = None
        widget.bind('<Enter>', self.show_tip)
        widget.bind('<Leave>', self.hide_tip)

    def show_tip(self, event=None):
        if self.tipwindow or not self.text:
            return
        x = self.widget.winfo_pointerx() + 20
        y = self.widget.winfo_pointery() + 10
        self.tipwindow = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f'+{x}+{y}')
        label = tk.Label(
            tw, text=self.text, justify='left',
            background=self.bg, relief='solid', borderwidth=1,
            font=self.font
        )
        label.pack(ipadx=1)

    def hide_tip(self, event=None):
        if self.tipwindow:
            self.tipwindow.destroy()
            self.tipwindow = None

