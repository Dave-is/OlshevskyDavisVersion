""" Simple GUI utils for MMS classification project.

(c) Vyacheslav Olshevsky (2019)

"""
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk

class ArrayGUI(tk.Frame):
    def __init__(self, arr, plot_item=None, master=None, vmin=-1, vmax=4, plot_all=None):
        #master.geometry('640x480')
        super().__init__(master)
        self.master = master

        self.arr = arr
        self.vmin = vmin
        self.vmax = vmax
        self.plot_item = plot_item
        self.plot_all = plot_all

        self.grid(column=4)
        self.create_vars()
        self.create_widgets()

        # Figure with all plots which we will close in destructor
        self.fig_all = self.plot_all()

    def btn_plot_all_click(self):
        self.fig_all = self.plot_all()

    def destroy(self):
        plt.close(self.fig_all)
        tk.Frame.destroy(self)

    def create_vars(self):
        self.varStart = tk.IntVar()
        self.varStart.set(0)
        self.varEnd = tk.IntVar()
        self.varEnd.set(self.arr.shape[0])
        self.varValue = tk.IntVar()
        self.varValue.set(-1)
        self.varValues = tk.StringVar()
        self.varValues.set('')
        self.varPlotIndex = tk.IntVar(value=0)

    def create_widgets(self):
        self.lblStart = tk.Label(self, text='Range start')
        self.lblStart.grid(column=0, row=1)

        self.edStart = tk.Entry(self, textvariable=self.varStart)
        self.edStart.grid(column=1, row=1)

        self.lblEnd = tk.Label(self, text='end')
        self.lblEnd.grid(column=2, row=1)

        self.edEnd = tk.Entry(self, textvariable=self.varEnd)
        self.edEnd.grid(column=3, row=1)

        self.lblValue = tk.Label(self, text='New value')
        self.lblValue.grid(column=0, row=2)

        self.edValue = tk.Entry(self, textvariable=self.varValue)
        self.edValue.grid(column=1, row=2)

        self.btnPrint = tk.Button(self, text='Print range', command=self.print_items)
        self.btnPrint.grid(column=2, row=2)

        self.btnApply = tk.Button(self, text="Set value", fg="red", command=self.set_values)
        self.btnApply.grid(column=4, row=2)

        self.btnPlot = tk.Button(self, text='Plot', command=self.plot_items)
        self.btnPlot.grid(column=0, row=3)

        self.edPlotIndex = tk.Entry(self, textvariable=self.varPlotIndex)
        self.edPlotIndex.grid(column=1, row=3)

        self.btnPlot = tk.Button(self, text='Plot all', command=self.btn_plot_all_click)
        self.btnPlot.grid(column=4, row=3)

        self.msgValues = tk.Message(self, textvariable=self.varValues, background='white')
        self.msgValues.grid(column=0, row=4, columnspan=4)

    def print_items(self):
        try:
            #print(self.arr[self.varStart.get():self.varEnd.get()])
            text = str(self.arr[self.varStart.get():self.varEnd.get()]).replace('\n', '')
            self.varValues.set(text)
        except IndexError:
            print('Start/End out of range. Possible values between', 0, 'and', self.arr.shape[0])

    def set_values(self):
        if (self.varValue.get() < self.vmin) or (self.varValue.get() > self.vmax):
            print('Error. Value should be between', self.vmin, 'and', self.vmax)
        else:
            try:
                self.arr[self.varStart.get():self.varEnd.get()] = self.varValue.get()
            except IndexError:
                print('Start/End out of range. Possible values between', 0, 'and', self.arr.shape[0])

    def plot_items(self):
        if self.plot_item != None:
            try:
                self.plot_item(self.varPlotIndex.get())
            except IndexError:
                print('Index out of range. Possible values between', 0, 'and', self.arr.shape[0])

def edit_array(arr, plot_item=None, plot_all=None, vmin=-1, vmax=4):
    window = tk.Tk()
    window.title('Array editor')
    app = ArrayGUI(arr, master=window, plot_item=plot_item, plot_all=plot_all, vmin=vmin, vmax=vmax)
    app.mainloop()
    return arr

if __name__ == "__main__":
    a = np.arange(1000)
    a = edit_array(a)