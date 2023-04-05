from __future__ import print_function
import numpy as np
import pandas as pd
import seaborn as sn
import cv2
from matplotlib import pyplot as plt
import tkinter as tk
from PIL import Image, ImageDraw
from sklearn import metrics

def plot_data(dataset, cols=5, rows=5, title='Title'):
    figure = plt.figure(num=title, figsize=(10, 8))
    
    for i in range(1, cols * rows + 1):
        sample_idx = np.random.randint(len(dataset), size=(1,)).item()
        img, label = dataset[sample_idx]
        
        figure.add_subplot(rows, cols, i)
        plt.title(label)
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap="gray")
    
    plt.show()

def plot_confusion_matrix(gt_labels, pred_labels, include_summaries=True):
    gt_labels = np.array(gt_labels).flatten()
    pred_labels = np.array(pred_labels).flatten()
    unique_labels = [str(l) for l in sorted(set(gt_labels))]
    
    confusion_matrix = metrics.confusion_matrix(gt_labels, pred_labels)
    
    num_total = np.sum(confusion_matrix)
    gt_sum = np.sum(confusion_matrix, axis=0)
    pred_sum = np.sum(confusion_matrix, axis=1)
    
    if include_summaries:
        # We add the cumulatives to the matrix
        confusion_matrix = np.concatenate((confusion_matrix, gt_sum.reshape(1, -1)), axis=0)
        pred_sum = np.concatenate((pred_sum, [num_total]))
        confusion_matrix = np.concatenate((confusion_matrix, pred_sum.reshape(-1, 1)), axis=1)
        
        index_labels = unique_labels + ['Pred-cumsum']
        column_labels = unique_labels + ['GT-cumsum']
    
    else:
        index_labels = unique_labels
        column_labels = unique_labels
    
    fig, ax = plt.subplots(figsize=(13, 10))
    
    df_cm = pd.DataFrame(confusion_matrix, index = index_labels,
                         columns = column_labels)

    sn.heatmap(df_cm, annot=True, fmt='d', ax=ax)
    ax.set(xlabel='CNN predictions', ylabel='GT labels')

    for t in ax.texts:
        t.set_text(t.get_text() + '\n' + '{0:.2%}'.format(int(t.get_text()) / num_total))
    
    plt.show()


def put_text(image, text, txt_color=tuple([0, 0, 0]), height=16,
             label_width=220, label_scale=0.75, label_thickness=1):
    label_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_COMPLEX_SMALL, label_scale, label_thickness)

    cv2.putText(image, text, tuple([label_width - label_size[0][0], int((height / 2) + (label_size[0][1] / 2))]),
                cv2.FONT_HERSHEY_COMPLEX_SMALL, label_scale, txt_color, label_thickness, cv2.LINE_AA)


def image_grid(images, labels=None, n_cols=None, n_rows=None, max_val=31):
    """
    images: list of images
    n_cols: number of columns
    n_rows: number of rows
    """
    if n_cols is None and n_rows is None:
        n_cols = int(np.round(np.sqrt(len(images))))
        n_rows = int(np.ceil(len(images) / n_cols))
    
    elif n_cols is None:
        n_cols = int(np.ceil(len(images) / n_rows))
    
    elif n_rows is None:
        n_rows = int(np.ceil(len(images) / n_cols))
    
    if len(images[0].shape) == 3:
        grid_image = np.full([n_rows*images[0].shape[0], n_cols*images[0].shape[1], images[0].shape[2]], max_val, np.uint8)
    else:
        grid_image = np.full([n_rows*images[0].shape[0], n_cols*images[0].shape[1]], max_val, np.uint8)

    for i, image in enumerate(images):
        row = i // n_cols
        col = i % n_cols
        
        if labels is not None:
            put_text(image=image, text=labels[i], txt_color=tuple([0, 0, max_val]))

        grid_image[row*image.shape[0]:(row+1)*image.shape[0], col*image.shape[1]:(col+1)*image.shape[1]] = image

    return grid_image


class DrawingCanvas(object):
    
    def __init__(self, title='MNIST Draw', save_event=None, size=(28, 28), pencil_thickness=2.5, display_multiplier=20, save_while_drawing=False):
        super().__init__()
        self.title = title
        self.size = size
        self.pencil_thickness = pencil_thickness
        self.display_multiplier = display_multiplier
        self.save_while_drawing = save_while_drawing
        
        if save_event is None:
            self.save_event = self._default_save
        else:
            self.save_event = save_event
        
        self.root = None
        self.canvas = None
        
        self.pil_image = None  # We will draw the canvas and the PIL image in parallel
        self.pil_draw = None
        
        self.b1_state = "up"
        self.x_old = None
        self.y_old = None
        
        # Create canvas
        self._create_canvas()
    
    def mainloop(self):
        self.root.mainloop()
    
    @staticmethod
    def _default_save(image):
        cv2.imwrite("my_drawing.png", image)
        
        fake_probs = [0.0] * 10
        
        return fake_probs

    def _create_canvas(self):
        self.root = tk.Tk()
        self.root.title(self.title)
        
        # Tkinter create a canvas to draw on
        self.canvas = tk.Canvas(self.root, width=self.size[0] * self.display_multiplier, height=self.size[1] * self.display_multiplier, background='BLACK')
        self.canvas.pack()
        
        # PIL create an empty image and draw object to draw on
        # memory only, not visible  (as we can't save the Tkinter canvas)
        self.pil_image = Image.new("RGB", (self.size[0] * self.display_multiplier, self.size[1] * self.display_multiplier), 'BLACK')
        self.pil_draw = ImageDraw.Draw(self.pil_image)
        
        self.canvas.bind("<Motion>", lambda event : self._motion_event(event))
        self.canvas.bind("<ButtonPress-1>", lambda event : self._b1_down_event(event))
        self.canvas.bind("<ButtonRelease-1>", lambda event : self._b1_up_event(event))
        
        self.results_label = tk.Label(self.root, text="DRAW A NUMBER!", font='Helvetica 12 bold')
        self.results_label.pack()
        
        if not self.save_while_drawing:  # We disable the button
            button_save = tk.Button(self.root, fg="green", text="Predict", command=lambda : self._save_event())
            button_save.pack(side=tk.RIGHT)
        
        button_clear = tk.Button(self.root, fg="red", text="Clear", command=lambda : self._clear_event())
        button_clear.pack(side=tk.LEFT)
    
    def _save_event(self):
        opencv_image = cv2.cvtColor(np.array(self.pil_image), cv2.COLOR_RGB2BGR)
        opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
        opencv_image = cv2.resize(opencv_image, dsize=self.size, interpolation=cv2.INTER_AREA)
        
        net_results = self.save_event(opencv_image)
        assert len(net_results), "Len of net_results should be 10. Provided len: {}".format(len(net_results))
        
        new_text = self._mnist_results_text(net_results)
        self.results_label.configure(text=new_text)
    
    @staticmethod
    def _mnist_results_text(net_results):
        text = []
        for i, result in enumerate(net_results):
            label = str(i)
            
            r_text = "[{}]: {:.1f}%".format(label, result*100.0)
            text.append(r_text)
        
        text = '  |  '.join(text)
        
        return text
        
    
    def _clear_event(self):
        self.pil_image.paste( (0, 0, 0), [0, 0, self.pil_image.size[0], self.pil_image.size[1]])
        self.canvas.delete("all")
        new_text = self._mnist_results_text([0.0] * 10)
        self.results_label.configure(text=new_text)
    
    def _b1_down_event(self, event):
        self.b1_state = "down"
        self._draw_circle(event.x, event.y)
        self.x_old = event.x
        self.y_old = event.y
    
    def _b1_up_event(self, event):
        self.b1_state = "up"
        self.x_old = None
        self.y_old = None
        if self.save_while_drawing:
            self._save_event()
    
    def _motion_event(self, event):
        if self.b1_state == "down":
            if self.x_old is not None and self.y_old is not None:
                self._draw_circle(self.x_old, self.y_old)
                self._draw_line(self.x_old, self.y_old, event.x, event.y)
                self._draw_circle(event.x, event.y)
                
            self.x_old = event.x
            self.y_old = event.y
    
    def _draw_circle(self, x_val, y_val):
        half_thick = (self.pencil_thickness * self.display_multiplier) // 2
        
        self.canvas.create_oval(x_val-half_thick, y_val-half_thick, x_val+half_thick, y_val+half_thick, width=1, fill='WHITE', outline='WHITE')  # do the Tkinter canvas drawings (visible)
        self.pil_draw.ellipse([x_val-half_thick, y_val-half_thick, x_val+half_thick, y_val+half_thick], width=1, fill='WHITE', outline='WHITE')  # do the PIL image/draw (in memory) drawings
        
    def _draw_line(self, x_start, y_start, x_end, y_end):
        thickness = int(self.pencil_thickness * self.display_multiplier)
        
        self.canvas.create_line(x_start, y_start, x_end, y_end, width=thickness, fill='WHITE')  # do the Tkinter canvas drawings (visible)
        self.pil_draw.line([x_start, y_start, x_end, y_end], width=thickness, fill='WHITE')  # do the PIL image/draw (in memory) drawings

