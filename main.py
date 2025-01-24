import csv
import os
import re
import shutil
import cv2
import numpy as np
import pydicom
import glob
import tensorflow as tf
from tqdm import tqdm
from PIL import Image, ImageTk
from skimage import measure, io
import tkinter as tk
from tkinter import filedialog, messagebox, ttk, StringVar
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import sys
import threading
import math
import trimesh
from tensorflow.keras.preprocessing import image as q1
from scipy.ndimage import label, binary_fill_holes
from textwrap import wrap

import pygame
from pygame.locals import *
from pygame.constants import *
from OpenGL.GL import *
from OpenGL.GLU import *

class OBJ:
    def __init__(self, filename, swapyz=False):
        self.vertices = []
        self.normals = []
        self.texcoords = []
        self.faces = []

        material = None
        for line in open(filename, "r"):
            if line.startswith('#'): continue
            values = line.split()
            if not values: continue
            if values[0] == 'v':
                v = list(map(float, values[1:4]))
                if swapyz:
                    v = [v[0], v[2], v[1]]
                self.vertices.append(v)
            elif values[0] == 'vn':
                v = list(map(float, values[1:4]))
                if swapyz:
                    v = [v[0], v[2], v[1]]
                self.normals.append(v)
            elif values[0] == 'vt':
                self.texcoords.append(list(map(float, values[1:3])))
            elif values[0] in ('usemtl', 'usemat'):
                material = values[1]
            elif values[0] == 'f':
                face = []
                texcoords = []
                norms = []
                for v in values[1:]:
                    w = v.split('/')
                    face.append(int(w[0]))
                    if len(w) >= 2 and len(w[1]) > 0:
                        texcoords.append(int(w[1]))
                    else:
                        texcoords.append(0)
                    if len(w) >= 3 and len(w[2]) > 0:
                        norms.append(int(w[2]))
                    else:
                        norms.append(0)
                self.faces.append((face, norms, texcoords, material))

        self.gl_list = glGenLists(1)
        glNewList(self.gl_list, GL_COMPILE)
        glEnable(GL_TEXTURE_2D)
        glFrontFace(GL_CCW)
        for face in self.faces:
            vertices, normals, texture_coords, material = face

            glBegin(GL_POLYGON)
            for i in range(len(vertices)):
                if normals[i] > 0:
                    glNormal3fv(self.normals[normals[i] - 1])
                if texture_coords[i] > 0:
                    glTexCoord2fv(self.texcoords[texture_coords[i] - 1])
                glVertex3fv(self.vertices[vertices[i] - 1])
            glEnd()
        glDisable(GL_TEXTURE_2D)
        glEndList()

class DicomConverterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Brain Seg")
        self.result = None
        self.areas = None
        self.predictions = None

        # Initialize the output_dir attribute
        self.output_dir = None

        # Main vertical paned window
        self.main_paned = tk.PanedWindow(root, orient=tk.VERTICAL)
        self.main_paned.pack(fill=tk.BOTH, expand=1)

        # Top horizontal paned window
        self.top_paned = tk.PanedWindow(self.main_paned, orient=tk.HORIZONTAL)
        self.main_paned.add(self.top_paned)

        # Frame Setup
        self.frame_upload = tk.Frame(self.top_paned, relief='solid', bd=1)
        self.frame_segment = tk.Frame(self.top_paned, relief='solid', bd=1)
        self.frame_results = tk.Frame(self.top_paned, relief='solid', bd=1)

        self.top_paned.add(self.frame_upload)
        self.top_paned.add(self.frame_segment)
        self.top_paned.add(self.frame_results)

        # Title
        self.label_title = tk.Label(self.frame_segment, text="Brain Seg", font=("Helvetica", 16), fg="blue")
        self.label_title.pack(pady=10)

        # Upload Directory Button
        self.btn_upload = tk.Button(self.frame_upload, text="Upload Directory", command=self.upload_directory)
        self.btn_upload.pack(pady=20)

        # Display Selected Directory Path
        self.label_directory_path = tk.Label(self.frame_upload, text="", wraplength=250)
        self.label_directory_path.pack(pady=10)

        # Convert DICOM to PNG Button
        self.btn_segment = tk.Button(self.frame_segment, text="Convert Dicom to PNG", command=self.segment_directory)
        self.btn_segment.pack(pady=20)

        # Run AI Button
        self.btn_run_ai = tk.Button(self.frame_segment, text="Perform Full Segmentation", command=self.run_ai)
        self.btn_run_ai.pack(pady=20)

        # Run Region Segmentation Button
        self.btn_run_region_ai = tk.Button(self.frame_segment, text="Run Region Segmentation", command=self.run_region_ai)
        self.btn_run_region_ai.pack(pady=20)

        # Export Results Dropdown Menu
        self.export_options = ["Original PNGs", "Segmented Images", "Intensity Histogram", "Patient Information", "Region Segmentation Images"]
        self.export_var = StringVar(value=self.export_options[0])
        self.dropdown_export = ttk.Combobox(self.frame_segment, textvariable=self.export_var, values=self.export_options)
        self.dropdown_export.config(width=20)
        self.dropdown_export.pack(pady=20)

        # Export Button
        self.btn_export = tk.Button(self.frame_segment, text="Export Results", command=self.export_results)
        self.btn_export.pack(pady=20)

        # Notebook for tabs
        self.notebook = ttk.Notebook(self.frame_results)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Tab for original PNG images
        self.tab_png = tk.Frame(self.notebook)
        self.notebook.add(self.tab_png, text="Original PNGs")

        # Tab for segmented images
        self.tab_segmented = tk.Frame(self.notebook)
        self.notebook.add(self.tab_segmented, text="Segmented Images")

        # Tab for intensity histogram
        self.tab_histogram = tk.Frame(self.notebook)
        self.notebook.add(self.tab_histogram, text="Intensity Histogram")

        # Tab for patient information
        self.tab_patient_info = tk.Frame(self.notebook)
        self.notebook.add(self.tab_patient_info, text="Patient Information")

        # Tab for 3D model
        self.tab_3d_model = tk.Frame(self.notebook)
        self.notebook.add(self.tab_3d_model, text="View 3D Model")

        # Tab for region segmentation
        self.tab_region_segmented = tk.Frame(self.notebook)
        self.notebook.add(self.tab_region_segmented, text="Region Segmentation")

        # Button to view 3D model
        self.btn_view_3d_model = tk.Button(self.tab_3d_model, text="View 3D Model", command=self.view_3d_model)
        self.btn_view_3d_model.pack(pady=20)

        # Display PNG Results
        self.canvas_png = tk.Canvas(self.tab_png, scrollregion=(0, 0, 1000, 1000))
        self.vbar_png = tk.Scrollbar(self.tab_png, orient=tk.VERTICAL, command=self.canvas_png.yview)
        self.vbar_png.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas_png.config(yscrollcommand=self.vbar_png.set)
        self.canvas_png.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)

        # Display Segmented Results
        self.canvas_segmented = tk.Canvas(self.tab_segmented, scrollregion=(0, 0, 1000, 1000))
        self.vbar_segmented = tk.Scrollbar(self.tab_segmented, orient=tk.VERTICAL, command=self.canvas_segmented.yview)
        self.vbar_segmented.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas_segmented.config(yscrollcommand=self.vbar_segmented.set)
        self.canvas_segmented.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)

        # Display Region Segmented Results
        self.canvas_region_segmented = tk.Canvas(self.tab_region_segmented, scrollregion=(0, 0, 1000, 1000))
        self.vbar_region_segmented = tk.Scrollbar(self.tab_region_segmented, orient=tk.VERTICAL, command=self.canvas_region_segmented.yview)
        self.vbar_region_segmented.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas_region_segmented.config(yscrollcommand=self.vbar_region_segmented.set)
        self.canvas_region_segmented.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)

        # Display Histogram Results
        self.canvas_histogram = tk.Canvas(self.tab_histogram, scrollregion=(0, 0, 1000, 1000))
        self.vbar_histogram = tk.Scrollbar(self.tab_histogram, orient=tk.VERTICAL, command=self.canvas_histogram.yview)
        self.hbar_histogram = tk.Scrollbar(self.tab_histogram, orient=tk.HORIZONTAL, command=self.canvas_histogram.xview)
        self.vbar_histogram.pack(side=tk.RIGHT, fill=tk.Y)
        self.hbar_histogram.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas_histogram.config(yscrollcommand=self.vbar_histogram.set, xscrollcommand=self.hbar_histogram.set)
        self.canvas_histogram.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)

        # Store references to images
        self.image_refs_png = []
        self.image_refs_segmented = []
        self.image_refs_region_segmented = []

        # Text box for terminal output
        self.text_output = tk.Text(self.main_paned, height=10, bg='white')
        self.main_paned.add(self.text_output)

        # Text widget for patient information
        self.text_patient_info = tk.Text(self.tab_patient_info, bg='white')
        self.text_patient_info.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Redirect stdout and stderr
        sys.stdout = self.RedirectText(self.text_output)
        sys.stderr = self.RedirectText(self.text_output)

        self.frame_results.bind("<Configure>", self.on_frame_configure)

    def create_dir(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

    def delete_all_files(self, folder_path):
        for root, dirs, files in os.walk(folder_path):
            for file_name in files:
                file_path = os.path.join(root, file_name)
                if os.path.isfile(file_path):
                    os.remove(file_path)

    def upload_directory(self):
        self.dicom_dir = filedialog.askdirectory()
        if self.dicom_dir:
            self.label_directory_path.config(text=self.dicom_dir)

    def segment_directory(self):
        self.output_dir = 'PNG_images'
        self.create_dir(self.output_dir)
        self.delete_all_files(self.output_dir)

        if self.dicom_dir:
            for file in os.listdir(self.dicom_dir):
                if file.lower().endswith('.dcm'):
                    dicom_path = os.path.join(self.dicom_dir, file)
                    self.convert_dicom_to_png(dicom_path, self.output_dir)
            self.display_results(self.canvas_png, self.image_refs_png, self.output_dir)

    def convert_dicom_to_png(self, dicom_path, output_dir):
        try:
            ds = pydicom.dcmread(dicom_path)
            img = ds.pixel_array

            if img.dtype != np.uint8:
                img = img.astype(np.float32)
                img = (np.maximum(img, 0) / img.max()) * 255.0
                img = np.uint8(img)

            img = Image.fromarray(img)
            png_filename = os.path.splitext(os.path.basename(dicom_path))[0] + '.png'
            png_path = os.path.join(output_dir, png_filename)
            img.save(png_path)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to convert {dicom_path}: {e}")

    def display_results(self, canvas, image_refs, image_dir, result=None, areas=None, predictions=None):
        canvas.delete("all")
        image_refs.clear()

        width = self.frame_results.winfo_width()
        max_columns = max(1, width // 110)
        x_position = 10
        y_position = 10
        column = 0

        filenames = sorted(os.listdir(image_dir), key=lambda x: int(re.findall(r'\d+', x)[0]))

        for i, filename in enumerate(filenames):
            if filename.lower().endswith('.png'):
                img_path = os.path.join(image_dir, filename)
                img = Image.open(img_path)
                img.thumbnail((100, 100))
                img = ImageTk.PhotoImage(img)
                canvas.create_image(x_position, y_position, anchor=tk.NW, image=img)

                if result is not None and areas is not None and predictions is not None:
                    text = f"INT: {int(result[i])} Area: {int(areas[i])} {predictions[i]}"
                    wrapped_text = "\n".join(wrap(text, width=20))
                    text_id = canvas.create_text(x_position, y_position + 110, anchor=tk.NW, text=wrapped_text, width=100)
                    image_refs.append((img, text_id))
                else:
                    image_refs.append(img)

                if column < max_columns - 1:
                    column += 1
                    x_position += 110
                else:
                    column = 0
                    x_position = 10
                    y_position += 150  # Increased from 130 to provide more space between rows

        canvas.config(scrollregion=(0, 0, width, y_position + 150))  # Updated to match new y_position increment
        # Bind the resize event to update the canvas
        self.root.bind("<Configure>", self.on_frame_configure)

    def on_frame_configure(self, event):
        try:
            if self.output_dir:
                self.display_results(self.canvas_png, self.image_refs_png, self.output_dir)
            self.display_segmented_results(self.canvas_segmented, self.image_refs_segmented, 'segmentation')
            self.display_results(self.canvas_region_segmented, self.image_refs_region_segmented, 'multiclass_images')
        except TypeError as e:
            if "missing 1 required positional argument" not in str(e):
                raise e  # Re-raise if it's not the specific TypeError

    def run_ai(self):
        threading.Thread(target=self.run_ai_thread).start()

    def run_ai_thread(self):
        import time
        import re
        import cv2
        import numpy as np
        import glob
        import tensorflow as tf
        from tqdm import tqdm
        from tensorflow.keras.preprocessing import image as q1
        from scipy.ndimage import label

        inputdir = self.dicom_dir
        temp = os.path.normpath('temp')

        self.create_dir(temp)
        self.delete_all_files(temp)

        self.dicom_to_png(inputdir, temp)
        
        outdir = os.path.normpath("PNG_images")
        self.create_dir(outdir)
        self.delete_all_files(outdir)

        self.find_and_copy_png_files(temp, outdir)

        items = os.listdir(inputdir)
        first_item_path = ""
        if items:
            first_item = items[0]
            first_item_path = os.path.abspath(os.path.join(inputdir, first_item))
            first_item_path_raw = r"{}".format(first_item_path)
            print("Full path to the first item (raw string):", first_item_path_raw)
        else:
            print("The folder is empty.")

        dicom_file_path = first_item_path
        dicom_data = pydicom.dcmread(dicom_file_path)

        patient_info = {
            "Patient Name": dicom_data.get("PatientName", "N/A"),
            "Patient ID": dicom_data.get("PatientID", "N/A"),
            "Patient Birth Date": dicom_data.get("PatientBirthDate", "N/A"),
            "Patient Sex": dicom_data.get("PatientSex", "N/A"),
            "Patient Age": dicom_data.get("PatientAge", "N/A"),
            "Patient Weight": dicom_data.get("PatientWeight", "N/A"),
            "Patient Address": dicom_data.get("PatientAddress", "N/A"),
        }

        self.display_patient_info(patient_info)

        test_path = "PNG_images"
        directory = test_path

        def natural_sort_key(s):
            return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

        files = [f for f in os.listdir(directory) if f.endswith('.png')]
        files.sort(key=natural_sort_key)

        for i, filename in enumerate(files):
            new_name = f"{i}.png"
            os.rename(os.path.join(directory, filename), os.path.join(directory, new_name))

        dataset_path = os.path.join("dataset", "non-aug")
        files_dir = os.path.join("files", "non-aug")
        model_file = os.path.join(files_dir, "unet_FULL_BRAIN.h5")
        prediction_file = os.path.join(files_dir, "brainmodel.h5")

        self.create_dir(files_dir)

        save_path = "prediction"
        self.create_dir(save_path)
        self.delete_all_files(save_path)
        model2 = tf.keras.models.load_model(model_file)
        model2.summary()

        test_x = sorted(glob.glob(os.path.join(outdir, "*")))

        print(len(test_x))
        time_taken = []

        for x in tqdm(test_x):
            name = os.path.basename(x)

            x_img = cv2.imread(x, cv2.IMREAD_COLOR)
            if x_img is None:
                print(f"Failed to read image: {x}")
                continue
            x_img = x_img / 255.0
            x_img = np.expand_dims(x_img, axis=0)

            start_time = time.time()
            p = model2.predict(x_img)[0]
            total_time = time.time() - start_time
            time_taken.append(total_time)
            p = p > 0.5
            p = (p * 255).astype(np.uint8)

            save_path_full = os.path.join(save_path, name)
            os.makedirs(os.path.dirname(save_path_full), exist_ok=True)

            success = cv2.imwrite(save_path_full, p)

        input_folder_fill = save_path
        output_folder_fill = os.path.join('temp_bin_images')
        self.create_dir(output_folder_fill)
        self.delete_all_files(output_folder_fill)
        self.process_folder(input_folder_fill, output_folder_fill)
        save_path = os.path.join(output_folder_fill, 'PNG_images')  # Update save_path to the correct directory

        mask_folder = save_path
        output_folder = os.path.join("segmentation")
        self.create_dir(output_folder)
        self.delete_all_files(output_folder)

        self.composite_images(test_path, mask_folder, output_folder)
        image_folder = output_folder

        image_files = glob.glob(os.path.join(image_folder, '*.*g'))

        for image_path in image_files:
            img = cv2.imread(image_path)

            if img.ndim == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            _, bw = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            labeled_array, num_features = label(bw)

            if num_features > 5:
                os.remove(image_path)
                print(f'Deleted {os.path.basename(image_path)}')
                mask_image_path = os.path.join(mask_folder, os.path.basename(image_path))

                if os.path.exists(mask_image_path):
                    os.remove(mask_image_path)
                    print(f'Deleted mask {os.path.basename(mask_image_path)}')

        print('Processing complete.')

        black_threshold = 100
        percentage_threshold = 95

        image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]

        for image_file in image_files:
            image_path = os.path.join(image_folder, image_file)

            img = cv2.imread(image_path)
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            total_pixels = img.size
            num_black_pixels = np.sum(img < black_threshold)
            black_percentage = (num_black_pixels / total_pixels) * 100
            if black_percentage > percentage_threshold:

                os.remove(image_path)
                print(f'Deleted {image_file}')

                mask_image_path = os.path.join(mask_folder, image_file)

                if os.path.exists(mask_image_path):
                    os.remove(mask_image_path)
                    print(f'Deleted mask {image_file}')

        print('Processing complete.')

        model_pred = tf.keras.models.load_model(prediction_file)
        model_pred.summary()

        results = []
        folder_path = output_folder
        for filename in os.listdir(folder_path):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                img_path = os.path.join(folder_path, filename)
                img = q1.load_img(img_path, target_size=(128, 128))
                img_array = q1.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)
                img_array = img_array / 255.0

                predictions = model_pred.predict(img_array)

                if predictions[0][0] > 0.5:
                    results.append(('CN', str(round(((predictions[0][0] - 0.5) / 0.5) * 100, 2)) + "%"))
                else:
                    results.append(('AD', str(round(((0.5 - predictions[0][0]) / 0.5) * 100, 2)) + "%"))

        area_array = []
        png_arra = []
        total_value = []
        image_directory = mask_folder
        filenames = os.listdir(image_directory)
        sorted_filenames = sorted(filenames, key=lambda x: int(re.findall(r'\d+', x)[0]))

        for filename in sorted_filenames:
            file_path = os.path.join(image_directory, filename)
            png_arra.append(file_path)
            image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            white_pixel_count = cv2.countNonZero(image)
            area_array.append(white_pixel_count)

        image_directory = output_folder
        filenames = os.listdir(image_directory)
        sorted_filenames = sorted(filenames, key=lambda x: int(re.findall(r'\d+', x)[0]))

        for filename in sorted_filenames:
            file_path = os.path.join(image_directory, filename)
            png_arra.append(file_path)
            image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            total = np.sum(image)
            total_value.append(total)

        array1 = np.array(area_array)
        array2 = np.array(total_value)

        print("Shape of array1:", array1.shape)
        print("Shape of array2:", array2.shape)
        self.result = np.where(array2 != 0, array2 / array1, 0)
        self.areas = array1
        self.predictions = results

        # Create png_images2 directory with remaining grayscale images
        self.create_png_images2_directory(test_path, output_folder)

        areas = array1

        image_directory = test_path
        filenames = os.listdir(image_directory)
        sorted_filenames = sorted(filenames, key=lambda x: int(re.findall(r'\d+', x)[0]))

        num_images = len(sorted_filenames)
        num_cols = 10
        num_rows = math.ceil(num_images / num_cols)

        fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(20, num_rows * 2))

        for ax_row, row_index in zip(axes, range(num_rows)):
            for ax, filename in zip(ax_row, sorted_filenames[row_index * num_cols:]):
                file_path = os.path.join(image_directory, filename)
                image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                ax.imshow(image, cmap='gray')
                ax.set_title("INPUT")
                ax.axis('off')

        for i in range(num_images, num_rows * num_cols):
            axes.flatten()[i].axis('off')

        plt.tight_layout()
        plt.show()

        for key, value in patient_info.items():
            print(f"{key}: {value}")

        image_directory = output_folder
        filenames = os.listdir(image_directory)
        sorted_filenames = sorted(filenames, key=lambda x: int(re.findall(r'\d+', x)[0]))

        num_images = len(sorted_filenames)
        num_cols = 10
        num_rows = math.ceil(num_images / num_cols)

        fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(20, num_rows * 2))

        for ax_row, row_index in zip(axes, range(num_rows)):
            for ax, filename, intentsity, area, pred in zip(ax_row, sorted_filenames[row_index * num_cols:], self.result[row_index * num_cols:], areas[row_index * num_cols:], self.predictions[row_index * num_cols:]):
                file_path = os.path.join(image_directory, filename)
                image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                ax.imshow(image, cmap='gray')
                title_l = "INT: " + str(int(intentsity)) + " Area : " + str(int(area)) + str(pred)
                wrapped_title = "\n".join(wrap(title_l, width=20))
                ax.set_title(wrapped_title)
                ax.axis('off')

        for i in range(num_images, num_rows * num_cols):
            axes.flatten()[i].axis('off')

        plt.tight_layout()
        plt.show()
        print(self.predictions[0])

        self.display_segmented_results(output_folder, self.result, self.areas, self.predictions)
        self.plot_histogram(self.result)

        volume = self.load_images(self.rename_directory(output_folder))
        segmented_volume = self.segment_brain(volume, 90)
        verts, faces, normals = self.extract_surface(segmented_volume)
        self.plot_3d(verts, faces)
        self.write_obj("output_brain_model.obj", verts, faces, normals)

        self.adjust_3d_model_orientation("output_brain_model.obj", "adjusted_model_trimesh_treated.obj")

        self.add_material_lines("adjusted_model_trimesh_treated.obj", "adjusted_model_trimesh_treated_mat_lines.obj")

        print("All imports were successful!")

    def create_png_images2_directory(self, original_dir, segmented_dir):
        self.create_dir('png_images2')
        for filename in os.listdir(original_dir):
            if filename in os.listdir(segmented_dir):
                shutil.copy2(os.path.join(original_dir, filename), 'png_images2')

    def display_patient_info(self, patient_info):
        self.text_patient_info.delete('1.0', tk.END)
        for key, value in patient_info.items():
            self.text_patient_info.insert(tk.END, f"{key}: {value}\n")

    def display_segmented_results(self, directory, result, areas, predictions):
        self.canvas_segmented.delete("all")
        self.image_refs_segmented.clear()

        width = self.frame_results.winfo_width()
        max_columns = max(1, width // 110)
        x_position = 10
        y_position = 10
        column = 0

        filenames = os.listdir(directory)
        sorted_filenames = sorted(filenames, key=lambda x: int(re.findall(r'\d+', x)[0]))

        for i, (filename, intensity, area, prediction) in enumerate(zip(sorted_filenames, result, areas, predictions)):
            if filename.lower().endswith('.png'):
                img_path = os.path.join(directory, filename)
                img = Image.open(img_path)
                img.thumbnail((100, 100))
                img = ImageTk.PhotoImage(img)
                self.canvas_segmented.create_image(x_position, y_position, anchor=tk.NW, image=img)
                text = f"INT: {int(intensity)} Area: {int(area)} {prediction}"
                wrapped_text = "\n".join(wrap(text, width=20))
                text_id = self.canvas_segmented.create_text(x_position, y_position + 110, anchor=tk.NW, text=wrapped_text, width=100)

                if column < max_columns - 1:
                    column += 1
                    x_position += 110
                else:
                    column = 0
                    x_position = 10
                    y_position += 150  # Increased from 130 to provide more space between rows

                self.image_refs_segmented.append((img, text_id))

        self.canvas_segmented.config(scrollregion=(0, 0, width, y_position + 150))  # Updated to match new y_position increment

        # Bind the resize event to update the canvas
        self.root.bind("<Configure>", self.on_frame_configure)

    def export_results(self):
        selected_option = self.export_var.get()
        export_dir = filedialog.askdirectory(title=f"Select Directory to Save {selected_option}")
        if not export_dir:
            return

        if selected_option == "Original PNGs":
            self.copy_files(self.output_dir, export_dir)
        elif selected_option == "Segmented Images":
            self.copy_files('segmentation', export_dir)
        elif selected_option == "Intensity Histogram":
            histogram_path = os.path.join(export_dir, "histogram.png")
            shutil.copy2("histogram.png", histogram_path)
            messagebox.showinfo("Export Complete", f"Histogram saved to {histogram_path}")
        elif selected_option == "Patient Information":
            self.export_patient_info(export_dir)
            self.export_calculations(export_dir)
        elif selected_option == "Region Segmentation Images":
            self.copy_files('multiclass_images', export_dir)
        else:
            messagebox.showerror("Error", "Invalid export option selected.")

    def export_patient_info(self, export_dir):
        patient_info_path = os.path.join(export_dir, "patient_info.txt")
        with open(patient_info_path, "w") as f:
            f.write(self.text_patient_info.get("1.0", tk.END))
        messagebox.showinfo("Export Complete", f"Patient information saved to {patient_info_path}")

    def export_calculations(self, export_dir):
        calculations_path = os.path.join(export_dir, "calculations.csv")
        with open(calculations_path, "w", newline="") as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(["Image", "Intensity", "Area", "Prediction"])
            for i, (intensity, area, prediction) in enumerate(zip(self.result, self.areas, self.predictions)):
                image_name = f"{i}.png"
                csvwriter.writerow([image_name, int(intensity), int(area), prediction])
        messagebox.showinfo("Export Complete", f"Calculations saved to {calculations_path}")

    def copy_files(self, src_dir, dest_dir):
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
        for file_name in os.listdir(src_dir):
            full_file_name = os.path.join(src_dir, file_name)
            if os.path.isfile(full_file_name):
                shutil.copy(full_file_name, dest_dir)
        messagebox.showinfo("Export Complete", f"Files saved to {dest_dir}")

    def dicom_to_png(self, dicom_path, output_folder):
        self.create_dir(output_folder)
        dicom_files = [os.path.join(dicom_path, f) for f in os.listdir(dicom_path) if f.endswith('.dcm')]

        for dicom_file in dicom_files:
            ds = pydicom.dcmread(dicom_file)
            img_array = ds.pixel_array
            img_name = os.path.splitext(os.path.basename(dicom_file))[0] + '.png'
            plt.imsave(os.path.join(output_folder, img_name), img_array, cmap='gray')

    def find_and_copy_png_files(self, source_dir, destination_dir):
        for root, dirs, files in os.walk(source_dir):
            for file in files:
                if file.lower().endswith('.png'):
                    shutil.copy2(os.path.join(root, file), destination_dir)

    def process_folder(self, input_folder, output_folder, threshold_value=128):
        output_subfolder = os.path.join(output_folder, 'PNG_images')
        os.makedirs(output_subfolder, exist_ok=True)
        for filename in os.listdir(input_folder):
            if filename.lower().endswith('.png'):
                input_path = os.path.join(input_folder, filename)
                output_path = os.path.join(output_subfolder, filename)  # Ensure the correct path
                self.process_image(input_path, output_path, threshold_value)

    def process_image(self, image_path, output_path, threshold_value=128):
        grayscale_image = Image.open(image_path).convert('L')
        grayscale_array = np.array(grayscale_image)
        binary_image = grayscale_array > threshold_value
        filled_binary_image = binary_fill_holes(binary_image)
        filled_grayscale_array = (filled_binary_image * 255).astype(np.uint8)
        filled_grayscale_image = Image.fromarray(filled_grayscale_array)
        filled_grayscale_image.save(output_path)

    def composite_images(self, image_folder, mask_folder, output_folder):
        image_files = os.listdir(image_folder)

        for image_file in image_files:
            image_path = os.path.join(image_folder, image_file)
            mask_file = image_file.split('.')[0] + '.png'
            mask_path = os.path.join(mask_folder, mask_file)

            if os.path.exists(mask_path):  # Ensure the mask exists before processing
                image = Image.open(image_path)
                mask = Image.open(mask_path)

                result = Image.composite(image, Image.new('RGB', image.size, (0, 0, 0)), mask)

                output_file = os.path.join(output_folder, image_file)
                self.create_dir(os.path.dirname(output_file))  # Ensure the directory exists
                result.save(output_file)
            else:
                print(f"Mask not found for image: {image_file}")

    def plot_histogram(self, data):
        fig, ax = plt.subplots(figsize=(20, 6))
        ax.bar(range(len(data)), data, align='center', edgecolor='black')
        ax.set_xlabel('Slice')
        ax.set_ylabel('Gluecose Intensity')
        ax.set_title('Intensity Of Slices 0 - ' + str(len(data)))
        ax.set_xticks(range(0, len(data), 5))
        ax.grid(False)
        plt.tight_layout()

        fig.savefig('histogram.png')

        plt.show()

        self.show_histogram()

    def show_histogram(self):
        self.canvas_histogram.delete("all")

        img = Image.open('histogram.png')
        img_width, img_height = img.size
        img = ImageTk.PhotoImage(img)
        self.canvas_histogram.create_image(0, 0, anchor=tk.NW, image=img)
        self.canvas_histogram.image = img
        self.canvas_histogram.config(scrollregion=(0, 0, img_width, img_height))

    def load_images(self, directory):
        def extract_index(filename):
            match = re.search(r'(\d+)', filename)
            return int(match.group(1)) if match else -1

        files = sorted(
            [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith('.png')],
            key=lambda x: extract_index(os.path.basename(x))
        )

        images = [io.imread(file, as_gray=True) for file in files]
        volume = np.stack(images, axis=-1)
        return volume

    def segment_brain(self, volume, percentile=90):
        threshold = np.percentile(volume, percentile)
        segmented = volume > threshold
        return segmented.astype(np.float32)

    def rename_directory(self, directory_path):
        directory = directory_path
        files = [f for f in os.listdir(directory) if f.endswith('.png')]
        files.sort(key=self.natural_sort_key)  # Use 'key=' keyword argument

        for i, filename in enumerate(files):
            new_name = f"{i}.png"
            os.rename(os.path.join(directory, filename), os.path.join(directory, new_name))

        return directory_path

    def extract_surface(self, volume, level=0.5):
        verts, faces, normals, values = measure.marching_cubes(volume, level=level)
        return verts, faces, normals

    def write_obj(self, filename, verts, faces, normals):
        with open(filename, 'w') as file:
            for v in verts:
                file.write(f"v {v[0]} {v[1]} {v[2]}\n")
            for face in faces:
                file.write("f {} {} {}\n".format(face[0] + 1, face[1] + 1, face[2] + 1))
            for v_normal in normals:
                file.write(f"vn {v_normal[0]} {v_normal[1]} {v_normal[2]}\n")

    def plot_3d(self, verts, faces):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2], cmap='bone', lw=1)
        plt.show()

    def natural_sort_key(self, s):
        return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

    def adjust_3d_model_orientation(self, input_model, output_model):
        mesh = trimesh.load(input_model)
        centroid = mesh.centroid
        mesh.vertices -= centroid
        theta = -np.pi / 2
        rotation_matrix = np.array([
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)]
        ])
        rotated_vertices = np.dot(mesh.vertices, rotation_matrix.T)
        centroid = np.mean(rotated_vertices, axis=0)
        centered_vertices = rotated_vertices - centroid
        adjusted_mesh = trimesh.Trimesh(vertices=centered_vertices, faces=mesh.faces, vertex_normals=mesh.vertex_normals)
        adjusted_mesh.invert()
        adjusted_mesh.export(output_model)

    def add_material_lines(self, input_model, output_model):
        with open(input_model, "r") as f:
            lines = f.readlines()

        with open(output_model, "w") as n:
            insert_string = "mtllib generic_material.mtl\nusemtl BRAISEG\ns off\n"
            n.write(insert_string)
            for line in lines:
                n.write(line)

    def view_3d_model(self):
        threading.Thread(target=self.view_3d_model_thread).start()

    def view_3d_model_thread(self):
        pygame.init()
        viewport = (800, 600)
        hx = viewport[0] // 2
        hy = viewport[1] // 2
        srf = pygame.display.set_mode(viewport, OPENGL | DOUBLEBUF)

        glLightfv(GL_LIGHT0, GL_POSITION, (-40, 200, 100, 0.0))
        glLightfv(GL_LIGHT0, GL_AMBIENT, (0.2, 0.2, 0.2, 1.0))
        glLightfv(GL_LIGHT0, GL_DIFFUSE, (0.5, 0.5, 0.5, 1.0))
        glEnable(GL_LIGHT0)
        glEnable(GL_LIGHTING)
        glEnable(GL_COLOR_MATERIAL)
        glEnable(GL_DEPTH_TEST)
        glShadeModel(GL_SMOOTH)

        obj = OBJ("adjusted_model_trimesh_treated_mat_lines.obj", swapyz=True)

        clock = pygame.time.Clock()

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        width, height = viewport
        gluPerspective(90.0, width / float(height), 1, 100.0)
        glEnable(GL_DEPTH_TEST)
        glMatrixMode(GL_MODELVIEW)

        rx, ry = (0, 0)
        tx, ty = (0, 0)
        zpos = 5
        rotate = move = False

        custom_exit = False
        while not custom_exit:
            clock.tick(30)
            pressed = pygame.key.get_pressed()

            if pressed[pygame.K_LEFT]:
                rx += -1.25

            if pressed[pygame.K_RIGHT]:
                rx += 1.25

            if pressed[pygame.K_UP]:
                ry += -1.25

            if pressed[pygame.K_DOWN]:
                ry += 1.25

            for e in pygame.event.get():
                if e.type == QUIT:
                    custom_exit = True
                elif e.type == KEYDOWN and e.key == K_ESCAPE:
                    custom_exit = True
                elif e.type == MOUSEBUTTONDOWN:
                    if e.button == 4:
                        zpos = max(1, zpos - 1)
                    elif e.button == 5:
                        zpos += 1
                    elif e.button == 1:
                        rotate = True
                    elif e.button == 3:
                        move = True
                elif e.type == MOUSEBUTTONUP:
                    if e.button == 1:
                        rotate = False
                    elif e.button == 3:
                        move = False
                elif e.type == MOUSEMOTION:
                    i, j = e.rel
                    if rotate:
                        rx += i
                        ry += j
                    if move:
                        tx += i
                        ty -= j

            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            glLoadIdentity()

            glTranslate(tx / 20., ty / 20., -zpos)
            glTranslate(0, 0, -60)
            glRotate(ry, 1, 0, 0)
            glRotate(rx, 0, 1, 0)
            glRotate(-90, 1, 0, 0)
            glRotate(0, 0, 90, 1)

            glFrontFace(GL_CW)
            glColor3f(0.24, 0.58, 0.87)
            glCallList(obj.gl_list)

            pygame.display.flip()

        pygame.quit()


    def run_region_ai(self):
        threading.Thread(target=self.run_region_ai_thread).start()

    def run_region_ai_thread(self):
        import numpy as np
        import os
        import re
        import matplotlib.pyplot as plt
        from tensorflow.keras.preprocessing.image import load_img, img_to_array
        from tensorflow.keras.models import load_model
        import cv2
        from pathlib import Path
        import glob
        import math
        from textwrap import wrap

        def natural_sort_key(s):
            return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

        def load_test_images(image_dir, image_size):
            images = []
            image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')], key=natural_sort_key)
            for img_file in image_files:
                img_path = os.path.join(image_dir, img_file)
                img = load_img(img_path, color_mode="grayscale", target_size=image_size)
                img = img_to_array(img) / 255.0
                images.append(img)
            images = np.array(images)
            return images, image_files

        def decode_predictions(preds):
            return np.argmax(preds, axis=-1)

        def apply_custom_color_map(prediction, mask_classes):
            colored_mask = np.zeros((*prediction.shape, 3))
            for class_idx, color in mask_classes.items():
                colored_mask[prediction == class_idx] = color
            return colored_mask

        mask_classes = {
            0: (0, 0, 0),
            1: (255, 0, 0),
            2: (0, 255, 0),
            3: (0, 0, 255),
            4: (255, 255, 0),
            5: (255, 0, 255)
        }

        # Directory for segmented masks
        test_image_dir = 'segmentation'
        # Directory for grayscale PNG images
        grayscale_img_dir = 'png_images2'  # Updated to use png_images2
        image_size = (128, 128)
        test_images, test_image_files = load_test_images(test_image_dir, image_size)

        # Load the trained model
        model = load_model('unet_model.h5')

        # Predict the segmentation masks
        predictions = model.predict(test_images)

        # Decode predictions
        decoded_predictions = decode_predictions(predictions)

        # Save predicted segmentation masks
        output_dir = os.path.join('multiclass_images')
        self.create_dir(output_dir)
        self.delete_all_files(output_dir)
        for i, pred in enumerate(decoded_predictions):
            colored_mask = apply_custom_color_map(pred, mask_classes)
            plt.imshow(test_images[i], cmap='gray')
            plt.imshow(colored_mask, alpha=0.5)
            plt.axis('off')
            output_path = os.path.join(output_dir, f'prediction_{i}.png')
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
            plt.close()

        print("Segmentation results saved successfully.")

        # Creating binary masks
        input_dir = 'segmentation'
        threshold_folder = os.path.join("threshold")
        self.create_dir(threshold_folder)
        self.delete_all_files(threshold_folder)
        Path(threshold_folder).mkdir(parents=True, exist_ok=True)

        threshold_value = 100

        for filename in os.listdir(input_dir):
            if filename.endswith(".png") or filename.endswith(".jpg"):
                img = cv2.imread(os.path.join(input_dir, filename), cv2.IMREAD_GRAYSCALE)
                _, mask = cv2.threshold(img, threshold_value, 255, cv2.THRESH_BINARY)
                mask_filename = os.path.join(threshold_folder, filename)
                cv2.imwrite(mask_filename, mask)

        print("Segmentation masks created and saved successfully.")

        def resize_image(image, size=(128, 128)):
            return cv2.resize(image, size, interpolation=cv2.INTER_AREA)

        def combine_images(color_img, grayscale_img, mask_img):
            color_img = resize_image(color_img, size=grayscale_img.shape[:2])
            mask_img = resize_image(mask_img, size=grayscale_img.shape[:2])
            _, mask = cv2.threshold(mask_img, 127, 255, cv2.THRESH_BINARY)
            grayscale_img_colored = cv2.cvtColor(grayscale_img, cv2.COLOR_GRAY2BGR)
            combined_img = np.where(mask[..., None] == 255, color_img, grayscale_img_colored)
            return combined_img

        color_img_dir = output_dir
        mask_img_dir = threshold_folder
        output_img_dir = os.path.join("refined_region_mask")
        self.create_dir(output_img_dir)
        self.delete_all_files(output_img_dir)

        color_img_paths = sorted(glob.glob(os.path.join(color_img_dir, '*.png')), key=lambda x: int(re.findall(r'\d+', x)[0]))
        grayscale_img_paths = sorted(glob.glob(os.path.join(grayscale_img_dir, '*.png')), key=lambda x: int(re.findall(r'\d+', x)[0]))
        mask_img_paths = sorted(glob.glob(os.path.join(mask_img_dir, '*.png')), key=lambda x: int(re.findall(r'\d+', x)[0]))

        assert len(color_img_paths) == len(grayscale_img_paths) == len(mask_img_paths)

        for i in range(len(color_img_paths)):
            color_img = cv2.imread(color_img_paths[i])
            grayscale_img = cv2.imread(grayscale_img_paths[i], cv2.IMREAD_GRAYSCALE)
            mask_img = cv2.imread(mask_img_paths[i], cv2.IMREAD_GRAYSCALE)
            combined_img = combine_images(color_img, grayscale_img, mask_img)
            output_img_name = f'combined_{i}.png'
            output_img_path = os.path.join(output_img_dir, output_img_name)
            cv2.imwrite(output_img_path, combined_img)
            print(f'Saved combined image: {output_img_path}')

        image_directory = output_img_dir
        filenames = os.listdir(image_directory)
        sorted_filenames = sorted(filenames, key=lambda x: int(re.findall(r'\d+', x)[0]))

        num_images = len(sorted_filenames)
        num_cols = 10
        num_rows = math.ceil(num_images / num_cols)

        fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(20, num_rows * 2))

        for ax_row, row_index in zip(axes, range(num_rows)):
            for ax, filename, intensity, area, pred in zip(ax_row, sorted_filenames[row_index * num_cols:], self.result[row_index * num_cols:], self.areas[row_index * num_cols:], self.predictions[row_index * num_cols:]):
                file_path = os.path.join(image_directory, filename)
                image = cv2.imread(file_path, cv2.IMREAD_COLOR)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                ax.imshow(image)
                title_l = f" Area: {int(area)} {pred}"
                wrapped_title = "\n".join(wrap(title_l, width=20))
                ax.set_title(wrapped_title)
                ax.axis('off')

        for i in range(num_images, num_rows * num_cols):
            axes.flatten()[i].axis('off')

        plt.tight_layout()
        plt.show()

        self.display_results(self.canvas_region_segmented, self.image_refs_region_segmented, output_img_dir, self.result, self.areas, self.predictions)

    class SafeImage:
        def __init__(self, pil_image):
            self.pil_image = pil_image
            self.tk_image = ImageTk.PhotoImage(pil_image)
            self._tk = self.tk_image.tk
            self._name = self.tk_image.__str__()

        def __del__(self):
            if hasattr(self, '_tk') and hasattr(self, '_name'):
                try:
                    self._tk.call('image', 'delete', self._name)
                except RuntimeError:
                    # This error is expected when the main loop is not running
                    pass

        def get(self):
            return self.tk_image

    class RedirectText:
        def __init__(self, text_widget):
            self.output = text_widget

        def write(self, string):
            # List of keywords or phrases to filter out
            filter_out_keywords = [
                "TypeError: DicomConverterApp.display_segmented_results() missing 1 required positional argument",
                "RuntimeError: main thread is not in main loop",
                "Exception ignored in",
                "Clipping input data to the valid range for imshow with RGB data",
                "<function Image.__del__",
                "<function Variable.__del__",
                "Traceback (most recent call last):",
                "Could not load dynamic library 'cudart64_110.dll'",
                "Ignore above cudart dlerror if you do not have a GPU set up on your machine.",
                "pygame 2.5.2 (SDL 2.28.3, Python 3.10.7)",
                "Hello from the pygame community.",
                "AttributeError",
                "<function DicomConverterApp.safe_variable_del at 0x000001C902DA9510>",
                "RuntimeError",
                "File",
                "self.tk.call",
                "<function",
                "self._tk.",
                ": '_tkinter",
                ": main thread",
                "# Override the",
                ": '_tkinter.tkapp",
                ": '_tkinter.tkapp' object has no attribute 'after'",
                ": main thread is not in main loop",
                "    try:",
                "main thread is not in main loop",
                "object has no attribute",
                ":"




            ]

                # Check if the string contains any of the filter out keywords or is empty/whitespace
            if any(keyword in string for keyword in filter_out_keywords) or string.strip() == "":
                return

            self.output.insert(tk.END, string)
            self.output.see(tk.END)

            # Check if the string contains any of the filter out keywords
            if any(keyword in string for keyword in filter_out_keywords):
                return

            self.output.insert(tk.END, string)
            self.output.see(tk.END)

        def flush(self):
            pass

    # Override the __del__ method of the tkinter Variable class
    def safe_variable_del(self):
        try:
            if hasattr(self, '_tk') and hasattr(self, '_name'):
                self._tk.after(0, self._tk.call, 'unset', self._name)
        except RuntimeError:
            pass

    tk.Variable.__del__ = safe_variable_del

    # Ensure that all calls to tkinter methods that manipulate the GUI happen in the main thread
    def safe_tk_call(func):
        def wrapper(*args, **kwargs):
            try:
                func(*args, **kwargs)
            except RuntimeError as e:
                if 'main thread is not in main loop' in str(e):
                    return
                else:
                    raise
        return wrapper

    # Override the __del__ method of the Image class
    from PIL import Image as PILImage

    @safe_tk_call
    def safe_image_del(self):
        try:
            if hasattr(self, 'tk') and hasattr(self, 'name'):
                self.tk.call('image', 'delete', self.name)
        except RuntimeError:
            pass

    PILImage.__del__ = safe_image_del


if __name__ == "__main__":
    root = tk.Tk()
    app = DicomConverterApp(root)
    root.mainloop()
