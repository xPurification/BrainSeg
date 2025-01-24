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
from tkinter import filedialog, messagebox, ttk, StringVar
import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import sys
import threading
import math
import shutil
from tensorflow.keras.preprocessing import image as q1
from scipy.ndimage import label
from textwrap import wrap

class DicomConverterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Brain Seg")

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

        # Export Results Dropdown Menu
        self.export_options = ["Original PNGs", "Segmented Images", "Intensity Histogram", "Patient Information"]
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

    def display_results(self, canvas, image_refs, image_dir):
        canvas.delete("all")
        image_refs.clear()  # Clear the image reference list

        width = self.frame_results.winfo_width()
        max_columns = max(1, width // 110)
        x_position = 10
        y_position = 10
        column = 0

        for file in os.listdir(image_dir):
            if file.lower().endswith('.png'):
                img_path = os.path.join(image_dir, file)
                img = Image.open(img_path)
                img.thumbnail((100, 100))
                img = ImageTk.PhotoImage(img)
                canvas.create_image(x_position, y_position, anchor=tk.NW, image=img)

                if column < max_columns - 1:
                    column += 1
                    x_position += 110
                else:
                    column = 0
                    x_position = 10
                    y_position += 110

                image_refs.append(img)  # Store a reference to the image
        canvas.config(scrollregion=(0, 0, width, y_position + 110))

    def on_frame_configure(self, event):
        self.display_results(self.canvas_png, self.image_refs_png, self.output_dir)
        self.display_segmented_results(self.canvas_segmented, self.image_refs_segmented, 'segmentation')

    def run_ai(self):
        threading.Thread(target=self.run_ai_thread).start()

    def run_ai_thread(self):
        inputdir = self.dicom_dir
        temp = 'temp'
        self.create_dir(temp)
        self.delete_all_files(temp)

        self.dicom_to_png(inputdir, temp)
        self.find_and_copy_png_files(temp, "PNG_images")

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

        test_x = sorted(glob.glob(os.path.join(test_path, "*")))

        print(len(test_x))
        time_taken = []
        import time
        for x in tqdm(test_x):
            name = x.split("/")[-1]

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

            simplified_p = np.zeros_like(p)
            simplified_save_path = os.path.join(save_path, f"simplified_{name}")
            simplified_success = cv2.imwrite(simplified_save_path, simplified_p)

            save_path_full = os.path.join(save_path, name)
            os.makedirs(os.path.dirname(save_path_full), exist_ok=True)

            success = cv2.imwrite(save_path_full, p)

        test_image = np.zeros((128, 128, 1), dtype=np.uint8)
        test_image_path = os.path.join(save_path, "test_image.png")
        if cv2.imwrite(test_image_path, test_image):
            print("Successfully wrote test image")
        else:
            print("Failed to write test image")

        mask_folder = os.path.join(save_path, "PNG_images")
        self.create_dir(mask_folder)
        output_folder = os.path.join("segmentation")
        self.create_dir(output_folder)
        self.delete_all_files(output_folder)

        self.composite_images(test_path, mask_folder, output_folder)

        image_folder = output_folder  # Change this to your folder path

        # Get a list of all image files in the folder
        image_files = glob.glob(os.path.join(image_folder, '*.*g'))  # This will include jpg, png, etc.

        # Loop through each image file in the folder
        image_files = glob.glob(os.path.join(image_folder, '*.*g'))  # This will include jpg, png, etc.

        # Loop through each image file in the folder
        for image_path in image_files:
            # Read the image
            img = cv2.imread(image_path)

            # Convert the image to grayscale if it is not already
            if img.ndim == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Convert the image to a binary image
            _, bw = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Identify the connected components in the binary image
            labeled_array, num_features = label(bw)

            # Check if the number of islands is greater than 5
            if num_features > 5:
                # Delete the image if it has more than 5 islands
                os.remove(image_path)
                print(f'Deleted {os.path.basename(image_path)}')
                mask_image_path = os.path.join(mask_folder, os.path.basename(image_path))

                # Check if the mask image exists and delete it
                if os.path.exists(mask_image_path):
                    os.remove(mask_image_path)
                    print(f'Deleted mask {os.path.basename(mask_image_path)}')

        print('Processing complete.')

        # Threshold for considering a pixel as black (0-255 scale)
        black_threshold = 100  # Adjust this value as needed

        # Percentage threshold for considering an image mostly black
        percentage_threshold = 95  # Adjust this value as needed

        # Get a list of all image files in the folder
        image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]

        # Loop through each image file in the folder
        for image_file in image_files:
            # Get the full path of the image
            image_path = os.path.join(image_folder, image_file)

            # Read the image
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
        # Iterate over each image in the folder
        for filename in os.listdir(folder_path):
            if filename.endswith('.jpg') or filename.endswith('.png'):  # Check for image file types
                # Load and preprocess the image
                img_path = os.path.join(folder_path, filename)
                img = q1.load_img(img_path, target_size=(128, 128))
                img_array = q1.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)
                img_array = img_array / 255.0

                # Make a prediction
                predictions = model_pred.predict(img_array)

                # Save the filename and prediction to the results list
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
        result = np.where(array2 != 0, array2 / array1, 0)

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

        #self.delete_all_files("PNG_images")
        for key, value in patient_info.items():
            print(f"{key}: {value}")

        image_directory = output_folder
        # Array containing titles corresponding to each image
        # Get all filenames in the directory
        filenames = os.listdir(image_directory)
        # Sort filenames using regular expressions to sort numerically
        sorted_filenames = sorted(filenames, key=lambda x: int(re.findall(r'\d+', x)[0]))

        num_images = len(sorted_filenames)
        num_cols = 10  # Number of columns
        num_rows = math.ceil(num_images / num_cols)  # Calculate the number of rows needed

        fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(20, num_rows * 2))  # Dynamic figsize based on num_rows

        for ax_row, row_index in zip(axes, range(num_rows)):
            for ax, filename, intentsity, area, pred in zip(ax_row, sorted_filenames[row_index * num_cols:], result[row_index * num_cols:], areas[row_index * num_cols:], results[row_index * num_cols:]):
                file_path = os.path.join(image_directory, filename)
                image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                ax.imshow(image, cmap='gray')
                title_l = "INT: " + str(int(intentsity)) + " Area : " + str(int(area)) + str(pred)
                wrapped_title = "\n".join(wrap(title_l, width=20))
                ax.set_title(wrapped_title)  # Set title from the results array
                ax.axis('off')  # Turn off axis

        # Hide empty subplots if any
        for i in range(num_images, num_rows * num_cols):
            axes.flatten()[i].axis('off')

        plt.tight_layout()  # Adjust layout to prevent overlap
        plt.show()
        print(results[0])

        self.display_segmented_results(output_folder, result, areas, results)
        self.plot_histogram(result)

        #self.delete_all_files(save_path)

        volume = self.load_images(self.rename_directory(output_folder))
        segmented_volume = self.segment_brain(volume, 90)
        verts, faces = self.extract_surface(segmented_volume)
        self.plot_3d(verts, faces)
        self.write_obj("output_brain_model.obj", verts, faces)

        #self.delete_all_files(mask_folder)
        #self.delete_all_files(output_folder)
        print("All imports were successful!")

    def display_patient_info(self, patient_info):
        self.text_patient_info.delete('1.0', tk.END)
        for key, value in patient_info.items():
            self.text_patient_info.insert(tk.END, f"{key}: {value}\n")

    def display_segmented_results(self, directory, result, areas, predictions):
        self.canvas_segmented.delete("all")
        self.image_refs_segmented = []

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
        else:
            messagebox.showerror("Error", "Invalid export option selected.")

    def export_patient_info(self, export_dir):
        patient_info_path = os.path.join(export_dir, "patient_info.txt")
        with open(patient_info_path, "w") as f:
            f.write(self.text_patient_info.get("1.0", tk.END))
        messagebox.showinfo("Export Complete", f"Patient information saved to {patient_info_path}")

    def copy_files(self, src_dir, dest_dir):
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
        for file_name in os.listdir(src_dir):
            full_file_name = os.path.join(src_dir, file_name)
            if os.path.isfile(full_file_name):
                shutil.copy(full_file_name, dest_dir)
        messagebox.showinfo("Export Complete", f"Files saved to {dest_dir}")

    def create_dir(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

    def delete_all_files(self, folder_path):
        for root, dirs, files in os.walk(folder_path):
            for file_name in files:
                file_path = os.path.join(root, file_name)
                if os.path.isfile(file_path):
                    os.remove(file_path)

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

    def composite_images(self, image_folder, mask_folder, output_folder):
        image_files = os.listdir(image_folder)

        for image_file in image_files:
            image_path = os.path.join(image_folder, image_file)
            mask_file = image_file.split('.')[0] + '.png'
            mask_path = os.path.join(mask_folder, mask_file)

            image = Image.open(image_path)
            mask = Image.open(mask_path)

            result = Image.composite(image, Image.new('RGB', image.size, (0, 0, 0)), mask)

            output_file = os.path.join(output_folder, image_file)
            self.create_dir(os.path.dirname(output_file))  # Ensure the directory exists
            result.save(output_file)

    # Update the plot_histogram and show_histogram methods
    def plot_histogram(self, data):
        fig, ax = plt.subplots(figsize=(20, 6))
        ax.bar(range(len(data)), data, align='center', edgecolor='black')
        ax.set_xlabel('Slice')
        ax.set_ylabel('Gluecose Intensity')
        ax.set_title('Intensity Of Slices 0 - ' + str(len(data)))
        ax.set_xticks(range(0, len(data), 5))
        ax.grid(False)
        plt.tight_layout()

        # Save the figure to a file
        fig.savefig('histogram.png')

        # Also display the histogram in a popup window
        plt.show()

        # Load the figure into the "Intensity Histogram" tab
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
        files.sort(key=self.natural_sort_key)

        for i, filename in enumerate(files):
            new_name = f"{i}.png"
            os.rename(os.path.join(directory, filename), os.path.join(directory, new_name))

        return directory_path

    def extract_surface(self, volume, level=0.5):
        verts, faces, normals, values = measure.marching_cubes(volume, level=level)
        return verts, faces

    def write_obj(self, filename, verts, faces):
        with open(filename, 'w') as file:
            for v in verts:
                file.write(f"v {v[0]} {v[1]} {v[2]}\n")
            for face in faces:
                file.write("f {} {} {}\n".format(face[0] + 1, face[1] + 1, face[2] + 1))

    def plot_3d(self, verts, faces):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2], cmap='bone', lw=1)
        plt.show()

    def natural_sort_key(self, s):
        return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

    class RedirectText:
        def __init__(self, text_widget):
            self.output = text_widget

        def write(self, string):
            self.output.insert(tk.END, string)
            self.output.see(tk.END)

        def flush(self):
            pass


if __name__ == "__main__":
    root = tk.Tk()
    app = DicomConverterApp(root)
    root.mainloop()
