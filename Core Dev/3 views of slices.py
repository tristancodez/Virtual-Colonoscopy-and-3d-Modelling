import os
import pydicom
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk

# Global variables for storing data
selected_series = None
volume = None
axial_slider = None
coronal_slider = None
sagittal_slider = None

# Function to load and group selected DICOM files by SeriesInstanceUID
def load_and_group_dicom_files(file_paths):
    grouped_files = {}
    for file_path in file_paths:
        ds = pydicom.dcmread(file_path)
        series_instance_uid = ds.SeriesInstanceUID
        if series_instance_uid not in grouped_files:
            grouped_files[series_instance_uid] = []
        grouped_files[series_instance_uid].append((ds, ds.pixel_array))
    
    # Sort slices within each group based on the SliceLocation attribute
    for series_instance_uid in grouped_files:
        grouped_files[series_instance_uid].sort(key=lambda x: x[0].SliceLocation)
    
    return grouped_files

# Function to stack slices into a 3D volume
def stack_slices(slices):
    pixel_arrays = [s[1] for s in slices]
    volume = np.stack(pixel_arrays, axis=2)
    return volume

# Function to convert a slice to Pillow format
def dicom_to_pillow(pixel_array):
    # Rescale to 0-255
    img = ((pixel_array - np.min(pixel_array)) / (np.max(pixel_array) - np.min(pixel_array)) * 255).astype(np.uint8)
    # Convert to Pillow Image
    img = Image.fromarray(img)
    return img

# Function to display axial, coronal, and sagittal views with interactive adjustments
def display_views(event=None):  # Accept an event argument
    global volume, axial_slider, coronal_slider, sagittal_slider

    if volume is None:
        print("Error: volume is None")
        return

    axial_index = axial_slider.get()
    coronal_index = coronal_slider.get()
    sagittal_index = sagittal_slider.get()

    num_slices_axial = volume.shape[2]
    num_slices_coronal = volume.shape[1]
    num_slices_sagittal = volume.shape[0]

    # Axial view
    axial_slice = volume[:, :, axial_index]
    axial_img = dicom_to_pillow(axial_slice)
    axial_img = axial_img.resize((400, 400))  # Resize to a larger size
    axial_photo = ImageTk.PhotoImage(axial_img)
    axial_canvas.create_image(0, 0, anchor=tk.NW, image=axial_photo)
    axial_canvas.image = axial_photo
    axial_canvas.config(scrollregion=axial_canvas.bbox(tk.ALL))

    # Coronal view
    coronal_slice = volume[:, coronal_index, :]
    coronal_img = dicom_to_pillow(coronal_slice)
    coronal_img = coronal_img.rotate(90, expand=True)
    coronal_img = coronal_img.resize((400, 400))  # Resize to a larger size
    coronal_photo = ImageTk.PhotoImage(coronal_img)
    coronal_canvas.create_image(0, 0, anchor=tk.NW, image=coronal_photo)
    coronal_canvas.image = coronal_photo
    coronal_canvas.config(scrollregion=coronal_canvas.bbox(tk.ALL))

    # Sagittal view
    sagittal_slice = volume[sagittal_index, :, :]
    sagittal_img = dicom_to_pillow(sagittal_slice)
    sagittal_img = sagittal_img.rotate(90, expand=True)
    sagittal_img = sagittal_img.transpose(Image.FLIP_LEFT_RIGHT)  # Horizontal flip
    sagittal_img = sagittal_img.resize((400, 400))  # Resize to a larger size
    sagittal_photo = ImageTk.PhotoImage(sagittal_img)
    sagittal_canvas.create_image(0, 0, anchor=tk.NW, image=sagittal_photo)
    sagittal_canvas.image = sagittal_photo
    sagittal_canvas.config(scrollregion=sagittal_canvas.bbox(tk.ALL))

# Function to handle loading DICOM files
def load_dicom_files():
    global selected_series, volume, axial_slider, coronal_slider, sagittal_slider

    file_paths = filedialog.askopenfilenames(title='Select DICOM Files', filetypes=[("DICOM files", "*.dcm")])
    if file_paths:
        try:
            grouped_files = load_and_group_dicom_files(file_paths)
            series_instance_uids = list(grouped_files.keys())

            if len(series_instance_uids) == 1:
                # Only one series, no need to ask the user
                selected_series = series_instance_uids[0]
                volume = stack_slices(grouped_files[selected_series])
                create_sliders()
            else:
                # Create a new window for series selection
                series_selection_window = tk.Toplevel(root)
                series_selection_window.title("Series Selection")

                # Create a dropdown menu for series selection
                series_var = tk.StringVar(series_selection_window)
                series_var.set(series_instance_uids[0])  # Set the default value

                dropdown = ttk.Combobox(series_selection_window, textvariable=series_var, values=series_instance_uids)
                dropdown.pack(padx=10, pady=10)

                def on_select():
                    global selected_series, volume
                    selected_series = series_var.get()
                    volume = stack_slices(grouped_files[selected_series])
                    series_selection_window.destroy()
                    create_sliders()

                select_button = tk.Button(series_selection_window, text="Select", command=on_select)
                select_button.pack(pady=10)

                series_selection_window.transient(root)
                series_selection_window.grab_set()
                root.wait_window(series_selection_window)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load DICOM files: {str(e)}")

# Function to create sliders for axial, coronal, and sagittal views
def create_sliders():
    global axial_slider, coronal_slider, sagittal_slider

    num_slices_axial = volume.shape[2]
    num_slices_coronal = volume.shape[1]
    num_slices_sagittal = volume.shape[0]

    # Remove existing sliders if any
    if axial_slider:
        axial_slider.destroy()
    if coronal_slider:
        coronal_slider.destroy()
    if sagittal_slider:
        sagittal_slider.destroy()

    # Sliders for axial, coronal, and sagittal views
    axial_slider = tk.Scale(frame_sliders, length=400, from_=0, to=num_slices_axial - 1, orient=tk.HORIZONTAL, command=display_views)
    axial_slider.pack(side=tk.LEFT, fill=tk.X, padx=10, pady=10)
    coronal_slider = tk.Scale(frame_sliders, length=400, from_=0, to=num_slices_coronal - 1, orient=tk.HORIZONTAL, command=display_views)
    coronal_slider.pack(side=tk.LEFT, fill=tk.X, padx=10, pady=10)
    sagittal_slider = tk.Scale(frame_sliders, length=400, from_=0, to=num_slices_sagittal - 1, orient=tk.HORIZONTAL, command=display_views)
    sagittal_slider.pack(side=tk.LEFT, fill=tk.X, padx=10, pady=10)

    # Display initial images
    display_views()

# Create main application window
root = tk.Tk()
root.title("DICOM Viewer")

# Create a PanedWindow for organizing the views
paned_window = tk.PanedWindow(orient=tk.HORIZONTAL)
paned_window.pack(fill=tk.BOTH, expand=True)

# Frame for Axial, Coronal, and Sagittal views
frame_views = tk.Frame(paned_window)
frame_views.pack(fill=tk.BOTH, expand=True)

# Canvas for Axial view
axial_canvas = tk.Canvas(frame_views, width=400, height=400)
axial_canvas.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.BOTH, expand=True)
axial_scrollbar = tk.Scrollbar(frame_views, orient=tk.HORIZONTAL, command=axial_canvas.xview)
axial_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
axial_canvas.configure(xscrollcommand=axial_scrollbar.set)

# Canvas for Coronal view
coronal_canvas = tk.Canvas(frame_views, width=400, height=400)
coronal_canvas.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.BOTH, expand=True)
coronal_scrollbar = tk.Scrollbar(frame_views, orient=tk.HORIZONTAL, command=coronal_canvas.xview)
coronal_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
coronal_canvas.configure(xscrollcommand=coronal_scrollbar.set)

# Canvas for Sagittal view
sagittal_canvas = tk.Canvas(frame_views, width=400, height=400)
sagittal_canvas.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.BOTH, expand=True)
sagittal_scrollbar = tk.Scrollbar(frame_views, orient=tk.HORIZONTAL, command=sagittal_canvas.xview)
sagittal_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
sagittal_canvas.configure(xscrollcommand=sagittal_scrollbar.set)

# Frame for sliders
frame_sliders = tk.Frame(paned_window)
frame_sliders.pack(fill=tk.Y, expand=False)

# Load button for DICOM files
load_dicom_button = tk.Button(frame_sliders, text="Load DICOM Files", command=load_dicom_files)
load_dicom_button.pack(pady=10)

# Run the GUI main loop
root.mainloop()
