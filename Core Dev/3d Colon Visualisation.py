import tkinter as tk
from tkinter import filedialog, messagebox
import nibabel as nib
import pyvista as pv
from pyvista import examples

# Global variable for storing the segmentation
segmentation = None

# Function to handle loading NIfTI file for segmentation
def load_nifti_file():
    global segmentation

    nifti_file = filedialog.askopenfilename(title='Select NIfTI File', filetypes=[("NIfTI files", "*.nii;*.nii.gz")])
    if nifti_file:
        try:
            nifti_data = nib.load(nifti_file)
            segmentation = nifti_data.get_fdata()

            # Visualize the segmentation using PyVista
            volume = pv.wrap(segmentation)
            contours = volume.contour([0.3])
            smoothed_contours = contours.smooth(n_iter=30, relaxation_factor=0.3)
            plotter = pv.Plotter()
            plotter.add_mesh(smoothed_contours, color="pink", opacity=1.0)
            plotter.show()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load NIfTI file: {str(e)}")

# Create main application window
root = tk.Tk()
root.title("NIfTI Viewer")

# Create and place load button for NIfTI file
load_nifti_button = tk.Button(root, text="Load NIfTI File", command=load_nifti_file)
load_nifti_button.pack(pady=20)

# Run the GUI main loop
root.mainloop()
