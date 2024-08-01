import pyvista as pv
import vtk
 
def load_nifti_as_surface(nifti_file, contour_value):
    """Load NIfTI file and convert to a surface using Marching Cubes."""
    reader = vtk.vtkNIFTIImageReader()
    reader.SetFileName(nifti_file)
    reader.Update()
 
    # Create a surface using the Marching Cubes algorithm
    marching_cubes = vtk.vtkMarchingCubes()
    marching_cubes.SetInputConnection(reader.GetOutputPort())
    marching_cubes.SetValue(0, contour_value)  # Set the contour value
    marching_cubes.Update()
 
    output = marching_cubes.GetOutput()
    if output.GetNumberOfPoints() == 0:
        print(f"Warning: No surface was created with contour value {contour_value}.")
        return None
 
    return pv.wrap(output)
 
def main():
    model_file = "new seg.nii 1.gz"
    centerline_file = "centerline.vtp"
 
    # Try different contour values within the range
    possible_contour_values = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for contour_value in possible_contour_values:
        print(f"Trying contour value: {contour_value}")
        model_surface = load_nifti_as_surface(model_file, contour_value)
        if model_surface is not None:
            break
 
    if model_surface is None:
        print("Failed to create a surface with any of the contour values.")
        return
 
    centerline_surface = pv.read(centerline_file)
 
    plotter = pv.Plotter()
    plotter.add_mesh(model_surface, color='gray', opacity=.5)
    plotter.add_mesh(centerline_surface, color='red', line_width=5)
    plotter.camera_position = 'iso'
    plotter.show()
 
if __name__ == "__main__":
    main()