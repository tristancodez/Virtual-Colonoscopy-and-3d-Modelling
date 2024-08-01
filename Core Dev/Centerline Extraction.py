import vtk
import vmtk.vmtkscripts as vmtk
import pyvista as pv

def extract_network_and_center_curves(nifti_file, output_network_file, output_centerline_file, reduction_factor=0.5, label_value=1, smoothing_iterations=20, relaxation_factor=0.1, verbose=False):
    try:
        if verbose:
            print(f"Reading NIFTI file: {nifti_file}")
        reader = vtk.vtkNIFTIImageReader()
        reader.SetFileName(nifti_file)
        reader.Update()
        
        if verbose:
            print("Converting the image to a VTK PolyData surface using Marching Cubes")
        marchingCubes = vtk.vtkMarchingCubes()
        marchingCubes.SetInputConnection(reader.GetOutputPort())
        marchingCubes.SetValue(0, label_value)
        marchingCubes.Update()
        
        surface = marchingCubes.GetOutput()

        if verbose:
            print("Smoothing the surface using vtkSmoothPolyDataFilter")
        smoother = vtk.vtkSmoothPolyDataFilter()
        smoother.SetInputData(surface)
        smoother.SetNumberOfIterations(smoothing_iterations)
        smoother.SetRelaxationFactor(relaxation_factor)
        smoother.FeatureEdgeSmoothingOff()
        smoother.BoundarySmoothingOn()
        smoother.Update()

        smoothed_surface = smoother.GetOutput()
        
        if verbose:
            print(f"Decimating the surface with a reduction factor of {reduction_factor}")
        decimate = vtk.vtkDecimatePro()
        decimate.SetInputData(smoothed_surface)
        decimate.SetTargetReduction(reduction_factor)
        decimate.PreserveTopologyOn()
        decimate.Update()
        
        decimated_surface = decimate.GetOutput()
        
        if verbose:
            print("Visualizing the decimated surface for verification")
        visualize_surface(decimated_surface)
        
        if verbose:
            print("Extracting network curves using vmtk")
        network_filter = vmtk.vmtkNetworkExtraction()
        network_filter.Surface = decimated_surface
        network_filter.Execute()
        
        network_curves = network_filter.Network
        
        if verbose:
            print(f"Writing network curves to the output file: {output_network_file}")
        writer_network = vtk.vtkXMLPolyDataWriter()
        writer_network.SetFileName(output_network_file)
        writer_network.SetInputData(network_curves)
        writer_network.Write()

        if verbose:
            print("Extracting centerline using vmtk")
        centerline_filter = vmtk.vmtkCenterlines()
        centerline_filter.Surface = decimated_surface
        centerline_filter.Execute()
        
        centerlines = centerline_filter.Centerlines
        
        if verbose:
            print(f"Writing centerline to the output file: {output_centerline_file}")
        writer_centerline = vtk.vtkXMLPolyDataWriter()
        writer_centerline.SetFileName(output_centerline_file)
        writer_centerline.SetInputData(centerlines)
        writer_centerline.Write()

        if verbose:
            print("Centerline extraction and writing completed successfully")
    except Exception as e:
        print(f"An error occurred: {e}")

def visualize_surface(surface):
    """Utility function to visualize the surface for debugging."""
    pv_surface = pv.wrap(surface)
    plotter = pv.Plotter()
    plotter.add_mesh(pv_surface, color='lightblue', show_edges=True)
    plotter.show()
 
# def visualize_curves(curves_file):
#     """Utility function to visualize the curves in PyVista."""
#     curves = pv.read(curves_file)
#     plotter = pv.Plotter()
#     plotter.add_mesh(curves, color='red', line_width=2)
#     plotter.camera_position = 'xy'
#     plotter.show()

# Usage example
nifti_file = "D:\\dicom\\combined_segmentation.nii.gz"
output_network_file = 'network_curves5.vtp'
output_centerline_file = 'centerline5.vtp'
extract_network_and_center_curves(nifti_file, output_network_file, output_centerline_file, reduction_factor=0.3, label_value=1, smoothing_iterations=20, relaxation_factor=0.1, verbose=True)

# Visualize the resulting network curves and centerline
# visualize_curves(output_network_file)
# visualize_curves(output_centerline_file)
