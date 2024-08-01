
import pyvista as pv
import vtk
import numpy as np
from scipy.ndimage import gaussian_filter1d

def load_nifti_as_surface(nifti_file, contour_value, n_iter=50, relaxation_factor=0.5):
    reader = vtk.vtkNIFTIImageReader()
    reader.SetFileName(nifti_file)
    reader.Update()

    discrete_marching_cubes = vtk.vtkDiscreteMarchingCubes()
    discrete_marching_cubes.SetInputConnection(reader.GetOutputPort())
    discrete_marching_cubes.SetValue(0, contour_value)
    discrete_marching_cubes.Update()

    surface = discrete_marching_cubes.GetOutput()
    if surface.GetNumberOfPoints() == 0:
        print(f"Warning: No surface was created with contour value {contour_value}.")
        return None

    surface = pv.wrap(surface)
    smoothed_surface = surface.smooth(n_iter=n_iter, relaxation_factor=relaxation_factor)
    return smoothed_surface

def smooth_centerline_points(points, sigma=5):
    smoothed_points = gaussian_filter1d(points, sigma=sigma, axis=0)
    return smoothed_points

def resample_centerline(centerline_points, num_points=500):
    length = np.cumsum(np.sqrt(np.sum(np.diff(centerline_points, axis=0) ** 2, axis=1)))
    length = np.insert(length, 0, 0)
    uniform_length = np.linspace(0, length[-1], num_points)
    resampled_points = np.zeros((num_points, centerline_points.shape[1]))
    for dim in range(centerline_points.shape[1]):
        resampled_points[:, dim] = np.interp(uniform_length, length, centerline_points[:, dim])
    return resampled_points

def compute_frenet_serret(points):
    tangents = np.gradient(points, axis=0)
    tangents /= np.linalg.norm(tangents, axis=1)[:, np.newaxis]

    normals = np.gradient(tangents, axis=0)
    normals /= np.linalg.norm(normals, axis=1)[:, np.newaxis]

    binormals = np.cross(tangents, normals)
    binormals /= np.linalg.norm(binormals, axis=1)[:, np.newaxis]

    return tangents, normals, binormals





