import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QHBoxLayout, QWidget
from pyvistaqt import QtInteractor
import pyvista as pv
import vtk
import numpy as np
from scipy.ndimage import gaussian_filter1d
import visualize2

class VisualizationWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Flythrough Standard")

        # Set up the main widget and layout
        main_widget = QWidget()
        layout = QHBoxLayout(main_widget)
        self.setCentralWidget(main_widget)

        # Set up the left visualization
        self.left_view = QtInteractor(self)
        layout.addWidget(self.left_view.interactor)

        # Set up the right visualization
        self.right_view = QtInteractor(self)
        layout.addWidget(self.right_view.interactor)

        self.left_view.set_background('black')

        # Load and display the first model
        mesh1 = visualize2.load_nifti_as_surface('new seg.nii.gz', 1.0)
        centerline = pv.read('centerline.vtp')

        start_point = centerline.points[0]

        self.sphere = pv.Sphere(center=start_point, radius=4)
        
       
        self.left_view.add_mesh(mesh1, color=(240, 125, 100),opacity=0.5,pickable = False)
        # self.left_view.add_mesh(centerline,color = 'green',line_width=5,pickable = False)
        self.sphere_actor = self.left_view.add_mesh(self.sphere, pickable=False, color='green')
        # Load and display the second model and centerline
        mesh2 = visualize2.load_nifti_as_surface('new seg.nii.gz', 0.5)
        
        self.right_view.add_mesh(mesh2, color=(240, 125, 100), opacity=1.0, specular=1.0,specular_power=100, smooth_shading=False) 
        self.right_view.add_mesh(centerline, color='green', line_width=5)

        # Interactive flythrough
        self.flythrough(centerline)

        # Start the renderers
        self.left_view.show()
        self.right_view.show()

    def create_camera_path(self, centerline_surface, num_points=2000):
        smooth_points = centerline_surface.points
        tangents, normals, binormals = visualize2.compute_frenet_serret(smooth_points)

        vtk_points = vtk.vtkPoints()
        for point in centerline_surface.points:
            vtk_points.InsertNextPoint(point)

        spline = vtk.vtkParametricSpline()
        spline.SetPoints(vtk_points)

        function_source = vtk.vtkParametricFunctionSource()
        function_source.SetParametricFunction(spline)
        function_source.SetUResolution(num_points - 1)
        function_source.Update()

        spline_points = function_source.GetOutput().GetPoints()
        points = np.array([spline_points.GetPoint(i) for i in range(spline_points.GetNumberOfPoints())])
        return points

    def flythrough(self, centerline):
        resampled_points = visualize2.resample_centerline(centerline.points, num_points=150)
        smoothed_points = visualize2.smooth_centerline_points(resampled_points, sigma=2)
        camera_positions = self.create_camera_path(pv.PolyData(smoothed_points))

        camera_index = [0]

        fov = 140

        def move_camera_w():

            camera_index[0] = min(camera_index[0] + 1, len(camera_positions) - 1)
            update_camera()

        def move_camera_s():

            camera_index[0] = max(camera_index[0] - 1, 0)
            update_camera()

        def update_camera():
            position = camera_positions[camera_index[0]]
            next_position = camera_positions[min(camera_index[0] + 1, len(camera_positions) - 1)]

            print(f"Moving camera to position: {position}, focal point: {next_position}")

            self.right_view.camera_position = [position, next_position, (0, 1, 0)]  # Fixed up vector
            self.right_view.camera.focal_point = next_position
            self.right_view.camera.view_angle = fov
            self.right_view.camera.view_up = (0, 1, 0)  # Fixed up vector

            new_sphere = pv.Sphere(center=position, radius=4)
            self.left_view.remove_actor(self.sphere_actor)
            self.sphere_actor = self.left_view.add_mesh(new_sphere, pickable=False, color='yellow')


            self.left_view.update()
            self.left_view.render()

            self.right_view.update()
            self.right_view.render()


        self.right_view.add_key_event('c', move_camera_w)
        self.right_view.add_key_event('v', move_camera_s)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VisualizationWindow()
    window.show()
    sys.exit(app.exec_())

    
