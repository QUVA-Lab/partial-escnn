import vtk
from vtkmodules.util import numpy_support
import numpy as np

# Create a 4D array (4, 28, 28, 28)
array_4d = np.random.rand(28, 28, 28, 4)

# Convert the array to RGBA, assuming R, G, B are the first three channels and A is the fourth
rgba = np.zeros((28, 28, 28, 4), dtype=np.uint8)
rgba[..., :3] = (array_4d[..., :3] * 255).astype(np.uint8)  # R, G, B channels
rgba[..., 3] = 125  # Alpha channel

# Create VTK array from the numpy array
vtk_array = numpy_support.numpy_to_vtk(
    rgba.ravel(), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR
)

# Create VTK image data object
image_data = vtk.vtkImageData()
image_data.SetDimensions(28, 28, 28)
image_data.GetPointData().SetScalars(vtk_array)

# Create color transfer function
color_func = vtk.vtkColorTransferFunction()
color_func.AddRGBPoint(0, 0.0, 0.0, 0.0)
color_func.AddRGBPoint(255, 1.0, 1.0, 1.0)

# Create opacity transfer function
alpha_func = vtk.vtkPiecewiseFunction()
alpha_func.AddPoint(0, 0.0)
alpha_func.AddPoint(255, 0.5)

# Create volume property
volume_property = vtk.vtkVolumeProperty()
volume_property.SetColor(color_func)
volume_property.SetScalarOpacity(alpha_func)

# Create volume mapper
mapper = vtk.vtkSmartVolumeMapper()
mapper.SetInputData(image_data)

# Create volume actor
volume = vtk.vtkVolume()
volume.SetMapper(mapper)
volume.SetProperty(volume_property)

# Create renderer
renderer = vtk.vtkRenderer()

# Create render window
render_window = vtk.vtkRenderWindow()
render_window.AddRenderer(renderer)

# Create render window interactor
render_window_interactor = vtk.vtkRenderWindowInteractor()
render_window_interactor.SetRenderWindow(render_window)

# Add volume to renderer
renderer.AddVolume(volume)

# Set background color
renderer.SetBackground(0.0, 0.0, 0.0)

# Reset the camera to show the full object
renderer.ResetCamera()

# Start the interaction
render_window.Render()
render_window_interactor.Start()
