import numpy as np

from vispy import app, scene, io


def get_3d_canvas(vol: np.ndarray):
    # Prepare canvas
    canvas = scene.SceneCanvas(keys='interactive', size=(800, 600), show=True)

    # Set up a viewbox to display the image with interactive pan/zoom
    view = canvas.central_widget.add_view()

    # Create the visuals
    volume = scene.visuals.Volume(
        vol,
        parent=view.scene,
        threshold=0.225,
        texture_format="auto",  # OpenGL
    )

    # Create and set the camera
    fov = 60.
    cam = scene.cameras.TurntableCamera(
        parent=view.scene,
        fov=fov,
        name='Turntable'
    )
    view.camera = cam

    # # Implement key presses
    # @canvas.events.key_press.connect
    # def on_key_press(event):
        # if event.text in 'xyzo':
            # ...

    return canvas

if __name__ == '__main__':

    vol = np.load(io.load_data_file('volume/stent.npz'))['arr_0']
    get_3d_canvas(vol)
    app.run()
