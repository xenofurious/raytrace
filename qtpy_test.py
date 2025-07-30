import sys
# Setting the Qt bindings for QtPy
from qtpy import QtWidgets
import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor, MainWindow
import pandas as pd
import os
os.environ["QT_API"] = "pyqt5"
pl = pv.Plotter()



raytrace_data = pd.read_csv('generated_ray_data.csv')
#setting up axes

#table = ax3.table((2, 4), 2, 2)
row_num = raytrace_data.shape[0]
col_num = raytrace_data.shape[1]
class MyMainWindow(MainWindow):

    def __init__(self, parent=None, show=True):
        QtWidgets.QMainWindow.__init__(self, parent)

        # create the frame
        self.frame = QtWidgets.QFrame()
        vlayout = QtWidgets.QVBoxLayout()

        # add the pyvista interactor object
        self.plotter = QtInteractor(self.frame, shape=(2, 2))
        vlayout.addWidget(self.plotter.interactor)
        self.signal_close.connect(self.plotter.close)

        self.frame.setLayout(vlayout)
        self.setCentralWidget(self.frame)



        # simple menu to demo functions
        mainMenu = self.menuBar()
        fileMenu = mainMenu.addMenu('File')
        exitButton = QtWidgets.QAction('Exit', self)
        exitButton.setShortcut('Ctrl+Q')
        exitButton.triggered.connect(self.close)
        fileMenu.addAction(exitButton)

        # allow adding a sphere
        meshMenu = mainMenu.addMenu('Mesh')
        self.add_sphere_action = QtWidgets.QAction('Add Sphere', self)
        self.add_sphere_action.triggered.connect(self.add_sphere)
        meshMenu.addAction(self.add_sphere_action)

        # allow adding a cube
        self.add_cube_action = QtWidgets.QAction('Add Cube', self)
        self.add_cube_action.triggered.connect(self.add_cube)
        meshMenu.addAction(self.add_cube_action)

        # add sabrina
        self.add_sabrina()

        # add pyvista plot

        self.plot_dataframe(raytrace_data, 1)
        self.add_overlay()
        if show:
            self.show()


    def add_sphere(self):
        """ add a sphere to the pyqt frame """
        self.plotter.subplot(1, 1)
        sphere = pv.Sphere()
        self.plotter.add_mesh(sphere, show_edges=True)
        self.plotter.reset_camera()

    def add_cube(self):
        """ add a cube to the pyqt frame """
        self.plotter.subplot(0, 1)
        cube = pv.Cube()
        self.plotter.add_mesh(cube, show_edges=True)
        self.plotter.reset_camera()

    def add_sabrina(self):
        """ add sabrina carpenter to the pyqt frame"""
        self.plotter.subplot(1, 0)
        sabrina = pv.read("sabrina.obj")
        self.plotter.add_mesh(sabrina)
        self.plotter.reset_camera()

    def plot_dataframe(self, dataframe, id):
        self.plotter.subplot(1, 0)
        start_point = np.array(parse_coord_data(dataframe['start_point'].iloc[0]))
        pl.add_points(start_point, point_size=20.0, color='red')

        points = []
        lines = []

        for index, row in dataframe.iterrows():
            try:
                coord_index = dataframe.columns.get_loc('interaction_type_1')
                previous_point = start_point
                point_colour = 'black'
                while coord_index < col_num and pd.isna(row.iloc[coord_index]) == False:
                    if type(row.iloc[coord_index]) != int:
                        current_point = np.array(parse_coord_data(row.iloc[coord_index]))

                        # pl.add_points(current_point, render_points_as_spheres=True, point_size=10.0, color=point_colour)
                        points.append(current_point)
                        my_line = pv.Line(previous_point, current_point)
                        lines.append(my_line)
                        # pl.add_mesh(my_line, color='black')
                        previous_point = current_point
                    coord_index += 1
                end_point = np.array(parse_coord_data(row['end_point']))
                self.plotter.add_points(end_point, point_size=15.0, color='black')
                end_line = pv.Line(previous_point, end_point)
                lines.append(end_line)
                # pl.add_mesh(end_line)
            except:
                print("a row was omitted for this")
        ## TEMPORARY DATA!! ##
        # essentially the reason why this is here is because of ui conflicts.
        # i don't know whether to have the points represent the interaction type or the ray id yet.
        # the code before (which gets overwritten) represents the former
        # the code here (which overwrites the former) represents the latter.
        # i'll have to decide later.
        match id:
            case 1:
                point_colour = 'blue'
            case 2:
                point_colour = 'lightgreen'
            case 3:
                point_colour = 'yellow'
            case other:
                point_colour = 'purple'
        ## END OF TEMPORARY CODE ##

        self.plotter.add_points(np.array(points), render_points_as_spheres=True, point_size=10.0, color=point_colour)
        # merging the lines into one mesh. this is for performance uplift
        combined = lines[0]
        for line in lines[1:]:
            combined = combined.merge(line)
        self.plotter.add_mesh(combined, color='black', line_width=2)
        self.plotter.reset_camera()

    def add_overlay(self):
        model_name = input(
            "Enter the model you want to overlay over the simulation. Leave blank if you don't want to: ")
        if model_name != '':
            model = pv.read(model_name)
            actor = self.plotter.add_mesh(
                model,
                color='lightblue',
                opacity=0.4,
                specular=1.0,
                smooth_shading=True
            )


def parse_coord_data(coords):
    coords = coords.strip(')(').split(')(')
    coords = [float(i) for i in coords]
    return coords

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = MyMainWindow()
    sys.exit(app.exec_())