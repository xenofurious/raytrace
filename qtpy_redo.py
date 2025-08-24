import sys

from PyQt6.QtWidgets import QHBoxLayout, QSpinBox
# Setting the Qt bindings for QtPy
from qtpy import QtWidgets
import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor, MainWindow
from qtpy.QtWidgets import (QDockWidget, QWidget, QPushButton, QCheckBox, QFileDialog, QInputDialog, QLineEdit, QDialog, QInputDialog, QSpinBox, QLabel,
                            QVBoxLayout, QHBoxLayout, QGridLayout)
from qtpy.QtCore import Qt, Signal
import pandas as pd
import os
import xml.etree.ElementTree as ET

# importing scripts (look at me oooh)
import simulate
import parse_bsm

os.environ["QT_API"] = "pyqt6"
pl = pv.Plotter()


# initialising folders
if not os.path.exists("generated_files"):
    os.mkdir("generated_files")
os.chdir("generated_files")
if not os.path.exists("csv"): os.mkdir("csv")
if not os.path.exists("obj"): os.mkdir("obj")
os.chdir("..")


# universal global variables
raytrace_data = pd.DataFrame()
unique_ids = []


class MyMainWindow(MainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("simulation window")
        self.resize(1000, 700)

        # create the frame
        self.frame = QtWidgets.QFrame()
        vlayout = QtWidgets.QVBoxLayout()

        # add the pyvista interactor object
        self.plotter = QtInteractor(self.frame, shape=(2, 2))
        vlayout.addWidget(self.plotter.interactor)
        self.signal_close.connect(self.plotter.close)
        self.frame.setLayout(vlayout)
        self.setCentralWidget(self.frame)

        # add some properties
        self.added_actors = []
        self.plots = []


        # set up sidebar
        self.sidebar = MySidebar()
        self.addDockWidget(Qt.RightDockWidgetArea, self.sidebar)
        self.sidebar.simulate.connect(self.simulate)
        self.sidebar.add_obj.connect(self.add_obj)
        self.sidebar.remove_objs.connect(self.remove_objs)
        self.sidebar.add_cube.connect(self.add_cube)
        self.sidebar.add_sphere.connect(self.add_sphere)
        self.sidebar.toggle_plot.connect(self.toggle_plot)
        self.sidebar.parse_bsm.connect(self.parse_bsm)

        # menu options
        mainMenu = self.menuBar()
        meshMenu = mainMenu.addMenu('Options')
        self.add_sphere_action = QtWidgets.QAction('Add Sphere', self)
        self.add_sphere_action.triggered.connect(self.add_sphere)
        meshMenu.addAction(self.add_sphere_action)

        # add sabrina
        self.add_sabrina()
        self.show()

    def simulate(self):
        """add simulation data from a csv file"""
        global raytrace_data
        self.plotter.subplot(0, 0)
        root_projdir = os.getcwd()
        # this script shows that the thing is being called correctly. i just find it funny
        dlg = QDialog(self)
        dlg.setWindowTitle("Hello !")
        dlg.resize(400, 200)
        dlg.exec()

        # okay now for it to call simulate.py and generate a csv file called generated_ray_data.csv
        os.chdir("generated_files/csv")
        simulate.create_csv(id=1, model_used=root_projdir+"/models/cube.obj", no_of_sources=1, start_strength=10000, max_reflections=10)
        raytrace_data = pd.read_csv("generated_ray_data.csv")

        # define the ray tracing data new parameters
        unique_ids = raytrace_data["id"].unique()

        # clear old plot and add new plot
        self.plotter.remove_actor(self.plots)
        plots = []
        print(unique_ids)
        for i in unique_ids:
            subdataframe = raytrace_data[raytrace_data["id"] == i]
            plots.append(self.plot_dataframe(subdataframe, i))
        self.plots = plots

    def add_sphere(self):
        """ add a sphere to the pyqt frame """
        self.plotter.subplot(0, 0)
        sphere = pv.Sphere(radius=0.5)
        self.plotter.add_mesh(sphere,
                              color='lightblue',
                              opacity=0.4,
                              specular=1.0,
                              smooth_shading=True,
                              show_edges=True)
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
        sabrina = pv.read("models/sabrina.obj")
        self.plotter.add_mesh(sabrina)
        self.plotter.reset_camera()

    def toggle_plot(self, plot_index, plots):
        pass

    def add_obj(self):
        """ add a file of your choice to visualise"""
        self.plotter.subplot(0, 0)
        start_dir = os.getcwd()

        file_dialogs, _ = QFileDialog.getOpenFileNames(
            self,
            "Open OBJ File(s)",
            start_dir,
            "OBJ Files (*.obj);;All Files (*)"
        )
        for file_dialog in file_dialogs:
            imported_mesh = pv.read(file_dialog)
            actor = self.plotter.add_mesh(
                imported_mesh,
                color='lightblue',
                opacity=0.4,
                specular=1.0,
                smooth_shading=True
            )
            self.added_actors.append(actor)
        self.plotter.reset_camera()
        print(len(self.added_actors))

    def remove_objs(self):
        """removes all added obj overlays"""
        for actor in self.added_actors:
            self.plotter.remove_actor(actor)

    def plot_dataframe(self, dataframe, colour):
        """ adds the plot"""
        col_num = dataframe.shape[1]
        actors = []

        self.plotter.subplot(0, 0)
        start_point = np.array(parse_coord_data(dataframe['start_point'].iloc[0]))
        actors.append(self.plotter.add_points(start_point, point_size=20.0, color='red'))

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
                        points.append(current_point)
                        my_line = pv.Line(previous_point, current_point)
                        lines.append(my_line)
                        previous_point = current_point
                    coord_index += 1
                end_point = np.array(parse_coord_data(row['end_point']))
                actors.append(self.plotter.add_points(end_point, point_size=15.0, color='black'))
                end_line = pv.Line(previous_point, end_point)
                actors.append(lines.append(end_line))
                # pl.add_mesh(end_line)
            except:
                print("a row was omitted for this")
        ## TEMPORARY DATA!! ##
        # essentially the reason why this is here is because of ui conflicts.
        # i don't know whether to have the points represent the interaction type or the ray id yet.
        # the code before (which gets overwritten) represents the former
        # the code here (which overwrites the former) represents the latter.
        # i'll have to decide later.
        match colour:
            case 1:
                point_colour = 'blue'
            case 2:
                point_colour = 'lightgreen'
            case 3:
                point_colour = 'yellow'
            case other:
                point_colour = 'purple'
        ## END OF TEMPORARY CODE ##

        actors.append(self.plotter.add_points(np.array(points), render_points_as_spheres=True, point_size=8.0,
                                              color=point_colour))
        # merging the lines into one mesh. this is for performance uplift
        combined = lines[0]
        for line in lines[1:]:
            combined = combined.merge(line)
        actors.append(self.plotter.add_mesh(combined, color='black', line_width=2))
        actors.append(self.plotter.reset_camera())
        return actors

    def parse_bsm(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Open BSM File(s)",
            "",
            "BSM file (*.bsm);;All Files (*)"
        )
        if filepath != "":
            tree = ET.parse(filepath)
            root = tree.getroot()
            floor_collection = root[9]

            text, ok = QInputDialog.getText(
                self,
                "Enter Name Of File",
                "Please enter the name of the file / folder",
                QLineEdit.Normal
            )
            parse_bsm.write_floor_collection_to_simple_obj(text + ".obj", floor_collection)
            print("Written to " + text + ".obj!")




class MySidebar(QDockWidget):

    # custom signals
    toggle_plot = Signal(int, list)
    simulate = Signal()
    add_sphere = Signal()
    add_cube = Signal()
    add_obj = Signal()
    remove_objs = Signal()
    parse_bsm = Signal()


    def __init__(self):
        super().__init__("Sidebar")
        sidebar_widget = QWidget()
        sidebar_layout = QVBoxLayout()



        # sidebar objs
        button_simulate = QPushButton("Add Simulation (csv)")
        button_simulate.clicked.connect(self.simulate.emit)
        button_sphere = QPushButton("Add Sphere")
        button_sphere.clicked.connect(self.add_sphere.emit)
        button_cube = QPushButton("Add Cube")
        button_cube.clicked.connect(self.add_cube.emit)
        button_add_model = QPushButton("Add 3D model overlay")
        button_add_model.clicked.connect(lambda: self.add_obj.emit())
        button_clear_models = QPushButton("Clear all 3D overlays")
        button_clear_models.clicked.connect(lambda: self.remove_objs.emit())
        button_parse_bsm = QPushButton("Select .bsm file")
        button_parse_bsm.clicked.connect(self.parse_bsm.emit)

        #sidebar_layout.addWidget(self.checkboxes_parent_widget)
        sidebar_layout.addWidget(button_simulate)
        sidebar_layout.addWidget(button_sphere)
        sidebar_layout.addWidget(button_cube)
        sidebar_layout.addWidget(button_add_model)
        sidebar_layout.addWidget(button_clear_models)
        sidebar_layout.addWidget(button_parse_bsm)
        sidebar_widget.setLayout(sidebar_layout)
        sidebar_layout.addStretch()
        self.setWidget(sidebar_widget)

class Checkboxes_Widget(QWidget):
    def __init__(self):
        super().__init__()


class SimulateDialog1(QDialog):
    def __init__(self):
        super().__init__()
        self.transmitter_no = 0
        self.receiver_no = 0

        self.setWindowTitle("simulation parameters")
        dialog_sim1_layout = QGridLayout()
        transmitter_selection_label = QLabel("Number of transmitters (TX):")
        receiver_selection_label = QLabel("Number of receivers(RX):")
        transmitter_selection_widget = QSpinBox(minimum=1, maximum=20, value=1)
        receiver_selection_widget = QSpinBox(minimum=1, maximum=20, value=1)
        dialog_sim1_layout.addWidget(transmitter_selection_label, 0, 0)
        dialog_sim1_layout.addWidget(receiver_selection_label, 1, 0)
        dialog_sim1_layout.addWidget(transmitter_selection_widget, 0, 1)
        dialog_sim1_layout.addWidget(receiver_selection_widget, 1, 1)


        self.setLayout(dialog_sim1_layout)



class SimulateDialog2(QDialog):
    def __init__(self):
        super().__init__()



def parse_coord_data(coords):
    coords = coords.strip(')(').split(')(')
    coords = [float(i) for i in coords]
    return coords

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = MyMainWindow()
    sys.exit(app.exec_())