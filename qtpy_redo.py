import sys
from pickletools import bytes_types

from PyQt6.QtWidgets import QHBoxLayout, QSpinBox, QDialogButtonBox
# Setting the Qt bindings for QtPy
from qtpy import QtWidgets
import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor, MainWindow
from qtpy.QtWidgets import (QDockWidget, QWidget, QPushButton, QCheckBox, QFileDialog, QLineEdit, QDialog, QInputDialog, QSpinBox, QDoubleSpinBox, QLabel,
                            QVBoxLayout, QHBoxLayout, QGridLayout, QFrame)
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
        self.frame = QFrame()
        vlayout = QVBoxLayout()

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

        dialog1 = SimulateDialog1()
        if dialog1.exec():

            transmitter_no, receiver_no = dialog1.get_values()

            dialog2 = SimulateDialog2(transmitter_no, receiver_no)
            if dialog2.exec():
                # okay now for it to call simulate.py and generate a csv file called generated_ray_data.csv
                transmitter_id_input, transmitter_pos_input_transmitter, transmitter_direction_input, transmitter_no_of_rays_input, transmitter_no_of_reflections_input = dialog2.transmitter_get_values()
                receiver_pos_input, receiver_radius = dialog2.receiver_get_values()
                print("id input = ", transmitter_id_input)
                print("pos input = ", transmitter_pos_input_transmitter)
                print("direction input = ", transmitter_direction_input)
                print("no of rays input = ", transmitter_no_of_rays_input)
                print("no of reflections input = ", transmitter_no_of_reflections_input)
                print("------------------")
                print("receiver pos input = ", receiver_pos_input)
                print("receiver radius = ", receiver_radius)

                os.chdir("generated_files/csv")
                simulate.create_csv(id=1, model_used=root_projdir+"/models/cube.obj", no_of_sources=1, start_strength=10000, max_reflections=10)
                raytrace_data = pd.read_csv("generated_ray_data.csv")

                # define the ray tracing data new parameters
                unique_ids = raytrace_data["id"].unique()

                # clear old plot and add new plot
                self.plotter.remove_actor(self.plots)
                plots = []
                for i in unique_ids:
                    subdataframe = raytrace_data[raytrace_data["id"] == i]
                    plots.append(self.plot_dataframe(subdataframe, i))
                self.plots = plots




            else:
                print("your mother 2")
        else:
            print("your mother")



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

        widget_layout = QVBoxLayout()

        # inputs
        dialog_sim1_widget = QWidget()
        dialog_sim1_layout = QGridLayout()
        transmitter_selection_label = QLabel("Number of transmitters (TX):")
        receiver_selection_label = QLabel("Number of receivers(RX):")
        self.transmitter_selection_widget = QSpinBox(minimum=1, maximum=20, value=1)
        self.receiver_selection_widget = QSpinBox(minimum=1, maximum=20, value=1)
        dialog_sim1_layout.addWidget(transmitter_selection_label, 0, 0)
        dialog_sim1_layout.addWidget(receiver_selection_label, 1, 0)
        dialog_sim1_layout.addWidget(self.transmitter_selection_widget, 0, 1)
        dialog_sim1_layout.addWidget(self.receiver_selection_widget, 1, 1)

        dialog_sim1_widget.setLayout(dialog_sim1_layout)
        widget_layout.addWidget(dialog_sim1_widget)

        # okay and cancel buttons
        button_widget = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_widget.accepted.connect(self.accept)
        button_widget.rejected.connect(self.reject)
        widget_layout.addWidget(button_widget)
        self.setLayout(widget_layout)

    def get_values(self):
        return self.transmitter_selection_widget.value(), self.receiver_selection_widget.value()



class SimulateDialog2(QDialog):
    def __init__(self, transmitter_no, receiver_no):
        super().__init__()
        self.transmitter_no = transmitter_no
        self.receiver_no = receiver_no

        self.setWindowTitle("more simulation parameters")
        #self.model =
        self.receiver_id = []
        self.receiver_start_pos = []
        self.transmission_type = ""
        self.transmission_direction = "random"
        self.receiver_start_strength = []
        self.max_reflection = []


        dialog_layout = QHBoxLayout()
        self.sidebar_thing = SimulateDialog2SidebarThing(transmitter_no, receiver_no, self)
        dialog_layout.addWidget(self.sidebar_thing)

        plotter_preview_window = QtInteractor()
        plotter_preview_window.add_mesh(pv.Sphere())
        dialog_layout.addWidget(plotter_preview_window)

        self.setLayout(dialog_layout)

    def transmitter_get_values(self):
        self.transmitter_id_input, self.transmitter_pos_input, self.transmitter_direction_input, self.transmitter_no_of_rays_input, self.transmitter_no_of_reflections_input = self.sidebar_thing.transmitter_get_values()
        return self.transmitter_id_input, self.transmitter_pos_input, self.transmitter_direction_input, self.transmitter_no_of_rays_input, self.transmitter_no_of_reflections_input

    def receiver_get_values(self):
        self.receiver_pos_input, self.receiver_radius = self.sidebar_thing.receiver_get_values()
        return self.receiver_pos_input, self.receiver_radius

    def add_obj(self):
        """the code for this should be nigh identical to what i have in the main window btdubs"""
        pass


class SimulateDialog2SidebarThing(QWidget):

    add_model = Signal()

    def __init__(self, transmitter_no, receiver_no, parent=None):
        super().__init__(parent)
        self.transmitter_no = transmitter_no
        self.receiver_no = receiver_no

        dialog_layout = QVBoxLayout()
        button_add_model = QPushButton("Add a .obj file to the right hand side to adjust location of simulation")
        button_add_model.clicked.connect(self.add_model.emit)
        dialog_layout.addWidget(button_add_model)

        simulate_dialog2_container = QWidget()
        simulate_dialog2_layout = QHBoxLayout()

        transmitter_container = QWidget()
        receiver_container = QWidget()
        transmitter_container_layout = QVBoxLayout()
        receiver_container_layout = QVBoxLayout()

        self.transmitter_input_box  = TransmitterInputBox(1)
        transmitter_container_layout.addWidget(self.transmitter_input_box)
        transmitter_container.setLayout(transmitter_container_layout)

        self.receiver_input_box = ReceiverInputBox()
        receiver_container_layout.addWidget(self.receiver_input_box)
        receiver_container.setLayout(receiver_container_layout)

        simulate_dialog2_layout.addWidget(transmitter_container)
        simulate_dialog2_layout.addWidget(receiver_container)
        simulate_dialog2_container.setLayout(simulate_dialog2_layout)

        button_widget = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_widget.accepted.connect(parent.accept)
        button_widget.rejected.connect(parent.reject)

        dialog_layout.addWidget(simulate_dialog2_container)
        dialog_layout.addWidget(button_widget)

        self.setLayout(dialog_layout)

    def transmitter_get_values(self):
        self.id_input, self.pos_input, self.direction_input, self.no_of_rays_input, self.no_of_reflections_input = self.transmitter_input_box.get_values()
        return self.id_input, self.pos_input, self.direction_input, self.no_of_rays_input, self.no_of_reflections_input

    def receiver_get_values(self):
        self.receiver_pos_input, self.receiver_radius = self.receiver_input_box.get_values()
        return self.receiver_pos_input, self.receiver_radius


class TransmitterInputBox(QWidget):
    def __init__(self, default_id):
        super().__init__()
        self.default_id = default_id

        transmitter_input_box_layout = QGridLayout()

        # id
        id_label = QLabel("id:")
        transmitter_input_box_layout.addWidget(id_label, 0, 0)

        self.id_input = QSpinBox(minimum=1, maximum=20, value=default_id)
        transmitter_input_box_layout.addWidget(self.id_input, 0, 1)

        # position
        position_label = QLabel("position:")
        transmitter_input_box_layout.addWidget(position_label, 1, 0)

        self.pos_input = CoordinateInputTemplate()
        transmitter_input_box_layout.addWidget(self.pos_input, 1, 1)

        # transmission
        transmission_label = QLabel("transmission:")
        transmitter_input_box_layout.addWidget(transmission_label, 2, 0)

        # some stuff goes here? idk.

        # direction
        direction_label = QLabel("direction:")
        transmitter_input_box_layout.addWidget(direction_label, 3, 0)

        self.direction_input = CoordinateInputTemplate()
        transmitter_input_box_layout.addWidget(self.direction_input, 3, 1)


        # no of rays
        no_of_rays_label = QLabel("no of rays:")
        transmitter_input_box_layout.addWidget(no_of_rays_label, 4, 0)

        self.no_of_rays_input = QSpinBox(minimum=1, maximum=1000, value=1)
        transmitter_input_box_layout.addWidget(self.no_of_rays_input, 4, 1)

        # no of reflections
        no_of_reflections_label = QLabel("no of reflections:")
        transmitter_input_box_layout.addWidget(no_of_reflections_label, 5, 0)

        self.no_of_reflections_input = QSpinBox(minimum=1, maximum=1000, value=1)
        transmitter_input_box_layout.addWidget(self.no_of_reflections_input, 5, 1)

        # start strength
        start_strength_label = QLabel("start strength:")
        transmitter_input_box_layout.addWidget(start_strength_label, 6, 0)

        self.start_strength_input = QDoubleSpinBox(minimum=0, maximum=1000000, singleStep=10, decimals=3, value=10000)
        transmitter_input_box_layout.addWidget(self.start_strength_input, 6, 1)

        # end
        self.setLayout(transmitter_input_box_layout)

    def get_values(self):
        return self.id_input.value(), self.pos_input.get_values(), self.direction_input.get_values(), self.no_of_rays_input.value(), self.no_of_reflections_input.value()


class ReceiverInputBox(QWidget):
    def __init__(self):
        super().__init__()

        receiver_input_box_layout = QGridLayout()

        # position
        position_label = QLabel("position:")
        receiver_input_box_layout.addWidget(position_label, 0, 0)

        self.pos_input = CoordinateInputTemplate()
        receiver_input_box_layout.addWidget(self.pos_input, 0, 1)

        # radius
        radius_label = QLabel("radius:")
        receiver_input_box_layout.addWidget(radius_label, 1, 0)

        self.radius_input = QDoubleSpinBox()
        receiver_input_box_layout.addWidget(self.radius_input, 1, 1)

        self.setLayout(receiver_input_box_layout)

    def get_values(self):
        return self.pos_input.get_values(), self.radius_input.value()



class CoordinateSpinBox(QDoubleSpinBox):
    def __init__(self):
        super().__init__()
        self.setRange(-10000, 10000)
        self.setSingleStep(1)
        self.setDecimals(3)

class CoordinateInputTemplate(QWidget):
    def __init__(self):
        super().__init__()
        template_layout = QHBoxLayout()

        self.x = CoordinateSpinBox()
        self.y = CoordinateSpinBox()
        self.z = CoordinateSpinBox()

        template_layout.addWidget(self.x)
        template_layout.addWidget(self.y)
        template_layout.addWidget(self.z)

        self.setLayout(template_layout)

    def get_values(self):
        x_val = self.x.value()
        y_val = self.y.value()
        z_val = self.z.value()
        values = np.array([x_val, y_val, z_val])
        return values

def parse_coord_data(coords):
    coords = coords.strip(')(').split(')(')
    coords = [float(i) for i in coords]
    return coords

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = MyMainWindow()
    sys.exit(app.exec_())