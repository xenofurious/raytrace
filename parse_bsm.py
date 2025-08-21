import xml.etree.ElementTree as ET
import numpy as np
import os
import shutil
from pathlib import Path

filepath = "Demo/Demo/Building/B1.bsm"
tree = ET.parse(filepath)
root = tree.getroot()

#print(root[9][0][10][0])
sample_entity = root[9][0][10][0] #this is the first "wall"

def return_face_obj(sample_entity):
    face_corner_list = []
    thickness = sample_entity[4] #self explanatory
    for corner in sample_entity[6]: #sample_entity[6] refers to the wall corners
        coordinate = [float(corner[0][0].text), float(corner[0][1].text), float(corner[0][2].text)]
        face_corner_list.append(coordinate)

    #now convert it to an obj by extruding it as a prism
    vertex_list = []
    face_list = []
    norm = calculate_norm_from_face(face_corner_list)
    wall_face_arr = np.array(face_corner_list) #for vectorised operations
    vertex_list.extend(wall_face_arr+norm)
    n = len(vertex_list)
    vertex_list.extend(wall_face_arr-norm)

    #get faces working
    #face 1:
    face_1 = list(range(1, n+1))
    face_2 = list(range(n+1, 2*n+1))
    face_list.append(face_1)
    face_list.append(face_2)
    for i in range(1, n+1):
        third_corner = i+n+1
        fourth_corner = i+1
        if third_corner> 2*n:
            third_corner = i+1
            fourth_corner = 1
        new_face = [i, i+n, third_corner, fourth_corner]
        face_list.append(new_face)

    material = sample_entity[5]
    return vertex_list, face_list, material


def calculate_norm_from_face(face_of_vertices):
    face_of_vertices_arr = np.array(face_of_vertices) #for vectorised operations
    vector1 = face_of_vertices_arr[1]-face_of_vertices_arr[0]
    vector2 = face_of_vertices_arr[2]-face_of_vertices_arr[1]
    norm = np.cross(vector1, vector2)
    norm = norm/np.linalg.norm(norm)
    return  norm

def write_to_obj(filename, vertices, faces):
    with open(filename, 'w') as f:
        #const
        number_of_faces = len(vertices)

        #writing metadata
        f.write("#Kasper Epic Technologies Ltd.\n")
        f.write("#I love writing placeholder metadata!\n")

        for vertex in vertices:
            f.write('v '+str(vertex[0]) + ' ' + str(vertex[1]) + ' ' + str(vertex[2]) + '\n')

        for face in faces:
            f.write('f ' + str(face[0]))
            for vertex_id in face[1:]:
                f.write(' ' + str(vertex_id))
            f.write('\n')

def write_floor_to_obj(folder_name, file_name, floor):
    cwd = Path.cwd()
    if os.path.exists(folder_name):
        shutil.rmtree(folder_name)
    os.mkdir(folder_name)
    os.chdir(folder_name)
    entities = floor[10]
    for index in range(len(entities)):
       entity = entities[index]
       file_str = file_name+str(index)+".obj"
       vertex_list, face_list, material = return_face_obj(entity)
       write_to_obj(file_str, vertex_list, face_list)
    os.chdir(cwd)


def write_floor_collection_to_obj(folder_name_outer, folder_name_inner, file_name, floor_collection):
    cwd = Path.cwd()
    if os.path.exists(folder_name_outer):
        shutil.rmtree(folder_name_outer)
    os.mkdir(folder_name_outer)
    os.chdir(folder_name_outer)
    for index in range(len(floor_collection)):
        floor = floor_collection[index]
        folder_str = folder_name_inner+str(index)+".obj"
        file_str = file_name+str(index)+"_"
        write_floor_to_obj(folder_str, file_str, floor)
    os.chdir(cwd)


def write_floor_collection_to_simple_obj(file_name, floor_collection):
    vertex_list = []
    face_list = []
    no_of_vertices_added = 0
    floor_height_change = 0
    for index in range(len(floor_collection)):
        floor = floor_collection[index]
        for entity in floor[10]:
            new_vertices, new_faces, material = return_face_obj(entity)
            #add however much to each vertices / face thing.

            new_vertices = [coordinate + np.array([0, 0, floor_height_change])for coordinate in new_vertices]
            new_faces = [[x + no_of_vertices_added for x in sublist] for sublist in new_faces]
            vertex_list.extend(new_vertices)
            face_list.extend(new_faces)
            no_of_vertices_added += len(new_vertices)

        floor_height = float(floor[5].text)
        floor_height_change+=floor_height
    write_to_obj(file_name, vertex_list, face_list)



#sample_floor = root[9][0]
sample_floor_collection = root[9]
#sample_vertex_list, sample_face_list, sample_material = return_face_obj(sample_entity)
#print(sample_face_list)
#write_wall_face_to_obj("sybau.obj", sample_vertex_list, sample_face_list)

#write_floor_to_obj("sybau", "sybau", sample_floor)
write_floor_collection_to_obj("collection_outer", "collection_inner", "entity", sample_floor_collection)
#write_floor_collection_to_simple_obj("sybau_final.obj", sample_floor_collection)