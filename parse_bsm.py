import xml.etree.ElementTree as ET
import numpy as np

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
    face_1 = list(range(n))
    face_2 = list(range(n, 2*n))
    face_list.append(face_1)
    face_list.append(face_2)
    for i in range(n):
        new_face = [i, i+n, i+n+1, i+1]
        face_list.append(new_face)

    material = sample_entity[5]
    return vertex_list, face_list, material


def calculate_norm_from_face(face_of_vertices):
    face_of_vertices_arr = np.array(face_of_vertices) #for vectorised operations
    vector1 = face_of_vertices_arr[1]-face_of_vertices_arr[0]
    vector2 = face_of_vertices_arr[2]-face_of_vertices_arr[1]
    norm = np.linalg.cross(vector1, vector2)
    norm = norm/np.linalg.norm(norm)
    return  norm

def write_wall_face_to_obj(filename, vertices, faces):
    with open(filename, 'w') as f:
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

sample_vertex_list, sample_face_list, sample_material = return_face_obj(sample_entity)
write_wall_face_to_obj("sybau.txt", sample_vertex_list, sample_face_list)