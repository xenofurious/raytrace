import numpy as np
import pandas as pd
import random
import trimesh

# constants
c = 299792458
ray_advance = 20

#model_used = "models/"+input("Input the filename of the mesh you want to carry out simulations on: ")
#mesh = trimesh.load_mesh(model_used, process=True)
#if isinstance(mesh, trimesh.Scene):
#    mesh = trimesh.util.concatenate(mesh.dump())
#mesh.ray = trimesh.ray.ray_triangle.RayMeshIntersector(mesh)
#normals = mesh.face_normals



class transmitter:
    def __init__(self, location, no_of_rays, mode, direction=None, spread=None):
        self.location = location
        self.no_of_rays = no_of_rays
        self.mode = mode

class receiver:
    def __init__(self, location, detection_radius):
        self.location = location
        self.detection_radius = detection_radius

def generate_random_ray_lists(start_point, no_of_rays):
    ray_origins = [start_point for i in range(no_of_rays)]
    ray_directions = []
    for i in range(no_of_rays):
        vec_x = random.uniform(-1, 1)
        vec_y = random.uniform(-1, 1)
        vec_z = random.uniform(-1, 1)
        direction = np.array([vec_x, vec_y, vec_z])
        direction = direction/np.linalg.norm(direction)
        ray_directions.append(direction)
    return ray_origins, ray_directions

def generate_directional_ray_lists(start_point, no_of_rays, direction, spread):
    ray_origins = [start_point for i in range(no_of_rays)]
    ray_directions = []
    for i in range(no_of_rays):
        pass

def package_coord(coord):
    if not np.all(np.isnan(coord)):
        return_str = "(" + str(coord[0]) + ")(" + str(coord[1]) + ")(" + str(coord[2]) + ")"
    else:
        return_str = np.nan
    return return_str

test_receiver = receiver(location=np.array([0, 0, 0]), detection_radius=10)
detection_sphere = (test_receiver.location, test_receiver.detection_radius)


def create_dataframe(id, transmitter, receiver, start_strength, max_reflections, normals):
    start_point = transmitter.location
    end_point = receiver.location
    no_of_rays = transmitter.no_of_rays

def ray_sphere_intersection(ray_origin, ray_direction, receiver):
    my_ray_origin, my_ray_direction = ray_origin.copy(), ray_direction.copy()
    my_ray_direction = my_ray_direction/np.linalg.norm(my_ray_direction)
    sphere_centre = receiver.location
    sphere_radius = receiver.detection_radius
    s = my_ray_origin - sphere_centre
    b = np.dot(s, my_ray_direction)
    c = np.dot(s, s) - sphere_radius * sphere_radius
    h = b * b - c
    if (h < 0):
        return np.nan
    h = np.sqrt(h)
    t = -b - h

    return max(t, 0)

def ray_sphere_intersections(ray_origins, ray_directions, receiver): #returns an array of distances
    my_ray_origins, my_ray_directions = ray_origins.copy(), ray_directions.copy()
    my_ray_directions = my_ray_directions/np.linalg.norm(my_ray_directions, axis=1)[:, None] # the end bit of code reshapes the array so it doesn't break
    sphere_centre = receiver.location
    sphere_radius = receiver.detection_radius
    s = my_ray_origins - sphere_centre
    b = np.einsum('ij, ij->i', s, my_ray_directions)
    c = np.einsum('ij, ij->i', s, s) - sphere_radius * sphere_radius
    h = b * b - c

    h = np.where(h>=0, np.sqrt(h), np.nan)  # np.where works like: "condition", cond=True, cond=False
    # the code above works perfectly fine, but you get an annoying runtime warning.
    t = -b-h
    return t



# DEBUG
# some sample debug values
sample_receiver = receiver(np.array([0, 0, 0]), 5)
sample_ray_origin = np.array([10, 100, 10])
sample_ray_direction = np.array([-1, -1, -1])

sample_ray_origins = np.array([np.array([10, 11, 10]), np.array([-10, -9, -10]), np.array([20, -21, 20]), np.array([200, 20, 20])])
sample_ray_directions = np.array([[-10, -10, -10], [10, 10, 10], [-20, 20, -20], [20, 20, 20]]) #valid valid valid invalid
sample_ray_directions2 = np.array([[10, 0, 0], [10, 0, 0], [10, 0, 0], [10, 0, 0]]) #invalid values

print(ray_sphere_intersections(sample_ray_origins, sample_ray_directions, sample_receiver))

