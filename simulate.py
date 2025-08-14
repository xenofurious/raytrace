import numpy as np
import pandas as pd
import random
import trimesh

# constants
c = 299792458
ray_advance = 20
model_used = "models/"+input("Input the filename of the mesh you want to carry out simulations on: ")

mesh = trimesh.load_mesh(model_used, process=True)
if isinstance(mesh, trimesh.Scene):
    mesh = trimesh.util.concatenate(mesh.dump())
mesh.ray = trimesh.ray.ray_triangle.RayMeshIntersector(mesh)
normals = mesh.face_normals



class transmitter:
    def __init__(self, location, no_of_rays, mode, direction=None, spread=None):
        self.location = location
        self.no_of_rays = no_of_rays
        self.mode = mode

class receiver:
    def __init__(self, location, detection_radius, receive_number):
        self.location = location
        self.detection_radius = detection_radius
        self.receive_number = receive_number


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

