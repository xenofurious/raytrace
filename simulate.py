import numpy as np
import pandas as pd
import random
import trimesh
from pooch.utils import cache_location

# constants
c = 299792458
ray_advance = 20
model_used = "models/cube.obj"

mesh = trimesh.load_mesh(model_used, process=True)
if isinstance(mesh, trimesh.Scene):
    mesh = trimesh.util.concatenate(mesh.dump())
mesh.ray = trimesh.ray.ray_triangle.RayMeshIntersector(mesh)
normals = mesh.face_normals



class Transmitter:
    def __init__(self, location, no_of_rays, mode, direction=None, spread=None):
        self.location = location
        self.no_of_rays = no_of_rays
        self.mode = mode

class Receiver:
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

def generate_directional_ray_lists(start_point, no_of_rays, direction, spread):#unfinished!
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

test_receiver = Receiver(location=np.array([0, 0, 0]), detection_radius=10)
detection_sphere = (test_receiver.location, test_receiver.detection_radius)

def ray_sphere_intersection(ray_origin, ray_direction, receiver):
    my_ray_origin, my_ray_direction = ray_origin.copy(), ray_direction.copy()
    my_ray_direction = my_ray_direction/np.linalg.norm(my_ray_direction, axis=1)
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
    t = np.where(t<0, np.nan, t) # this is to avoid "collisions" backwards.
    return t


def calculate_sphere_collision_points_from_distance(previous_points_, new_directions_, distance_):
    previous_points, new_directions, distance = previous_points_.copy(), new_directions_.copy(), distance_.copy()
    new_directions = new_directions / np.linalg.norm(new_directions, axis=1)[:, None]
    sphere_collision_points = previous_points + new_directions * distance[:, None]
    return sphere_collision_points

def create_dataframe(id, transmitter, receiver, start_strength, max_reflections, normals):

    no_of_rays = transmitter.no_of_rays
    start_point = transmitter.location
    ray_origins, ray_directions = generate_random_ray_lists(start_point, no_of_rays)
    ray_origins = np.array(ray_origins)
    ray_directions = np.array(ray_directions)

    id = [id for i in range(no_of_rays)]
    start_strength = [start_strength for i in range(no_of_rays)]
    distance_arr = np.array([np.float64(0) for i in range(no_of_rays)])
    reflections = 0
    signal_factor =1

    start_point_packaged = list(map(package_coord, ray_origins.tolist()))

    data = {
        'id': id,
        'start_strength': start_strength,
        'end_strength': [],
        'traversal_time_ns': [],
        'start_point': start_point_packaged
    }

    end_points_arr = np.full((no_of_rays, 3), np.nan)
    while reflections < max_reflections and np.sum(np.isfinite(ray_directions).all(axis=1))>0:
        #need to filter out the nans from ray_origins and ray_directions before we input them - since mesh.ray.intersects doesn't accept nans
        prev_nan_mask = np.isfinite(ray_directions).all(axis=1)
        prev_nan_mask_indices = np.where(prev_nan_mask)[0]
        filtered_ray_origins = ray_origins[prev_nan_mask]
        filtered_ray_directions = ray_directions[prev_nan_mask]

        # calculating the ray-mesh intersections
        coord_arr, ray_index_arr, tri_index_arr = mesh.ray.intersects_location(
            filtered_ray_origins, filtered_ray_directions, multiple_hits=False
        )


        # sorting the rays - due to parallel processing constraints, the rays are returned in a jumbled mess
        n = len(ray_index_arr)
        for i in range(n):
            for j in range(0, n - i - 1):
                if ray_index_arr[j] > ray_index_arr[j + 1]:
                    temp = ray_index_arr[j].copy()
                    ray_index_arr[j], ray_index_arr[j + 1] = ray_index_arr[j + 1], temp
                    temp = coord_arr[j].copy()
                    coord_arr[j], coord_arr[j+1] = coord_arr[j+1], temp
                    temp = tri_index_arr[j].copy()
                    tri_index_arr[j], tri_index_arr[j + 1] = tri_index_arr[j + 1], temp

        dummy_coord_arr = np.full((no_of_rays, 3), np.nan)
        dummy_coord_arr[prev_nan_mask] = coord_arr.copy()
        coord_arr = dummy_coord_arr.copy()

        dummy_tri_index_arr = np.full(no_of_rays, np.nan)
        dummy_tri_index_arr[prev_nan_mask] = tri_index_arr.copy()
        tri_index_arr = dummy_tri_index_arr.copy()

        ray_index_arr = np.array(range(no_of_rays))



        # BY THIS POINT ThE MESH INTERSECTIONS HAVE BEEN CALCULATED.
        # NOW WE CALCULATE THE SPHERE COLLISIONS!!!!
        # NEXT TASK IS TO MOVE THE STUFF HERE.
        receiver_distances = ray_sphere_intersections(ray_origins, ray_directions, receiver) #this might not be in the correct place
        receiver_collision_points = calculate_sphere_collision_points_from_distance(ray_origins, ray_directions,
                                                                                    receiver_distances)
        # this is a block of code that loops through the face normals and "expands it" into a full array with nans. this is not vectorised, so it may need to be rewritten later.
        face_normals = np.full((no_of_rays, 3), np.nan)
        for x in range(no_of_rays):
            a = tri_index_arr[ray_index_arr[x]]
            if not np.isnan(a):
                face_normals[x] = normals[int(a)]

        # this code is for calculating the new ray directions
        epsilon = 0.00001 #this is for avoiding floating_point errors/face collisions
        dot_products = np.einsum('ij,ij->i', ray_directions, face_normals)[:, np.newaxis]

        coords_before = ray_origins.copy()
        ray_directions = ray_directions - 2 * dot_products * face_normals

        # this code is for calculating the new ray origins (it's already in coord_arr)
        ray_origins = np.array([i for i in coord_arr])
        ray_origins += ray_directions*epsilon
        reflections += 1

        # this code is for calculating the distance
        coords_after = ray_origins.copy()
        new_distance_arr = coords_after-coords_before

        # comparing the distances of ray-mesh and ray-sphere
        mesh_distances = np.linalg.norm(new_distance_arr, axis=1)

        # set the next ray directions to np.nan where comparison (mask) = True
        # and then make sure that the collision point was set to the sphere collision
        receiver_collision_mask = mesh_distances > receiver_distances
        valid_receiver_points_mask = np.isfinite(receiver_collision_points).all(axis=1)
        final_mask = receiver_collision_mask & valid_receiver_points_mask
        ray_origins[final_mask] = np.nan
        end_points_arr[final_mask] = receiver_collision_points[final_mask]

        # NOW WE NEED TO WRITE A BLOCK OF CODE THAT SETS ray_directions TO 0
        ray_directions[final_mask] = np.nan

        # check for whether maximum distance has been exceeded
        mask = np.all(np.isnan(new_distance_arr), axis=1)
        new_distance_arr[mask] = [0, 0, 0]
        distance_arr += np.linalg.norm(new_distance_arr, axis=1) #distance_arr shows how far the ray has travelled
        distance_mask = (start_strength - distance_arr) <0
        excess_distances = np.linalg.norm(new_distance_arr[distance_mask].copy(), axis=1)
        distance_arr[distance_mask] -= excess_distances

        # changing the coordinates with maximum distance to equal nan
        if np.any(distance_mask):
            ray_origins[distance_mask] = np.nan
            ray_directions[distance_mask] = np.nan

        title_a = 'interaction_type_'+str(reflections)
        data[title_a] = ['2' for i in range(no_of_rays)]
        title_b = 'point_' + str(reflections)
        ray_origins_packaged = list(map(package_coord, ray_origins.tolist()))
        data[title_b] = ray_origins_packaged

    indices = np.where(np.isnan(end_points_arr).any(axis=1))[0]
    end_points_list = list(map(package_coord, end_points_arr.tolist()))
    print(end_points_list)
    data['end_strength'] = (data['start_strength']-distance_arr*signal_factor).tolist() # alter later to match a realistic simulation model
    data['traversal_time_ns'] = (distance_arr*(10**9)/c).tolist()
    data = pd.DataFrame(data)
    data['end_point'] = end_points_list
    print(indices)
    data = data.drop(data.index[indices])
    return data

def create_csv(no_of_sources):
    for i in range(no_of_sources):
        #actual process
        start_strength = 10000
        max_reflections = 10
        df = create_dataframe(1, sample_transmitter, sample_receiver, start_strength, max_reflections, normals)
        if i==0:
            combined_df = df
        else:
            combined_df = pd.concat([combined_df, df], ignore_index=True, sort=False)

    combined_df.to_csv('generated_ray_data.csv', index=False)


# DEBUG
# some sample debug values
sample_transmitter = Transmitter(np.array([0.9, 0, 0]), no_of_rays=20, mode="random")
sample_receiver = Receiver(np.array([0, 0, 0]), 0.5)
#sample_ray_origin = np.array([10, 100, 10])
#sample_ray_direction = np.array([-1, -1, -1])
#
#sample_ray_origins = np.array([np.array([10, 11, 10]), np.array([-10, -9, -10]), np.array([20, -21, 20]), np.array([200, 20, 20])])
#sample_ray_directions = np.array([[-10, -10, -10], [10, 10, 10], [-20, 20, -20], [20, 20, 20]]) #valid valid valid invalid
#sample_ray_directions2 = np.array([[10, 0, 0], [10, 0, 0], [10, 0, 0], [10, 0, 0]]) #invalid values
#
#print(ray_sphere_intersections(sample_ray_origins, sample_ray_directions, sample_receiver))
#

#create_dataframe(id=1, transmitter=sample_transmitter, receiver=sample_receiver, start_strength=10000, max_reflections=6, normals=normals)
create_csv(1)