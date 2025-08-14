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

def generate_ray_lists(start_point, no_of_rays):
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

def package_coord(coord):
    if not np.all(np.isnan(coord)):
        return_str = "(" + str(coord[0]) + ")(" + str(coord[1]) + ")(" + str(coord[2]) + ")"
    else:
        return_str = np.nan
    return return_str


def create_dataframe(id, start_point, start_strength, no_of_rays, max_reflections, normals):
    ray_origins, ray_directions = generate_ray_lists(start_point, no_of_rays)
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

    while reflections < max_reflections and np.sum(np.isfinite(ray_origins).all(axis=1))>0:
        ray_index_arr_total = np.array([i for i in range(no_of_rays)])

        #need to filter out the nans from ray_origins and ray_directions before we input them - since mesh.ray.intersects doesn't accept nans
        mask = np.isfinite(ray_origins).all(axis=1)
        filtered_ray_origins = ray_origins[mask]
        filtered_ray_directions = ray_directions[mask]

        coord_arr, ray_index_arr, tri_index_arr = mesh.ray.intersects_location(
            filtered_ray_origins, filtered_ray_directions, multiple_hits=False
        )
        no_of_nans = no_of_rays-len(coord_arr)
        arr_nans_vec = np.full((no_of_nans, 3), np.nan)
        arr_nans_float = np.array([np.nan for i in range(no_of_nans)])

        diff_rayindex = np.setdiff1d(ray_index_arr_total, ray_index_arr)

        if coord_arr.size ==0:
            coord_arr = arr_nans_vec
            tri_index_arr = arr_nans_float
        else:
            coord_arr = np.concatenate((coord_arr, arr_nans_vec), axis=0)
            tri_index_arr = np.concatenate((tri_index_arr, arr_nans_float), axis=0)

        ray_index_arr = np.concatenate((ray_index_arr, diff_rayindex), axis=0)

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

        epsilon = 0.00001 #this is for avoiding floating_point errors/face collisions

        face_normals = np.full((no_of_rays, 3), np.nan)
        for x in range(no_of_rays):
            a = tri_index_arr[ray_index_arr[x]]
            if not np.isnan(a):
                face_normals[x] = normals[int(a)]

        dot_products = np.einsum('ij,ij->i', ray_directions, face_normals)[:, np.newaxis]

        coords_before = ray_origins

        ray_directions = ray_directions - 2 * dot_products * face_normals
        ray_origins = np.array([i for i in coord_arr])
        ray_origins += ray_directions*epsilon
        reflections += 1

        coords_after = ray_origins
        new_distance_arr = coords_after-coords_before

        # check for whether maximum distance has been exceeded
        mask = np.all(np.isnan(new_distance_arr), axis=1)
        new_distance_arr[mask] = [0, 0, 0]
        distance_arr += np.linalg.norm(new_distance_arr, axis=1)
        distance_mask = (start_strength - distance_arr) <0
        excess_distances = np.linalg.norm(new_distance_arr[distance_mask].copy(), axis=1)
        distance_arr[distance_mask] -= excess_distances

        # changing the coordinates with maximum distance to equal nan
        if np.any(distance_mask):
            ray_origins[distance_mask] = np.full((distance_mask.sum(), 3), np.nan)

        title_a = 'interaction_type_'+str(reflections)
        data[title_a] = ['2' for i in range(no_of_rays)]
        title_b = 'point_' + str(reflections)
        ray_origins_packaged = list(map(package_coord, ray_origins.tolist()))
        data[title_b] = ray_origins_packaged


    data['end_strength'] = (data['start_strength']-distance_arr*signal_factor).tolist() # alter later to match a realistic simulation model
    data['traversal_time_ns'] = (distance_arr*(10**9)/c).tolist()
    data = pd.DataFrame(data)

    data['end_point'] = data.bfill(axis=1).iloc[:, -1]

    return data

def create_csv(no_of_sources):
    for i in range(no_of_sources):
        #actual process


        start_point = input("What is the start point of the rays from source "+str(i+1)+" you want to generate? Type your values separated by spaces and lave blank for origin: ")
        if start_point == '':
            start_point = test_start_point
        else:
            start_point = np.array(start_point.split()).astype(float)
        start_strength = input("What is the start strength of the rays from source "+str(i+1)+" you want to generate? Leave blank for 10,000: ")
        if start_strength == '':
            start_strength = 10000
        no_of_rays = input("What is the number of rays for source "+str(i+1)+" you want to generate? Leave blank for 10: ")
        if no_of_rays == '':
            no_of_rays = 10
        max_reflections = input("What is the maximum number of reflections for source "+str(i+1)+" you want your rays to simulate? Leave blank for 10: ")
        if max_reflections == '':
            max_reflections = 10

        start_strength = int(start_strength)
        no_of_rays = int(no_of_rays)
        max_reflections = int(max_reflections)
        df = create_dataframe(i+1, start_point, start_strength, no_of_rays, max_reflections, normals=normals)
        if i==0:
            combined_df = df
        else:
            combined_df = pd.concat([combined_df, df], ignore_index=True, sort=False)

    combined_df.to_csv('generated_ray_data.csv', index=False)




#df = create_dataframe(id=i+1, start_point, start_strength, no_of_rays, max_reflections, normals=normals)
#print(df.to_string())
#df.to_csv('generated_ray_data.csv', index=False)
test_start_point = np.array([0, 0, 0])
test_start_point2 = np.array([1, 0, 0])


no_of_sources = input("Enter the number of ray sources you want to use. Leave blank for 1: ")
if no_of_sources == '':
    no_of_sources = 1
else:
    try:
        no_of_sources = int(no_of_sources)
    except:
        print("You didn't input an integer 🤡")
if type(no_of_sources) == int:
    create_csv(no_of_sources)
    print("Simulation successful")