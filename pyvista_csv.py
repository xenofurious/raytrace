import pyvista as pv
import numpy as np
import pandas as pd
raytrace_data = pd.read_csv('generated_ray_data.csv')

#setting up axes
pl = pv.Plotter(shape=(2, 2))

#table = ax3.table((2, 4), 2, 2)
row_num = raytrace_data.shape[0]
col_num = raytrace_data.shape[1]

##################
#RAYCAST PLOTTING#
##################

def parse_coord_data(coords):
    coords = coords.strip(')(').split(')(')
    coords = [float(i) for i in coords]
    return coords

print(raytrace_data.to_string())
start_point = np.array(parse_coord_data(raytrace_data['start_point'][0]))
pl.add_points(start_point, point_size=20.0, color='red')


row_number = raytrace_data.shape[0]

for index, row in raytrace_data.iterrows():
    try:
        coord_index = raytrace_data.columns.get_loc('interaction_type_1')
        previous_point = start_point
        point_colour = 'black'
        while coord_index <col_num and pd.isna(row.iloc[coord_index])==False:
            match row.iloc[coord_index]:
                case 1:
                    point_colour = 'blue'
                case 2:
                    point_colour = 'lightgreen'
                case 3:
                    point_colour = 'yellow'
                case other:
                    current_point = np.array(parse_coord_data(row.iloc[coord_index]))
                    pl.add_points(current_point, render_points_as_spheres=True, point_size=10.0, color=point_colour)
                    my_line = pv.Line(previous_point, current_point)
                    pl.add_mesh(my_line, color='black')
                    previous_point = current_point
            coord_index +=1
        end_point = np.array(parse_coord_data(row['end_point']))
        pl.add_points(end_point, point_size=15.0, color='black')
        end_line = pv.Line(previous_point, end_point)
        pl.add_mesh(end_line)
    except:
        print("a row was omitted for this")


model_name = input("Enter the model you want to overlay over the simulation. Leave blank if you don't want to: ")
if model_name != '':
    model = pv.read(model_name)
    pl.add_mesh(
        model,
        color='lightblue',
        opacity=0.4,
        specular=1.0,
        smooth_shading=True
    )





##############################
#POWER DELAY PROFILE PLOTTING#
##############################

pl.subplot(0, 1)

pl.show_bounds(
    grid='front',
    location='outer',
    all_edges=True,
    ticks='both',
    xtitle='end strength', ytitle='traversal time', ztitle='',
    font_size=14,
    color='black',
)

pdp_df = raytrace_data[['end_strength', 'traversal_time_ns']].astype(float)
#ax2.axhline(0, color='gray', linewidth=1.5, zorder=0)

#pdp profile calculations
def traversal_time_ns_mean_calc(pdp_df):
    return np.sqrt((pdp_df['end_strength']*pdp_df['traversal_time_ns']).mean())
def rms_delay_calc(pdp_df):
    pathloss_sum = pdp_df['end_strength'].sum()
    traversal_time_ns_mean = traversal_time_ns_mean_calc(pdp_df)
    squared_traversal_time = (((pdp_df['traversal_time_ns']-traversal_time_ns_mean)**2)*pdp_df['end_strength']).sum()
    rms_delay = np.sqrt(squared_traversal_time/pathloss_sum)
    return rms_delay

traversal_time_ns_mean = traversal_time_ns_mean_calc(pdp_df)
rms_delay = rms_delay_calc(pdp_df)


for row in pdp_df.itertuples(index=False):
    row_x = row[0]
    row_y = row[1]
    my_stem = pv.Line([row_x, 0, 0], [row_x, row_y, 0])
    pl.add_mesh(my_stem, color='black', line_width=2)

pl.view_xy()

##############
#PLOTTING OBJ#
##############

pl.subplot(1, 0)
custom_model = pv.read('modelname.obj')
pl.add_mesh(custom_model)

pl.subplot(1, 1)
sabrina = pv.read('sabrina.obj')
pl.add_mesh(sabrina)


pl.show()