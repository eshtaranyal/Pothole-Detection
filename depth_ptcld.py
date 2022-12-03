import sys
import numpy as np
import open3d as o3d
import matplotlib as plt
import pyransac3d as pyrsc
import collections

def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)
    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0, 0, 1])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])
    

d=[]

pcd_load = o3d.io.read_point_cloud("point_cloud/eartsciencepothole.ply")

points = np.asarray(pcd_load.points)
plano1 = pyrsc.Plane()
length=points.shape[0]

print("Statistical outlier removal")
cl, ind = pcd_load.remove_statistical_outlier(nb_neighbors=1000, std_ratio=2)
inlier_cloud = pcd_load.select_by_index(ind)
o3d.io.write_point_cloud("point_cloud/inlier.ply", inlier_cloud)

pcd_load = o3d.io.read_point_cloud("point_cloud/inlier.ply")
#display_inlier_outlier(pcd_load, ind)
points = np.asarray(pcd_load.points)
length=points.shape[0]

best_eq, best_inliers = plano1.fit(points, 0.005, maxIteration=10000)
#print(best_inliers)
nx=best_eq[0]
ny=best_eq[1]
nz=best_eq[2]


#phdry_crop_mesh.ply
# nx=-0.0841306
# ny=-0.337601
# nz=0.937522
#phdry_crop.ply
# nx=-0.0776269
# ny=-0.378461
# nz=0.922356
# lawn_pot_crop.ply 
nxl=-0.124241
nyl=0.775119
nzl=0.619479
print(nx," ,", ny, " , " ,nz)
for i in range(int(length)):
	dist=(nx*points[i][0]+ny*points[i][1]+nz*points[i][2])/((nx**2+ny**2+nz**2)**0.5)
	dist=round(dist, 3)
	d.append(abs(dist))
	
ctr = collections.Counter(d)
freq=ctr.most_common(1)[0][0]
least_common = ctr.most_common()[-1]
index_max_dist=np.where(d == freq)
print(freq)
#print(points[index_max_dist])
d.sort()

print(d[0]," : ",d[-1])
print("height : ",abs(d[0]-freq))
print("depth : ",abs(d[-1]-freq)+.015)

plane = pcd_load.select_by_index(best_inliers)
not_plane = pcd_load.select_by_index(best_inliers, invert=True)
o3d.visualization.draw_geometries([ plane,not_plane])
