import open3d as o3d
import numpy as np
import time
from open3d.examples.geometry.point_cloud_outlier_removal_statistical import display_inlier_outlier

def add_noise_in_point_cloud(filename):
    """ Загрузка облака точек из файла. Open3D"""
    pcd = o3d.io.read_point_cloud(filename)
    original_size = len(np.asarray(pcd.points))
    print(original_size)
    pcd.estimate_normals()
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
    mesh.paint_uniform_color([1, 0.706, 0])
    vertices = np.asarray(mesh.vertices)
    vertices += np.random.uniform(0, 3, size=vertices.shape)
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    pcd1 = mesh.sample_points_uniformly(number_of_points=original_size)
    o3d.io.write_point_cloud("anchor_01.ply", pcd1)
    return pcd

def copy_point_cloud(point_cloud):
    """ Копирование облака точек. """
    copied_cloud = o3d.geometry.PointCloud()
    copied_cloud.points = o3d.utility.Vector3dVector(np.array(point_cloud.points))
    if point_cloud.has_normals():
        copied_cloud.normals = o3d.utility.Vector3dVector(np.array(point_cloud.normals))
    if point_cloud.has_colors():
        copied_cloud.colors = o3d.utility.Vector3dVector(np.array(point_cloud.colors))
    return copied_cloud

def load_point_cloud(filename):
    """ Загрузка облака точек из файла. Open3D"""
    pcd = o3d.io.read_point_cloud(filename)
    return pcd


def mesh_reconstruction(point_cloud, depth=9):
    """ Реконструкция поверхности методом Poisson. """
    point_cloud.estimate_normals()
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(point_cloud, depth=depth)
    o3d.visualization.draw_geometries([mesh],
                                      zoom=0.8,
                                      front=[ 0.10298039599678206, -0.9430397827479553, 0.31634001674627754 ],
                                      lookat=[ 1.8899999999999999, 3.2595999999999998, 0.9284 ],
                                      up=[ 0.42202188330344015, 0.32941096768937156, 0.84462177593226273 ])
    return mesh


def estimate_normals(point_cloud, radius=0.1, max_nn=30):
    """ Оценка нормалей точек в облаке. """
    point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
    return point_cloud

def remove_statistical_outliers(point_cloud, nb_neighbors=20, std_ratio=2.0):
    """ Удаление статистических выбросов из облака точек. """
    start_time = time.time()
    clean_cloud, ind = point_cloud.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    elapsed_time = time.time() - start_time
    clean_cloud.paint_uniform_color([0, 0, 0])
    o3d.visualization.draw_geometries([clean_cloud],
                                      zoom=0.8,
                                      front=[0.079982930507723496, -0.075192128688307422, 0.99395617338528452],
                                      lookat=[1.8899999999999999, 3.2595999999999998, 0.9284],
                                      up=[0.089932090137603005, 0.99362853040788313, 0.067930572814844326])
    # display_inlier_outlier(point_cloud, ind)
    # mesh_reconstruction(clean_cloud)
    return clean_cloud,elapsed_time

def remove_radius_outliers(point_cloud, nb_points=10, radius=3):
    """ Удаление выбросов на основе радиуса. """
    start_time = time.time()
    clean_cloud, ind = point_cloud.remove_radius_outlier(nb_points=nb_points, radius=radius)
    elapsed_time = time.time() - start_time
    clean_cloud.paint_uniform_color([0, 0, 0])
    o3d.visualization.draw_geometries([clean_cloud],
                                      zoom=0.8,
                                      front=[0.079982930507723496, -0.075192128688307422, 0.99395617338528452],
                                      lookat=[1.8899999999999999, 3.2595999999999998, 0.9284],
                                      up=[0.089932090137603005, 0.99362853040788313, 0.067930572814844326])
    return clean_cloud,elapsed_time

def downsample_cloud(point_cloud, voxel_size=0.02):
    """ Уменьшение количества точек в облаке с помощью воксельной решетки. """
    start_time = time.time()
    point_cloud.estimate_normals()
    downsampled_cloud = point_cloud.voxel_down_sample(voxel_size=voxel_size)
    elapsed_time = time.time() - start_time
    o3d.visualization.draw_geometries([downsampled_cloud],
                                      zoom=0.8,
                                      front=[0.10298039599678206, -0.9430397827479553, 0.31634001674627754],
                                      lookat=[1.8899999999999999, 3.2595999999999998, 0.9284],
                                      up=[0.42202188330344015, 0.32941096768937156, 0.84462177593226273])
    return downsampled_cloud,elapsed_time

def filter_fun_simple(point_cloud,original_size,number_iteration):
    start_time = time.time()
    point_cloud.estimate_normals()
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(point_cloud, depth=9)
    mesh_out = mesh.filter_smooth_simple(number_of_iterations=number_iteration)
    elapsed_time = time.time() - start_time
    pcd2 = mesh_out.sample_points_uniformly(number_of_points=original_size)
    pcd2.paint_uniform_color([0, 0, 0])
    o3d.visualization.draw_geometries([pcd2],
                                      zoom=0.8,
                                      front=[0.079982930507723496, -0.075192128688307422, 0.99395617338528452],
                                      lookat=[1.8899999999999999, 3.2595999999999998, 0.9284],
                                      up=[0.089932090137603005, 0.99362853040788313, 0.067930572814844326])
    return pcd2, elapsed_time


def filter_fun_laplacian(point_cloud,original_size,number_iteration):
    start_time = time.time()
    point_cloud.estimate_normals()
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(point_cloud, depth=9)
    mesh_out = mesh.filter_smooth_laplacian(number_of_iterations=number_iteration)
    elapsed_time = time.time() - start_time
    pcd2 = mesh_out.sample_points_uniformly(number_of_points=original_size)
    pcd2.paint_uniform_color([0, 0, 0])
    o3d.visualization.draw_geometries([pcd2],
                                      zoom=0.8,
                                      front=[0.079982930507723496, -0.075192128688307422, 0.99395617338528452],
                                      lookat=[1.8899999999999999, 3.2595999999999998, 0.9284],
                                      up=[0.089932090137603005, 0.99362853040788313, 0.067930572814844326])
    return pcd2, elapsed_time

def filter_fun_taubin(point_cloud,original_size,number_iteration):
    start_time = time.time()
    point_cloud.estimate_normals()
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(point_cloud, depth=9)
    mesh_out = mesh.filter_smooth_taubin(number_of_iterations=number_iteration)
    elapsed_time = time.time() - start_time
    pcd2 = mesh_out.sample_points_uniformly(number_of_points=original_size)
    pcd2.paint_uniform_color([0, 0, 0])
    o3d.visualization.draw_geometries([pcd2],
                                      zoom=0.8,
                                      front=[0.079982930507723496, -0.075192128688307422, 0.99395617338528452],
                                      lookat=[1.8899999999999999, 3.2595999999999998, 0.9284],
                                      up=[0.089932090137603005, 0.99362853040788313, 0.067930572814844326])
    return pcd2, elapsed_time

def analyze_methods(point_cloud):
    """ Анализ методов и вывод результатов. """
    original_size = len(np.asarray(point_cloud.points))
    print(f"Original Point Count: {original_size}")
    point_cloud.paint_uniform_color([0, 0, 0])
    o3d.visualization.draw_geometries([point_cloud],
                                      zoom=0.8,
                                      front=[ 0.079982930507723496, -0.075192128688307422, 0.99395617338528452 ],
                                      lookat=[ 1.8899999999999999, 3.2595999999999998, 0.9284 ],
                                      up=[ 0.089932090137603005, 0.99362853040788313, 0.067930572814844326 ])

    cleaned_cloud, time_stat_outliers = remove_statistical_outliers(copy_point_cloud(point_cloud),20,2.0)
    cleaned_size = len(np.asarray(cleaned_cloud.points))
    print(f"Cleaned with Statistical Outlier Removal: {cleaned_size} points, Time: {time_stat_outliers:.4f} sec")


    cleaned_cloud_radius, time_radius_outliers = remove_radius_outliers(copy_point_cloud(point_cloud),15)
    cleaned_radius_size = len(np.asarray(cleaned_cloud_radius.points))
    print(f"Cleaned with Radius Outlier Removal: {cleaned_radius_size} points, Time: {time_radius_outliers:.4f} sec")

    # downsampled_cloud, time_downsample = downsample_cloud( copy_point_cloud(point_cloud))
    # downsampled_size = len(np.asarray(downsampled_cloud.points))
    # print(f"Downsampled Point Count: {downsampled_size}, Time: {time_downsample:.4f} sec")

    filt_simple_cloud, time_simple = filter_fun_simple( copy_point_cloud(point_cloud), original_size,1)
    # filt_simple_size = len(np.asarray(filt_simple_cloud.points))
    # filt_cloud, time_stat_outliers = remove_statistical_outliers(filt_simple_cloud, 20, 2.0)
    print(f"Simple Filter Point Count:  Time: {time_simple:.4f} sec")


    filt_laplacian_cloud, time_laplacian = filter_fun_laplacian(copy_point_cloud(point_cloud), original_size,100)
    # filt_laplacian_size = len(np.asarray(filt_laplacian_cloud.points))
    # filt_cloud, time_stat_outliers = remove_statistical_outliers(filt_laplacian_cloud, 20, 2.0)
    print(f"Laplacian Filter Point Count: , Time: {time_laplacian:.4f} sec")


    filt_taubin_cloud, time_taubin = filter_fun_taubin(copy_point_cloud(point_cloud), original_size,10)
    # filt_cloud, time_stat_outliers = remove_statistical_outliers(filt_taubin_cloud, 20, 2.0)
    print(f"Taubin Filter Point Count:  Time: {time_taubin:.4f} sec")


# mesh = mesh_reconstruction(copy_point_cloud(point_cloud))
    # print(f"Mesh Vertices Count: {len(mesh.vertices)}")

def main():
    filename = "dc_4.ply"
    pcd = load_point_cloud(filename)
    analyze_methods(pcd)

if __name__ == "__main__":
    main()
