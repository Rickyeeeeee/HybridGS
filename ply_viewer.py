import open3d as o3d
import sys

def main(filepath):
    cloud = o3d.io.read_point_cloud(filepath) # Read point cloud
    o3d.visualization.draw_geometries([cloud]) # Visualize point cloud

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script_name.py path_to_ply_file")
        exit()
    
    filepath = sys.argv[1]
    main(filepath)