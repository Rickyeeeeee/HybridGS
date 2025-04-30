import os

scenes = [24, 37, 40, 55, 63, 65, 69, 83, 97, 105, 106, 110, 114, 118, 122]
data_base_path='/workspace/data/replica_sclike_colmap_dnsplatter/dtu_dataset/dtu'
out_base_path='/workspace/work/Outputs/dtu'
eval_path='/workspace/data/replica_sclike_colmap_dnsplatter/dtu_dataset/MVS_Data'
out_name='pgsr'
gpu_id=1

for scene in scenes:
    cmd = f'rm -rf {out_base_path}/{out_name}/dtu_scan{scene}/*'
    print(cmd)
    os.system(cmd)

    cmd = f'cp -rf {data_base_path}/scan{scene}/sparse/0/* {data_base_path}/scan{scene}/sparse/'
    print(cmd)
    os.system(cmd)

    common_args = "--quiet -r2 --ncc_scale 0.5"
    cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python train.py -s {data_base_path}/scan{scene} -m {out_base_path}/{out_name}/dtu_scan{scene} {common_args}'
    print(cmd)
    os.system(cmd)

    cmd = 'conda install open3d==0.18.0 -y'
    print(cmd)
    os.system(cmd)

    common_args = "--quiet --num_cluster 1 --voxel_size 0.002 --max_depth 5.0"
    cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python render.py -m {out_base_path}/{out_name}/dtu_scan{scene} {common_args}'
    print(cmd)
    os.system(cmd)

    cmd = f"CUDA_VISIBLE_DEVICES={gpu_id} python scripts/eval_dtu/evaluate_single_scene.py " + \
          f"--input_mesh {out_base_path}/{out_name}/dtu_scan{scene}/mesh/tsdf_fusion_post.ply " + \
          f"--scan_id {scene} --output_dir {out_base_path}/{out_name}/dtu_scan{scene}/mesh " + \
          f"--mask_dir {data_base_path} " + \
          f"--DTU {eval_path}"
    print(cmd)
    os.system(cmd)

    cmd = f"CUDA_VISIBLE_DEVICES={gpu_id} python metrics.py " + \
          f"-m {out_base_path}/{out_name}/dtu_scan{scene} "
    print(cmd)
    os.system(cmd)