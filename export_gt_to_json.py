import json

from tqdm import tqdm

from nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes

from data.vector_map import VectorizedLocalMap
from data.rasterize import rasterize_map

import mmcv


def main(args):
    patch_h = args.ybound[1] - args.ybound[0]
    patch_w = args.xbound[1] - args.xbound[0]
    canvas_h = int(patch_h / args.ybound[2])
    canvas_w = int(patch_w / args.xbound[2])
    patch_size = (patch_h, patch_w)
    canvas_size = (canvas_h, canvas_w)

    submission = {
        "meta": {
            "use_camera": True,
            "use_lidar": False,
            "use_radar": False,
            "use_external": False,
            "vector": not args.raster
        },
        "results": {}
    }

    scenes = create_splits_scenes()[args.eval_set]
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)
    vector_map = VectorizedLocalMap(args.dataroot, patch_size=patch_size, canvas_size=canvas_size)

    samples = [samp for samp in nusc.sample if nusc.get('scene', samp['scene_token'])['name'] in scenes]
    for rec in tqdm(samples):
        location = nusc.get('log', nusc.get('scene', rec['scene_token'])['log_token'])['location']
        ego_pose = nusc.get('ego_pose', nusc.get('sample_data', rec['data']['LIDAR_TOP'])['ego_pose_token'])
        vectors = vector_map.gen_vectorized_samples(location, ego_pose['translation'], ego_pose['rotation'])
        if args.raster:
            pred_map, confidence_level = rasterize_map(vectors, patch_size, canvas_size, args.max_channel, args.thickness)
            submission['results'][rec['token']] = {
                'map': pred_map,
                'confidence_level': confidence_level
            }
        else:
            for vector in vectors:
                vector['confidence_level'] = 1
                vector['pts'] = vector['pts'].tolist()
            submission['results'][rec['token']] = vectors

    with open(args.output, 'w') as f:
        json.dump(submission, f)
    print(f"exported to {args.output}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Demo Script to turn GT into json submission.')
    parser.add_argument('--dataroot', type=str, default='dataset/nuScenes/')
    parser.add_argument('--version', type=str, default='v1.0-mini', choices=['v1.0-trainval', 'v1.0-mini'])
    parser.add_argument('--eval_set', type=str, default='mini_val', choices=['train', 'val', 'test', 'mini_train', 'mini_val'])
    parser.add_argument('--output', type=str, default='submission.json')
    parser.add_argument('--thickness', type=int, default=1)
    parser.add_argument('--max_channel', type=int, default=3)
    parser.add_argument("--xbound", nargs=3, type=float, default=[-30.0, 30.0, 0.15])
    parser.add_argument("--ybound", nargs=3, type=float, default=[-15.0, 15.0, 0.15])
    parser.add_argument('--raster', action='store_true', default=False)
    args = parser.parse_args()

    main(args)
