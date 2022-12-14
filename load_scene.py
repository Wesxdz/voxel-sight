load_scene = "/home/aeri/il/minerl/voxel_sight/data/train/voxel_scene_0"
with open(load_scene) as scene:
    voxels = [line.strip() for line in scene.readlines()]
    print(voxels)