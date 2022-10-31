# This script runs in Blender's Python environment!

import opensimplex
import random
import bpy
from bpy import context
from mathutils import Vector
import numpy as np
import math
import bmesh

class TerrainNoiseLayer():
    scale = 1.0
    multiplier = 1.0
    seed = random.randint(0, 10000000)
    def __init__(self, scale, multiplier):
        self.scale = scale
        self.multiplier = multiplier
        
    def set_random_seed(self):
        self.seed = random.randint(0, 10000000)
        
voxel_type_seed = random.randint(0, 10000000)
    
noise_layers = [TerrainNoiseLayer(1.0, 0.5), TerrainNoiseLayer(0.1, 3.0), TerrainNoiseLayer(0.02, 10.0)]

pool_collection_name = "voxels"
block_size = 0.0
chunk_size = 16

# TODO: Will need to create a distinct placer for each voxel type, or encode objects in vertex data
placer = bpy.data.objects["placer"]

def choose_random_object_from_pool(objects):
    return random.choice(objects)

def spawn_voxel_occlusion_heightmap(noise, verts, colors, rgb, start_x, start_y, width, height, scale):
    for y in range(height):
        for x in range(width):
            n = noise[int(start_y) + y][int(start_x) + x]
            location = Vector(((start_x + x), (start_y + y), round(n)))
            byte_color = [int(start_x+x), int(start_y+y), int(random.randint(0, 255))]
            color = [byte_color[0]/255.0, byte_color[1]/255.0, byte_color[2]/255.0, 1.0]
            verts.append(location)
            colors.append(color)
            rgb.append(byte_color)
            
def spawn_voxel_heightmap(noise, verts, start_x, start_y, width, height, scale):
    for y in range(height):
        for x in range(width):
            n = noise[int(start_y) + y][int(start_x) + x]
            location = Vector(((start_x + x), (start_y + y), round(n)))
            verts.append(location)

def get_heightmap_baseline():
    pass

def spawn_occlusion_chunks(radius, noise, d):
    mesh = bpy.data.meshes.new("example")
    ob = bpy.data.objects.new("placer", mesh)
    bpy.context.collection.objects.link(ob)
    ob.location = Vector((0,0,0))
    verts = []
    rgb = []
    colors = []
    for y in range(0, radius*2):
        for x in range(0, radius*2):
            chunk_origin = Vector((x*chunk_size, y*chunk_size, 0.0))
            spawn_voxel_occlusion_heightmap(noise, verts, colors, rgb, chunk_origin.x, chunk_origin.y, chunk_size, chunk_size, 1.0)
    mesh.from_pydata(verts, [], [])
    bpy.context.view_layer.objects.active = ob
    bpy.ops.object.modifier_add(type='NODES')
    bpy.context.object.modifiers['GeometryNodes'].node_group = bpy.data.node_groups['Geometry Nodes']
    bpy.context.object.modifiers['GeometryNodes']['Input_2'] = bpy.data.objects['occlusion_voxel']
    bpy.ops.geometry.color_attribute_add(name="color", color=(1.0, 0, 0.0, 1))
    player_floor = noise[chunk_size*radius][chunk_size*radius]
    ob.location.z = -player_floor*block_size
    ob.scale = Vector((block_size, block_size, block_size))
    for i, cd in enumerate(bpy.context.active_object.data.attributes['color'].data):
        cd.color = colors[i]
    with open('data/voxel_scene_{}'.format(d), 'w+', encoding='utf-8') as vs:
        for i in range(len(verts)):
            vs.write("{}/{}/{}\n".format(str(i), str(rgb[i]), str(int((verts[i].z-player_floor)))))
    return ob

def spawn_chunks(radius, noise):
    terrain = []
    verts = []
    for y in range(0, radius*2):
        for x in range(0, radius*2):
            chunk_origin = Vector((x*chunk_size, y*chunk_size, 0.0))
            spawn_voxel_heightmap(noise, verts, chunk_origin.x, chunk_origin.y, chunk_size, chunk_size, 1.0)
    voxel_types = ['dirt', 'grass', 'sand', 'stone']
    voxel_verts = {}
    for vt in voxel_types:
        voxel_verts[vt] = []
    for vert in verts:
        voxel_verts[voxel_types[random.randint(0, len(voxel_types)-1)]].append(vert)
    for i, vt in enumerate(voxel_types):
        mesh = bpy.data.meshes.new(vt + "voxel_geo")
        ob = bpy.data.objects.new(vt + "_placer", mesh)
        bpy.context.collection.objects.link(ob)
        ob.location = Vector((0,0,0))
        mesh.from_pydata(voxel_verts[vt], [], [])
        bpy.context.view_layer.objects.active = ob
        bpy.ops.object.modifier_add(type='NODES')
        bpy.context.object.modifiers['GeometryNodes'].node_group = bpy.data.node_groups['Geometry Nodes.001']
        bpy.context.object.modifiers['GeometryNodes']['Input_2'] = bpy.data.objects[vt + '_voxel']
        ob.scale = Vector((block_size, block_size, block_size))
    
        player_floor = noise[chunk_size*radius][chunk_size*radius]
        ob.location.z = -player_floor*block_size
        terrain.append(ob)
    bpy.data.objects["Camera"].location.x = chunk_size*radius*block_size
    bpy.data.objects["Camera"].location.y = chunk_size*radius*block_size
    return terrain
        
bpy.context.scene.render.image_settings.color_depth = "16"
bpy.context.scene.render.image_settings.compression = 0

dataset_size = 32
for d in range(dataset_size):
    # Recreate noise layers for random seeds!
    for layer in noise_layers:
        layer.set_random_seed() # TODO: Use a deterministic seed chain for replicability 
    # Make voxels random sizes to make the prediction robust to distinct voxel grid sizes
    # 0.1 to 0.5
    block_size = max(0.1, random.random()/2.0)
#    block_size = 0.2
    bpy.data.objects["Camera"].location.z = block_size * 1.8

    bpy.context.scene.view_settings.view_transform = 'Filmic'
    # Rotate camera randomly (always pointing forward to ensure reward is met)
    bpy.data.objects["Camera"].rotation_euler.x = math.radians(random.randint(45, 115))
    bpy.data.objects["Camera"].rotation_euler.z = math.radians(random.randint(-44, 44))
    # Spawn random terrain
    radius = 2
    noise = np.zeros(shape=[chunk_size*radius*2, chunk_size*radius*2])
    for layer in noise_layers:
        opensimplex.seed(layer.seed)
        ix, iy = np.array([x * layer.scale for x in range(chunk_size*radius*2)]), np.array([y * layer.scale for y in range(chunk_size*radius*2)])
        noise = np.add(noise, opensimplex.noise2array(ix, iy)*layer.multiplier)
    # Render voxel terrain
    terrain = spawn_chunks(radius, noise)
    bpy.context.scene.render.filepath = "data/voxels_%d.png" % d
    bpy.ops.render.render(write_still = True)
    for ob in terrain:
        bpy.data.objects.remove(ob, do_unlink=True)
    # Destroy textured voxels
    # Render occlusion terrain
    occlusion = spawn_occlusion_chunks(radius, noise, d)
    bpy.context.scene.view_settings.view_transform = 'Raw'
    bpy.context.scene.render.filepath = "data/occlusion_elements_%d.png" % d
    bpy.ops.render.render(write_still = True)
    bpy.data.objects.remove(occlusion, do_unlink=True)