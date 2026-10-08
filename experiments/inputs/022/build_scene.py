import bpy
import math
import os
from mathutils import Vector

out = r'/home/mfk01/Projects/econ-systems/demo/experiments/inputs/022'
assets = r'/home/mfk01/Projects/econ-systems/demo/artifacts/blender_pose_sanity'
os.makedirs(out, exist_ok=True)
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)

scene = bpy.context.scene
available_engines = {item.identifier for item in scene.render.bl_rna.properties["engine"].enum_items}
scene.render.engine = "BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in available_engines else "BLENDER_EEVEE"
scene.render.resolution_x = 640
scene.render.resolution_y = 480
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "PNG"
scene.render.image_settings.color_mode = "RGB"
scene.render.filepath = os.path.join(out, "input.png")
scene.render.film_transparent = False
scene.view_settings.view_transform = "Standard"
scene.view_settings.look = "None"
scene.view_settings.exposure = 0.0
scene.view_settings.gamma = 1.0
bpy.context.preferences.filepaths.save_version = 0

world = bpy.data.worlds.new("Neutral background")
world.use_nodes = True
world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.52, 0.52, 0.52, 1.0)
world.node_tree.nodes["Background"].inputs["Strength"].default_value = 1.0
scene.world = world

camera_data = bpy.data.cameras.new("Sanity camera")
camera_data.lens = 500.0 * 36.0 / 640
camera_data.sensor_width = 36.0
camera_data.sensor_fit = "HORIZONTAL"
camera_data.clip_start = 0.01
camera_data.clip_end = 10.0
camera = bpy.data.objects.new("Sanity camera", camera_data)
scene.collection.objects.link(camera)
camera.location = (0.0, 0.0, 0.0)
camera.rotation_euler = (0.0, 0.0, 0.0)  # Blender cameras look along local -Z.
scene.camera = camera

def tag_material(image_path, name):
    image = bpy.data.images.load(image_path, check_existing=True)
    image.colorspace_settings.name = "Non-Color"
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    texture = nodes.new("ShaderNodeTexImage")
    texture.image = image
    texture.interpolation = "Closest"
    emission = nodes.new("ShaderNodeEmission")
    output = nodes.new("ShaderNodeOutputMaterial")
    material.node_tree.links.new(texture.outputs["Color"], emission.inputs["Color"])
    material.node_tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return material

specs = [(0, 0.095, (-0.1614, 0.0163, 0.8495), (0.3004, -0.084, 0.0738)), (4, 0.06725, (0.2015, -0.0752, 0.9888), (0.2465, -0.2226, 0.0889))]
for tag_id, size, cv_position in [(s[0], s[1], s[2]) for s in specs]:
    pass
for tag_id, size, cv_position, rot_xyz in specs:
    cv_x, cv_y, cv_z = cv_position
    # Blender world coordinates converted from OpenCV camera coordinates.
    center = (cv_x, -cv_y, -cv_z)
    h = size / 2.0
    mesh = bpy.data.meshes.new("AprilTag %d mesh" % tag_id)
    mesh.from_pydata([(-h,-h,0), (h,-h,0), (h,h,0), (-h,h,0)], [], [(0,1,2,3)])
    mesh.update()
    uv = mesh.uv_layers.new(name="UVMap")
    for loop_index, uv_coord in zip(mesh.polygons[0].loop_indices, [(0,0),(1,0),(1,1),(0,1)]):
        uv.data[loop_index].uv = uv_coord
    obj = bpy.data.objects.new("AprilTag ID %d" % tag_id, mesh)
    scene.collection.objects.link(obj)
    obj.location = center
    obj.rotation_euler = rot_xyz
    obj.data.materials.append(tag_material(os.path.join(assets, "tag_%d.png" % tag_id), "Tag %d emission" % tag_id))

bpy.ops.wm.save_as_mainfile(filepath=os.path.join(out, "scene.blend"))
bpy.ops.render.render(write_still=True)
