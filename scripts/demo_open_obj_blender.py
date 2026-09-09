# Open TripoSR OBJ in Blender with Gradio display orientation
# (Rx -90°, scale Z -1, Ry +90°) so vertex colors match upright view.
import math
import sys
from pathlib import Path

import bpy
from mathutils import Matrix


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for block in bpy.data.meshes:
        if block.users == 0:
            bpy.data.meshes.remove(block)


def import_obj(path: Path):
    # Blender 4+/5: wm.obj_import; older: import_scene.obj
    if hasattr(bpy.ops.wm, "obj_import"):
        bpy.ops.wm.obj_import(filepath=str(path))
    else:
        bpy.ops.import_scene.obj(filepath=str(path))


def gradio_orientation(obj):
    # Match TripoSR `to_gradio_3d_orientation` in tsr/utils.py:
    # Rx(-90°) then Ry(+90°).
    rx = Matrix.Rotation(-math.pi / 2.0, 4, "X")
    ry = Matrix.Rotation(math.pi / 2.0, 4, "Y")
    obj.matrix_world = ry @ rx @ obj.matrix_world
    bpy.context.view_layer.update()


def frame_view():
    for area in bpy.context.screen.areas:
        if area.type != "VIEW_3D":
            continue
        for region in area.regions:
            if region.type != "WINDOW":
                continue
            with bpy.context.temp_override(area=area, region=region):
                bpy.ops.view3d.view_all(center=False)
            return


def main():
    argv = sys.argv
    if "--" not in argv:
        raise SystemExit("usage: blender --python demo_open_obj_blender.py -- <mesh.obj>")
    path = Path(argv[argv.index("--") + 1]).resolve()
    if not path.is_file():
        raise SystemExit(f"OBJ not found: {path}")

    clear_scene()
    import_obj(path)
    for obj in bpy.context.selected_objects:
        if obj.type == "MESH":
            gradio_orientation(obj)
            # Prefer solid vertex-color viewport shading
            for mat in obj.data.materials:
                if mat is None:
                    continue
                mat.use_nodes = True
    frame_view()
    # Viewport: Material Preview shows vertex colors better than Solid gray
    for area in bpy.context.screen.areas:
        if area.type == "VIEW_3D":
            for space in area.spaces:
                if space.type == "VIEW_3D":
                    space.shading.type = "SOLID"
                    space.shading.color_type = "VERTEX"


if __name__ == "__main__":
    main()
