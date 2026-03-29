#!/usr/bin/env python3
"""
Render STL/PLY files in a CAD-style shaded look using Blender (headless).

Usage:
  blender -b -P render_stls_blender.py -- \
    --stl-dir /path/to/stls \
    --output-dir /path/to/out \
    --size 512 \
    --overwrite \
    --num-objects 100
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import bpy
import mathutils


# -----------------------------
# Defaults (match your script)
# -----------------------------
# DEFAULT_STL_DIR = Path("/home/kevin/datasets/persam_real_coco/mesh")
# DEFAULT_OUTPUT_DIR = Path("/home/kevin/datasets/persam_real_coco/rendered_blender")
DEFAULT_STL_DIR = Path("/sata1/data/kevin/realworld_datasets/3d_printing_meshes/stls")
DEFAULT_OUTPUT_DIR = Path("/sata1/data/kevin/realworld_datasets/3d_printing_meshes/renders_2442_0323_test")

# Toggle mesh format (STL vs PLY)
USE_PLY = False
MESH_LABEL = "PLY" if USE_PLY else "STL"
MESH_GLOBS = ("*.ply", "*.PLY") if USE_PLY else ("*.stl", "*.STL")
DEFAULT_MESH_DIR = DEFAULT_STL_DIR

# Output filename base (keep legacy naming even when rendering PLY)
OUTPUT_BASE_SUFFIX = "stl_base"

# When rendering PLY, cache an STL conversion under this subdir of the mesh dir.
CACHE_STL_SUBDIR = "_stl_cache"
# Your camera radius
CAMERA_RADIUS = 2.0

# Material (similar to your pyrender baseColorFactor gray + high roughness)
BASE_COLOR_RGBA = (0.4, 0.4, 0.4, 1.0)
ROUGHNESS = 0.98
METALLIC = 0.05

# Background: you used CAD_BACKGROUND = [0,0,0,0] (transparent)
TRANSPARENT_BG = True

# Lights (angle, elevation_deg, intensity) — approximates your pyrender setup
LIGHT_SETUPS = [
    # (math.radians(30.0), 35.0, 1.0),
    # (math.radians(210.0), 35.0, 0.8),
    (math.radians(0.0), 80.0, 1.0),
    (math.radians(180.0), 80.0, 1.0),
]

# Soft camera-aligned fill light
CAMERA_LIGHT_ENABLED = True
CAMERA_LIGHT_ENERGY = 0.1
CAMERA_LIGHT_SIZE = 1.5

# Shadow control (soften / reduce)
SHADOWS_ENABLED = True
SUN_ANGLE_DEG = 1.0  # larger angle -> softer, less pronounced shadows

# Material AO (screen-space) to help cavity contrast in Eevee
MATERIAL_AO_ENABLED = True
MATERIAL_AO_DISTANCE = 1.35
MATERIAL_AO_STRENGTH = 1.6


# -----------------------------
# CLI
# -----------------------------
def parse_args() -> argparse.Namespace:
    # Blender passes its own args; everything after `--` is ours
    argv = []
    if "--" in os.sys.argv:
        argv = os.sys.argv[os.sys.argv.index("--") + 1 :]

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stl-dir", type=Path, default=DEFAULT_MESH_DIR)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--overwrite", default=True)
    p.add_argument("--num-objects", type=int, default=None)
    return p.parse_args(argv)


# -----------------------------
# Scene utilities
# -----------------------------
def clear_scene() -> None:
    # Delete all objects
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    # Purge orphan data blocks (optional, but helps batch runs)
    for _ in range(2):
        bpy.ops.outliner.orphans_purge(do_recursive=True)


def setup_render(size: int) -> None:
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = size
    scene.render.resolution_y = size
    scene.render.resolution_percentage = 100

    # Output RGBA
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.film_transparent = bool(TRANSPARENT_BG)

    # Eevee settings (fast, CAD-like)
    ee = scene.eevee
    # ee.use_soft_shadows = True
    # ee.shadow_cube_size = "1024"
    # ee.shadow_cascade_size = "1024"

    # Ambient occlusion handled in material nodes (see make_cad_material).

    # World background (only visible if film_transparent False)
    world = scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs[0].default_value = (0.9, 0.9, 0.9, 1.0)
        bg.inputs[1].default_value = 1.0

    # setup_freestyle()


def setup_freestyle() -> None:
    # Enable Freestyle line rendering for clear edges independent of shadows.
    scene = bpy.context.scene
    view_layer = bpy.context.view_layer
    scene.render.use_freestyle = True
    view_layer.use_freestyle = True

    fs = getattr(view_layer, "freestyle_settings", None)
    if fs is None:
        return

    # Create or reuse a line set and style.
    linesets = getattr(fs, "linesets", None)
    if linesets is None:
        return
    if linesets:
        lineset = linesets[0]
    else:
        lineset = linesets.new("LineSet")

    # Prefer a dedicated linestyle if supported.
    linestyle = None
    if bpy.data.linestyles:
        linestyle = bpy.data.linestyles[0]
    else:
        linestyle = bpy.data.linestyles.new("LineStyle")
    lineset.linestyle = linestyle

    # Select edge types for internal edges and contours.
    for attr in (
        "select_silhouette",
        "select_border",
        "select_crease",
        "select_edge_mark",
        "select_material_boundary",
        "select_contour",
        "select_external_contour",
        "select_intersection",
    ):
        setattr(lineset, attr, True)

    # Style: thin, solid black lines.
    if linestyle is not None:
        linestyle.color = (0.0, 0.0, 0.0)
        linestyle.thickness = 1.0


def ensure_camera() -> bpy.types.Object:
    cam_data = bpy.data.cameras.new("Camera")
    cam = bpy.data.objects.new("Camera", cam_data)
    bpy.context.collection.objects.link(cam)
    bpy.context.scene.camera = cam

    # Similar to pyrender yfov = pi/4
    cam_data.lens_unit = "FOV"
    cam_data.angle = math.pi / 4.0
    return cam


def make_cad_material(name: str = "CAD_Gray") -> bpy.types.Material:
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes.get("Principled BSDF")
    if bsdf is None:
        bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = BASE_COLOR_RGBA
    bsdf.inputs["Roughness"].default_value = float(ROUGHNESS)
    bsdf.inputs["Metallic"].default_value = float(METALLIC)
    # Dial down specular to make it more matte across Blender versions.
    # bsdf.inputs["Specular"].default_value = 0.01
    # bsdf.inputs["Specular IOR Level"].default_value = 0.01
    # bsdf.inputs["Clearcoat"].default_value = 0.0
    if MATERIAL_AO_ENABLED:
        nodes = nt.nodes
        links = nt.links
        rgb = nodes.new("ShaderNodeRGB")
        rgb.outputs["Color"].default_value = BASE_COLOR_RGBA
        ao = nodes.new("ShaderNodeAmbientOcclusion")
        ao.inputs["Distance"].default_value = float(MATERIAL_AO_DISTANCE)
        mix = nodes.new("ShaderNodeMixRGB")
        mix.blend_type = "MULTIPLY"
        mix.inputs["Fac"].default_value = float(MATERIAL_AO_STRENGTH)
        links.new(rgb.outputs["Color"], mix.inputs["Color1"])
        links.new(ao.outputs["Color"], mix.inputs["Color2"])
        links.new(mix.outputs["Color"], bsdf.inputs["Base Color"])
    # Double-sided look: in Blender you can disable backface culling
    mat.use_backface_culling = False
    return mat


def add_sun_light(name: str, direction: mathutils.Vector, intensity: float) -> bpy.types.Object:
    light_data = bpy.data.lights.new(name=name, type="SUN")
    light_data.energy = float(intensity)
    light_data.use_shadow = bool(SHADOWS_ENABLED)
    light_data.angle = math.radians(SUN_ANGLE_DEG)
    light_obj = bpy.data.objects.new(name=name, object_data=light_data)
    bpy.context.collection.objects.link(light_obj)

    # Point the sun in the given direction (world space):
    # In Blender, light looks along its -Z axis.
    direction = direction.normalized()
    rot = direction.to_track_quat("-Z", "Y").to_euler()
    light_obj.rotation_euler = rot
    return light_obj


def add_camera_light(cam: bpy.types.Object) -> bpy.types.Object:
    # Large area light parented to the camera for a soft fill from camera direction.
    light_data = bpy.data.lights.new(name="CamArea", type="AREA")
    light_data.energy = float(CAMERA_LIGHT_ENERGY)
    light_data.size = float(CAMERA_LIGHT_SIZE)
    light_data.use_shadow = True
    light_obj = bpy.data.objects.new(name="CamArea", object_data=light_data)
    bpy.context.collection.objects.link(light_obj)
    light_obj.parent = cam
    light_obj.location = mathutils.Vector((0.0, 0.0, 0.0))
    light_obj.rotation_euler = mathutils.Euler((0.0, 0.0, 0.0))
    return light_obj


def camera_pose_from_angle(angle: float, radius: float = 2.0, elevation_deg: float = 45.0):
    # Match your pyrender version (Z-up)
    elevation = math.radians(elevation_deg)
    x = radius * math.cos(angle) * math.cos(elevation)
    y = radius * math.sin(angle) * math.cos(elevation)
    z = radius * math.sin(elevation)

    pos = mathutils.Vector((x, y, z))
    target = mathutils.Vector((0.0, 0.0, 0.0))
    up = mathutils.Vector((0.0, 0.0, 1.0))

    forward = (target - pos).normalized()
    right = forward.cross(up).normalized()
    true_up = right.cross(forward).normalized()

    # Blender camera looks down -Z in its local frame; +Y is up in camera local.
    # Build a rotation matrix whose columns are local axes in world:
    # local X -> right, local Y -> up, local -Z -> forward  => local Z -> -forward
    rot = mathutils.Matrix((
        right,
        true_up,
        -forward,
    )).transposed()

    return pos, rot.to_euler()


def import_stl(path: Path) -> bpy.types.Object:
    # Blender 4/5 moved STL import to wm.stl_import; keep legacy fallback.
    if hasattr(bpy.ops.wm, "stl_import"):
        bpy.ops.wm.stl_import(filepath=str(path))
    else:
        if not hasattr(bpy.ops.import_mesh, "stl"):
            try:
                bpy.ops.preferences.addon_enable(module="io_mesh_stl")
            except Exception:
                pass
        if hasattr(bpy.ops.import_mesh, "stl"):
            bpy.ops.import_mesh.stl(filepath=str(path))
        else:
            raise RuntimeError("STL import operator not found (wm.stl_import or import_mesh.stl).")
    # imported objects are selected
    objs = [o for o in bpy.context.selected_objects if o.type == "MESH"]
    if not objs:
        raise RuntimeError(f"No mesh object imported from {path}")
    # If multiple, join them
    if len(objs) > 1:
        bpy.context.view_layer.objects.active = objs[0]
        bpy.ops.object.join()
        obj = bpy.context.view_layer.objects.active
    else:
        obj = objs[0]
    return obj


def import_ply(path: Path) -> bpy.types.Object:
    # Blender 4/5 uses wm.ply_import; keep legacy fallback.
    if hasattr(bpy.ops.wm, "ply_import"):
        bpy.ops.wm.ply_import(filepath=str(path))
    else:
        if not hasattr(bpy.ops.import_mesh, "ply"):
            try:
                bpy.ops.preferences.addon_enable(module="io_mesh_ply")
            except Exception:
                pass
        if hasattr(bpy.ops.import_mesh, "ply"):
            bpy.ops.import_mesh.ply(filepath=str(path))
        else:
            raise RuntimeError("PLY import operator not found (wm.ply_import or import_mesh.ply).")
    # imported objects are selected
    objs = [o for o in bpy.context.selected_objects if o.type == "MESH"]
    if not objs:
        raise RuntimeError(f"No mesh object imported from {path}")
    # If multiple, join them
    if len(objs) > 1:
        bpy.context.view_layer.objects.active = objs[0]
        bpy.ops.object.join()
        obj = bpy.context.view_layer.objects.active
    else:
        obj = objs[0]
    return obj


def export_stl(obj: bpy.types.Object, path: Path) -> None:
    # Ensure only this object is selected for export.
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    path.parent.mkdir(parents=True, exist_ok=True)

    if hasattr(bpy.ops.wm, "stl_export"):
        try:
            bpy.ops.wm.stl_export(filepath=str(path), use_selection=True)
        except TypeError:
            bpy.ops.wm.stl_export(filepath=str(path))
    else:
        if not hasattr(bpy.ops.export_mesh, "stl"):
            try:
                bpy.ops.preferences.addon_enable(module="io_mesh_stl")
            except Exception:
                pass
        if hasattr(bpy.ops.export_mesh, "stl"):
            bpy.ops.export_mesh.stl(filepath=str(path), use_selection=True)
        else:
            raise RuntimeError("STL export operator not found (wm.stl_export or export_mesh.stl).")


def ensure_cached_stl(ply_path: Path, cache_dir: Path) -> Path:
    cache_path = cache_dir / f"{ply_path.stem}.stl"
    if cache_path.exists() and cache_path.stat().st_mtime >= ply_path.stat().st_mtime:
        return cache_path

    obj = import_ply(ply_path)
    export_stl(obj, cache_path)

    # Remove the temporary PLY mesh objects.
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.ops.object.delete(use_global=False)

    return cache_path


def normalize_object(obj: bpy.types.Object) -> None:
    # Apply transforms to mesh data first
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=False)

    # Center by geometry centroid and scale to max extent = 1
    # Use evaluated mesh bounds in local space
    mesh = obj.data
    coords = [v.co for v in mesh.vertices]
    if not coords:
        return
    min_v = mathutils.Vector((min(v.x for v in coords), min(v.y for v in coords), min(v.z for v in coords)))
    max_v = mathutils.Vector((max(v.x for v in coords), max(v.y for v in coords), max(v.z for v in coords)))
    center = (min_v + max_v) * 0.5
    extents = mathutils.Vector((
        max_v.x - min_v.x,
        max_v.y - min_v.y,
        max_v.z - min_v.z,
    ))
    extent = max(extents.x, extents.y, extents.z)
    if extent <= 0:
        extent = 1.0

    # Translate vertices so centroid is at origin
    for v in mesh.vertices:
        v.co -= center

    # Orient largest bbox face "up" => align smallest extent axis to +Z.
    min_axis = min(range(3), key=lambda i: extents[i])
    if min_axis == 0:
        rot = mathutils.Matrix.Rotation(math.radians(90.0), 3, "Y")
    elif min_axis == 1:
        rot = mathutils.Matrix.Rotation(math.radians(-90.0), 3, "X")
    else:
        rot = None
    if rot is not None:
        for v in mesh.vertices:
            v.co = rot @ v.co

    # Scale vertices so max extent becomes 1.0
    scale = 1.0 / extent
    for v in mesh.vertices:
        v.co *= scale

    mesh.update()


def assign_material(obj: bpy.types.Object, mat: bpy.types.Material) -> None:
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)


def set_smooth_shading(obj: bpy.types.Object, smooth: bool = True, autosmooth_angle_deg: float = 30.0) -> None:
    # Smooth shading similar to pyrender smooth=True
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.shade_smooth() if smooth else bpy.ops.object.shade_flat()

    # Blender 5+ uses the operator for angle-based auto smooth.
    bpy.ops.object.shade_auto_smooth(angle=math.radians(autosmooth_angle_deg))


def render_to(path: Path) -> None:
    bpy.context.scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)


def make_mask_from_alpha(rgba_path: Path, mask_path: Path) -> None:
    # Use Blender compositor nodes: faster than PIL? But simplest is PIL.
    from PIL import Image

    im = Image.open(rgba_path).convert("RGBA")
    arr = im.load()
    w, h = im.size

    mask = Image.new("RGB", (w, h), (0, 0, 0))
    mpx = mask.load()
    for y in range(h):
        for x in range(w):
            a = arr[x, y][3]
            if a > 0:
                mpx[x, y] = (255, 0, 0)
    mask.save(mask_path)


# -----------------------------
# Main batch render
# -----------------------------
def main() -> int:
    args = parse_args()
    mesh_dir = args.stl_dir.expanduser()
    out_dir = args.output_dir.expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    meshes = []
    for pattern in MESH_GLOBS:
        meshes.extend(sorted(mesh_dir.glob(pattern)))
    if args.num_objects is not None:
        meshes = meshes[: max(0, args.num_objects)]
    if not meshes:
        print(f"[warn] No {MESH_LABEL} files found in {mesh_dir}")
        return 0

    # Static scene setup
    clear_scene()
    setup_render(args.size)
    cam = ensure_camera()
    mat = make_cad_material()

    # Lights
    # Convert your light poses to directions (lights are directional in world).
    # In pyrender you place a directional light at a pose; here we aim by direction from origin.
    for i, (angle, elev, intensity) in enumerate(LIGHT_SETUPS):
        pos, _ = camera_pose_from_angle(angle, radius=2.5, elevation_deg=elev)
        # Light points from its position toward origin => direction is (origin - pos)
        direction = (mathutils.Vector((0.0, 0.0, 0.0)) - pos)
        add_sun_light(f"Sun_{i}", direction=direction, intensity=intensity * 0.5)  # scale up a bit for Eevee
    if CAMERA_LIGHT_ENABLED:
        add_camera_light(cam)

    # Preset views (same as your script)
    layer_specs = [
        (80.0, 2),
        (30.0, 4),
        (-30.0, 4),
        (-80.0, 2),
    ]
    preset_views: list[tuple[float, float]] = []
    for layer_idx, (elevation, num_views) in enumerate(layer_specs):
        start_deg = (layer_idx * 45.0) % 360.0
        angle_step = 360.0 / max(1, num_views)
        for view_idx in range(num_views):
            angle_deg = (start_deg + angle_step * view_idx) % 360.0
            preset_views.append((math.radians(angle_deg), elevation))

    cache_dir = mesh_dir / CACHE_STL_SUBDIR if USE_PLY else None

    for mesh_path in meshes:
        stem = mesh_path.stem
        print(f"[info] rendering {mesh_path.name}")

        if not args.overwrite:
            all_done = True
            for idx, _ in enumerate(preset_views):
                suffix = f"{OUTPUT_BASE_SUFFIX}_{idx:02d}"
                out_path = out_dir / f"{stem}_{suffix}.png"
                mask_path = out_dir / f"{stem}_{suffix}_mask.png"
                if not (out_path.exists() and mask_path.exists()):
                    all_done = False
                    break
            if all_done:
                continue

        # Remove previous mesh objects but keep camera/lights
        for obj in list(bpy.data.objects):
            if obj.type == "MESH":
                bpy.data.objects.remove(obj, do_unlink=True)

        try:
            load_path = ensure_cached_stl(mesh_path, cache_dir) if USE_PLY else mesh_path
            obj = import_stl(load_path)
        except Exception as e:
            print(f"[warn] failed to import {mesh_path}: {e}")
            continue

        normalize_object(obj)
        assign_material(obj, mat)
        set_smooth_shading(obj, smooth=True, autosmooth_angle_deg=50.0)

        for idx, (angle, elevation) in enumerate(preset_views):
            suffix = f"{OUTPUT_BASE_SUFFIX}_{idx:02d}"
            out_path = out_dir / f"{stem}_{suffix}.png"
            mask_path = out_dir / f"{stem}_{suffix}_mask.png"

            if out_path.exists() and mask_path.exists() and (not args.overwrite):
                continue

            cam_loc, cam_rot = camera_pose_from_angle(
                angle,
                radius=CAMERA_RADIUS,
                elevation_deg=elevation,
            )
            cam.location = cam_loc
            cam.rotation_euler = cam_rot

            render_to(out_path)
            make_mask_from_alpha(out_path, mask_path)

            print(f"[info] saved {out_path.name}")
            print(f"[info] saved {mask_path.name}")

    print(f"[done] Rendered {len(meshes)} {MESH_LABEL} meshes to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
