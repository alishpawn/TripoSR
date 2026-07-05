import argparse
import logging
import os
import time

import numpy as np
import rembg
import torch
import trimesh
import xatlas
from PIL import Image

from tsr.system import TSR
from tsr.utils import (
    clean_foreground_alpha,
    limit_image_size,
    remove_background,
    prepare_mesh_for_ar,
    prepare_normals_for_ar,
    prepare_vertices_for_ar,
    infer_ar_orientation,
    resize_foreground,
    save_video,
    to_gradio_3d_orientation,
    to_gradio_3d_orientation_arrays,
)
from tsr.bake_texture import bake_texture, create_textured_visual


class Timer:
    def __init__(self):
        self.items = {}
        self.time_scale = 1000.0  # ms
        self.time_unit = "ms"

    def start(self, name: str) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.items[name] = time.time()
        logging.info(f"{name} ...")

    def end(self, name: str) -> float:
        if name not in self.items:
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = self.items.pop(name)
        delta = time.time() - start_time
        t = delta * self.time_scale
        logging.info(f"{name} finished in {t:.2f}{self.time_unit}.")


timer = Timer()


def alpha_coverage(image):
    image = np.array(image)
    if image.shape[-1] != 4:
        return None
    alpha = image[:, :, 3]
    if not np.any(alpha > 8):
        return None
    ys, xs = np.where(alpha > 8)
    height = ys.max() - ys.min() + 1
    width = xs.max() - xs.min() + 1
    return max(height / image.shape[0], width / image.shape[1])


def adaptive_foreground_ratio(image, foreground_ratio):
    coverage = alpha_coverage(image)
    if coverage is None:
        return foreground_ratio
    if coverage > 0.9:
        return min(foreground_ratio, 0.75)
    if coverage > 0.75:
        return min(foreground_ratio, 0.8)
    if coverage < 0.55:
        return max(foreground_ratio, 0.9)
    return foreground_ratio


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)
parser = argparse.ArgumentParser()
parser.add_argument("image", type=str, nargs="+", help="Path to input image(s).")
parser.add_argument(
    "--device",
    default="cuda:0",
    type=str,
    help="Device to use. If no CUDA-compatible device is found, will fallback to 'cpu'. Default: 'cuda:0'",
)
parser.add_argument(
    "--pretrained-model-name-or-path",
    default="stabilityai/TripoSR",
    type=str,
    help="Path to the pretrained model. Could be either a huggingface model id is or a local path. Default: 'stabilityai/TripoSR'",
)
parser.add_argument(
    "--chunk-size",
    default=int(os.getenv("TRIPOSR_RENDERER_CHUNK_SIZE", "2048")),
    type=int,
    help="Evaluation chunk size for surface extraction and rendering. Smaller chunk size reduces VRAM usage but increases computation time. 0 for no chunking. Default: 2048",
)
parser.add_argument(
    "--mc-resolution",
    default=int(os.getenv("TRIPOSR_MC_RESOLUTION", "256")),
    type=int,
    help="Marching cubes grid resolution. Default: 256"
)
parser.add_argument(
    "--no-remove-bg",
    action="store_true",
    help="If specified, the background will NOT be automatically removed from the input image, and the input image should be an RGB image with gray background and properly-sized foreground. Default: false",
)
parser.add_argument(
    "--foreground-ratio",
    default=0.85,
    type=float,
    help="Ratio of the foreground size to the image size. Only used when --no-remove-bg is not specified. Default: 0.85",
)
parser.add_argument(
    "--output-dir",
    default="output/",
    type=str,
    help="Output directory to save the results. Default: 'output/'",
)
parser.add_argument(
    "--model-save-format",
    default="obj",
    type=str,
    choices=["obj", "glb"],
    help="Format to save the extracted mesh. Default: 'obj'",
)
parser.add_argument(
    "--bake-texture",
    action="store_true",
    help="Bake a texture atlas for the extracted mesh, instead of vertex colors",
)
parser.add_argument(
    "--texture-resolution",
    default=2048,
    type=int,
    help="Texture atlas resolution, only useful with --bake-texture. Default: 2048"
)
parser.add_argument(
    "--texture-brightness",
    default=float(os.getenv("TRIPOSR_TEXTURE_BRIGHTNESS", "1.1")),
    type=float,
    help="Brightness multiplier for baked textures. Default: 1.1",
)
parser.add_argument(
    "--density-threshold",
    default=float(os.getenv("TRIPOSR_DENSITY_THRESHOLD", "25.0")),
    type=float,
    help="Surface density threshold. Increase for a thinner mesh; decrease for a fuller mesh. Default: 25",
)
parser.add_argument(
    "--min-component-area-ratio",
    default=float(os.getenv("TRIPOSR_MIN_COMPONENT_AREA_RATIO", "0.005")),
    type=float,
    help="Remove disconnected fragments smaller than this fraction of total surface area. Use 0 to disable. Default: 0.005",
)
parser.add_argument(
    "--ar-ready",
    action="store_true",
    help="Center, ground, and scale a GLB for Y-up AR placement",
)
parser.add_argument(
    "--ar-size-meters",
    default=float(os.getenv("TRIPOSR_AR_SIZE_METERS", "0.25")),
    type=float,
    help="Longest horizontal size of an AR-ready model in meters. Default: 0.25",
)
parser.add_argument(
    "--ar-orientation",
    choices=["auto", "flat", "upright"],
    default="auto",
    help="AR placement orientation. Auto lays top-down food flat and keeps front views upright. Default: auto",
)
parser.add_argument(
    "--render",
    action="store_true",
    help="If specified, save a NeRF-rendered video. Default: false",
)
args = parser.parse_args()
if not 1.0 <= args.density_threshold <= 100.0:
    parser.error("--density-threshold must be between 1 and 100")
if not 0.0 <= args.min_component_area_ratio <= 0.1:
    parser.error("--min-component-area-ratio must be between 0 and 0.1")
if not 0.01 <= args.ar_size_meters <= 10.0:
    parser.error("--ar-size-meters must be between 0.01 and 10")
if args.ar_ready and args.model_save_format != "glb":
    parser.error("--ar-ready requires --model-save-format glb")

output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)

device = args.device
if not torch.cuda.is_available():
    device = "cpu"

timer.start("Initializing model")
model = TSR.from_pretrained(
    args.pretrained_model_name_or_path,
    config_name="config.yaml",
    weight_name="model.ckpt",
)
model.renderer.set_chunk_size(args.chunk_size)
model.eval()
model.to(device)
timer.end("Initializing model")

timer.start("Processing images")
images = []
ar_orientations = []

if args.no_remove_bg:
    rembg_session = None
else:
    rembg_session = rembg.new_session()

for i, image_path in enumerate(args.image):
    os.makedirs(os.path.join(output_dir, str(i)), exist_ok=True)
    if args.no_remove_bg:
        image = np.array(limit_image_size(Image.open(image_path).convert("RGB")))
    else:
        image = remove_background(Image.open(image_path), rembg_session)
        image = clean_foreground_alpha(image)
        image = resize_foreground(image, adaptive_foreground_ratio(image, args.foreground_ratio))
        image = np.array(image).astype(np.float32) / 255.0
        image = image[:, :, :3] * image[:, :, 3:4] + (1 - image[:, :, 3:4]) * 0.5
        image = limit_image_size(Image.fromarray((image * 255.0).astype(np.uint8)))
        image.save(os.path.join(output_dir, str(i), "input.png"))
    images.append(image)
    ar_orientations.append(
        infer_ar_orientation(image) if args.ar_orientation == "auto" else args.ar_orientation
    )
timer.end("Processing images")

for i, image in enumerate(images):
    logging.info(f"Running image {i + 1}/{len(images)} ...")

    timer.start("Running model")
    with torch.inference_mode():
        scene_codes = model([image], device=device)
    timer.end("Running model")

    if args.render:
        timer.start("Rendering")
        render_images = model.render(scene_codes, n_views=30, return_type="pil")
        for ri, render_image in enumerate(render_images[0]):
            render_image.save(os.path.join(output_dir, str(i), f"render_{ri:03d}.png"))
        save_video(
            render_images[0], os.path.join(output_dir, str(i), "render.mp4"), fps=30
        )
        timer.end("Rendering")

    timer.start("Extracting mesh")
    meshes = model.extract_mesh(
        scene_codes,
        not args.bake_texture,
        resolution=args.mc_resolution,
        threshold=args.density_threshold,
        min_component_area_ratio=args.min_component_area_ratio,
    )
    timer.end("Extracting mesh")

    out_mesh_path = os.path.join(output_dir, str(i), f"mesh.{args.model_save_format}")
    if args.bake_texture:
        out_texture_path = os.path.join(output_dir, str(i), "texture.png")

        timer.start("Baking texture")
        bake_output = bake_texture(
            meshes[0],
            model,
            scene_codes[0],
            args.texture_resolution,
            args.texture_brightness,
        )
        timer.end("Baking texture")

        vertices = meshes[0].vertices[bake_output["vmapping"]]
        faces = bake_output["indices"]
        uvs = bake_output["uvs"]
        normals = meshes[0].vertex_normals[bake_output["vmapping"]]
        vertices, normals = to_gradio_3d_orientation_arrays(vertices, normals)
        if args.ar_ready:
            vertices = prepare_vertices_for_ar(vertices, args.ar_size_meters, ar_orientations[i])
            normals = prepare_normals_for_ar(normals, ar_orientations[i])
        texture_image = Image.fromarray(
            (bake_output["colors"] * 255.0).astype(np.uint8)
        ).transpose(Image.FLIP_TOP_BOTTOM)

        timer.start("Exporting mesh and texture")
        if args.model_save_format == "glb":
            visual = create_textured_visual(uvs, texture_image)
            textured_mesh = trimesh.Trimesh(
                vertices=vertices,
                faces=faces,
                vertex_normals=normals,
                visual=visual,
                process=False,
            )
            textured_mesh.export(out_mesh_path, file_type="glb")
        else:
            xatlas.export(out_mesh_path, vertices, faces, uvs, normals)
        texture_image.save(out_texture_path)
        timer.end("Exporting mesh and texture")
    else:
        timer.start("Exporting mesh")
        meshes[0] = to_gradio_3d_orientation(meshes[0])
        if args.ar_ready:
            meshes[0] = prepare_mesh_for_ar(
                meshes[0], args.ar_size_meters, ar_orientations[i]
            )
        meshes[0].export(out_mesh_path)
        timer.end("Exporting mesh")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
