import argparse
import logging
import os
import uuid
from pathlib import Path

import numpy as np
import rembg
import torch
import trimesh
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from PIL import Image

from tsr.bake_texture import bake_texture
from tsr.system import TSR
from tsr.utils import remove_background, resize_foreground


def alpha_coverage(image):
    image = np.array(image)
    if image.shape[-1] != 4:
        return None
    alpha = image[:, :, 3]
    if not np.any(alpha > 0):
        return None
    ys, xs = np.where(alpha > 0)
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


def preprocess(input_image, do_remove_background, foreground_ratio, rembg_session):
    def fill_background(image):
        image = np.array(image).astype(np.float32) / 255.0
        image = image[:, :, :3] * image[:, :, 3:4] + (1 - image[:, :, 3:4]) * 0.5
        return Image.fromarray((image * 255.0).astype(np.uint8))

    if do_remove_background:
        image = input_image.convert("RGB")
        image = remove_background(image, rembg_session)
        image = resize_foreground(image, adaptive_foreground_ratio(image, foreground_ratio))
        return fill_background(image)

    image = input_image
    if image.mode == "RGBA" and image.getextrema()[3][0] < 255:
        image = resize_foreground(image, adaptive_foreground_ratio(image, foreground_ratio))
        image = fill_background(image)
    return image


def load_model(model_name_or_path: str, device: str):
    model = TSR.from_pretrained(
        model_name_or_path,
        config_name="config.yaml",
        weight_name="model.ckpt",
    )
    model.renderer.set_chunk_size(int(os.getenv("TRIPOSR_RENDERER_CHUNK_SIZE", "2048")))
    model.to(device)
    return model


def create_app(model, device: str, output_dir: Path):
    app = FastAPI(title="TripoSR API")
    artifacts_dir = output_dir.resolve()
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/artifacts", StaticFiles(directory=str(artifacts_dir)), name="artifacts")
    rembg_session = None

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/generate")
    async def generate(
        image: UploadFile = File(...),
        remove_bg: bool = Form(True),
        foreground_ratio: float = Form(0.85),
        mc_resolution: int = Form(256),
        bake_texture_output: bool = Form(False),
        model_save_format: str = Form("glb"),
    ):
        if model_save_format not in {"obj", "glb"}:
            raise HTTPException(status_code=400, detail="model_save_format must be obj or glb")

        input_image = Image.open(image.file).convert("RGBA")
        nonlocal rembg_session
        if remove_bg and rembg_session is None:
            rembg_session = rembg.new_session()
        processed = preprocess(input_image, remove_bg, foreground_ratio, rembg_session)
        scene_codes = model([processed], device=device)
        meshes = model.extract_mesh(scene_codes, not bake_texture_output, resolution=mc_resolution)

        job_id = uuid.uuid4().hex
        job_dir = artifacts_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        texture_url = None

        if bake_texture_output:
            bake_output = bake_texture(meshes[0], model, scene_codes[0], 2048)
            vertices = meshes[0].vertices[bake_output["vmapping"]]
            faces = bake_output["indices"]
            uvs = bake_output["uvs"]
            normals = meshes[0].vertex_normals[bake_output["vmapping"]]
            texture_path = job_dir / "texture.png"
            texture_image = Image.fromarray((bake_output["colors"] * 255.0).astype(np.uint8)).transpose(Image.FLIP_TOP_BOTTOM)
            texture_image.save(texture_path)
            texture_url = f"/artifacts/{job_id}/{texture_path.name}"

            mesh_path = job_dir / f"mesh.{model_save_format}"
            if model_save_format == "glb":
                visual = trimesh.visual.texture.TextureVisuals(uv=uvs, image=texture_image)
                textured_mesh = trimesh.Trimesh(
                    vertices=vertices,
                    faces=faces,
                    vertex_normals=normals,
                    visual=visual,
                    process=False,
                )
                textured_mesh.export(mesh_path, file_type="glb")
            else:
                import xatlas

                xatlas.export(str(mesh_path), vertices, faces, uvs, normals)
        else:
            mesh_path = job_dir / f"mesh.{model_save_format}"
            meshes[0].export(mesh_path)

        return {
            "job_id": job_id,
            "mesh_path": str(mesh_path),
            "mesh_url": f"/artifacts/{job_id}/{mesh_path.name}",
            "texture_url": texture_url,
        }

    return app


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="stabilityai/TripoSR", type=str)
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--host", default="0.0.0.0", type=str)
    parser.add_argument("--port", default=8000, type=int)
    parser.add_argument("--output-dir", default="output", type=str)
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    model = load_model(args.model, device)
    app = create_app(model, device, Path(args.output_dir))

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)
