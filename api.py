import argparse
import asyncio
import logging
import os
import time
import uuid
from io import BytesIO
from pathlib import Path

import numpy as np
import rembg
import torch
import trimesh
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from PIL import Image, ImageOps, UnidentifiedImageError

from tsr.bake_texture import bake_texture, create_textured_visual
from tsr.system import TSR
from tsr.utils import (
    remove_background,
    resize_foreground,
    to_gradio_3d_orientation,
    to_gradio_3d_orientation_arrays,
)


DEFAULT_RENDERER_CHUNK_SIZE = int(os.getenv("TRIPOSR_RENDERER_CHUNK_SIZE", "2048"))
DEFAULT_MC_RESOLUTION = int(os.getenv("TRIPOSR_MC_RESOLUTION", "256"))
DEFAULT_TEXTURE_RESOLUTION = int(os.getenv("TRIPOSR_TEXTURE_RESOLUTION", "2048"))
DEFAULT_TEXTURE_BRIGHTNESS = float(os.getenv("TRIPOSR_TEXTURE_BRIGHTNESS", "1.1"))


def configure_runtime(cpu_threads: int):
    if cpu_threads > 0:
        torch.set_num_threads(cpu_threads)
        try:
            torch.set_num_interop_threads(max(1, min(cpu_threads, 2)))
        except RuntimeError:
            pass


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


def has_transparency(image):
    return image.mode == "RGBA" and image.getextrema()[3][0] < 255


def preprocess(input_image, do_remove_background, foreground_ratio, rembg_session):
    def fill_background(image):
        image = np.array(image).astype(np.float32) / 255.0
        image = image[:, :, :3] * image[:, :, 3:4] + (1 - image[:, :, 3:4]) * 0.5
        return Image.fromarray((image * 255.0).astype(np.uint8))

    image = ImageOps.exif_transpose(input_image)
    if has_transparency(image):
        image = resize_foreground(image, adaptive_foreground_ratio(image, foreground_ratio))
        return fill_background(image)

    if do_remove_background:
        image = remove_background(image.convert("RGB"), rembg_session)
        image = resize_foreground(image, adaptive_foreground_ratio(image, foreground_ratio))
        return fill_background(image)

    return image.convert("RGB")


def load_model(model_name_or_path: str, device: str):
    model = TSR.from_pretrained(
        model_name_or_path,
        config_name="config.yaml",
        weight_name="model.ckpt",
    )
    model.renderer.set_chunk_size(DEFAULT_RENDERER_CHUNK_SIZE)
    model.eval()
    model.to(device)
    return model


def resolve_device(device: str) -> str:
    return device if torch.cuda.is_available() else "cpu"


def create_app_from_env():
    device = resolve_device(os.getenv("TRIPOSR_DEVICE", "cuda:0"))
    configure_runtime(int(os.getenv("TRIPOSR_CPU_THREADS", "4")))
    model = load_model(os.getenv("TRIPOSR_MODEL", "stabilityai/TripoSR"), device)
    output_dir = Path(os.getenv("TRIPOSR_OUTPUT_DIR", "output"))
    return create_app(
        model,
        device,
        output_dir,
        max_concurrent_jobs=int(os.getenv("TRIPOSR_MAX_CONCURRENT_JOBS", "1")),
        queue_size=int(os.getenv("TRIPOSR_QUEUE_SIZE", "8")),
    )


def create_app(model, device: str, output_dir: Path, max_concurrent_jobs: int = 1, queue_size: int = 8):
    app = FastAPI(title="TripoSR API")
    artifacts_dir = output_dir.resolve()
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/artifacts", StaticFiles(directory=str(artifacts_dir)), name="artifacts")
    rembg_session = None
    queue_lock = asyncio.Lock()
    job_slots = asyncio.Semaphore(max_concurrent_jobs)
    queued_jobs = 0
    running_jobs = 0

    @app.get("/health")
    async def health():
        async with queue_lock:
            return {
                "status": "ok",
                "queued": queued_jobs,
                "running": running_jobs,
                "max_concurrent_jobs": max_concurrent_jobs,
                "queue_size": queue_size,
                "renderer_chunk_size": DEFAULT_RENDERER_CHUNK_SIZE,
                "default_mc_resolution": DEFAULT_MC_RESOLUTION,
                "default_texture_resolution": DEFAULT_TEXTURE_RESOLUTION,
                "default_texture_brightness": DEFAULT_TEXTURE_BRIGHTNESS,
                "device": device,
            }

    def generate_sync(
        image_bytes: bytes,
        remove_bg: bool,
        foreground_ratio: float,
        mc_resolution: int,
        bake_texture_output: bool,
        texture_resolution: int,
        texture_brightness: float,
        model_save_format: str,
    ):
        nonlocal rembg_session
        try:
            input_image = Image.open(BytesIO(image_bytes))
            input_image.load()
        except (UnidentifiedImageError, OSError) as exc:
            raise ValueError("uploaded file is not a valid image") from exc

        input_image = ImageOps.exif_transpose(input_image)
        if remove_bg and not has_transparency(input_image) and rembg_session is None:
            rembg_session = rembg.new_session()
        processed = preprocess(input_image, remove_bg, foreground_ratio, rembg_session)
        with torch.inference_mode():
            scene_codes = model([processed], device=device)
            meshes = model.extract_mesh(scene_codes, not bake_texture_output, resolution=mc_resolution)

        job_id = uuid.uuid4().hex
        job_dir = artifacts_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        processed_path = job_dir / "processed_input.png"
        processed.save(processed_path)

        texture_url = None

        if bake_texture_output:
            bake_output = bake_texture(
                meshes[0], model, scene_codes[0], texture_resolution, texture_brightness
            )
            vertices = meshes[0].vertices[bake_output["vmapping"]]
            faces = bake_output["indices"]
            uvs = bake_output["uvs"]
            normals = meshes[0].vertex_normals[bake_output["vmapping"]]
            vertices, normals = to_gradio_3d_orientation_arrays(vertices, normals)
            texture_path = job_dir / "texture.png"
            texture_image = Image.fromarray((bake_output["colors"] * 255.0).astype(np.uint8)).transpose(Image.FLIP_TOP_BOTTOM)
            texture_image.save(texture_path)
            texture_url = f"/artifacts/{job_id}/{texture_path.name}"

            mesh_path = job_dir / f"mesh.{model_save_format}"
            if model_save_format == "glb":
                visual = create_textured_visual(uvs, texture_image)
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
            meshes[0] = to_gradio_3d_orientation(meshes[0])
            meshes[0].export(mesh_path)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {
            "job_id": job_id,
            "mesh_path": str(mesh_path),
            "mesh_url": f"/artifacts/{job_id}/{mesh_path.name}",
            "processed_image_url": f"/artifacts/{job_id}/{processed_path.name}",
            "texture_url": texture_url,
        }

    @app.post("/generate")
    async def generate(
        image: UploadFile = File(...),
        remove_bg: bool = Form(True),
        foreground_ratio: float = Form(0.85),
        mc_resolution: int = Form(DEFAULT_MC_RESOLUTION),
        bake_texture_output: bool = Form(True),
        texture_resolution: int = Form(DEFAULT_TEXTURE_RESOLUTION),
        texture_brightness: float = Form(DEFAULT_TEXTURE_BRIGHTNESS),
        model_save_format: str = Form("glb"),
    ):
        if model_save_format not in {"obj", "glb"}:
            raise HTTPException(status_code=400, detail="model_save_format must be obj or glb")
        if not 0.5 <= foreground_ratio <= 1.0:
            raise HTTPException(status_code=400, detail="foreground_ratio must be between 0.5 and 1.0")
        if not 32 <= mc_resolution <= 320:
            raise HTTPException(status_code=400, detail="mc_resolution must be between 32 and 320")
        if not 256 <= texture_resolution <= 4096:
            raise HTTPException(status_code=400, detail="texture_resolution must be between 256 and 4096")
        if not 0.5 <= texture_brightness <= 2.0:
            raise HTTPException(status_code=400, detail="texture_brightness must be between 0.5 and 2.0")

        image_bytes = await image.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="image is empty")

        nonlocal queued_jobs, running_jobs
        async with queue_lock:
            if queued_jobs + running_jobs >= max_concurrent_jobs + queue_size:
                raise HTTPException(status_code=429, detail="generation queue is full; try again later")
            queued_jobs += 1

        queued_at = time.monotonic()
        acquired_slot = False
        try:
            async with job_slots:
                queue_wait_seconds = round(time.monotonic() - queued_at, 3)
                async with queue_lock:
                    queued_jobs -= 1
                    running_jobs += 1
                    acquired_slot = True
                try:
                    result = await run_in_threadpool(
                        generate_sync,
                        image_bytes,
                        remove_bg,
                        foreground_ratio,
                        mc_resolution,
                        bake_texture_output,
                        texture_resolution,
                        texture_brightness,
                        model_save_format,
                    )
                except ValueError as exc:
                    raise HTTPException(status_code=400, detail=str(exc)) from exc
                result["queue_wait_seconds"] = queue_wait_seconds
                return result
        finally:
            async with queue_lock:
                if acquired_slot and running_jobs > 0:
                    running_jobs -= 1
                elif not acquired_slot and queued_jobs > 0:
                    queued_jobs -= 1

    return app


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="stabilityai/TripoSR", type=str)
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--host", default="0.0.0.0", type=str)
    parser.add_argument("--port", default=8000, type=int)
    parser.add_argument("--output-dir", default="output", type=str)
    parser.add_argument("--cpu-threads", default=int(os.getenv("TRIPOSR_CPU_THREADS", "4")), type=int)
    parser.add_argument("--max-concurrent-jobs", default=int(os.getenv("TRIPOSR_MAX_CONCURRENT_JOBS", "1")), type=int)
    parser.add_argument("--queue-size", default=int(os.getenv("TRIPOSR_QUEUE_SIZE", "8")), type=int)
    parser.add_argument(
        "--workers",
        default=int(os.getenv("WEB_CONCURRENCY", "1")),
        type=int,
        help="Number of Uvicorn worker processes. Each worker loads its own model copy.",
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be 1 or greater")
    if args.max_concurrent_jobs < 1:
        parser.error("--max-concurrent-jobs must be 1 or greater")
    if args.queue_size < 0:
        parser.error("--queue-size must be 0 or greater")

    configure_runtime(args.cpu_threads)

    import uvicorn

    if args.workers > 1:
        os.environ["TRIPOSR_MODEL"] = args.model
        os.environ["TRIPOSR_DEVICE"] = args.device
        os.environ["TRIPOSR_OUTPUT_DIR"] = args.output_dir
        os.environ["TRIPOSR_CPU_THREADS"] = str(args.cpu_threads)
        os.environ["TRIPOSR_MAX_CONCURRENT_JOBS"] = str(args.max_concurrent_jobs)
        os.environ["TRIPOSR_QUEUE_SIZE"] = str(args.queue_size)
        uvicorn.run(
            "api:create_app_from_env",
            factory=True,
            host=args.host,
            port=args.port,
            workers=args.workers,
        )
    else:
        device = resolve_device(args.device)
        model = load_model(args.model, device)
        app = create_app(
            model,
            device,
            Path(args.output_dir),
            max_concurrent_jobs=args.max_concurrent_jobs,
            queue_size=args.queue_size,
        )
        uvicorn.run(app, host=args.host, port=args.port)
