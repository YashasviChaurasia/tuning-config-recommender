from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import os
from loguru import logger
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from tuning_config_recommender.adapters import FMSAdapter
from pathlib import Path
from datetime import datetime, timezone
import uuid
import yaml
import asyncio

app = FastAPI(title="Recommender API")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def delete_files(file_paths: list[str]) -> None:
    await asyncio.sleep(600)
    for file_path in file_paths:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            logger.error(f"Failed to delete file {file_path}: {e}")

class RecommendationsRequest(BaseModel):
    tuning_config: Optional[dict] = None
    tuning_data_config: Optional[dict] = None
    compute_config: Optional[dict] = None
    accelerate_config: Optional[dict] = None
    skip_estimator: Optional[bool] = False

def generate_unique_stamps():
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    random_id = uuid.uuid4().hex[:8]
    return f"{timestamp}_{random_id}"


@app.post("/recommend")
async def recommend(
    background_tasks: BackgroundTasks,
    req: RecommendationsRequest,
):
    err_msg = (
        "Generation failed, please provide correct inputs or report it to the team!"
    )
    try:
        paths_to_delete = []
        base_dir = Path(__file__).parent
        output_dir = base_dir / "outputs" / generate_unique_stamps()

        fms_adapter = FMSAdapter(
            base_dir=output_dir, additional_actions=[]
        )

        fms_adapter.execute(
            train_config=req.tuning_config,
            compute_config=req.compute_config,
            dist_config=req.accelerate_config,
            data_config=req.tuning_data_config,
            unique_tag="",
            paths={},
            skip_estimator=req.skip_estimator,
        )
        response = {
            "tuning_config": yaml.safe_load(str(output_dir / "tuning_config.yaml")),
            "tuning_data_config": yaml.safe_load(str(output_dir / "data_config.yaml")),
            "compute_config": yaml.safe_load(str(output_dir / "compute_config.yaml")),
            "accelerate_config": yaml.safe_load(str(output_dir / "accelerate_config.yaml")),
            "paths": {
            "tuning_config": output_dir / "tuning_config.yaml",
            "tuning_data_config": output_dir / "data_config.yaml",
            "compute_config": output_dir / "compute_config.yaml",
            "accelerate_config": output_dir / "accelerate_config.yaml",
            }
        }
        paths_to_delete = [
            output_dir / "tuning_config.yaml",
            output_dir / "data_config.yaml",
            output_dir / "compute_config.yaml",
            output_dir / "accelerate_config.yaml"
        ]
        background_tasks.add_task(delete_files, paths_to_delete)
        return response
    except Exception as e:
        logger.error(e)
        return JSONResponse(
            status_code=500,
            content=jsonable_encoder({"message": err_msg}),
        )
