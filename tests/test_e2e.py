from tuning_config_recommender.adapters import FMSAdapter


def test_FMSAdapter_execution_on_hf_model_and_datasets():
    fms_adapter = FMSAdapter(base_dir="tmp/fms_full")
    fms_adapter.execute(
        tuning_config={
            "model_name_or_path": "ibm-granite/granite-3.1-8b-base",
            "training_data_path": "tatsu-lab/alpaca",
            "tuning_strategy": "full",
        },
        compute_config={},
        accelerate_config={},
        data_config={},
        unique_tag="gpq12df-fms",
        paths={},
    )
