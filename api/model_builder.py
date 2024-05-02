import shared.ml_config_core as ml_config_core


def build_model() -> ml_config_core.ModelTrainingResult:
    try:
        result = ml_config_core.ModelTrainingResult.load_serialize_model(
            model_key="LGBM_AUC_Base_Features",
            target_folder="../Notebooks/.production_models",
        )
    except:
        result = ml_config_core.ModelTrainingResult.load_serialize_model(
            model_key="LGBM_AUC_Base_Features",
            target_folder="Notebooks/.production_models",
        )
    return result


if __name__ == "__main__":
    build_model()
