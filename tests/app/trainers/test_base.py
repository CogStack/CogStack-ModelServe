from trainers.base import TrainerCommon


def test_get_experiment_name():
    assert TrainerCommon.get_experiment_name("SNOMED model") == "SNOMED_model"
    assert TrainerCommon.get_experiment_name("SNOMED model", "unsupervised") == "SNOMED_model_unsupervised"
