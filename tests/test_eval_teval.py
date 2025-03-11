import sys
import types
import importlib
import pytest
from contextlib import nullcontext

def setup_dummy_modules(valid=True):
    """
    setup_dummy_modules(valid=True)
    Setup dummy modules for mmengine.config and opencompass configs.
    If valid is True, all dummy models have a valid meta_template.round.
    If valid is False, we simulate an invalid model that misses meta_template.
    """
    # Create a dummy mmengine.config module with read_base as a no-op context manager.
    dummy_mmengine_config = types.ModuleType("mmengine.config")
    dummy_mmengine_config.read_base = lambda: nullcontext()
    sys.modules["mmengine.config"] = dummy_mmengine_config

    # Create dummy dataset modules.
    dummy_teval_en = types.ModuleType("opencompass.configs.datasets.teval.teval_en_gen_1ac254")
    dummy_teval_en.teval_datasets = [{"name": "dummy_en_dataset"}]
    sys.modules["opencompass.configs.datasets.teval.teval_en_gen_1ac254"] = dummy_teval_en

    dummy_teval_zh = types.ModuleType("opencompass.configs.datasets.teval.teval_zh_gen_1ac254")
    dummy_teval_zh.teval_datasets = [{"name": "dummy_zh_dataset"}]
    sys.modules["opencompass.configs.datasets.teval.teval_zh_gen_1ac254"] = dummy_teval_zh

    # Create dummy model modules.
    # For qwen-7b-chat-hf: if valid==True, we provide a valid HUMAN round.
    dummy_qwen = types.ModuleType("opencompass.configs.models.qwen.hf_qwen_7b_chat")
    if valid:
        dummy_qwen.models = [{
            "abbr": "qwen-7b-chat-hf",
            "meta_template": {"round": [{"role": "HUMAN", "begin": "qwen_human_begin", "end": "qwen_human_end"}]}
        }]
    else:
        # Simulate missing meta_template key to trigger ValueError.
        dummy_qwen.models = [{"abbr": "qwen-7b-chat-hf"}]
    sys.modules["opencompass.configs.models.qwen.hf_qwen_7b_chat"] = dummy_qwen

    # For internlm2-chat-7b-hf: valid dummy model with a HUMAN round.
    dummy_internlm = types.ModuleType("opencompass.configs.models.hf_internlm.hf_internlm2_chat_7b")
    dummy_internlm.models = [{
        "abbr": "internlm2-chat-7b-hf",
        "meta_template": {"round": [{"role": "HUMAN", "begin": "internlm_human_begin", "end": "internlm_human_end"}]}
    }]
    sys.modules["opencompass.configs.models.hf_internlm.hf_internlm2_chat_7b"] = dummy_internlm

    # For llama-2-7b-chat-hf: already has a SYSTEM round; no change is needed.
    dummy_llama = types.ModuleType("opencompass.configs.models.hf_llama.hf_llama2_7b_chat")
    dummy_llama.models = [{
        "abbr": "llama-2-7b-chat-hf",
        "meta_template": {"round": [
            {"role": "HUMAN", "begin": "llama_human_begin", "end": "llama_human_end"},
            {"role": "SYSTEM", "begin": "llama_system_begin", "end": "llama_system_end"}
        ]}
    }]
    sys.modules["opencompass.configs.models.hf_llama.hf_llama2_7b_chat"] = dummy_llama

    # Also, add dummy modules required by the 'with read_base()' block in eval_teval.
    # (They were already added above for mmengine.config.)

@pytest.fixture(autouse=True)
def clean_eval_teval_module():
    # Ensure that the eval_teval module is removed from sys.modules so that our dummy modules are used fresh.
    yield
    sys.modules.pop("configs.eval_teval", None)

def test_eval_teval_valid(capsys):
    """
    Test eval_teval with dummy valid modules:
    - Checks that the models list is correctly patched: for models without an existing SYSTEM round,
        a SYSTEM round is appended.
    - Verifies that datasets concatenation and work_dir are correctly set.
    """
    setup_dummy_modules(valid=True)
    # Import (or reload) the eval_teval module.
    import configs.eval_teval as eval_teval
    importlib.reload(eval_teval)

    # There should be 3 models in total.
    models = eval_teval.models
    assert len(models) == 3

    # Check the model 'internlm2-chat-7b-hf': since it is in meta_template_system_patches,
    # it should have a SYSTEM round appended (so 2 rounds in total).
    internlm_model = [m for m in models if m["abbr"] == "internlm2-chat-7b-hf"][0]
    assert len(internlm_model["meta_template"]["round"]) == 2
    # The appended round should exactly be the patch from meta_template_system_patches.
    appended_round = internlm_model["meta_template"]["round"][-1]
    expected_patch = {'role': 'SYSTEM', 'begin': '<|im_start|>system\n', 'end': '<|im_end|>\n'}
    assert appended_round == expected_patch

    # Check the model 'qwen-7b-chat-hf': not present in meta_template_system_patches, so it should clone a HUMAN round.
    qwen_model = [m for m in models if m["abbr"] == "qwen-7b-chat-hf"][0]
    assert len(qwen_model["meta_template"]["round"]) == 2
    # The appended round in qwen_model should have role SYSTEM (cloned from HUMAN).
    assert qwen_model["meta_template"]["round"][-1]["role"].upper() == "SYSTEM"

    # Check the model 'llama-2-7b-chat-hf': already had a SYSTEM round so no new round added.
    llama_model = [m for m in models if m["abbr"] == "llama-2-7b-chat-hf"][0]
    assert len(llama_model["meta_template"]["round"]) == 2

    # Verify datasets (should be the concatenation of en and zh datasets).
    datasets = eval_teval.datasets
    assert isinstance(datasets, list)
    assert len(datasets) == 2
    # Verify work_dir.
    assert eval_teval.work_dir == "./outputs/teval"

    # Optionally, capture and test printed outputs from the module initialization.
    captured = capsys.readouterr().out
    assert "model qwen-7b-chat-hf is using the following meta_template:" in captured

def test_eval_teval_invalid():
    """
    Test eval_teval loading when at least one dummy model is invalid (missing meta_template).
    Expect a ValueError.
    """
    setup_dummy_modules(valid=False)
    # Use importlib.reload to attempt to reimport the eval_teval module.
    with pytest.raises(ValueError) as excinfo:
        import configs.eval_teval as eval_teval_invalid
        importlib.reload(eval_teval_invalid)
    assert "no meta_template.round" in str(excinfo.value)
def test_eval_teval_missing_round_key():
    """Test eval_teval raising ValueError when a model's meta_template exists but is missing the 'round' key."""
    # Set up dummy modules with valid models first
    setup_dummy_modules(valid=True)

    # Override the qwen model with a meta_template missing the 'round' key
    import types
    dummy_qwen = types.ModuleType("opencompass.configs.models.qwen.hf_qwen_7b_chat")
    dummy_qwen.models = [{"abbr": "qwen-7b-chat-hf", "meta_template": {}}]
    import sys
    sys.modules["opencompass.configs.models.qwen.hf_qwen_7b_chat"] = dummy_qwen

    # Remove the cached eval_teval module if it exists
    sys.modules.pop("configs.eval_teval", None)

    import importlib
    with pytest.raises(ValueError) as excinfo:
        import configs.eval_teval as eval_teval
        importlib.reload(eval_teval)

    assert "no meta_template.round in qwen-7b-chat-hf" in str(excinfo.value)
def test_eval_teval_deepcopy(capsys):
    setup_dummy_modules(valid=True)
    """
    Test that the models in eval_teval are deep copies of the dummy models.
    Changing the processed models does not affect the original dummy models.
    """
    from configs import eval_teval
    import sys
    # Retrieve the original dummy internlm model from its module.
    dummy_internlm = sys.modules["opencompass.configs.models.hf_internlm.hf_internlm2_chat_7b"].models[0]
    # Retrieve the processed model from eval_teval.
    internlm_eval = [m for m in eval_teval.models if m["abbr"] == "internlm2-chat-7b-hf"][0]
    # They must be distinct objects since a deepcopy was performed.
    assert id(dummy_internlm) != id(internlm_eval)
    # Verify that the original dummy model’s meta_template was not modified.
    original_round_length = len(dummy_internlm.get("meta_template", {}).get("round", []))
    assert original_round_length == 1
    # In the processed model, a SYSTEM round should have been appended.
    assert len(internlm_eval["meta_template"]["round"]) == original_round_length + 1

def test_eval_teval_print_output_all_models(capsys):
    setup_dummy_modules(valid=True)
    """
    Test that the module initialization prints log messages for all models.
    Verifies that each model defined in the dummy modules logs its meta_template.
    """
    # Import the module to trigger its print statements.
    import configs.eval_teval as eval_teval
    import re
    captured = capsys.readouterr().out
    # These are the expected model abbreviations.
    model_abbrs = ["qwen-7b-chat-hf", "internlm2-chat-7b-hf", "llama-2-7b-chat-hf"]
    for abbr in model_abbrs:
        pattern = re.compile(rf"model {abbr} is using the following meta_template:")
        assert pattern.search(captured) is not None
def test_eval_teval_empty_round_error():
    """Test that an empty meta_template.round for a model not in meta_template_system_patches raises IndexError."""
    setup_dummy_modules(valid=True)
    import types, sys, importlib, pytest
    # Override qwen module with an empty round list.
    dummy_qwen = types.ModuleType("opencompass.configs.models.qwen.hf_qwen_7b_chat")
    dummy_qwen.models = [{"abbr": "qwen-7b-chat-hf", "meta_template": {"round": []}}]
    sys.modules["opencompass.configs.models.qwen.hf_qwen_7b_chat"] = dummy_qwen
    sys.modules.pop("configs.eval_teval", None)
    with pytest.raises(IndexError):
        import configs.eval_teval as mod
        importlib.reload(mod)

def test_eval_teval_round_not_list_error():
    """Test that a non-list meta_template.round (e.g., a string) raises a TypeError."""
    setup_dummy_modules(valid=True)
    import types, sys, importlib, pytest
    # Override qwen module with round as a non-list value.
    dummy_qwen = types.ModuleType("opencompass.configs.models.qwen.hf_qwen_7b_chat")
    dummy_qwen.models = [{"abbr": "qwen-7b-chat-hf", "meta_template": {"round": "not_a_list"}}]
    sys.modules["opencompass.configs.models.qwen.hf_qwen_7b_chat"] = dummy_qwen
    sys.modules.pop("configs.eval_teval", None)
    with pytest.raises(TypeError):
        import configs.eval_teval as mod
        importlib.reload(mod)