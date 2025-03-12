import pytest
from opencompass.utils.datasets_info import DATASETS_MAPPING, DATASETS_URL

def test_datasets_mapping_keys_and_values():
    """Test that all entries in DATASETS_MAPPING adhere to the expected structure."""
    for key, value in DATASETS_MAPPING.items():
        # Each value must be a dictionary with at least 'ms_id', 'hf_id', and 'local'
        assert isinstance(value, dict), f"Value for key {key} is not a dict"
        assert 'local' in value, f"'local' key missing for {key}"
        assert 'ms_id' in value, f"'ms_id' key missing for {key}"
        assert 'hf_id' in value, f"'hf_id' key missing for {key}"

        # Optionally, some dataset entries have additional keys (e.g., 'om_id') 
        # Verify ms_id and hf_id are string if not None or empty string.
        if value['ms_id'] not in [None, ""]:
            assert isinstance(value['ms_id'], str), f"ms_id for {key} is not a string"
        if value['hf_id'] not in [None, ""]:
            assert isinstance(value['hf_id'], str), f"hf_id for {key} is not a string"

        # For the 'opencompass/gsm8k' dataset, check that the optional 'om_id' key exists
        if key == "opencompass/gsm8k":
            assert 'om_id' in value, f"'om_id' key missing for {key}"

def test_datasets_mapping_unique_keys():
    """Test that DATASETS_MAPPING has unique keys (ensuring duplicate keys are handled correctly)."""
    keys = list(DATASETS_MAPPING.keys())
    assert len(keys) == len(set(keys)), "There are duplicate keys in DATASETS_MAPPING"

def test_datasets_url_structure():
    """Test that each entry in DATASETS_URL has a valid URL and md5 checksum."""
    for key, data in DATASETS_URL.items():
        assert isinstance(data, dict), f"Value for URL key {key} is not a dict"
        assert 'url' in data, f"Missing 'url' key for {key}"
        assert data['url'].startswith("http://"), f"URL for {key} does not start with 'http://'"
        assert 'md5' in data, f"Missing 'md5' key for {key}"
        assert len(data['md5']) > 0, f"Empty md5 for {key}"

def test_specific_dataset():
    """Test that a specific dataset returns the expected local path."""
    mapping = DATASETS_MAPPING.get("opencompass/advglue-dev")
    assert mapping is not None, "'opencompass/advglue-dev' dataset is missing"
    assert mapping["local"] == "./data/adv_glue/dev_ann.json", "The local path does not match for 'opencompass/advglue-dev'"

def test_specific_dataset_url():
    """Test that a specific dataset URL entry has the correct md5 checksum."""
    url_data = DATASETS_URL.get("/longbenchv2")
    assert url_data is not None, "'/longbenchv2' dataset URL entry is missing"
    expected_md5 = "09b7e06e6f98c5cca8ad597b3d7b42f0"
    assert url_data["md5"] == expected_md5, f"Expected md5 '{expected_md5}' but got '{url_data['md5']}'"
def test_gsm8k_optional_key():
    """Test that the 'opencompass/gsm8k' dataset has the optional 'om_id' key with the expected value."""
    ds = DATASETS_MAPPING.get("opencompass/gsm8k")
    assert ds is not None, "Dataset 'opencompass/gsm8k' is missing"
    assert "om_id" in ds, "'om_id' key is missing for 'opencompass/gsm8k'"
    assert ds["om_id"] == "OpenCompass/gsm8k", f"Expected 'om_id' to be 'OpenCompass/gsm8k' but got {ds['om_id']}"

def test_advglue_none_ids():
    """Test that 'opencompass/advglue-dev' dataset has ms_id and hf_id set to None."""
    ds = DATASETS_MAPPING.get("opencompass/advglue-dev")
    assert ds is not None, "Dataset 'opencompass/advglue-dev' is missing"
    assert ds["ms_id"] is None, f"Expected ms_id to be None but got {ds['ms_id']}"
    assert ds["hf_id"] is None, f"Expected hf_id to be None but got {ds['hf_id']}"

def test_duplicate_keys_handling():
    """Test that duplicate keys in DATASETS_MAPPING result in a single unique key using the last value defined."""
    key = "opencompass/mmmlu_lite"
    ds = DATASETS_MAPPING.get(key)
    assert ds is not None, f"Dataset '{key}' is missing"
    # Because the same key is defined twice with identical content, we check that the 'local' field is as expected.
    assert ds["local"] == "./data/mmmlu_lite", f"Expected local path './data/mmmlu_lite' for key '{key}' but got {ds['local']}"

def test_non_existent_dataset_returns_none():
    """Test that requesting a non-existent dataset returns None."""
    ds = DATASETS_MAPPING.get("non_existent_dataset")
    assert ds is None, "Non-existent dataset should return None"

def test_dataset_url_md5_format():
    """Test that each md5 checksum value in DATASETS_URL is exactly 32 characters long."""
    for key, data in DATASETS_URL.items():
        md5_value = data.get("md5", "")
        assert isinstance(md5_value, str), f"md5 for {key} is not a string"
        assert len(md5_value) == 32, f"md5 for {key} is expected to be 32 characters long but got {len(md5_value)}"
def test_local_field_non_empty():
    """Test that each dataset's local field is a non-empty string."""
    for key, ds in DATASETS_MAPPING.items():
        local = ds.get("local", None)
        assert isinstance(local, str) and local.strip() != "", f"local field for {key} is empty or not a valid string"

def test_dataset_empty_ids():
    """Test that datasets with empty ms_id/hf_id are correctly set as empty strings (e.g., opencompass/korbench)."""
    ds_korbench = DATASETS_MAPPING.get("opencompass/korbench")
    assert ds_korbench is not None, "Missing dataset 'opencompass/korbench'"
    assert ds_korbench["ms_id"] == "", "ms_id for 'opencompass/korbench' should be an empty string"
    assert ds_korbench["hf_id"] == "", "hf_id for 'opencompass/korbench' should be an empty string"

def test_dataset_empty_ids_mmlu_pro():
    """Test that dataset 'opencompass/mmlu_pro' has empty ms_id/hf_id."""
    ds = DATASETS_MAPPING.get("opencompass/mmlu_pro")
    assert ds is not None, "Missing dataset 'opencompass/mmlu_pro'"
    assert ds["ms_id"] == "", "ms_id for 'opencompass/mmlu_pro' should be an empty string"
    assert ds["hf_id"] == "", "hf_id for 'opencompass/mmlu_pro' should be an empty string"

def test_dataset_url_md5_hex():
    """Test that each md5 checksum in DATASETS_URL is a valid lowercase hexadecimal string of 32 characters."""
    import re
    pattern = re.compile(r'^[0-9a-f]{32}$')
    for key, data in DATASETS_URL.items():
        md5_val = data.get("md5", "")
        assert pattern.match(md5_val), f"md5 for {key} is not a valid hexadecimal string: {md5_val}"