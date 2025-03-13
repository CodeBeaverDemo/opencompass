import pytest
from opencompass.utils.datasets_info import DATASETS_MAPPING, DATASETS_URL

def test_datasets_mapping_structure():
    """Test that DATASETS_MAPPING has proper structure and required fields."""
    assert isinstance(DATASETS_MAPPING, dict)
    for key, value in DATASETS_MAPPING.items():
        assert isinstance(key, str)
        assert isinstance(value, dict)
        # Check required keys in each mapping entry
        for required_key in ["ms_id", "hf_id", "local"]:
            assert required_key in value, f"Key {required_key} missing in {key}"
        # Check that the 'local' field is a string
        assert isinstance(value["local"], str)

def test_known_dataset_values():
    """Test specific known dataset entries in DATASETS_MAPPING."""
    advglue = DATASETS_MAPPING.get("opencompass/advglue-dev", {})
    assert advglue.get("local") == "./data/adv_glue/dev_ann.json", "AdvGlue dev dataset path mismatch"

    arc_test = DATASETS_MAPPING.get("opencompass/ai2_arc-test", {})
    expected_arc_test = "./data/ARC/ARC-c/ARC-Challenge-Test.jsonl"
    assert arc_test.get("local") == expected_arc_test, "ARC test dataset path mismatch"

def test_duplicate_dataset_keys():
    """Test that duplicate dataset keys (if any) are handled properly (last occurrence wins)."""
    # For the key "opencompass/mmmlu_lite", the dictionary should reflect the last definition.
    mmmlu = DATASETS_MAPPING.get("opencompass/mmmlu_lite")
    expected = {"ms_id": "", "hf_id": "", "local": "./data/mmmlu_lite"}
    assert mmmlu == expected, "Duplicate key 'opencompass/mmmlu_lite' did not get expected value"

def test_datasets_url_structure():
    """Test that DATASETS_URL has proper structure with 'url' and 'md5' keys."""
    assert isinstance(DATASETS_URL, dict)
    for key, value in DATASETS_URL.items():
        assert isinstance(key, str)
        assert isinstance(value, dict)
        for required_key in ["url", "md5"]:
            assert required_key in value, f"Key {required_key} missing in URL entry for {key}"
        # Check that the URL and md5 are strings
        assert isinstance(value["url"], str)
        assert isinstance(value["md5"], str)

def test_datasets_url_valid_md5():
    """Test that md5 values in DATASETS_URL are valid hex strings of expected length (32)."""
    for key, value in DATASETS_URL.items():
        md5_val = value.get("md5", "")
        # Check that md5 is exactly 32 hex digits
        assert len(md5_val) == 32, f"MD5 for {key} does not have length 32"
        try:
            int(md5_val, 16)
        except ValueError:
            pytest.fail(f"MD5 for {key} is not a valid hex string")
def test_optional_fields_in_datasets_mapping():
    """Test that optional fields in DATASETS_MAPPING are handled correctly.
    Specifically, check that ms_id and hf_id can be None, and that extra keys such as om_id are preserved.
    """
    # Check for the dataset with None values for ms_id and hf_id
    advglue = DATASETS_MAPPING.get("opencompass/advglue-dev")
    assert advglue["ms_id"] is None, "Expected ms_id to be None for advglue-dev"
    assert advglue["hf_id"] is None, "Expected hf_id to be None for advglue-dev"

    # Check that a dataset with an extra key (om_id) retains that key and its expected value.
    gsm = DATASETS_MAPPING.get("opencompass/gsm8k")
    assert "om_id" in gsm, "Expected key 'om_id' in gsm8k dataset"
    assert gsm["om_id"] == "OpenCompass/gsm8k", "Unexpected value for 'om_id' in gsm8k dataset"

def test_nonexistent_dataset_key():
    """Test that retrieving a non-existent dataset key returns None."""
    non_exist = DATASETS_MAPPING.get("opencompass/nonexistent_key")
    assert non_exist is None, "Expected None for a non-existent dataset key"

def test_datasets_url_http_prefix():
    """Test that each dataset URL in DATASETS_URL begins with 'http://' or 'https://'."""
    for key, value in DATASETS_URL.items():
        url_val = value.get("url", "")
        assert url_val.startswith("http://") or url_val.startswith("https://"), f"URL for {key} is invalid: {url_val}"
def test_dataset_local_path_format():
    """Test that all local paths in DATASETS_MAPPING are non-empty strings and seem like valid relative paths."""
    for key, value in DATASETS_MAPPING.items():
        local_path = value.get("local", "")
        assert local_path, f"Local path for {key} should not be empty"
        # Check that the local path starts with './' to ensure it's relative
        assert local_path.startswith("./"), f"Local path for {key} is not relative: {local_path}"

def test_datasets_mapping_field_types():
    """Test that the 'ms_id' and 'hf_id' fields, if not None, are strings."""
    for key, value in DATASETS_MAPPING.items():
        ms_id = value.get("ms_id")
        hf_id = value.get("hf_id")
        if ms_id is not None:
            assert isinstance(ms_id, str), f"ms_id for {key} is not a string"
        if hf_id is not None:
            assert isinstance(hf_id, str), f"hf_id for {key} is not a string"

def test_dataset_with_empty_ms_id_and_hf_id():
    """Test that datasets with empty ms_id and hf_id fields have empty strings and not None when expected."""
    # For key "opencompass/korbench", ms_id and hf_id are set as empty strings.
    korbench = DATASETS_MAPPING.get("opencompass/korbench", {})
    assert korbench.get("ms_id") == "", "Expected empty string for ms_id in korbench"
    assert korbench.get("hf_id") == "", "Expected empty string for hf_id in korbench"

def test_dataset_override_in_duplicate_keys():
    """Test that duplicate keys in DATASETS_MAPPING correctly override previous definitions."""
    # The key "opencompass/mmmlu_lite" should have the last defined value.
    mmmlu = DATASETS_MAPPING.get("opencompass/mmmlu_lite")
    expected_value = {"ms_id": "", "hf_id": "", "local": "./data/mmmlu_lite"}
    assert mmmlu == expected_value, "Duplicate key 'opencompass/mmmlu_lite' did not override correctly"