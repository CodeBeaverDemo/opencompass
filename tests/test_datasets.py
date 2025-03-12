import os
import json
import tempfile
import pytest
from opencompass.utils import datasets
from opencompass.utils.datasets import get_data_path, download_dataset, DEFAULT_DATA_FOLDER

class DummyLogger:
    def info(self, msg):
        pass

class TestDatasets:
    """Test suite for opencompass.utils.datasets."""

    @pytest.fixture(autouse=True)
    def setup_env(self, monkeypatch):
        # Setup user home to a temporary directory to avoid affecting real user environment
        temp_home = tempfile.mkdtemp()
        monkeypatch.setenv("HOME", temp_home)
        monkeypatch.setenv("COMPASS_DATA_CACHE", os.path.join(temp_home, "cache"))
        yield
        # Cleanup temporary directory after tests
        import shutil
        shutil.rmtree(temp_home)

    @pytest.fixture(autouse=True)
    def dummy_logger(self, monkeypatch):
        monkeypatch.setattr(datasets, "get_logger", lambda: DummyLogger())

    def test_get_data_path_absolute(self):
        """Test get_data_path returns the absolute path as is when dataset_id starts with a forward slash."""
        abs_path = "/some/absolute/path"
        result = get_data_path(abs_path)
        assert result == abs_path

    def test_get_data_path_local_mode_exists(self, monkeypatch):
        """Test get_data_path in local mode when the local file already exists."""
        test_id = "local_dataset"
        cache_dir = os.environ.get("COMPASS_DATA_CACHE")
        local_path = os.path.join(cache_dir, test_id)
        # Create the file/directory to simulate existence
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        with open(local_path, "w") as f:
            f.write("test content")

        result = get_data_path(test_id, local_mode=True)
        assert result == local_path

    def test_get_data_path_local_mode_not_exists(self, monkeypatch):
        """Test get_data_path in local mode when the local file does not exist, triggering download."""
        test_id = "local_dataset_missing"
        cache_dir = os.environ.get("COMPASS_DATA_CACHE")
        local_path = os.path.join(cache_dir, test_id)

        # Ensure the file does not exist
        if os.path.exists(local_path):
            os.remove(local_path)

        # Monkey patch download_dataset to simulate download behavior
        called = False
        def dummy_download(data_path, cache_dir_inner, remove_finished=True):
            nonlocal called
            called = True
            # Simulate that after download, the file now exists by creating it.
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            with open(local_path, "w") as f:
                f.write("downloaded content")
            return local_path

        monkeypatch.setattr(datasets, "download_dataset", dummy_download)

        result = get_data_path(test_id, local_mode=True)
        assert called is True
        assert result == local_path

    def test_get_data_path_modelscope(self, monkeypatch):
        """Test get_data_path with dataset_source set to ModelScope to return ms_id."""
        test_id = "test_dataset"
        # Inject a fake dataset mapping
        fake_mapping = {
            test_id: {
                "ms_id": "modelscope_123",
                "om_id": "openmind_123",
                "hf_id": "hf_123",
                "local": "local_test_dataset"
            }
        }
        monkeypatch.setattr(datasets, "DATASETS_MAPPING", fake_mapping)
        monkeypatch.setenv("DATASET_SOURCE", "ModelScope")
        result = get_data_path(test_id)
        assert result == "modelscope_123"

    def test_get_data_path_openmind(self, monkeypatch):
        """Test get_data_path with dataset_source set to OpenMind to return om_id."""
        test_id = "test_dataset"
        fake_mapping = {
            test_id: {
                "ms_id": "modelscope_123",
                "om_id": "openmind_456",
                "hf_id": "hf_123",
                "local": "local_test_dataset"
            }
        }
        monkeypatch.setattr(datasets, "DATASETS_MAPPING", fake_mapping)
        monkeypatch.setenv("DATASET_SOURCE", "OpenMind")
        result = get_data_path(test_id)
        assert result == "openmind_456"

    def test_get_data_path_hf(self, monkeypatch):
        """Test get_data_path with dataset_source set to HF to return hf_id."""
        test_id = "test_dataset"
        fake_mapping = {
            test_id: {
                "ms_id": "modelscope_123",
                "om_id": "openmind_456",
                "hf_id": "hf_789",
                "local": "local_test_dataset"
            }
        }
        monkeypatch.setattr(datasets, "DATASETS_MAPPING", fake_mapping)
        monkeypatch.setenv("DATASET_SOURCE", "HF")
        result = get_data_path(test_id)
        assert result == "hf_789"

    def test_get_data_path_default_local(self, monkeypatch):
        """Test get_data_path when DATASET_SOURCE is not set, so it uses local path from mapping."""
        test_id = "test_dataset"
        fake_mapping = {
            test_id: {
                "ms_id": None,
                "om_id": None,
                "hf_id": None,
                "local": "local_test_dataset_default"
            }
        }
        monkeypatch.setattr(datasets, "DATASETS_MAPPING", fake_mapping)
        monkeypatch.delenv("DATASET_SOURCE", raising=False)
        cache_dir = os.environ.get("COMPASS_DATA_CACHE")
        local_path = os.path.join(cache_dir, fake_mapping[test_id]["local"])

        # Simulate file does not exist and trigger download
        if os.path.exists(local_path):
            os.remove(local_path)

        # Monkey patch download_dataset
        called = False
        def dummy_download(data_path, cache_dir_inner, remove_finished=True):
            nonlocal called
            called = True
            # simulate download by creating the file
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            with open(local_path, "w") as f:
                f.write("downloaded default local content")
            return local_path

        monkeypatch.setattr(datasets, "download_dataset", dummy_download)

        result = get_data_path(test_id)
        assert called is True
        assert result == local_path

    def test_download_dataset_default_cache_exists(self, monkeypatch):
        """Test download_dataset returns try_default_path if exists in DEFAULT_DATA_FOLDER."""
        # Create a fake data file in DEFAULT_DATA_FOLDER
        fake_data_path = "data/fake_dataset"
        default_path = os.path.join(DEFAULT_DATA_FOLDER, fake_data_path)
        os.makedirs(os.path.dirname(default_path), exist_ok=True)
        with open(default_path, "w") as f:
            f.write("cached data")

        result = download_dataset(fake_data_path, os.environ.get("COMPASS_DATA_CACHE"))
        assert result == default_path

    def test_download_dataset_download(self, monkeypatch, tmp_path):
        # Monkey patch DATASETS_URL to include a valid dataset mapping for the fake data_path
        monkeypatch.setattr(datasets, "DATASETS_URL", {"nonexistent_dataset": {"url": "http://example.com/dataset.zip", "md5": "fake_md5"}})
        """Test download_dataset triggers downloading when file not exist in DEFAULT_DATA_FOLDER."""
        # Ensure try_default_path does not exist
        fake_data_path = "data/nonexistent_dataset"
        default_path = os.path.join(DEFAULT_DATA_FOLDER, fake_data_path)
        if os.path.exists(default_path):
            os.remove(default_path)

        # Simulate internal dataset JSON not loading: force the internal JSON file check to fail
        original_exists = os.path.exists
        def fake_exists(path):
            # Make sure internal dataset json file not found
            if ".OPENCOMPASS_INTERNAL_DATA_URL.json" in path:
                return False
            return original_exists(path)
        monkeypatch.setattr(os.path, "exists", fake_exists)

        # Monkey patch download_and_extract_archive to simulate download
        downloaded = False
        def dummy_download_and_extract_archive(url, download_root, md5, remove_finished):
            nonlocal downloaded
            downloaded = True
        monkeypatch.setattr(datasets, "download_and_extract_archive", dummy_download_and_extract_archive)

        cache_dir = os.environ.get("COMPASS_DATA_CACHE")
        result = download_dataset(fake_data_path, cache_dir)
        # Check that download was triggered and result path is returned
        assert downloaded is True
        expected_path = os.path.join(cache_dir, fake_data_path)
        assert result == expected_path
    def test_get_data_path_openmind_key_error(self, monkeypatch):
        """Test get_data_path raises KeyError when OpenMind mapping key is missing."""
        test_id = "test_dataset"
        fake_mapping = {
            test_id: {
                "ms_id": "modelscope_123",
                # Intentionally omit "om_id" to trigger the KeyError.
                "hf_id": "hf_123",
                "local": "local_test_dataset"
            }
        }
        monkeypatch.setattr(datasets, "DATASETS_MAPPING", fake_mapping)
        monkeypatch.setenv("DATASET_SOURCE", "OpenMind")
        with pytest.raises(KeyError, match=f"{test_id} is not supported in OpenMind"):
            get_data_path(test_id)

    def test_download_dataset_invalid_url(self, monkeypatch):
        """Test download_dataset raises AssertionError when no valid dataset url can be determined."""
        fake_data_path = "data/invalid_dataset"
        cache_dir = os.environ.get("COMPASS_DATA_CACHE")
        # Ensure that there is no cached file in the DEFAULT_DATA_FOLDER
        invalid_default_path = os.path.join(DEFAULT_DATA_FOLDER, fake_data_path)
        if os.path.exists(invalid_default_path):
            os.remove(invalid_default_path)
        # Set DATASETS_URL to an empty dictionary to simulate no available URL.
        monkeypatch.setattr(datasets, "DATASETS_URL", {})
        with pytest.raises(AssertionError, match=f'No valid url for {fake_data_path}!'):
            download_dataset(fake_data_path, cache_dir)
    def test_get_data_path_default_local_exists(self, monkeypatch):
        """Test get_data_path with default local file already existing so that download is not triggered."""
        test_id = "test_dataset_exists"
        fake_mapping = {
            test_id: {
                "ms_id": None,
                "om_id": None,
                "hf_id": None,
                "local": "local_exists_dataset"
            }
        }
        monkeypatch.setattr(datasets, "DATASETS_MAPPING", fake_mapping)
        monkeypatch.delenv("DATASET_SOURCE", raising=False)
        cache_dir = os.environ.get("COMPASS_DATA_CACHE")
        local_path = os.path.join(cache_dir, fake_mapping[test_id]["local"])
        # Create the file to simulate its existence
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        with open(local_path, "w") as f:
            f.write("existing content")

        # Monkey patch download_dataset to detect if download is triggered
        download_called = False
        def dummy_download(data_path, cache_dir_inner, remove_finished=True):
            nonlocal download_called
            download_called = True
            return local_path
        monkeypatch.setattr(datasets, "download_dataset", dummy_download)

        result = get_data_path(test_id)
        assert not download_called, "download_dataset should not be called if file exists"
        assert result == local_path, "get_data_path should return the existing local path"

    def test_download_dataset_internal_json(self, monkeypatch, tmp_path):
        """Test download_dataset loads internal JSON and updates DATASETS_URL before triggering download."""
        # Create a temporary user home and set USER_HOME in the datasets module
        temp_home = tmp_path / "temp_home"
        temp_home.mkdir()
        monkeypatch.setattr(datasets, "USER_HOME", str(temp_home))

        # Create internal JSON file with dummy dataset mapping
        internal_file = temp_home / ".OPENCOMPASS_INTERNAL_DATA_URL.json"
        dummy_internal = {"internal_test": {"url": "http://example.com/internal.zip", "md5": "internal_md5"}}
        internal_file.write_text(json.dumps(dummy_internal))

        # Force DATASETS_URL to be initially empty so that the internal JSON can update it
        monkeypatch.setattr(datasets, "DATASETS_URL", {})

        # Use a fake data path that contains the internal dataset key
        fake_data_path = "data/internal_test"
        cache_dir = os.environ.get("COMPASS_DATA_CACHE")

        # Ensure the default cache file does not exist
        default_path = os.path.join(DEFAULT_DATA_FOLDER, fake_data_path)
        if os.path.exists(default_path):
            os.remove(default_path)

        # Monkey patch download_and_extract_archive to simulate the download process
        archive_called = False
        def dummy_download_and_extract_archive(url, download_root, md5, remove_finished):
            nonlocal archive_called
            archive_called = True
        monkeypatch.setattr(datasets, "download_and_extract_archive", dummy_download_and_extract_archive)

        result = download_dataset(fake_data_path, cache_dir)
        assert archive_called, "download_and_extract_archive should be called when file is missing"
        expected_path = os.path.join(cache_dir, fake_data_path)
        assert result == expected_path, "download_dataset should return the correct local path after download"
    def test_get_data_path_invalid_dataset_mapping_modelscope(self, monkeypatch):
        """Test get_data_path raises KeyError if dataset_id is not in DATASETS_MAPPING for ModelScope."""
        test_id = "nonexistent_dataset"
        monkeypatch.setattr(datasets, "DATASETS_MAPPING", {})
        monkeypatch.setenv("DATASET_SOURCE", "ModelScope")
        with pytest.raises(KeyError):
            get_data_path(test_id)

    def test_get_data_path_invalid_dataset_mapping_local(self, monkeypatch):
        """Test get_data_path raises KeyError if dataset_id is not in DATASETS_MAPPING for local mode."""
        test_id = "nonexistent_dataset"
        monkeypatch.setattr(datasets, "DATASETS_MAPPING", {})
        monkeypatch.delenv("DATASET_SOURCE", raising=False)
        with pytest.raises(KeyError):
            get_data_path(test_id)

    def test_get_data_path_hf_none(self, monkeypatch):
        """Test get_data_path raises AssertionError when hf_id is None for HF source."""
        test_id = "test_dataset"
        fake_mapping = {
            test_id: {
                "ms_id": "modelscope_123",
                "om_id": "openmind_456",
                "hf_id": None,
                "local": "local_test_dataset"
            }
        }
        monkeypatch.setattr(datasets, "DATASETS_MAPPING", fake_mapping)
        monkeypatch.setenv("DATASET_SOURCE", "HF")
        with pytest.raises(AssertionError, match=f'{test_id} is not supported in HuggingFace'):
            get_data_path(test_id)

    def test_download_dataset_internal_json_exception(self, monkeypatch, tmp_path):
        """Test download_dataset handles exception when loading internal JSON file."""
        # Set up a temporary user home and create an internal JSON file with invalid content.
        temp_home = tmp_path / "temp_home"
        temp_home.mkdir()
        monkeypatch.setattr(datasets, "USER_HOME", str(temp_home))
        internal_file = temp_home / ".OPENCOMPASS_INTERNAL_DATA_URL.json"
        internal_file.write_text("invalid json")
        # Set DATASETS_URL to a dummy mapping so that download can proceed.
        fake_datasets_url = {"dummy": {"url": "http://example.com/dummy.zip", "md5": "dummy_md5"}}
        monkeyatch_copy = fake_datasets_url.copy()  # Just to ensure a copy is used.
        monkeypatch.setattr(datasets, "DATASETS_URL", monkeyatch_copy)
        fake_data_path = "data/dummy_dataset"
        cache_dir = os.environ.get("COMPASS_DATA_CACHE")
        default_path = os.path.join(DEFAULT_DATA_FOLDER, fake_data_path)
        if os.path.exists(default_path):
            os.remove(default_path)
        import json
        # Force json.load to throw an exception.
        monkeypatch.setattr(json, "load", lambda f: (_ for _ in ()).throw(ValueError("forced exception")))
        archive_called = False
        def dummy_download_and_extract_archive(url, download_root, md5, remove_finished):
            nonlocal archive_called
            archive_called = True
        monkeypatch.setattr(datasets, "download_and_extract_archive", dummy_download_and_extract_archive)
        result = download_dataset(fake_data_path, cache_dir)
        assert archive_called, "download_and_extract_archive should be called even if internal JSON load fails"
        expected_path = os.path.join(cache_dir, fake_data_path)
        assert result == expected_path, "download_dataset should return the correct local path after proceeding with download"