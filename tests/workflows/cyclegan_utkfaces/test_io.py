#    Copyright 2025, Stankevich Andrey, stankevich.as@phystech.edu

#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at

#        http://www.apache.org/licenses/LICENSE-2.0

#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


"""Unit tests for CycleGAN checkpoint I/O module.

Tests cover:
- CycleGANModels and TrainingState dataclasses
- CycleGANCheckpointManager save/load operations
- Error handling and edge cases
- Symlink creation and cleanup
- Auto-resume functionality
"""

from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from workflows.cyclegan_utkfaces.io import (
    auto_resume_training,
    CHECKPOINT_SIGNATURE,
    CHECKPOINT_VERSION,
    CycleGANCheckpointManager,
    CycleGANModels,
    TrainingState,
)


class _DTStub:
    """Datetime stub with call counter.

    Patching datetime.now with pytest-mock likely hits
    an immutable C object, which results in TypeError.
    This is a workaround.
    """
    calls = 0

    @classmethod
    def now(cls):
        cls.calls += 1

        class _Now:
            calls = 0

            def isoformat(self):
                _Now.calls += 1
                return "2023-01-01T00:00:00"
        return _Now()


# Fixtures for test dependencies
@pytest.fixture
def mock_models():
    """Meaningful model assembly out of lightweight components."""
    # generators and discriminators
    G_A2B = nn.Linear(10, 10)
    G_B2A = nn.Linear(10, 10)
    D_A = nn.Linear(10, 1)
    D_B = nn.Linear(10, 1)

    # optimizers
    optimizer_G = Adam(list(G_A2B.parameters()) + list(G_B2A.parameters()), lr=0.0002)
    optimizer_D_A = Adam(D_A.parameters(), lr=0.0002)
    optimizer_D_B = Adam(D_B.parameters(), lr=0.0002)

    # schedulers
    scheduler_G = StepLR(optimizer_G, step_size=10, gamma=0.5)
    scheduler_D_A = StepLR(optimizer_D_A, step_size=10, gamma=0.5)
    scheduler_D_B = StepLR(optimizer_D_B, step_size=10, gamma=0.5)

    return CycleGANModels(
        G_A2B=G_A2B,
        G_B2A=G_B2A,
        D_A=D_A,
        D_B=D_B,
        optimizer_G=optimizer_G,
        optimizer_D_A=optimizer_D_A,
        optimizer_D_B=optimizer_D_B,
        scheduler_G=scheduler_G,
        scheduler_D_A=scheduler_D_A,
        scheduler_D_B=scheduler_D_B
    )


@pytest.fixture
def mock_models_no_schedulers():
    """Meaningful model assembly w/o schedulers."""
    G_A2B = nn.Linear(10, 10)
    G_B2A = nn.Linear(10, 10)
    D_A = nn.Linear(10, 1)
    D_B = nn.Linear(10, 1)

    optimizer_G = Adam(list(G_A2B.parameters()) + list(G_B2A.parameters()), lr=0.0002)
    optimizer_D_A = Adam(D_A.parameters(), lr=0.0002)
    optimizer_D_B = Adam(D_B.parameters(), lr=0.0002)

    return CycleGANModels(
        G_A2B=G_A2B,
        G_B2A=G_B2A,
        D_A=D_A,
        D_B=D_B,
        optimizer_G=optimizer_G,
        optimizer_D_A=optimizer_D_A,
        optimizer_D_B=optimizer_D_B
    )


@pytest.fixture
def sample_training_state():
    """Training state with redefined fields."""
    state = TrainingState()
    state.epoch = 25
    state.global_step = 1500
    state.best_fid_score = 42.5
    state.best_lpips_score = 0.3

    # Add some loss history
    state.loss_history['loss_G_A2B'] = [1.0, 0.8, 0.6]
    state.loss_history['loss_G_B2A'] = [1.2, 0.9, 0.7]
    state.loss_history['loss_D_A'] = [0.5, 0.4, 0.3]

    return state


@pytest.fixture
def temp_checkpoint_dir(tmp_path):
    """Derivative from from tmp_path fixture."""
    return tmp_path / "checkpoints"


@pytest.fixture
def checkpoint_manager(temp_checkpoint_dir):
    """Checkpoint manager that writes into temp_checkpoint_dir."""
    return CycleGANCheckpointManager(
        checkpoint_dir=temp_checkpoint_dir,
        save_top_k=3,
        save_last=True
    )


@pytest.fixture
def mock_torch_save(mocker):
    """Mock torch.save call."""
    return mocker.patch(
        "workflows.cyclegan_utkfaces.io.torch.save"
    )

@pytest.fixture
def mock_torch_load(mocker):
    """Mock torch.load call."""
    return mocker.patch(
        "workflows.cyclegan_utkfaces.io.torch.load"
    )

class TestCycleGANModels:
    """CycleGANModels dataclass testing suite."""

    def test_models_creation(self, mock_models):
        """CycleGANModels mock has all required fields."""
        assert isinstance(mock_models.G_A2B, nn.Module)
        assert isinstance(mock_models.G_B2A, nn.Module)
        assert isinstance(mock_models.D_A, nn.Module)
        assert isinstance(mock_models.D_B, nn.Module)
        assert mock_models.optimizer_G is not None
        assert mock_models.optimizer_D_A is not None
        assert mock_models.optimizer_D_B is not None

    def test_models_with_schedulers(self, mock_models):
        """Schedulers are present if provided as kwargs."""
        assert mock_models.scheduler_G is not None
        assert mock_models.scheduler_D_A is not None
        assert mock_models.scheduler_D_B is not None

    def test_models_without_schedulers(self, mock_models_no_schedulers):
        """Scheduler defaults are None."""
        models = mock_models_no_schedulers
        assert models.scheduler_G is None
        assert models.scheduler_D_A is None
        assert models.scheduler_D_B is None


class TestTrainingState:
    """TrainingState dataclass testing suite."""

    def test_default_initialization(self):
        """TrainingState gets expected defaults."""
        state = TrainingState()
        assert state.epoch == 0
        assert state.global_step == 0
        assert state.best_fid_score == float('inf')
        assert state.best_lpips_score == float('inf')
        assert isinstance(state.loss_history, dict)
        assert len(state.loss_history) == 10

    def test_loss_history_structure(self):
        """Loss history dictionary keys match the expected set."""
        state = TrainingState()
        expected_keys = {
            'loss_G_A2B', 'loss_G_B2A', 'loss_D_A', 'loss_D_B',
            'loss_cycle_A', 'loss_cycle_B', 'loss_identity_A',
            'loss_identity_B', 'loss_G_total', 'loss_D_total'
        }
        assert set(state.loss_history.keys()) == expected_keys

    def test_metadata_fields(self):
        """Metadata fields are populated."""
        state = TrainingState()
        assert state.timestamp is not None
        assert state.pytorch_version is not None
        assert isinstance(state.device_info, str)

    def test_custom_values(self, sample_training_state):
        """TrainingState fields should be properly set via assignment."""
        state = sample_training_state
        assert state.epoch == 25
        assert state.global_step == 1500
        assert state.best_fid_score == 42.5
        assert len(state.loss_history['loss_G_A2B']) == 3


class TestCycleGANCheckpointManager:
    """CycleGANCheckpointManager class tesing suite."""

    def test_initialization(self, temp_checkpoint_dir):
        """Initialization populates proper attributes."""
        manager = CycleGANCheckpointManager(
            checkpoint_dir=temp_checkpoint_dir,
            save_top_k=5,
            save_last=False
        )

        assert manager.checkpoint_dir == Path(temp_checkpoint_dir)
        assert manager.save_top_k == 5
        assert manager.save_last is False
        assert temp_checkpoint_dir.exists()

    def test_initialization_creates_directory(self, tmp_path):
        """Initialization creates checkpoint directory."""
        checkpoint_dir = tmp_path / "new_checkpoints"
        assert not checkpoint_dir.exists()

        CycleGANCheckpointManager(checkpoint_dir)
        assert checkpoint_dir.exists()

    def test_save_checkpoint_basic(
        self,
        checkpoint_manager, mock_models, sample_training_state,
        mock_torch_save, monkeypatch
    ):
        """CheckpointManager.save_checkpoint keeps contract and calls proper methods."""
        # replace datetime with a stub
        import workflows.cyclegan_utkfaces.io as io_mod
        monkeypatch.setattr(io_mod, "datetime", _DTStub)

        # act
        saved_path = checkpoint_manager.save_checkpoint(
            models=mock_models,
            state=sample_training_state,
            metrics={"fid": 35.2, "lpips": 0.25}
        )

        # assert
        mock_torch_save.assert_called_once()
        assert _DTStub.calls == 1

        save_call_args = mock_torch_save.call_args[0]
        saved_data = save_call_args[0]

        expected_filename = (
            f"checkpoint_epoch_{sample_training_state.epoch:04d}"
            f"_step_{sample_training_state.global_step:04d}.pth"
        )

        assert all(key in saved_data for key in CHECKPOINT_SIGNATURE)
        assert saved_data['checkpoint_version'] == CHECKPOINT_VERSION
        assert (
            saved_data['training_state'] == sample_training_state.__dict__
        )
        assert saved_data['metrics'] == {"fid": 35.2, "lpips": 0.25}
        assert saved_data['save_timestamp'] == "2023-01-01T00:00:00"
        assert saved_path.name == expected_filename

    def test_save_checkpoint_custom_name(
        self, checkpoint_manager, mock_models, sample_training_state,
        mock_torch_save,
    ):
        """Providing alternative name creates proper save path."""
        custom_name = "my_custom_checkpoint.pth"

        saved_path = checkpoint_manager.save_checkpoint(
            models=mock_models,
            state=sample_training_state,
            checkpoint_name=custom_name
        )

        assert saved_path.name == custom_name
        mock_torch_save.assert_called_once()

    def test_save_checkpoint_with_schedulers(
        self, checkpoint_manager, mock_models, sample_training_state,
        mock_torch_save
    ):
        """Saved checkpoint includes scheduler state dicts."""
        checkpoint_manager.save_checkpoint(
            models=mock_models,
            state=sample_training_state
        )

        saved_data = mock_torch_save.call_args[0][0]

        # Verify scheduler state dicts are included
        assert saved_data['scheduler_G_state_dict'] is not None
        assert saved_data['scheduler_D_A_state_dict'] is not None
        assert saved_data['scheduler_D_B_state_dict'] is not None

    def test_save_checkpoint_without_schedulers(
        self, checkpoint_manager, mock_models_no_schedulers,
        sample_training_state, mock_torch_save,
    ):
        """Checkpoint saving works if schedulers are missing."""
        checkpoint_manager.save_checkpoint(
            models=mock_models_no_schedulers,
            state=sample_training_state
        )

        saved_data = mock_torch_save.call_args[0][0]

        # Verify scheduler state dicts are None
        assert saved_data['scheduler_G_state_dict'] is None
        assert saved_data['scheduler_D_A_state_dict'] is None
        assert saved_data['scheduler_D_B_state_dict'] is None

    def test_save_checkpoint_handles_errors(
        self, checkpoint_manager, mock_models, sample_training_state,
        mock_torch_save
    ):
        """Torch.save does not fail silenlty."""
        mock_torch_save.side_effect = Exception("Save failed")
        with pytest.raises(Exception, match="Save failed"):
            checkpoint_manager.save_checkpoint(
                models=mock_models,
                state=sample_training_state
            )

    def test_save_checkpoint_creates_symlinks(
        self, checkpoint_manager, sample_training_state, mock_models,
        mocker
    ):
        """Method save_checkpoint calls _create_symlink helper exactly 2 times."""
        mock_symlink = mocker.patch.object(checkpoint_manager, '_create_symlink')

        checkpoint_manager.save_checkpoint(
            models=mock_models,
            state=sample_training_state,
            is_best=True
        )

        best_call, last_call = mock_symlink.call_args_list
        assert mock_symlink.call_count == 2  # best + last
        assert "best_checkpoint.pth" in str(best_call[0][1])
        assert "last_checkpoint.pth" in str(last_call[0][1])

    def test_cleanup_respects_symlink_targets_and_keeps_top_k(
        self, tmp_path
    ):
        """Symlink target files (best/last) are detected and preserved."""
        import time

        ckpt_dir = tmp_path / "ckpts"
        ckpt_dir.mkdir()

        files = []
        for i in range(5):
            f = ckpt_dir / f"checkpoint_epoch_{i:04d}_step_0000.pth"
            f.touch()
            files.append(f)
            time.sleep(0.01)

        best_link = ckpt_dir / "best_checkpoint.pth"
        last_link = ckpt_dir / "last_checkpoint.pth"
        best_link.symlink_to(files[1].name)
        last_link.symlink_to(files[3].name)

        # sanity check: reference targets are files[1] and files[3]
        assert best_link.is_symlink() and last_link.is_symlink()
        assert best_link.resolve() == files[1].resolve()
        assert last_link.resolve() == files[3].resolve()

        manager = CycleGANCheckpointManager(ckpt_dir, save_top_k=2)
        manager._cleanup_old_checkpoints()

        assert files[1].exists()
        assert files[3].exists()

        survivors = sorted([p for p in ckpt_dir.glob("*.pth") if p.is_file()])
        targets = {files[1].resolve(), files[3].resolve()}
        non_targets = [f for f in survivors if f.resolve() not in targets]
        non_targets.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        should_remain = set(non_targets[:2])

        remaining_real = {
            p for p in ckpt_dir.glob("*.pth") if p.is_file()
            and not p.is_symlink()
        }

        expected_remaining = {files[1], files[3]} | should_remain
        assert remaining_real == expected_remaining

    def test_load_checkpoint_basic(
        self, checkpoint_manager, sample_training_state, mock_models,
        mock_torch_load, temp_checkpoint_dir, mocker
    ):
        """Checkpoint loading calls load_state_dict on all components."""
        # create a fake checkpoint file
        checkpoint_path = temp_checkpoint_dir / "test_checkpoint.pth"
        checkpoint_path.touch()

        # mock the loaded data
        mock_checkpoint_data = {
            'G_A2B_state_dict': {'weight': torch.randn(10, 10)},
            'G_B2A_state_dict': {'weight': torch.randn(10, 10)},
            'D_A_state_dict': {'weight': torch.randn(10, 1)},
            'D_B_state_dict': {'weight': torch.randn(10, 1)},
            'optimizer_G_state_dict': {'param_groups': []},
            'optimizer_D_A_state_dict': {'param_groups': []},
            'optimizer_D_B_state_dict': {'param_groups': []},
            'scheduler_G_state_dict': {'step_size': 10},
            'scheduler_D_A_state_dict': {'step_size': 10},
            'scheduler_D_B_state_dict': {'step_size': 10},
            'training_state': sample_training_state.__dict__,
            'checkpoint_version': CHECKPOINT_VERSION
        }
        mock_torch_load.return_value = mock_checkpoint_data

        # mock load_state_dict_attributes
        patch_targets = [
            "G_A2B", "G_B2A", "D_A", "D_B",
            "optimizer_G", "optimizer_D_A", "optimizer_D_B",
            "scheduler_G", "scheduler_D_A", "scheduler_D_B",
        ]

        mocks = {}
        for target in patch_targets:
            mocks[target] = mocker.patch.object(
                getattr(mock_models, target), 'load_state_dict')

        # act
        loaded_state = checkpoint_manager.load_checkpoint(
            models=mock_models,
            checkpoint_path=checkpoint_path
        )

        # assert
        mock_torch_load.assert_called_once_with(
            checkpoint_path,
            map_location=None,
            weights_only=False
        )

        for m in mocks.values():
            m.assert_called_once()

        assert loaded_state.epoch == sample_training_state.epoch
        assert loaded_state.global_step == sample_training_state.global_step

    def test_load_checkpoint_file_not_found(self, checkpoint_manager, mock_models):
        """Non-existent checkpoint raises FileNotFoundError."""
        non_existent_path = Path("non_existent_checkpoint.pth")

        with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
            checkpoint_manager.load_checkpoint(
                models=mock_models,
                checkpoint_path=non_existent_path
            )

    def test_load_checkpoint_missing_keys(
        self, checkpoint_manager, mock_models,
        mock_torch_load, temp_checkpoint_dir
    ):
        """Mmissing required checkpoint keys raises KeyError."""
        checkpoint_path = temp_checkpoint_dir / "bad_checkpoint.pth"
        checkpoint_path.touch()

        # incomplete checkpoint data
        mock_torch_load.return_value = {
            'G_A2B_state_dict': {},
            'checkpoint_version': CHECKPOINT_VERSION
        }

        with pytest.raises(KeyError, match="Checkpoint missing required keys"):
            checkpoint_manager.load_checkpoint(
                models=mock_models,
                checkpoint_path=checkpoint_path
            )

    def test_load_checkpoint_handles_errors(
        self, checkpoint_manager, mock_models,
        mock_torch_load, temp_checkpoint_dir
    ):
        """Torch load does not fail silently."""
        mock_torch_load.side_effect = Exception("Load failed")
        checkpoint_path = temp_checkpoint_dir / "test_checkpoint.pth"
        checkpoint_path.touch()

        with pytest.raises(Exception, match="Load failed"):
            checkpoint_manager.load_checkpoint(
                models=mock_models,
                checkpoint_path=checkpoint_path
            )

    def test_save_inference_models(
        self, checkpoint_manager, mock_models, mock_torch_save, temp_checkpoint_dir
    ):
        """Inference-only checkpoint has no optimizers and schedulers."""
        save_path = temp_checkpoint_dir / "inference_models.pth"
        metadata = {"training_epochs": 100, "dataset": "test"}

        result_path = checkpoint_manager.save_inference_models(
            models=mock_models,
            save_path=save_path,
            metadata=metadata
        )

        assert result_path == save_path
        mock_torch_save.assert_called_once()

        saved_data = mock_torch_save.call_args[0][0]
        assert 'G_A2B_state_dict' in saved_data
        assert 'G_B2A_state_dict' in saved_data
        assert saved_data['inference_metadata'] == metadata
        assert 'D_A_state_dict' not in saved_data  # Discriminators not saved
        assert 'optimizer_G_state_dict' not in saved_data  # Optimizers not saved

    def test_load_inference_models(
        self, checkpoint_manager,
        mock_torch_load, temp_checkpoint_dir, mocker
    ):
        """inference-only load calls load_state_dict on generators."""
        checkpoint_path = temp_checkpoint_dir / "inference_checkpoint.pth"
        checkpoint_path.touch()

        # Create mock generators
        G_A2B = nn.Linear(10, 10)
        G_B2A = nn.Linear(10, 10)

        # Mock loaded data
        metadata = {"training_epochs": 100}
        mock_torch_load.return_value = {
            'G_A2B_state_dict': {'weight': torch.randn(10, 10)},
            'G_B2A_state_dict': {'weight': torch.randn(10, 10)},
            'inference_metadata': metadata
        }

        mocks = [
            mocker.patch.object(G_A2B, 'load_state_dict'),
            mocker.patch.object(G_B2A, 'load_state_dict'),
            mocker.patch.object(G_A2B, 'eval'),
            mocker.patch.object(G_B2A, 'eval')
        ]

        loaded_metadata = checkpoint_manager.load_inference_models(
            G_A2B=G_A2B,
            G_B2A=G_B2A,
            checkpoint_path=checkpoint_path
        )

        for m in mocks:
            m.assert_called_once()

        # Verify metadata
        assert loaded_metadata == metadata

    def test_list_checkpoints_empty_directory(self, checkpoint_manager):
        """Empty checkpoin directory -> empty set."""
        checkpoints = checkpoint_manager.list_checkpoints()
        assert checkpoints == {}

    def test_list_checkpoints_with_files(self, checkpoint_manager, temp_checkpoint_dir):
        """List_checkpoints discovers pt and pth files."""
        # Create some checkpoint files
        checkpoint1 = temp_checkpoint_dir / "checkpoint_epoch_0010_step_0500.pth"
        checkpoint2 = temp_checkpoint_dir / "checkpoint_epoch_0020_step_1000.pth"
        non_checkpoint = temp_checkpoint_dir / "config.yaml"

        checkpoint1.touch()
        checkpoint2.touch()
        non_checkpoint.touch()

        checkpoints = checkpoint_manager.list_checkpoints()

        assert len(checkpoints) == 2
        assert "checkpoint_epoch_0010_step_0500" in checkpoints
        assert "checkpoint_epoch_0020_step_1000" in checkpoints
        assert "config" not in checkpoints  # Non-.pth files ignored

    def test_get_latest_checkpoint_symlink(self, checkpoint_manager, temp_checkpoint_dir):
        """Method get_latest_checkpoint correctly resolves existing symlink."""
        # Create symlink
        latest_symlink = temp_checkpoint_dir / "last_checkpoint.pth"
        target_file = temp_checkpoint_dir / "checkpoint_epoch_0010_step_0500.pth"
        target_file.touch()

        # Create symlink
        latest_symlink.symlink_to(target_file.name)

        latest = checkpoint_manager.get_latest_checkpoint()
        assert latest == target_file

    def test_get_latest_checkpoint_fallback(self, checkpoint_manager, temp_checkpoint_dir):
        """Latest checkpoint is the newest file if there's no symlink."""
        # Create checkpoint files with different timestamps
        old_checkpoint = temp_checkpoint_dir / "checkpoint_epoch_0010_step_0500.pth"
        new_checkpoint = temp_checkpoint_dir / "checkpoint_epoch_0020_step_1000.pth"

        old_checkpoint.touch()
        import time
        time.sleep(0.01)  # Ensure different timestamps
        new_checkpoint.touch()

        latest = checkpoint_manager.get_latest_checkpoint()
        assert latest == new_checkpoint

    def test_get_latest_checkpoint_none(self, checkpoint_manager):
        """If there are no checkpoints, latest is None."""
        latest = checkpoint_manager.get_latest_checkpoint()
        assert latest is None

    def test_get_best_checkpoint_exists(self, checkpoint_manager, temp_checkpoint_dir):
        """Method get_best_checkpoint correctly resolves existing symlink."""
        # Create symlink
        best_symlink = temp_checkpoint_dir / "best_checkpoint.pth"
        target_file = temp_checkpoint_dir / "checkpoint_epoch_0015_step_0750.pth"
        target_file.touch()

        best_symlink.symlink_to(target_file.name)

        best = checkpoint_manager.get_best_checkpoint()
        assert best == target_file

    def test_get_best_checkpoint_none(self, checkpoint_manager):
        """If there are no checkpoints, best is None."""
        best = checkpoint_manager.get_best_checkpoint()
        assert best is None

    def test_create_symlink_overwrites_existing(
        self, checkpoint_manager, temp_checkpoint_dir
    ):
        """Symlink creation overwrites existing symlink."""
        target1 = temp_checkpoint_dir / "target1.pth"
        target2 = temp_checkpoint_dir / "target2.pth"
        symlink_path = temp_checkpoint_dir / "link.pth"

        target1.touch()
        target2.touch()

        # Create initial symlink
        checkpoint_manager._create_symlink(target1, symlink_path)
        assert symlink_path.resolve() == target1

        # Overwrite with new target
        checkpoint_manager._create_symlink(target2, symlink_path)
        assert symlink_path.resolve() == target2

    def test_create_symlink_handles_errors(
        self, checkpoint_manager, temp_checkpoint_dir, mocker
    ):
        """Symlink creation handles OS errors gracefully."""
        mock_logger = mocker.patch(
            'workflows.cyclegan_utkfaces.io.logger')
        mock_logger.warning = mocker.Mock()

        target_file = temp_checkpoint_dir / "target.pth"
        symlink_path = temp_checkpoint_dir / "link.pth"
        target_file.touch()

        mocker.patch(
            'workflows.cyclegan_utkfaces.io.Path.symlink_to',
            side_effect=OSError("Permission denied")
        )

        # act
        checkpoint_manager._create_symlink(target_file, symlink_path)

        # assert
        mock_logger.warning.assert_called_once()
        assert "Failed to create symlink" in str(mock_logger.warning.call_args)

    def test_cleanup_old_checkpoints_disabled(self, temp_checkpoint_dir):
        """If save_top_k <= 0, all checkpoints are kept."""
        manager = CycleGANCheckpointManager(temp_checkpoint_dir, save_top_k=0)
        for i in range(5):
            (temp_checkpoint_dir / f"checkpoint_{i}.pth").touch()

        manager._cleanup_old_checkpoints()

        assert len(list(temp_checkpoint_dir.glob("*.pth"))) == 5


class TestAutoResumeTraining:
    """Test suite for auto_resume_training utility function."""

    def test_resume_from_specific_path(
        self, checkpoint_manager, mock_models, sample_training_state,
        temp_checkpoint_dir, mocker
    ):
        """Auto resume with path calls load_checkpoint with said path."""
        checkpoint_path = temp_checkpoint_dir / "specific_checkpoint.pth"
        checkpoint_path.touch()

        mock_load = mocker.patch.object(
            checkpoint_manager, 'load_checkpoint',
        )
        mock_load.return_value = sample_training_state

        state, resumed = auto_resume_training(
            checkpoint_manager, mock_models,
            resume_from=str(checkpoint_path)
        )

        assert resumed is True
        assert state == sample_training_state
        mock_load.assert_called_once_with(mock_models, Path(str(checkpoint_path)))

    def test_resume_from_latest(
        self, checkpoint_manager, mock_models, sample_training_state,
        temp_checkpoint_dir, mocker
    ):
        """Call stack for resume_from_latest is correct."""
        latest_path = temp_checkpoint_dir / "latest_checkpoint.pth"
        latest_path.touch()

        with (
            patch.object(checkpoint_manager, 'get_latest_checkpoint') as mock_get_latest,
            patch.object(checkpoint_manager, 'load_checkpoint') as mock_load
        ):

            mock_get_latest.return_value = latest_path
            mock_load.return_value = sample_training_state

            state, resumed = auto_resume_training(checkpoint_manager, mock_models)

            assert resumed is True
            assert state == sample_training_state
            mock_get_latest.assert_called_once()
            mock_load.assert_called_once_with(mock_models, latest_path)

    def test_no_checkpoint_available(self, checkpoint_manager, mock_models):
        """If no checkpoints available, resumed is False, and state is fresh."""
        with patch.object(checkpoint_manager, 'get_latest_checkpoint') as mock_get_latest:
            mock_get_latest.return_value = None

            state, resumed = auto_resume_training(checkpoint_manager, mock_models)

            assert resumed is False
            assert isinstance(state, TrainingState)
            assert state.epoch == 0

    def test_checkpoint_file_not_exists(self, checkpoint_manager, mock_models):
        """If checkpoint path is nonexistent, resumed is False, and state is fresh."""
        non_existent_path = Path("non_existent.pth")

        with patch.object(checkpoint_manager, 'get_latest_checkpoint') as mock_get_latest:
            mock_get_latest.return_value = non_existent_path

            state, resumed = auto_resume_training(checkpoint_manager, mock_models)

            assert resumed is False
            assert isinstance(state, TrainingState)
            assert state.epoch == 0

class TestEdgeCases:
    """Umbrella test class for edge cases and error conditions."""

    def test_save_checkpoint_permission_error(
        self, checkpoint_manager, mock_models, sample_training_state,
        mock_torch_save
    ):
        """Permission errors do not fail silently."""
        mock_torch_save.side_effect = PermissionError("Access Denied")
        with pytest.raises(PermissionError):
            checkpoint_manager.save_checkpoint(mock_models, sample_training_state)

    def test_checkpoint_compatibility_version_mismatch(
        self, checkpoint_manager, mocker
    ):
        """Test checkpoint version mismatch handling."""
        mock_logger = mocker.patch(
            'workflows.cyclegan_utkfaces.io.logger')
        mock_logger.warning = mocker.Mock()

        checkpoint_data = {
            **{key: {} for key in CHECKPOINT_SIGNATURE},
            'checkpoint_verions': '2.0'  # different version (note typo in key)
        }

        # should not raise exception but log warning
        checkpoint_manager._verify_checkpoint_compatibility(checkpoint_data)
        mock_logger.warning.assert_called_once()

    def test_training_state_serialization_roundtrip(
        self, sample_training_state
    ):
        """TrainingState can be serialized and deserialized."""
        # convert to dict and back, filter only valid fields for reconstruction
        state_dict = sample_training_state.__dict__
        valid_fields = {
            k: v for k, v in state_dict.items()
            if k in TrainingState.__dataclass_fields__
        }

        reconstructed_state = TrainingState(**valid_fields)

        assert reconstructed_state.epoch == sample_training_state.epoch
        assert reconstructed_state.global_step == sample_training_state.global_step
        assert reconstructed_state.best_fid_score == sample_training_state.best_fid_score


if __name__ == "__main__":
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--cov=io",
        "--cov-report=term-missing"
    ])
