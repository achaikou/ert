import logging
import os
import shutil
from pathlib import Path
from textwrap import dedent
from datetime import datetime, UTC

import pytest

from ert.gui.experiments import RunDialog
from ert.run_models import EnsembleExperiment, SingleTestRun

from .conftest import _open_main_window, get_children


@pytest.mark.parametrize("experiment_type", [SingleTestRun, EnsembleExperiment])
def test_that_gui_uses_config_random_seed_when_specified(
    run_experiment, use_tmpdir, qtbot, caplog, experiment_type
):
    config_text = dedent(
        """
        NUM_REALIZATIONS 1
        RANDOM_SEED 12345
        """
    )
    Path("config.ert").write_text(config_text, encoding="utf-8")

    with (
        caplog.at_level(logging.INFO),
        _open_main_window("config.ert") as (gui, _, _),
    ):
        run_experiment(experiment_type, gui, upload_dir="first_test")

    seed_logs = [line for line in caplog.text.splitlines() if "'random_seed':" in line]
    assert len(seed_logs) == 1
    assert "'random_seed': 12345" in seed_logs[0]


@pytest.mark.timeout(700)
@pytest.mark.parametrize("experiment_type", [SingleTestRun, EnsembleExperiment])
def test_that_gui_generates_different_seeds_for_consecutive_runs(
    run_experiment, use_tmpdir, qtbot, caplog, experiment_type
):
    config_text = dedent(
        """
        NUM_REALIZATIONS 1
        RUNPATH gui_random_seed/realization-<IENS>/iter-<ITER>
        """
    )
    Path("config.ert").write_text(config_text, encoding="utf-8")

    tmp_img_storage = os.path.join(
        "/tmp/test_docs_screenshots", "conftest", "second_test", experiment_type.name()
    )
    os.makedirs(tmp_img_storage, exist_ok=True)

    def wait_for_experiment_completion(gui):
        qtbot.waitUntil(lambda: gui.findChild(RunDialog) is not None, timeout=10000)
        time = datetime.now(UTC).isoformat().replace(":", "-")
        path = qtbot.screenshot(gui, suffix=f"{experiment_type.name()}_{time}")
        print(f"Screenshot of GUI widget at timestamp {time} saved to: {path}")
        shutil.copy(path, tmp_img_storage)

        run_dialog = get_children(gui, RunDialog)[-1]
        time = datetime.now(UTC).isoformat().replace(":", "-")
        path = qtbot.screenshot(gui, suffix=f"{experiment_type.name()}_{time}")
        print(f"Screenshot of GUI widget at timestamp {time} saved to: {path}")
        shutil.copy(path, tmp_img_storage)

        qtbot.waitUntil(
            lambda dialog=run_dialog: dialog.is_experiment_done() is True,
            timeout=300000,
        )
        time = datetime.now(UTC).isoformat().replace(":", "-")
        path = qtbot.screenshot(gui, suffix=f"{experiment_type.name()}_{time}")
        print(f"Screenshot of GUI widget at timestamp {time} saved to: {path}")
        shutil.copy(path, tmp_img_storage)

        qtbot.waitUntil(
            lambda: run_dialog._tab_widget.currentWidget() is not None, timeout=10000
        )
        time = datetime.now(UTC).isoformat().replace(":", "-")
        path = qtbot.screenshot(gui, suffix=f"{experiment_type.name()}_{time}")
        print(f"Screenshot of GUI widget at timestamp {time} saved to: {path}")
        shutil.copy(path, tmp_img_storage)

    with (
        caplog.at_level(logging.INFO),
        _open_main_window("config.ert") as (gui, _, _),
    ):
        qtbot.addWidget(gui)
        run_experiment(experiment_type, gui, wait_done=False)
        wait_for_experiment_completion(gui)

        seed_logs = [line for line in caplog.text.splitlines() if "RANDOM_SEED" in line]
        first_seed_from_log = seed_logs[-1]

        # run_experiment expects the runpath to not exist
        shutil.rmtree("gui_random_seed")

        run_experiment(experiment_type, gui, wait_done=False)
        wait_for_experiment_completion(gui)

        seed_logs = [line for line in caplog.text.splitlines() if "RANDOM_SEED" in line]
        second_seed_from_log = seed_logs[-1]

    assert first_seed_from_log != second_seed_from_log

    seed_logs = [line for line in caplog.text.splitlines() if "'random_seed':" in line]
    assert len(seed_logs) == 2
