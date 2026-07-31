"""Tests for golfswing.paths — where the project's data actually lives."""

from pathlib import Path

from golfswing import paths


class TestProjectRoot:
    def test_root_is_the_directory_containing_the_package(self):
        assert (paths.PROJECT_ROOT / "golfswing" / "__init__.py").exists()

    def test_root_holds_the_project_config(self):
        """A sanity anchor: if this moves, every derived path is wrong."""
        assert (paths.PROJECT_ROOT / "pyproject.toml").exists()

    def test_paths_are_absolute(self):
        """The whole point. Relative paths made every caller depend on cwd, and
        a script moved into scripts/ silently resolved data/ beneath itself."""
        for path in (paths.RAW_DIR, paths.PROCESSED_DIR, paths.LABELS_DIR,
                     paths.PREVIEWS_DIR, paths.OUTPUTS_DIR, paths.DB_PATH,
                     paths.MODELS_DIR, paths.THRESHOLDS):
            assert path.is_absolute(), f"{path} is relative"

    def test_data_paths_sit_under_the_data_directory(self):
        for path in (paths.RAW_DIR, paths.PROCESSED_DIR, paths.LABELS_DIR,
                     paths.PREVIEWS_DIR):
            assert path.parent == paths.PROJECT_ROOT / "data"

    def test_they_resolve_the_same_from_any_working_directory(self, tmp_path,
                                                              monkeypatch):
        before = paths.RAW_DIR
        monkeypatch.chdir(tmp_path)
        import importlib
        importlib.reload(paths)
        assert paths.RAW_DIR == before


class TestEnvironmentOverride:
    def test_an_explicit_root_wins(self, tmp_path, monkeypatch):
        """Lets a second copy of the data live elsewhere without editing code."""
        monkeypatch.setenv("GOLFSWING_ROOT", str(tmp_path))
        import importlib
        importlib.reload(paths)
        try:
            assert paths.PROJECT_ROOT == tmp_path
            assert paths.RAW_DIR == tmp_path / "data" / "raw"
        finally:
            monkeypatch.delenv("GOLFSWING_ROOT")
            importlib.reload(paths)
