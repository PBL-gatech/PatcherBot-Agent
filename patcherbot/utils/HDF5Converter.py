"""Lossless CSV/cell/folder <-> HDF5 archives, usable independently of the application.

Individual CSVs become same-stem .h5 files. Patch-clamp sessions produce
cell_<first-recording-index>.h5, grouping recordings by exact stage XYZ within
each session. All recording indices, protocol files, images, and metadata remain.
Matching metadata rows are exposed in the cell_metadata JSON attribute; the
original shared metadata CSV is also retained for byte-exact restoration.
Archives preserve original file bytes; they are not numerical HDF5 tables.
Close recording writers before conversion. Originals are always retained.
"""

from contextlib import nullcontext
from decimal import Decimal, InvalidOperation
import csv
import json
import re
import shutil
import hashlib
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import tempfile

import h5py
import numpy as np


class HDF5Converter:
    """Stream large files with progress reporting and SHA-256 verification."""

    FORMAT_ID = "PatcherBot.CSVArchive"
    FORMAT_VERSION = 1
    PROTOCOL_VERSION = 2
    CELL_VERSION = 3
    PATCH_ROOTS = {"patch_clamp_data", "TEST_patch_clamp_data"}
    PROTOCOL_NAMES = {
        "CurrentProtocol", "VoltageProtocol", "HoldingProtocol", "OptogeneticProtocol",
        "LeakSubtraction", "CellMetadata",
    }
    FILE_TYPES = {".csv", ".png", ".webp", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif"}

    def __init__(self, compression_level=6, chunk_size=1024 * 1024):
        if not isinstance(compression_level, int) or not 0 <= compression_level <= 9:
            raise ValueError("compression_level must be an integer from 0 to 9")
        if not isinstance(chunk_size, int) or chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer")
        self.compression_level = compression_level
        self.chunk_size = chunk_size

    def convert(self, source, destination=None, *, cell_id=None, overwrite=False, progress=None):
        """Convert a CSV or protocol folder to HDF5, or restore an archive.

        Patch roots/sessions group exact stage XYZ matches into one archive,
        named cell_<lowest-recording-index>.h5. Selecting a protocol folder includes
        its siblings. Pass any recording index as cell_id to select its whole cell.
        Other folders retain the generic folder archive.
        Archives restore relative paths into their containing folder by default.

        progress(stage, completed_bytes, total_bytes) runs synchronously; stages
        are compress, verify, and restore. Use a worker for a loading screen.
        Return signals success; callback errors abort without replacing output.
        No-overwrite publication requires filesystem hard-link support (e.g. NTFS).
        """
        source = Path(source).expanduser().resolve(strict=True)
        if source.is_dir():
            if source.name in self.PROTOCOL_NAMES and self.is_patch_session(source.parent):
                source = source.parent
            if source.name in self.PATCH_ROOTS:
                if destination is not None or cell_id is not None:
                    raise ValueError("Select a session to specify an output or cell index")
                return self.convert_protocols(source, overwrite=overwrite, progress=progress)
            if cell_id is not None:
                return self._pack_protocol(source, destination, overwrite, progress, cell_id=int(cell_id))
            if self.is_patch_session(source):
                if destination is not None:
                    raise ValueError("Specify cell_id when choosing a cell archive destination")
                return [self.convert(source, cell_id=index, overwrite=overwrite, progress=progress)
                        for index in self.cell_groups(source)]
            return self._pack_protocol(source, destination, overwrite, progress)
        if cell_id is not None:
            raise ValueError("cell_id requires a patch-clamp session directory")
        suffix = source.suffix.lower()
        if not source.is_file() or suffix not in (".csv", ".h5", ".hdf5"):
            raise ValueError("Source must be a .csv, .h5, or .hdf5 file")
        if suffix != ".csv":
            with h5py.File(source, "r") as archive:
                if archive.attrs.get("kind") in ("protocol", "cell"):
                    return self._restore_protocol(source, destination, overwrite, progress)
        target_suffix = ".h5" if suffix == ".csv" else ".csv"
        destination = Path(destination).expanduser() if destination is not None else source.with_suffix(target_suffix)
        destination = destination.resolve()
        if source == destination or (destination.exists() and os.path.samefile(source, destination)):
            raise ValueError("Source and destination must be different files")
        if destination.exists() and not overwrite:
            raise FileExistsError(f"Destination already exists: {destination}")
        progress = progress if progress is not None else lambda stage, completed, total: None
        original_state = self._file_state(source)
        descriptor, temporary_name = tempfile.mkstemp(dir=destination.parent,
                                                      prefix=f".{destination.name}.", suffix=".tmp")
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            if suffix == ".csv":
                self._compress(source, temporary, progress)
                self._read_archive(temporary, progress)  # Verify before publishing.
            else:
                self._read_archive(source, progress, destination=temporary)
            if self._file_state(source) != original_state:
                raise RuntimeError("Source changed during conversion; stop and close its writer first")
            if overwrite:
                os.replace(temporary, destination)
            else:
                os.link(temporary, destination)  # Atomically refuses an existing target.
            return destination
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _file_state(path):
        stat = path.stat()
        return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns

    def _write_file(self, group, name, source, progress, completed, total):
        size = source.stat().st_size
        checksum = hashlib.sha256()
        data = group.create_dataset(name, shape=(size,), maxshape=(None,), dtype="u1",
                                    chunks=(self.chunk_size,), compression="gzip",
                                    compression_opts=self.compression_level)
        data.attrs["source_size"] = size
        with source.open("rb") as original:
            for offset in range(0, size, self.chunk_size):
                block_size = min(self.chunk_size, size - offset)
                block = original.read(block_size)
                if len(block) != block_size:
                    raise RuntimeError("Source changed during compression")
                data[offset:offset + block_size] = np.frombuffer(block, dtype=np.uint8)
                checksum.update(block)
                progress("compress", completed + offset + block_size, total)
        data.attrs["sha256"] = checksum.hexdigest()
        return data

    def _read_file(self, data, metadata, progress, stage, completed, total, output=None):
        if not isinstance(data, h5py.Dataset) or data.ndim != 1 or data.dtype != np.dtype("uint8"):
            raise ValueError("Invalid archived byte dataset")
        size = int(metadata.get("source_size", -1))
        expected = metadata.get("sha256", "")
        if size != len(data) or not isinstance(expected, str) or len(expected) != 64:
            raise ValueError("Invalid archive length or checksum metadata")
        checksum = hashlib.sha256()
        for offset in range(0, size, self.chunk_size):
            block = data[offset:offset + self.chunk_size].tobytes()
            checksum.update(block)
            if output is not None:
                output.write(block)
            progress(stage, completed + offset + len(block), total)
        if checksum.hexdigest() != expected:
            raise ValueError("Archive integrity check failed: SHA-256 mismatch")
        return size

    def _compress(self, source, destination, progress):
        total = source.stat().st_size
        progress("compress", 0, total)
        with h5py.File(destination, "w") as archive:
            archive.attrs.update(format_id=self.FORMAT_ID, format_version=self.FORMAT_VERSION,
                                 original_filename=source.name, source_size=total)
            data = self._write_file(archive, "csv_bytes", source, progress, 0, total)
            archive.attrs["sha256"] = data.attrs["sha256"]

    def _read_archive(self, source, progress, destination=None):
        stage = "restore" if destination is not None else "verify"
        output_context = destination.open("wb") if destination is not None else nullcontext()
        with output_context as output, h5py.File(source, "r") as archive:
            if (archive.attrs.get("format_id") != self.FORMAT_ID
                    or archive.attrs.get("format_version") != self.FORMAT_VERSION):
                raise ValueError("Not a supported HDF5Converter CSV archive")
            total = int(archive.attrs.get("source_size", -1))
            progress(stage, 0, total)
            self._read_file(archive.get("csv_bytes"), archive.attrs, progress, stage, 0, total, output)

    def _protocol_files(self, folder):
        files = sorted(path for path in folder.rglob("*")
                       if path.is_file() and path.suffix.lower() in self.FILE_TYPES)
        if any(path.is_symlink() for path in files):
            raise ValueError("Protocol input must contain regular files, not symbolic links")
        return {path: self._file_state(path) for path in files}

    @classmethod
    def is_patch_session(cls, folder):
        folder = Path(folder)
        return folder.is_dir() and any(child.name in cls.PROTOCOL_NAMES and child.is_dir()
                                       and not child.is_symlink() for child in folder.iterdir())

    @staticmethod
    def _metadata_rows(path):
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream, delimiter=";")
            if "index" not in (reader.fieldnames or []):
                raise ValueError(f"Metadata requires an index column: {path}")
            rows = list(reader)
        if any(not str(row.get("index", "")).isdigit() for row in rows):
            raise ValueError(f"Metadata has an invalid cell index: {path}")
        return rows

    def _cell_groups(self, session):
        """Resolve recording indices through metadata to exact session-local XYZ."""
        session = Path(session).expanduser().resolve(strict=True)
        recordings, positions, metadata = {}, {}, {}
        for path, state in self._protocol_files(session).items():
            if path.name.lower() == "cell_metadata.csv":
                metadata[path] = state
                for row in self._metadata_rows(path):
                    index = int(row["index"])
                    try:
                        xyz = tuple(Decimal(str(row[key])) for key in ("stage_x", "stage_y", "stage_z"))
                        if not all(value.is_finite() for value in xyz):
                            raise ValueError("Non-finite coordinate")
                    except (KeyError, InvalidOperation, ValueError) as error:
                        raise ValueError(f"Missing or invalid stage coordinates for recording index {index}: {path}") from error
                    if index in positions and positions[index] != xyz:
                        raise ValueError(f"Conflicting stage coordinates for recording index {index}: {path}")
                    positions[index] = xyz
                    recordings.setdefault(index, {})
                continue
            protocol = path.relative_to(session).parts[0]
            match = re.match(rf"^(?:cell|MembraneTest|{re.escape(protocol)})_(\d+)(?:_|$)", path.stem)
            if match is None:
                raise ValueError(f"Cannot determine cell index: {path}")
            recordings.setdefault(int(match[1]), {})[path] = state
        missing = set(recordings) - set(positions)
        if missing:
            raise ValueError(f"Missing stage coordinates for recording indices {sorted(missing)} in {session}")
        cells = {}
        for index in sorted(recordings):
            cell = cells.setdefault(positions[index], {"indices": [], "files": {}})
            cell["indices"].append(index)
            cell["files"].update(recordings[index])
        # Keep shared originals so any cell can restore the source CSV exactly.
        groups = {}
        for xyz, cell in cells.items():
            cell["files"].update(metadata)
            cell["coordinates"] = xyz
            groups[cell["indices"][0]] = cell
        return groups

    def cell_groups(self, session):
        """Map each cell's lowest recording index to all files at its exact XYZ."""
        return {index: cell["files"] for index, cell in self._cell_groups(session).items()}

    def _pack_protocol(self, folder, destination, overwrite, progress, cell_id=None):
        cell = None
        if cell_id is not None:
            cell = next((group for group in self._cell_groups(folder).values()
                         if cell_id in group["indices"]), None)
            if cell is None:
                raise ValueError(f"No recording index {cell_id} in {folder}")
            cell_id = cell["indices"][0]
        name = folder.name if cell_id is None else f"cell_{cell_id}"
        destination = Path(destination).expanduser() if destination is not None else folder / f"{name}.h5"
        destination = destination.resolve()
        if destination.suffix.lower() not in (".h5", ".hdf5"):
            raise ValueError("Protocol output must end in .h5 or .hdf5")
        if destination.exists() and not overwrite:
            raise FileExistsError(f"Destination already exists: {destination}")
        def input_files():
            return self._protocol_files(folder) if cell_id is None else self.cell_groups(folder).get(cell_id, {})

        originals = self._protocol_files(folder) if cell is None else cell["files"]
        if not originals:
            raise ValueError("Protocol folder contains no CSVs or supported images")
        progress = progress if progress is not None else lambda stage, completed, total: None
        total = sum(state[2] for state in originals.values())
        descriptor, name = tempfile.mkstemp(dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp")
        os.close(descriptor)
        temporary = Path(name)
        try:
            progress("compress", 0, total)
            with h5py.File(temporary, "w") as archive:
                archive.attrs.update(format_id=self.FORMAT_ID,
                                     format_version=self.PROTOCOL_VERSION if cell_id is None else self.CELL_VERSION,
                                     kind="protocol" if cell_id is None else "cell", original_filename=folder.name,
                                     source_size=total, file_count=len(originals))
                if cell_id is not None:
                    metadata = {path.relative_to(folder).as_posix():
                                [row for row in self._metadata_rows(path) if int(row["index"]) in cell["indices"]]
                                for path in originals if path.name.lower() == "cell_metadata.csv"}
                    archive.attrs.update(cell_id=cell_id, cell_metadata=json.dumps(metadata),
                                         cell_identity="stage_xyz", recording_indices=cell["indices"],
                                         stage_coordinates=[float(value) for value in cell["coordinates"]])
                files = archive.create_group("files")
                completed = 0
                for path in originals:
                    relative = path.relative_to(folder).as_posix()
                    self._write_file(files, relative, path, progress, completed, total)
                    completed += originals[path][2]
            if input_files() != originals:
                raise RuntimeError("Source changed during protocol conversion")
            self._read_protocol(temporary, progress)
            if input_files() != originals:
                raise RuntimeError("Source changed during protocol conversion")
            if overwrite:
                os.replace(temporary, destination)
            else:
                os.link(temporary, destination)
            return destination
        finally:
            temporary.unlink(missing_ok=True)

    def _read_protocol(self, source, progress, destination=None):
        """Validate every member before publishing any restored files."""
        stage = "restore" if destination is not None else "verify"
        with h5py.File(source, "r") as archive:
            if (archive.attrs.get("format_id") != self.FORMAT_ID
                    or archive.attrs.get("format_version") != (self.CELL_VERSION if archive.attrs.get("kind") == "cell"
                                                              else self.PROTOCOL_VERSION)):
                raise ValueError("Not a supported protocol archive")
            members = []
            archive["files"].visititems(lambda name, item: members.append((name, item))
                                      if isinstance(item, h5py.Dataset) else None)
            total = int(archive.attrs.get("source_size", -1))
            if len(members) != archive.attrs.get("file_count"):
                raise ValueError("Protocol archive member count mismatch")
            progress(stage, 0, total)
            completed, paths = 0, []
            for name, data in members:
                relative = PurePosixPath(name)
                if relative.is_absolute() or any(part in (".", "..") or ":" in part or "\\" in part
                                                 or part != part.rstrip(" .") or PureWindowsPath(part).is_reserved()
                                                 for part in relative.parts):
                    raise ValueError("Unsafe protocol archive path")
                path = Path(*relative.parts)
                if str(path).casefold() in {str(item).casefold() for item in paths}:
                    raise ValueError("Duplicate protocol archive path")
                paths.append(path)
                target = destination / path if destination is not None else None
                if target is not None:
                    target.parent.mkdir(parents=True, exist_ok=True)
                context = target.open("wb") if target is not None else nullcontext()
                with context as output:
                    completed += self._read_file(data, data.attrs, progress, stage, completed, total, output)
            if completed != total:
                raise ValueError("Protocol archive size mismatch")
            return paths

    def _restore_protocol(self, source, destination, overwrite, progress):
        destination = (Path(destination).expanduser() if destination is not None else source.parent).resolve()
        progress = progress if progress is not None else lambda stage, completed, total: None
        original_state = self._file_state(source)
        staging = Path(tempfile.mkdtemp(dir=destination.parent, prefix=".protocol_restore_")).resolve()
        assert staging.is_relative_to(destination.parent) and staging != destination
        payload = staging / "payload"
        recovery = staging / "recovery"
        payload.mkdir()
        recovery.mkdir()
        published, backups, created_directories = [], [], []
        cleanup = True
        try:
            paths = self._read_protocol(source, progress, payload)
            if self._file_state(source) != original_state:
                raise RuntimeError("Source changed during protocol restoration")
            with h5py.File(source, "r") as archive:
                shared = (set(json.loads(archive.attrs["cell_metadata"]))
                          if archive.attrs.get("kind") == "cell" else set())
            # Multiple cells share a source metadata CSV. Reuse an identical copy
            # so restoring the next cell does not require enabling overwrite.
            paths = [relative for relative in paths
                     if not (relative.as_posix() in shared and (destination / relative).is_file()
                             and self._same_bytes(payload / relative, destination / relative))]
            for relative in paths:
                target = destination / relative
                if not target.resolve().is_relative_to(destination) or target.resolve() == source:
                    raise ValueError("Unsafe protocol restoration target")
                if target.exists() and (not overwrite or not target.is_file()):
                    raise FileExistsError(f"Destination already exists: {target}")
            for index, relative in enumerate(paths):
                target = destination / relative
                missing = []
                parent = target.parent
                while not parent.exists():
                    missing.append(parent)
                    parent = parent.parent
                for directory in reversed(missing):
                    directory.mkdir()
                    created_directories.append(directory)
                if target.exists():
                    backup = recovery / str(index)
                    os.replace(target, backup)
                    backups.append((backup, target))
                os.link(payload / relative, target)
                published.append(target)
            return destination
        except Exception:
            try:
                for target in reversed(published):
                    target.unlink()
                for backup, target in reversed(backups):
                    os.replace(backup, target)
                for directory in reversed(created_directories):
                    directory.rmdir()
            except OSError as error:
                cleanup = False
                raise RuntimeError(f"Restoration rollback failed; recovery files retained at {staging}") from error
            raise
        finally:
            if cleanup:
                shutil.rmtree(staging)

    def _same_bytes(self, first, second):
        if first.stat().st_size != second.stat().st_size:
            return False
        with first.open("rb") as a, second.open("rb") as b:
            while True:
                block = a.read(self.chunk_size)
                if block != b.read(self.chunk_size):
                    return False
                if not block:
                    return True

    def convert_protocols(self, patch_clamp_folder, *, overwrite=False, progress=None):
        """Convert each session to cell archives (retained batch entry point)."""
        root = Path(patch_clamp_folder).expanduser().resolve(strict=True)
        return [archive for session in sorted(root.iterdir())
                if self.is_patch_session(session) and not session.is_symlink()
                for archive in self.convert(session, overwrite=overwrite, progress=progress)]


if __name__ == "__main__":
    import argparse

    # Edit this file path for no-argument use; an optional CLI path overrides it.
    DEFAULT_FILE = Path(__file__).resolve().parents[2] / (
        "experiments/Data/patch_clamp_data/2026_03_16-12_32"
    )
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source", nargs="?", default=DEFAULT_FILE)
    parser.add_argument("-o", "--output")
    parser.add_argument("--overwrite", action="store_true")
    arguments = parser.parse_args()

    def show_progress(stage, completed, total):
        percent = 100 * completed / total if total else 100
        print(f"\r{stage.capitalize()}: {percent:5.1f}%", end="", flush=True)

    try:
        result = HDF5Converter().convert(arguments.source, arguments.output,
                                         overwrite=arguments.overwrite, progress=show_progress)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"\nConversion failed: {error}\n")
    print(f"\nSaved: {result}\nOriginal retained: {arguments.source}")
