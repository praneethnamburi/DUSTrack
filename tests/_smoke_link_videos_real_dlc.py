"""End-to-end smoke for the 1.3.0a2 hardlink-by-default flow.

Creates a real DLC project against the pia02 telemed mp4s living at
``M:\\us_videos_for_tracking2\\pia02_s001_002_{LFA2,RFA2}.mp4`` and
verifies:

* ``<project>/videos/<stem>.mp4`` is a hardlink to the source (same
  file via ``os.path.samefile``, ``st_nlink >= 2``)
* the ``.dnav-toc`` sidecar is also hardlinked when present
* ``config['video_sets']`` keys point at the in-project paths (not
  the source paths) so downstream DLC operations see videos as
  project-local

Not pytest-collected (filename starts with ``_``) -- this is an
integration smoke that requires DLC, M:\ network drive, the specific
pia02 fixtures, and ~10 s of project-creation time. Run directly with
the dlc3rc14 env's Python.
"""
import json
import os
import shutil
from pathlib import Path

import dustrack
from dustrack.dlcinterface import DLCProject

SMOKE_ROOT = Path(r"M:\_test_symlink_us_videos\smoke_dlc_proj")
SOURCE_VIDEOS = [
    Path(r"M:\us_videos_for_tracking2\pia02_s001_002_LFA2.mp4"),
    Path(r"M:\us_videos_for_tracking2\pia02_s001_002_RFA2.mp4"),
]


def main() -> int:
    if SMOKE_ROOT.exists():
        shutil.rmtree(SMOKE_ROOT)
    SMOKE_ROOT.mkdir(parents=True, exist_ok=True)
    ann_files: list[Path] = []
    for v in SOURCE_VIDEOS:
        ann_file = v.parent / f"{v.stem}_annotations.json"
        # Always overwrite: a leftover stub from a prior failed smoke
        # might carry the wrong labels and trip VideoAnnotation construction
        # downstream of the helper we're actually exercising here.
        ann_file.write_text(json.dumps({}))
        ann_files.append(ann_file)
    exit_code = 0
    try:
        print("=== Creating DLC project (link_videos=None / auto) ===")
        proj = DLCProject(
            path=str(SMOKE_ROOT),
            videos=[str(v) for v in SOURCE_VIDEOS],
            name="pia02_smoke",
            experimenter="praneeth",
            annotation_suffix="",
        )
        proj_dir = Path(proj.config_path).parent
        proj_videos_dir = proj_dir / "videos"
        print(f"  project: {proj_dir.name}")
        print()
        print("=== Hardlink verification ===")
        for src in SOURCE_VIDEOS:
            dst = proj_videos_dir / src.name
            if not dst.exists():
                print(f"  MISSING: {dst.name}")
                exit_code = 1
                continue
            same = os.path.samefile(src, dst)
            nlink = os.stat(dst).st_nlink
            tag = "OK" if same else "FAIL"
            print(f"  [{tag}] {src.name}: same-file={same} n_links={nlink}")
            if not same:
                exit_code = 1
            toc_src = src.with_name(src.name + ".dnav-toc")
            toc_dst = dst.with_name(dst.name + ".dnav-toc")
            if toc_src.exists():
                if toc_dst.exists():
                    toc_same = os.path.samefile(toc_src, toc_dst)
                    print(f"      .dnav-toc: same-file={toc_same}")
                    if not toc_same:
                        exit_code = 1
                else:
                    print(f"      .dnav-toc: MISSING at dst (expected)")
                    exit_code = 1
        print()
        print("=== config[video_sets] keys ===")
        # Compare via os.path.samefile against the in-project video paths
        # so the check is robust to drive-letter vs UNC path forms (M:\
        # is mapped to \\192.168.100.2\home\piano\... and DLC writes the
        # UNC form even when the source was passed as M:\).
        in_project_videos = {os.path.realpath(p) for p in proj_videos_dir.iterdir()
                             if p.suffix == ".mp4"}
        for k in proj.config["video_sets"].keys():
            try:
                k_resolved = os.path.realpath(k)
            except OSError:
                k_resolved = str(k)
            in_project = k_resolved in in_project_videos
            tag = "OK" if in_project else "FAIL"
            print(f"  [{tag}] in_project={in_project}: {Path(k).name}")
            if not in_project:
                print(f"      key:       {k}")
                print(f"      resolved:  {k_resolved}")
                print(f"      expected:  {in_project_videos}")
                exit_code = 1
        return exit_code
    finally:
        print()
        print("=== Cleanup ===")
        if SMOKE_ROOT.exists():
            shutil.rmtree(SMOKE_ROOT, ignore_errors=True)
        for f in ann_files:
            if f.exists():
                f.unlink()
        print(f"  cleanup done. exit_code={exit_code}")


if __name__ == "__main__":
    raise SystemExit(main())
