# pia02_workflow

Scripts for annotating and tracking a piano-playing ultrasound dataset using
[DUSTrack](../dustrack) (a [DeepLabCut](https://github.com/DeepLabCut/DeepLabCut)-based
keypoint tracking wrapper).

Ultrasound videos are recorded from each pianist's wrists during piano performance,
separately for the **Left ForeArm (LFA)** and **Right ForeArm (RFA)** probes, across
`n` sessions per participant. Every session therefore produces one LFA video and one
RFA video. Filenames follow the pattern:

```
pia02_s{participant_id}_{session}_{LFA|RFA}2.mp4
e.g.  pia02_s015_004_LFA2.mp4
```

The pipeline starts from a single shared **general (foundation) DLC model**,
forks per-participant / per-hand DLC projects from it, then refines each via
manual annotation and transfer-learning.

## Shared paths

These network paths recur across most scripts in this folder:

| Path | Role |
|---|---|
| `\\192.168.1.104\home\piano\us_videos_for_tracking2` | Raw `.mp4` ultrasound videos used by DLC |
| `\\192.168.1.104\home\piano\DLC_MODELS\general\interosseous_pn24-x-2025-10-24` | General (foundation) DLC project |
| `\\192.168.1.104\home\piano\DLC_MODELS\participant_models_general` | Root for per-participant, per-hand DLC projects |
| `\\192.168.1.104\home\piano\data` | Original (pre-DLC) telemed ultrasound source data |

For every participant + hand, the derived DLC project lives at:

```
\\192.168.1.104\home\piano\DLC_MODELS\participant_models_general\{participant_id}\{hand}\interosseous_pn24-x-2025-10-24\
```

The `interosseous_pn24-x-2025-10-24` subfolder name is fixed (it is the general
model's project name, copied verbatim into each participant directory).

---

## 1. `video_inference.py`

**Purpose.** Single-process DLC inference over a directory of `.mp4` videos using
the general (foundation) model. No multiprocessing, no log files — just one GPU,
one pass.

**Inputs** (CLI args, all with defaults):

| Flag | Default |
|---|---|
| `--video-dir` | `\\192.168.1.104\home\piano\us_videos_for_tracking2` |
| `--config-path` | `\\192.168.1.104\home\piano\DLC_MODELS\general\interosseous_pn24-x-2025-10-24\config.yaml` |
| `--cuda-device` | `0` |
| `--batch-size` | `16` |
| `--iteration-num` | `0` |

**Outputs.** Predictions written by `DLCProject.analyze_videos()` into the general
project's `videos/iteration-{N}/` directory. No labeled overlay video is created
(`create_video=False`).

**Logic.**

1. Sets `CUDA_VISIBLE_DEVICES` *before* importing torch.
2. Validates that the config file and video directory exist.
3. Globs all `.mp4` files in `--video-dir` (sorted).
4. Instantiates `DLCProject(path=config_path)`.
5. Calls `dlcp.analyze_videos(iteration_num, create_video=False, batchsize, videos)`.

---

## 2. `dustrack_general_model_workflow_annotate.py`

**Purpose.** Launch the DUSTrack/DLC annotation GUI for a single
`{participant_id, hand, video_index}` so an annotator can manually label
keypoints in that video.

**Inputs** (hard-coded constants at the top of the file — edit before running):

```python
participant_root_directory = r"\\192.168.1.104\home\piano\DLC_MODELS\participant_models_general"
participant_id             = 's015'      # change per session
hand                       = 'LFA'       # 'LFA' or 'RFA'
video_index                = 4           # index into the project's videos/ list
```

**Outputs.** Annotation files written by the GUI into the participant project's
`labeled-data/` directory.

**Logic.**

1. Builds the project config path:
   ```
   {participant_root_directory}/{participant_id}/{hand}/interosseous_pn24-x-2025-10-24/config.yaml
   ```
2. Exits if the config file does not exist.
3. `DLCProject(path=config_path)` → `dlcp.annotate(video_index)` → `plt.show()`.

**Path note.** The effective project config path **changes whenever you change
`participant_id` or `hand`** at the top of the script. Only those two variables
move you between projects; the root directory and the
`interosseous_pn24-x-2025-10-24` subfolder name stay fixed.

---

## 3. `create_new_project_from_general_model_project.py`

**Purpose.** Bootstrap participant- and hand-specific DLC projects by copying
the general model project and filtering its videos down to the matching
participant + hand subset. Run once to populate `participant_models_general/`
for all participants discovered in the raw video directory.

**Inputs** (hard-coded paths at the top of the file):

```python
video_root_path                 = r"\\192.168.1.104\home\piano\us_videos_for_tracking2"
participant_models_general_path = r"\\192.168.1.104\home\piano\DLC_MODELS\participant_models_general"
general_model_project_path      = r"\\192.168.1.104\home\piano\DLC_MODELS\general\interosseous_pn24-x-2025-10-24"
```

**Outputs.** For every `{participant_id, hand}` combination found in the raw
video directory, creates the following directory tree:

```
{participant_models_general_path}/{participant_id}/{hand}/interosseous_pn24-x-2025-10-24/
├── dlc-models/                  # copied from general
├── dlc-models-pytorch/          # copied from general
├── evaluation-results-pytorch/  # copied from general
├── training-datasets/           # copied from general
├── labeled-data/                # created empty
├── videos/                      # filtered subset for this participant+hand
└── config.yaml                  # copied from general, video_sets re-registered
```

**Logic.**

1. Scans `video_root_path` for `.mp4` files and parses each filename
   `pia02_s{ID}_{session}_{LFA|RFA}2.mp4` into a
   `{participant_id: {LFA: [...], RFA: [...]}}` dict.
2. For each `{participant_id, hand}` pair, creates the directory tree above.
   Skips the pair if its `interosseous_pn24-x-2025-10-24` folder already exists.
3. `shutil.copytree`s the four model / training / evaluation directories from
   the general project, then creates an empty `labeled-data/` folder.
4. Walks the general project's `videos/` recursively and copies only files
   matching `pia02_{participant_id}_*` whose 4th underscore-separated token
   starts with the current `hand` (`LFA` or `RFA`).
5. Copies `config.yaml`, blanks its `video_sets` via
   `deeplabcut.auxiliaryfunctions.edit_config(..., edits={'video_sets': {}})`,
   then re-registers the filtered video list with
   `deeplabcut.add_new_videos(..., copy_videos=True)`.

---

## 4. `create_ultrasound_tracking_spreadsheet.py`

**Purpose.** Generate per-participant, per-hand CSV spreadsheets that map each
DLC-side video to its original telemed ultrasound video, with frame counts and
empty slots for tracking progress (iteration, snapshot, correction, notes).
Used as a manual progress-tracking sheet.

**Inputs** (hard-coded paths at the top of the file):

```python
dataset_root_path = r"\\192.168.1.104\home\piano\data"
model_root_path   = r"\\192.168.1.104\home\piano\DLC_MODELS\participant_models_general"
```

**Outputs.** One CSV per `{participant, hand}` at:

```
{model_root_path}\spreadsheets\{participant_folder}_{hand}_tracking.csv
```

with columns:

| Column | Meaning |
|---|---|
| `dlc_video_name` | Filename inside the DLC project's `videos/` |
| `original_video_name` | Matching original telemed video filename |
| `frame_count` | Frame count of the DLC video (via OpenCV) |
| `dlc_video_index` | 0-based index in the sorted DLC video list |
| `iteraition` | Empty — to be filled in by hand (sic — original spelling kept) |
| `snapshot` | Empty — to be filled in by hand |
| `correction` | Empty — to be filled in by hand |
| `notes` | Empty — to be filled in by hand |

A CSV is **only written if the file does not already exist**, so re-running
this script will not overwrite manual edits.

**Logic.**

1. Lists participant folders in `model_root_path` matching `s0*` (e.g. `s035`).
2. For each participant, for each hand (`LFA`, `RFA`):
   - Looks up originals at
     `{dataset_root_path}/{participant_id_without_s_prefix}/telemed/*.mp4`.
   - Looks up DLC videos at
     `{model_root_path}/{participant_folder}/{hand}/interosseous_pn24-x-2025-10-24/videos/*.mp4`.
   - Strips `_LFA2.mp4` / `_RFA2.mp4` from each DLC filename to find the
     matching original.
   - Reads frame counts via `cv2.VideoCapture(...).get(CAP_PROP_FRAME_COUNT)`;
     warns if frame counts mismatch or if zero / multiple original matches are
     found for a DLC video.
   - Builds a `pandas.DataFrame` and writes the CSV if it does not already exist.

---

## 5. `dustrack_general_model_workflow_training.py`

**Purpose.** Train (refine) a participant-specific DLC model by transfer-
learning from a snapshot of the general model. Targets a single
`{participant_id, hand}` per run.

**Inputs** (hard-coded constants at the top of the file — edit before running):

```python
participant_root_directory = r"\\192.168.1.104\home\piano\DLC_MODELS\participant_models_general"
participant_id             = 's029'      # change per run
hand                       = 'LFA'       # 'LFA' or 'RFA'
source_model_path          = r"\\192.168.1.104\home\piano\DLC_MODELS\participant_models_general\snapshot-best-270.pt"
# CUDA_VISIBLE_DEVICES is set to "1" in code, before torch is imported
```

**Outputs.** Trained snapshots written into the project's `dlc-models-pytorch/`
directory, plus DLC's standard training artefacts (loss logs, evaluation
results, analyzed-video predictions) updated under the project.

**Logic.**

1. Sets `CUDA_VISIBLE_DEVICES = "1"` before importing torch.
2. Builds the project config path:
   ```
   {participant_root_directory}/{participant_id}/{hand}/interosseous_pn24-x-2025-10-24/config.yaml
   ```
3. Exits if the config file does not exist.
4. `DLCProject(path=config_path)` →
   `dlcp.process(maxiters=100, analyse_batchsize=8, create_video=False, refine=source_model_path)`,
   which runs DLC's full extract-frames → train → evaluate → analyze pipeline,
   refining from `source_model_path` instead of training from scratch.

**Path note.** As with the annotate script, the effective project config path
**changes with `participant_id` and `hand`**; everything else
(`participant_root_directory`, `interosseous_pn24-x-2025-10-24`, the source
snapshot path) stays fixed across runs.
