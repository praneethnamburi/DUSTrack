### work in progress: add to dlcinterface.py later
# when trying to import DLC
from . import imagesimilarity

import sys
sys.path.append(str(Path(__file__).parent.parent))

if True:
    import json 
    class LabeledData:
        """Manage labeled data and provide methods to adjust labeled data."""
        def __init__(self, dlcp: DLCProject):
            self.dlcp = dlcp # parent DLCProject
            self.labeled_data = self.load_labeled_data()
            # if there is an image file without a corresponding position, don't add it to the image sequence

            fm = FileManager(str(dlcp.paths['labels']))
            fm.add("images", "*.png")
            all_images = []
            for fname in fm.all_files:
                video_stem = Path(fname).parent.stem 
                frame_num = int(Path(fname).stem[3:])
                if frame_num not in self.labeled_data[video_stem].frames:
                    print(f"Skipping {fname} in {video_stem} - perhaps the labels were deleted.")
                else:
                    all_images.append(fname)

            self.image_sequence = {}
            self.image_sequence["default"] = imagesimilarity.ImageSequence(all_images)

            # create an "_archive" folder in the main DLCproject folder
            self.temp_path = dlcp.paths['project'] / '_archive'
            if not self.temp_path.exists():
                self.temp_path.mkdir()

        def order(self, method="imagehash"):
            assert method == "all" or method in imagesimilarity.ORDERING_METHODS, f"Method {method} not recognized. Must be one of {imagesimilarity.ORDERING_METHODS} or 'all'."
            if method == "all":
                for m in imagesimilarity.ORDERING_METHODS:
                    if m not in self.image_sequence:
                        self.image_sequence[m] = self.image_sequence["default"].order(method=m)
                return self.image_sequence
            
            # skip if already ordered
            if method in self.image_sequence:
                return self.image_sequence[method]
            
            self.image_sequence[method] = self.image_sequence["default"].order(method=method)
            return self.image_sequence[method]
        
        def load_labeled_data(self):
            # load all the labeled data
            fm = FileManager(self.dlcp.paths['labels'])
            fm.add("CollectedData", "CollectedData_*.h5")
            ret = {}
            for fname in fm["CollectedData"]:
                video_stem = Path(fname).parent.stem 
                print(f'Loading labeled data from {fname}')
                ann = VideoAnnotation(fname)
                ann.name = video_stem
                # do a sanity check
                dangling_labels = set(ann.frames) - set(ann.frames_overlapping)
                if dangling_labels:
                    print(f"Dangling labels found in {video_stem}: {dangling_labels}")
                ret[video_stem] = ann
            return ret
        
        def prepare(self, ordering="default"):
            this_name = f'{self.dlcp.name}_labeled_data_{ordering}'

            # create a video with the current ordering
            if ordering not in self.image_sequence:
                self.order(method=ordering)
            video_name = str(self.temp_path / f'{this_name}.mp4')
            if os.path.exists(video_name):
                print(f'Video {video_name} already exists. Skipping video creation.')
            else:
                self.image_sequence[ordering].create_video(video_name, fps=1)
            
            # check that the same points are labeled in all the videos
            target_labels = None
            for ann in self.labeled_data.values():
                if target_labels is None:
                    target_labels = set(ann.labels)
                elif set(ann.labels) != target_labels:
                    raise ValueError("All labeled data files must have the same set of labels.")
            label_list = list(target_labels)

            # get the positions of labeled data in each frame
            fname_mapping_json = self.temp_path / f'{this_name}_frame_mapping.json'
            if ordering not in self.image_sequence:
                self.order(method=ordering)
            imseq = self.image_sequence[ordering]
            if os.path.exists(fname_mapping_json):
                with open(fname_mapping_json, 'r') as f:
                    frame_mapping = json.load(f)
                print(f'Loaded frame mapping from {fname_mapping_json}')
            else:
                frame_mapping = {frame_num: (Path(x.fpath).parent.stem, int(Path(x.fpath).stem[3:])) for frame_num, x in enumerate(imseq)}
                # save frame_mapping to a json file with extenson .framemapping
                with open(fname_mapping_json, 'w') as f:
                    json.dump(frame_mapping, f, indent=4)
                print(f'Saved frame mapping to {fname_mapping_json}')

            ann_fname = str(self.temp_path / f"{this_name}_annotations_current.json")
            if os.path.exists(ann_fname):
                print(f'Annotation file {ann_fname} already exists. Skipping annotation creation.')
                return
            
            ann = VideoAnnotation()
            ann.name = "current"
            ann.fstem = f"{this_name}_annotations_{ann.name}"
            ann.fname = ann_fname
            
            for label in label_list:
                for frame_num in range(len(imseq)):
                    video_stem, orig_frame_num = frame_mapping[frame_num]
                    pos = self.labeled_data[video_stem].data[label][orig_frame_num]
                    ann.data[label][frame_num] = pos
            
            ann.save()

        def prepare_all(self):
            for method in ["default"] + imagesimilarity.ORDERING_METHODS:
                self.prepare(ordering=method)
        
        def refine(self, ordering="default"):
            this_name = f'{self.dlcp.name}_labeled_data_{ordering}'
            video_name = str(self.temp_path / f'{this_name}.mp4')
            return DUSTrack(video_name, "current")

if __name__ == "__main__":
    # example usage
    dlcp = DLCProject(r"M:\DLC_MODELS\general\interosseous_pn23-x-2025-09-11\config.yaml")
    ld = LabeledData(dlcp)
    ld.prepare_all()
    a = 1