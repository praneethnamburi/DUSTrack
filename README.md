# DUSTrack

[![src](https://img.shields.io/badge/src-github-blue)](https://github.com/praneethnamburi/DUSTrack)
[![PyPI - Version](https://img.shields.io/pypi/v/DUSTrack.svg?logo=pypi&label=PyPI&logoColor=gold)](https://pypi.org/project/DUSTrack/)
[![Documentation Status](https://readthedocs.org/projects/DUSTrack/badge/?version=latest)](https://DUSTrack.readthedocs.io)
[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://raw.githubusercontent.com/praneethnamburi/DUSTrack/main/LICENSE)

*Semi-automated point tracking in videos. Designed for ultrasound videos, but works with natural videos as well.*

`DUSTrack` (Deep learning and optical flow-based toolkit for UltraSound Tracking) is a semi-automated framework for tracking arbitrary points in B-mode ultrasound videos. It combines deep learning with optical flow to deliver high-quality, robust tracking across diverse anatomical structures and motion patterns. The toolkit includes a graphical user interface that streamlines the generation of high-quality training data and supports iterative model refinement. It also implements a novel optical-flow-based filtering technique that reduces high-frequency frame-to-frame noise while preserving rapid tissue motion.

## Features

- **Hybrid approach**: Combines deep learning with optical flow for accurate tracking
- **User-friendly GUI**: Streamlines training data generation and model refinement
- **Noise reduction**: Novel optical-flow-based filtering preserves rapid motion while reducing frame-to-frame noise
- **Versatile**: Works with ultrasound and other video types
- **Flexible installation**: Use GUI + optical flow only, or add deep learning capabilities

Learn more about DUSTrack in our [preprint](https://arxiv.org/abs/2507.14368).

## Installation

### Option 1: GUI + Optical Flow Only (Recommended for Short Videos)

For tracking points in videos with a few hundred frames, this lightweight installation is sufficient:

```sh
pip install DUSTrack
```

**Troubleshooting dependencies**: If you encounter dependency issues, use conda with the provided `requirements.yml` file:

```sh
conda env create -n env-dustrack -f https://github.com/praneethnamburi/DUSTrack/raw/main/requirements.yml
conda activate env-dustrack
pip install DUSTrack
```

### Option 2: Full Installation (Including Deep Learning)

For longer videos or ultrasound videos with repetitive motions, deep learning significantly reduces manual effort and improves tracking quality:

1. Create a conda environment and install DeepLabCut by following [these instructions](https://deeplabcut.github.io/DeepLabCut/docs/installation.html)
2. Activate your DeepLabCut environment and install DUSTrack:
   ```sh
   conda activate <your-dlc-env>
   pip install DUSTrack
   ```

## Quick Start

```python
from dustrack import DUSTrack
import datanavigator

# Launch the GUI with an example video
video_path = datanavigator.get_example_video()  # or use your own video path
d = DUSTrack(video_path, "pn") 
# The second argument is the name of the "layer" for storing tracking annotations
```

**Next steps:**
- Use the GUI to mark points of interest in your video (see [Keyboard shortcuts](https://github.com/praneethnamburi/DUSTrack/raw/main/docs/source/resources/keyboard_shortcuts.png))
- Track points using optical flow and/or train a deep learning model
- Export tracking results as a `.json` file for further analysis

For detailed tutorials and examples, see the [documentation](https://DUSTrack.readthedocs.io).

## Documentation

Full documentation is available at [DUSTrack.readthedocs.io](https://DUSTrack.readthedocs.io).

## Citation

If you use DUSTrack in your research, please cite our paper:

```bibtex
@article{namburi2025dustrack,
  title={DUSTrack: Semi-automated point tracking in ultrasound videos},
  author={Namburi, Praneeth and Pallar{\`e}s-L{\'o}pez, Roger and Rosendorf, Jessica and Folgado, Duarte and Anthony, Brian W},
  journal={arXiv preprint arXiv:2507.14368},
  year={2025}
}
```

## Contributing

Contributions are welcome! Please feel free to:
- Submit a Pull Request with improvements or bug fixes
- Share your use cases and feedback ([contact](https://praneethnamburi.com/contact/))

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

[Praneeth Namburi](https://praneethnamburi.com)

Project Link: [https://github.com/praneethnamburi/DUSTrack](https://github.com/praneethnamburi/DUSTrack)


## Acknowledgments

[MIT.nano Immersion Lab](https://immersion.mit.edu)

[NCSOFT](https://nc.com)
