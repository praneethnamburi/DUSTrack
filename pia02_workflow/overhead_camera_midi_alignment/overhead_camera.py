import xml.etree.ElementTree as ET
from datetime import datetime
import cv2

from midi import format_time_range

NAMESPACE = {'nrt': 'urn:schemas-professionalDisc:nonRealTimeMeta:ver.2.20'}


class OverheadCamera:
    """Sony FX30 overhead camera recording metadata.

    Parses the XML sidecar file for recording metadata and the MP4 file
    for video properties (via cv2).

    Args:
        xml_path: Path to the Sony XML sidecar file.
        mp4_path: Path to the corresponding MP4 video file.
    """

    def __init__(self, xml_path, mp4_path):
        self.xml_path = xml_path
        self.mp4_path = mp4_path

        self._tree = ET.parse(xml_path)
        self._root = self._tree.getroot()

        self._parse_xml()
        self._parse_mp4()

    def _parse_xml(self):
        root = self._root
        ns = NAMESPACE

        # Duration in frames
        duration_elem = root.find('nrt:Duration', ns)
        self.duration_frames = int(duration_elem.get('value'))

        # CreationDate (timezone-aware)
        creation_elem = root.find('nrt:CreationDate', ns)
        self.creation_date = datetime.fromisoformat(creation_elem.get('value'))

        # LTC timecodes
        ltc_table = root.find('nrt:LtcChangeTable', ns)
        if ltc_table is not None:
            self.ltc_tc_fps = ltc_table.get('tcFps')
            self.ltc_changes = [
                {
                    'frameCount': int(lc.get('frameCount')),
                    'value': lc.get('value'),
                    'status': lc.get('status'),
                }
                for lc in ltc_table.findall('nrt:LtcChange', ns)
            ]
        else:
            self.ltc_tc_fps = None
            self.ltc_changes = []

        # VideoFrame: capture and format fps
        video_frame = root.find('.//nrt:VideoFrame', ns)
        self.capture_fps = float(video_frame.get('captureFps').rstrip('pi'))
        self.format_fps = float(video_frame.get('formatFps').rstrip('pi'))

        # VideoLayout
        video_layout = root.find('.//nrt:VideoLayout', ns)
        self.pixel = video_layout.get('pixel')
        self.num_vertical_lines = video_layout.get('numOfVerticalLine')
        self.aspect_ratio = video_layout.get('aspectRatio')

        # Device info
        device = root.find('nrt:Device', ns)
        if device is not None:
            self.manufacturer = device.get('manufacturer')
            self.model_name = device.get('modelName')
        else:
            self.manufacturer = None
            self.model_name = None

        # RecordingMode
        rec_mode = root.find('nrt:RecordingMode', ns)
        self.recording_mode = rec_mode.get('type') if rec_mode is not None else None

    def _parse_mp4(self):
        cap = cv2.VideoCapture(str(self.mp4_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.mp4_path}")
        self.mp4_fps = cap.get(cv2.CAP_PROP_FPS)
        self.mp4_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.mp4_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.mp4_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

    @property
    def duration(self):
        """Wall-clock duration in seconds.

        Uses capture_fps (real recording rate, e.g. 239.76fps) to convert
        frame count to actual elapsed time.
        """
        return self.duration_frames / self.capture_fps

    @property
    def playback_duration(self):
        """Playback duration in seconds (slow-motion, as stored in file).

        Uses format_fps (e.g. 23.98fps), so this is ~10x the wall-clock
        duration for 239.76fps captured into 23.98fps.
        """
        return self.duration_frames / self.format_fps

    def get_recording_time_range(self, fmt="datetime"):
        """Return (start_time, end_time, duration) for this camera recording.

        Start time: from XML CreationDate.
        Duration: wall-clock duration (duration_frames / capture_fps).
        End time: start + duration.

        Args:
            fmt: "datetime" returns datetime objects; "unix" returns float timestamps.

        Returns:
            (start, end, duration) tuple.
        """
        return format_time_range(self.creation_date, self.duration, fmt)

    def __repr__(self):
        return (
            f"OverheadCamera(model={self.model_name}, "
            f"capture_fps={self.capture_fps}, "
            f"duration={self.duration:.2f}s, "
            f"frames={self.duration_frames})"
        )
