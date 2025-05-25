from typing import NamedTuple, List, Tuple, Union


class NodeInfo(NamedTuple):
    type: str
    station_idx: int
    time: int


class RuntimeKey(NamedTuple):
    from_station_idx: int
    to_station_idx: int
    speed: int


class TrainData(NamedTuple):
    id: str
    speed: int
    stops: List[int]


class BlockSectionTime(NamedTuple):
    section_name: str  # e.g., "A-B"
    time_300: int  # time for speed 300
    time_350: int  # time for speed 350


Node = Union[str, NodeInfo]
ArcKey = Tuple[str, Node, Node]

STATIONS_DATA = ["A", "B", "C", "D", "E", "F", "G"]
STATION_MILEAGES_DATA = {"A": 0, "B": 50, "C": 100, "D": 170, "E": 210, "F": 250, "G": 300}
TRAIN_DATA = [
    TrainData(id="G1", speed=350, stops=[1, 0, 0, 0, 0, 1, 1]),
    TrainData(id="G3", speed=350, stops=[1, 1, 1, 1, 0, 0, 1]),
    TrainData(id="G5", speed=350, stops=[1, 1, 0, 1, 0, 1, 1]),
    TrainData(id="G7", speed=350, stops=[1, 1, 0, 0, 1, 0, 1]),
    TrainData(id="G9", speed=350, stops=[1, 1, 0, 0, 0, 1, 1]),
    TrainData(id="G11", speed=350, stops=[1, 1, 0, 0, 1, 0, 1]),
    TrainData(id="G13", speed=300, stops=[1, 1, 1, 1, 1, 1, 1]),
    TrainData(id="G15", speed=300, stops=[1, 1, 1, 1, 1, 1, 1]),
    TrainData(id="G17", speed=300, stops=[1, 1, 1, 1, 1, 1, 1]),
    TrainData(id="G19", speed=300, stops=[1, 1, 1, 1, 1, 1, 1]),
    TrainData(id="G21", speed=300, stops=[1, 1, 1, 1, 1, 1, 1]),
    TrainData(id="G23", speed=300, stops=[1, 1, 1, 1, 1, 1, 1]),
    TrainData(id="G25", speed=300, stops=[1, 1, 1, 1, 1, 1, 1]),
    TrainData(id="G27", speed=300, stops=[1, 1, 1, 1, 1, 1, 1]),
    TrainData(id="G29", speed=300, stops=[1, 1, 1, 1, 1, 1, 1]),
    TrainData(id="G31", speed=300, stops=[1, 1, 1, 1, 1, 1, 1]),
]

BLOCK_SECTION_TIMES_DATA = [
    BlockSectionTime(section_name="A-B", time_300=10, time_350=9),
    BlockSectionTime(section_name="B-C", time_300=20, time_350=18),
    BlockSectionTime(section_name="C-D", time_300=14, time_350=12),
    BlockSectionTime(section_name="D-E", time_300=8, time_350=7),
    BlockSectionTime(section_name="E-F", time_300=8, time_350=7),
    BlockSectionTime(section_name="F-G", time_300=10, time_350=8)
]
