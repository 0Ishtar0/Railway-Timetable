from typing import Optional, Iterable

import matplotlib.pyplot as plt

from data import TrainData, BlockSectionTime, NodeInfo, RuntimeKey, ArcKey


class RailwayTimetablingSolver:
    def __init__(self,
                 stations: list[str],
                 station_mileages: dict[str, int],
                 train_specs: list[TrainData],
                 block_section_times: list[BlockSectionTime],
                 t_max: int = 160,
                 headway: int = 5,
                 min_stop_dwell: int = 2,
                 max_stop_dwell: int = 15,
                 pass_dwell: int = 0):
        self.stations: list[str] = stations
        self.station_mileages: dict[str, int] = station_mileages
        self.num_stations: int = len(stations)
        self.train_specs_list: list[TrainData] = train_specs
        self.train_specs_map: dict[str, TrainData] = {ts.id: ts for ts in train_specs}
        self.train_ids: list[str] = [ts.id for ts in train_specs]
        self.block_section_times_list: list[BlockSectionTime] = block_section_times
        self.t_max: int = t_max
        self.time_points: range = range(t_max + 1)
        self.headway: int = headway
        self.min_stop_dwell: int = min_stop_dwell
        self.max_stop_dwell: int = max_stop_dwell
        self.pass_dwell: int = pass_dwell
        self.node_sigma: str = "sigma"
        self.node_tau: str = "tau"

        self.runtimes: dict[RuntimeKey, int] = {}
        for i in range(self.num_stations - 1):
            s_from, s_to = self.stations[i], self.stations[i + 1]
            key_pair = f"{s_from}-{s_to}"
            for entry in self.block_section_times_list:
                if entry.section_name == key_pair:
                    self.runtimes[RuntimeKey(i, i + 1, 300)] = entry.time_300
                    self.runtimes[RuntimeKey(i, i + 1, 350)] = entry.time_350
                    break

    def get_station_idx(self, station_name: str) -> int:
        return self.stations.index(station_name)

    def get_dwell_times(self, train_id: str, station_idx: int) -> Iterable[int]:
        if station_idx == 0 or station_idx == self.num_stations - 1:  # Origin or Destination
            return [0]

        train_spec = self.train_specs_map[train_id]
        if not isinstance(train_spec.stops, (list, tuple)) or station_idx >= len(train_spec.stops):
            return [self.pass_dwell]

        is_stop: bool = train_spec.stops[station_idx] == 1
        if is_stop:
            return range(self.min_stop_dwell, self.max_stop_dwell + 1)
        else:
            return [self.pass_dwell]

    def solve(self) -> Optional[dict[ArcKey, float]]:
        raise NotImplementedError("Solver not implemented yet.")

    def plot_timetable(self, solution: dict[ArcKey, float]) -> None:
        plt.figure(figsize=(18, 12))

        # Extract train_ids from the solution keys
        train_ids_list_param = sorted(list(set(key[0] for key in solution.keys())))

        train_colors = plt.colormaps.get_cmap('tab20')

        for i, train_id in enumerate(train_ids_list_param):
            color = train_colors(i)
            path_events = []

            current_node = self.node_sigma
            initial_dep_time_at_A = -1

            # Find the starting arc for the current train
            for sol_key, sol_val in solution.items():
                s_train, s_from_node, s_to_node = sol_key

                if s_train == train_id and s_from_node == self.node_sigma and sol_val > 0.5:
                    if isinstance(s_to_node, NodeInfo) and \
                            s_to_node.type == 'W' and s_to_node.station_idx == 0:
                        initial_dep_time_at_A = s_to_node.time
                        path_events.append({
                            'time': initial_dep_time_at_A,
                            'station_idx': 0,
                            'type': 'dep',
                            'raw_node': s_to_node
                        })
                        current_node = s_to_node
                        break

            if initial_dep_time_at_A == -1:
                continue

            visited_arcs_count = 0
            max_arcs_to_trace = self.num_stations * 4

            while current_node != self.node_tau and visited_arcs_count < max_arcs_to_trace:
                found_next_arc = False
                for sol_key, sol_val in solution.items():
                    s_train, s_from_node, s_to_node = sol_key
                    if s_train == train_id and s_from_node == current_node and sol_val > 0.5:

                        event_type = None
                        event_station_idx = None
                        event_time = None

                        if isinstance(s_to_node, NodeInfo):
                            event_station_idx = s_to_node.station_idx
                            event_time = s_to_node.time
                            if s_to_node.type == 'U':
                                event_type = 'arr'
                            elif s_to_node.type == 'W':
                                event_type = 'dep'
                        elif s_to_node == self.node_tau:
                            pass

                        if event_type and event_station_idx is not None and event_time is not None:
                            path_events.append({
                                'time': event_time,
                                'station_idx': event_station_idx,
                                'type': event_type,
                                'raw_node': s_to_node
                            })

                        current_node = s_to_node
                        found_next_arc = True
                        visited_arcs_count += 1
                        break

                if not found_next_arc and current_node != self.node_tau:
                    break

            path_events.sort(key=lambda e: (e['time'], 0 if e['type'] == 'arr' else 1))

            if not path_events:
                continue

            plot_points_x = []
            plot_points_y = []

            for event in path_events:
                if event['station_idx'] < self.num_stations:
                    plot_points_x.append(event['time'])
                    plot_points_y.append(self.station_mileages[self.stations[event['station_idx']]])

            if plot_points_x:
                plt.plot(plot_points_x, plot_points_y, color=color,
                         marker='o', markersize=3, linestyle='-')
                plt.text(plot_points_x[0] + 0.5, plot_points_y[0],
                         train_id, fontsize=7, color=color, va='center')

        plt.xlabel("Time (minutes)")
        plt.ylabel("Station / Space (km)")
        plt.title("Train Timetable")

        station_ticks_loc = [self.station_mileages[s] for s in self.stations]
        plt.yticks(station_ticks_loc, self.stations)

        plt.xticks(range(0, self.t_max + 1, 10))
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.gca().invert_yaxis()
        plt.tight_layout()

        try:
            plt.savefig("train_timetable_plot.pdf")
            print("Timetable plot saved to train_timetable_plot.pdf")
            plt.show()
        except Exception as e:
            print(f"Error saving or showing plot: {e}")
