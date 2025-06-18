from typing import Optional, Literal
from collections import defaultdict

import rustworkx as rx
import matplotlib.pyplot as plt

from base_solver import RailwayTimetablingSolver, ArcKey
from data import NodeInfo, RuntimeKey, TrainData, BlockSectionTime, Node


class GraphBasedSolver(RailwayTimetablingSolver):

    def __init__(self,
                 stations: list[str],
                 station_mileages: dict[str, int],
                 train_specs: list[TrainData],
                 block_section_times: list[BlockSectionTime],
                 t_max: int = 160,
                 headway: int = 5,
                 min_stop_dwell: int = 2,
                 max_stop_dwell: int = 15,
                 pass_dwell: int = 0,
                 profit_setting: Literal["max_trains", "min_total_runtime"] = "max_trains"):
        super().__init__(stations, station_mileages, train_specs, block_section_times,
                         t_max, headway, min_stop_dwell, max_stop_dwell, pass_dwell)

        self.profit_setting = profit_setting
        self.feassiability: list[float] = []

        # Pre-generate all possible arcs for each train
        self._train_potential_arcs_map: dict[str, list[tuple[Node, Node]]] = {}
        # All unique physical nodes (NodeInfo objects) that can be occupied
        self._all_occupiable_nodes: set[NodeInfo] = set()

        for train_id in self.train_ids:
            arcs = self._get_train_specific_arcs(train_id)
            self._train_potential_arcs_map[train_id] = arcs
            for _, to_node in arcs:
                if isinstance(to_node, NodeInfo):
                    self._all_occupiable_nodes.add(to_node)

        # Pre-calculate original arc profits
        self._arc_original_profits: dict[tuple[str, Node, Node], float] = {}
        for train_id, train_arcs in self._train_potential_arcs_map.items():
            train_spec = self.train_specs_map[train_id]
            for from_node, to_node in train_arcs:
                key = (train_id, from_node, to_node)
                profit = 0.0
                if self.profit_setting == "min_total_runtime":
                    if isinstance(from_node, NodeInfo) and isinstance(to_node, NodeInfo):
                        if from_node.type == 'W' and to_node.type == 'U' and \
                                from_node.station_idx != to_node.station_idx:
                            rt_key = RuntimeKey(from_node.station_idx,
                                                to_node.station_idx, train_spec.speed)
                            runtime = self.runtimes.get(rt_key)
                            if runtime is not None:
                                profit = -float(runtime)
                elif self.profit_setting == "max_trains":
                    if from_node == self.node_sigma:
                        profit = 1.0
                self._arc_original_profits[key] = profit

    def _get_arc_original_profit(self, train_id: str, from_node: Node, to_node: Node) -> float:
        return self._arc_original_profits.get((train_id, from_node, to_node), 0.0)

    def _get_train_specific_arcs(self, train_id: str) -> list[tuple[Node, Node]]:
        arcs: list[tuple[Node, Node]] = []
        train_spec = self.train_specs_map[train_id]
        speed_val = train_spec.speed

        # 1. start arc
        for t_dep in self.time_points:
            arcs.append((self.node_sigma, NodeInfo('W', 0, t_dep)))

        # 2. station and interval arcs
        for s_idx in range(self.num_stations):
            if s_idx > 0:
                for t_arr in self.time_points:
                    node_U = NodeInfo('U', s_idx, t_arr)
                    for dt in self.get_dwell_times(train_id, s_idx):
                        t_dep_new = t_arr + dt
                        if t_dep_new in self.time_points:
                            arcs.append((node_U, NodeInfo('W', s_idx, t_dep_new)))

            if s_idx < self.num_stations - 1:
                for t_dep in self.time_points:
                    node_W = NodeInfo('W', s_idx, t_dep)
                    rt_key = RuntimeKey(s_idx, s_idx + 1, speed_val)
                    run_time = self.runtimes.get(rt_key)
                    if run_time is not None:
                        t_arr_new = t_dep + run_time
                        if t_arr_new in self.time_points:
                            arcs.append((node_W, NodeInfo('U', s_idx + 1, t_arr_new)))

        # 3. end arc
        dest_station_idx = self.num_stations - 1
        for t_arr_final in self.time_points:
            arcs.append((NodeInfo('U', dest_station_idx, t_arr_final), self.node_tau))

        return arcs

    @staticmethod
    def _solve_longest_path_dag_dp(G: rx.PyDiGraph, node_indices: dict[Node, int],
                                   source_node: Node, target_node: Node, weight_attr: str):
        if source_node not in node_indices or target_node not in node_indices:
            return float('-inf'), []

        source_idx = node_indices[source_node]
        target_idx = node_indices[target_node]

        dist: dict[int, float] = defaultdict(lambda: float('-inf'))
        parent: dict[int, Optional[int]] = {i: None for i in range(len(G))}
        dist[source_idx] = 0.0

        try:
            topo_sorted_indices = rx.topological_sort(G)
        except rx.DAGHasCycle:
            return float('-inf'), []

        for curr_idx in topo_sorted_indices:
            if dist[curr_idx] == float('-inf'):
                continue

            for edge in G.out_edges(curr_idx):
                neighbor_idx = edge[1]
                edge_data = G.get_edge_data(edge[0], edge[1])
                arc_profit = edge_data.get(weight_attr, 0.0)

                if dist[curr_idx] + arc_profit > dist[neighbor_idx]:
                    dist[neighbor_idx] = dist[curr_idx] + arc_profit
                    parent[neighbor_idx] = curr_idx

        path_arcs_rev: list[tuple[Node, Node]] = []
        if dist[target_idx] == float('-inf'):
            return float('-inf'), []

        idx_to_node = {idx: node for node, idx in node_indices.items()}

        curr_idx = target_idx
        while curr_idx != source_idx and parent[curr_idx] is not None:
            prev_idx = parent[curr_idx]
            prev_node = idx_to_node[prev_idx]
            curr_node = idx_to_node[curr_idx]
            path_arcs_rev.append((prev_node, curr_node))
            curr_idx = prev_idx

        if curr_idx != source_idx and source_idx != target_idx:
            return float('-inf'), []

        return dist[target_idx], list(reversed(path_arcs_rev))

    def _calculate_feasibility_metric(self, solution_x_arcs: dict[ArcKey, float]) -> float:
        if not solution_x_arcs:
            return 0.0

        y_vx: dict[NodeInfo, int] = defaultdict(int)
        for arc_key, val in solution_x_arcs.items():
            if val > 0.5:
                _train_id, _, to_node = arc_key
                if isinstance(to_node, NodeInfo):
                    y_vx[to_node] += 1

        total_violation = 0.0
        headway_constraint_keys_q: list[tuple[str, int, int]] = []
        for s_idx in range(self.num_stations):
            for node_type_str in ['U', 'W']:
                for t0 in range(self.t_max - self.headway + 2):
                    headway_constraint_keys_q.append((node_type_str, s_idx, t0))

        for q_type, q_station, q_anchor_t in headway_constraint_keys_q:
            num_occupations_in_window = 0
            for t_offset in range(self.headway):
                time_in_window = q_anchor_t + t_offset
                if time_in_window > self.t_max:
                    continue

                node_in_window = NodeInfo(q_type, q_station, time_in_window)
                num_occupations_in_window += y_vx.get(node_in_window, 0)
            total_violation += max(0, num_occupations_in_window - 1)
        return total_violation

    def plot_feassibility(self, filename: str = "feasibility.pdf"):
        plt.clf()
        plt.plot(self.feassiability, linewidth=1)
        plt.title('Feasibility Over Iterations')
        plt.xlabel('Iteration')
        plt.ylabel('Feasibility')
        plt.grid()
        plt.savefig(filename)
        plt.close()
