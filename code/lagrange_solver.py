import time
from collections import defaultdict
from typing import Literal

import rustworkx as rx

from base_solver import ArcKey
from data import NodeInfo, TrainData, BlockSectionTime, Node
from graph_based_solver import GraphBasedSolver


class LagrangianRelaxationSolver(GraphBasedSolver):
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
                 max_iterations: int = 50,
                 initial_eta: float = 0.1,
                 eta_decay_factor: float = 0.995,
                 objective_type: Literal["max_trains", "min_total_runtime"] = "max_trains",
                 verbose: bool = False):
        super().__init__(stations, station_mileages, train_specs, block_section_times,
                         t_max, headway, min_stop_dwell, max_stop_dwell, pass_dwell,
                         profit_setting=objective_type)

        # Lagrange multipliers: Keyed by (node_type, station_idx, t_anchor_of_headway_window)
        self.lambdas: dict[tuple[str, int, int], float] = defaultdict(float)
        self.max_iterations = max_iterations
        self.initial_eta = initial_eta
        self.eta_decay_factor = eta_decay_factor
        self.objective_type = objective_type
        self.verbose = verbose

        # Define keys for all headway constraints q=(type, station, t_anchor)
        self.headway_constraint_keys: list[tuple[str, int, int]] = []
        for s_idx in range(self.num_stations):
            for node_type_str in ['U', 'W']:
                for t0 in range(self.t_max - self.headway + 2):
                    self.headway_constraint_keys.append((node_type_str, s_idx, t0))

        if self.verbose:
            print("Pre-generating potential arcs for all trains...")
            print("Arc pre-generation complete.")

    def _calculate_lambda_penalty_for_node(self, node_w: NodeInfo) -> float:
        penalty = 0.0
        for t_anchor_offset in range(self.headway):
            t_anchor = node_w.time - t_anchor_offset
            if t_anchor < 0:
                continue

            q_key = (node_w.type, node_w.station_idx, t_anchor)
            if q_key in self.lambdas:
                if t_anchor <= node_w.time < t_anchor + self.headway:
                    penalty += self.lambdas.get(q_key, 0.0)
        return penalty

    def _solve_train_subproblem_spp(self, train_id: str, train_potential_arcs: list[tuple[Node, Node]]) \
            -> tuple[list[ArcKey], float, float, set[NodeInfo]]:
        G_sub = rx.PyDiGraph()

        node_indices: dict[Node, int] = {}
        unique_nodes: set[Node] = set()
        for u_node, v_node in train_potential_arcs:
            unique_nodes.add(u_node)
            unique_nodes.add(v_node)

        for node in unique_nodes:
            idx = G_sub.add_node(node)
            node_indices[node] = idx

        for u_node, v_node in train_potential_arcs:
            original_profit = self._get_arc_original_profit(train_id, u_node, v_node)
            penalty = 0.0
            if isinstance(v_node, NodeInfo):  # Penalty is associated with occupying/arriving at v_node
                penalty = self._calculate_lambda_penalty_for_node(v_node)

            modified_profit = original_profit - penalty

            u_idx = node_indices[u_node]
            v_idx = node_indices[v_node]
            G_sub.add_edge(u_idx, v_idx, {'mod_profit': modified_profit,
                                          'orig_profit': original_profit})

        # Find the longest path based on modified_profit
        max_modified_profit, path_uv_tuples = self._solve_longest_path_dag_dp(
            G_sub, node_indices, self.node_sigma, self.node_tau, 'mod_profit')

        if max_modified_profit == float('-inf'):  # No path found
            return [], 0.0, 0.0, set()

        # Reconstruct path details
        final_path_arcs: list[ArcKey] = []
        total_original_profit_on_path = 0.0
        nodes_occupied_by_path: set[NodeInfo] = set()

        for u_n, v_n in path_uv_tuples:
            final_path_arcs.append((train_id, u_n, v_n))
            u_idx = node_indices[u_n]
            v_idx = node_indices[v_n]
            arc_data = G_sub.get_edge_data(u_idx, v_idx)
            total_original_profit_on_path += arc_data.get('orig_profit', 0.0)
            # Collect physical nodes occupied by this path
            if isinstance(u_n, NodeInfo):
                nodes_occupied_by_path.add(u_n)
            if isinstance(v_n, NodeInfo):
                nodes_occupied_by_path.add(v_n)

        return final_path_arcs, total_original_profit_on_path, max_modified_profit, nodes_occupied_by_path

    def _primal_heuristic(self,
                          train_subproblem_results: list[tuple[str,
                                                               list[ArcKey], float, float, set[NodeInfo]]]
                          ) -> tuple[dict[ArcKey, float], float, set[NodeInfo]]:
        sorted_trains_data = sorted(train_subproblem_results, key=lambda x: x[3], reverse=True)

        feasible_timetable_arcs: dict[ArcKey, float] = {}  # Stores the final feasible paths
        # Tracks nodes taken by trains in this heuristic
        occupied_nodes_in_heuristic: set[NodeInfo] = set()
        total_heuristic_original_profit = 0.0

        for train_id, _, _, _, _ in sorted_trains_data:  # We only need train_id for SPP
            G_heuristic = rx.PyDiGraph()
            train_potential_arcs = self._train_potential_arcs_map[train_id]

            # Create node mappings for this train's graph
            node_indices: dict[Node, int] = {}
            heuristic_spp_nodes: set[Node] = set()
            for u_n, v_n in train_potential_arcs:
                heuristic_spp_nodes.add(u_n)
                heuristic_spp_nodes.add(v_n)

            # Add nodes to graph and create mappings
            for n in heuristic_spp_nodes:
                idx = G_heuristic.add_node(n)
                node_indices[n] = idx

            # Add valid arcs with original profits
            for u_node, v_node in train_potential_arcs:
                valid_arc = True
                if isinstance(v_node, NodeInfo):
                    for existing_occ_node in occupied_nodes_in_heuristic:
                        if (v_node.type == existing_occ_node.type and
                                v_node.station_idx == existing_occ_node.station_idx and
                                abs(v_node.time - existing_occ_node.time) < self.headway):
                            valid_arc = False
                            break
                if not valid_arc:
                    continue

                # Also check u_node if it's a physical departure node (W type)
                if isinstance(u_node, NodeInfo) and u_node.type == 'W':
                    for existing_occ_node in occupied_nodes_in_heuristic:
                        if (u_node.type == existing_occ_node.type and
                                u_node.station_idx == existing_occ_node.station_idx and
                                abs(u_node.time - existing_occ_node.time) < self.headway):
                            valid_arc = False
                            break
                if not valid_arc:
                    continue

                original_profit = self._get_arc_original_profit(train_id, u_node, v_node)
                u_idx = node_indices[u_node]
                v_idx = node_indices[v_node]
                G_heuristic.add_edge(u_idx, v_idx, {'orig_profit_h': original_profit})

            current_train_profit, path_uv_tuples_h = self._solve_longest_path_dag_dp(
                G_heuristic, node_indices, self.node_sigma, self.node_tau, 'orig_profit_h'
            )

            if current_train_profit > float('-inf'):
                total_heuristic_original_profit += current_train_profit
                newly_occupied_by_this_train: set[NodeInfo] = set()
                for u_n, v_n in path_uv_tuples_h:
                    feasible_timetable_arcs[(train_id, u_n, v_n)] = 1.0
                    if isinstance(u_n, NodeInfo):
                        newly_occupied_by_this_train.add(u_n)
                    if isinstance(v_n, NodeInfo):
                        newly_occupied_by_this_train.add(v_n)
                occupied_nodes_in_heuristic.update(newly_occupied_by_this_train)
            elif self.verbose:
                print(f"Train {train_id} skipped in primal heuristic (no valid conflict-free path).")

        return feasible_timetable_arcs, total_heuristic_original_profit, occupied_nodes_in_heuristic

    def solve(self) -> dict[ArcKey, float]:
        start_solve_time = time.time()

        upper_bound = float('inf')
        lower_bound = float('-inf')
        best_feasible_solution_arcs: dict[ArcKey, float] = {}

        eta = self.initial_eta

        if self.verbose:
            print(
                f"Starting Lagrangian Relaxation: MaxIter={self.max_iterations}, Eta0={self.initial_eta}, Profit={self.objective_type}")
            print(
                f"Number of headway constraint groups (lambdas): {len(self.headway_constraint_keys)}")

        for k_iter in range(self.max_iterations):
            iter_start_time = time.time()

            current_L_val_sum_spp_duals = 0.0
            subproblem_results_this_iter: list[tuple[str,
                                                     list[ArcKey], float, float, set[NodeInfo]]] = []
            all_nodes_occupied_in_subproblems: set[NodeInfo] = set()

            for train_id in self.train_ids:
                train_arcs = self._train_potential_arcs_map[train_id]
                path_arcs, path_orig_profit, path_dual_profit, path_nodes = \
                    self._solve_train_subproblem_spp(train_id, train_arcs)

                subproblem_results_this_iter.append(
                    (train_id, path_arcs, path_orig_profit, path_dual_profit, path_nodes)
                )
                current_L_val_sum_spp_duals += path_dual_profit
                all_nodes_occupied_in_subproblems.update(path_nodes)

            current_ub_val = current_L_val_sum_spp_duals + sum(self.lambdas.get(q_key, 0.0)
                                                               for q_key in self.headway_constraint_keys)
            upper_bound = min(upper_bound, current_ub_val)

            heuristic_arcs, heuristic_obj, _ = self._primal_heuristic(subproblem_results_this_iter)

            if heuristic_obj > lower_bound:
                lower_bound = heuristic_obj
                best_feasible_solution_arcs = heuristic_arcs

            iter_time = time.time() - iter_start_time
            if self.verbose or (k_iter % 5 == 0):
                print(
                    f"Iter {k_iter:3d}: LB={lower_bound:8.2f}, UB={upper_bound:8.2f}, Eta={eta:.5f}, Time={iter_time:5.2f}s")

            feas = self._calculate_feasibility_metric(best_feasible_solution_arcs)
            self.feassiability.append(feas)
            if lower_bound > float('-inf') and upper_bound < float('inf'):
                if upper_bound - lower_bound <= 0.1 * abs(upper_bound):
                    if feas == 0:
                        if self.verbose:
                            print("Stopping: UB-LB tolerance met with feasible solution.")
                        break

            if k_iter == self.max_iterations - 1:
                if self.verbose:
                    print("Stopping: Max iterations reached.")
                break

            for q_key in self.headway_constraint_keys:
                q_type, q_station, q_anchor_t = q_key

                num_trains_in_q_window_subproblem = 0
                for t_offset in range(self.headway):
                    time_in_window = q_anchor_t + t_offset
                    if time_in_window > self.t_max:
                        continue

                    node_in_window = NodeInfo(q_type, q_station, time_in_window)
                    if node_in_window in all_nodes_occupied_in_subproblems:
                        num_trains_in_q_window_subproblem += 1

                violation = num_trains_in_q_window_subproblem - 1.0
                self.lambdas[q_key] = max(0, self.lambdas.get(q_key, 0.0) + eta * violation)

            eta *= self.eta_decay_factor

        total_cpu_time = time.time() - start_solve_time
        print(f"\nLagrangian Relaxation Finished. Total CPU Time: {total_cpu_time:.2f} seconds.")

        final_feasibility = self._calculate_feasibility_metric(best_feasible_solution_arcs)
        print(f"Final reported solution's feasibility (sum of violations): {final_feasibility:.2f}")
        print(f"Final Lower Bound (Objective value): {lower_bound:.2f}")
        print(f"Final Upper Bound: {upper_bound:.2f}")

        return best_feasible_solution_arcs


if __name__ == "__main__":
    from data import TRAIN_DATA, STATION_MILEAGES_DATA, STATIONS_DATA, BLOCK_SECTION_TIMES_DATA

    lr_solver = LagrangianRelaxationSolver(
        stations=STATIONS_DATA,
        station_mileages=STATION_MILEAGES_DATA,
        train_specs=TRAIN_DATA,
        block_section_times=BLOCK_SECTION_TIMES_DATA,
        t_max=160,
        headway=5,
        min_stop_dwell=2,
        max_stop_dwell=15,
        pass_dwell=0,
        max_iterations=1000,
        initial_eta=0.05,
        eta_decay_factor=0.98,
        objective_type="max_trains",
        verbose=True
    )
    lr_solution = lr_solver.solve()
    lr_solver.plot_timetable(lr_solution)
    lr_solver.plot_feassibility()
