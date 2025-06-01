import time
from collections import defaultdict
from typing import Literal, Optional

import rustworkx as rx

from base_solver import ArcKey
from data import NodeInfo, TrainData, BlockSectionTime, Node
from graph_based_solver import GraphBasedSolver


class ADMMSolver(GraphBasedSolver):
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
                 max_iterations: int = 100,
                 rho: float = 1.0,
                 profit_setting: Literal["max_trains", "min_total_runtime"] = "max_trains",
                 verbose: bool = False):
        super().__init__(stations, station_mileages, train_specs, block_section_times,
                         t_max, headway, min_stop_dwell, max_stop_dwell, pass_dwell,
                         profit_setting=profit_setting)

        self.max_iterations = max_iterations
        self.rho = rho
        self.verbose = verbose

        self.z_global: dict[tuple[str, NodeInfo], int] = defaultdict(int)
        self.mu: dict[tuple[str, NodeInfo], float] = defaultdict(float)

    def _x_update(
            self,
            current_z_global: dict[tuple[str, NodeInfo], int],
            current_mu: dict[tuple[str, NodeInfo], float]
    ) -> tuple[dict[ArcKey, float], dict[tuple[str, NodeInfo], int], float]:
        """Solves the x-subproblem for all trains."""
        all_paths_arcs: dict[ArcKey, float] = {}
        z_local_new: dict[tuple[str, NodeInfo], int] = defaultdict(int)
        total_objective_from_x = 0.0

        for train_id in self.train_ids:
            G_x = rx.PyDiGraph()
            train_potential_arcs = self._train_potential_arcs_map[train_id]

            # Create node mappings
            node_indices: dict[Node, int] = {}
            spp_nodes_x: set[Node] = set()
            for u, v in train_potential_arcs:
                spp_nodes_x.add(u)
                spp_nodes_x.add(v)

            # Add nodes to graph and create mappings
            for n in spp_nodes_x:
                idx = G_x.add_node(n)
                node_indices[n] = idx

            for u_node, v_node in train_potential_arcs:
                original_profit = self._get_arc_original_profit(train_id, u_node, v_node)
                cost_penalty_at_v = 0.0
                if isinstance(v_node, NodeInfo):  # Penalty applies if arc leads to physical node v_node
                    key_jv = (train_id, v_node)
                    cost_penalty_at_v = current_mu.get(key_jv, 0.0) + \
                        (self.rho / 2.0) * (1 - 2 * current_z_global.get(key_jv, 0.0))

                arc_effective_profit = original_profit - cost_penalty_at_v

                u_idx = node_indices[u_node]
                v_idx = node_indices[v_node]
                G_x.add_edge(u_idx, v_idx, {
                    'effective_profit': arc_effective_profit,
                    'original_profit': original_profit
                })

            _, path_uv_tuples = self._solve_longest_path_dag_dp(
                G_x, node_indices, self.node_sigma, self.node_tau, 'effective_profit')

            current_train_orig_obj = 0.0
            if path_uv_tuples:
                for u_n, v_n in path_uv_tuples:
                    all_paths_arcs[(train_id, u_n, v_n)] = 1.0
                    u_idx = node_indices[u_n]
                    v_idx = node_indices[v_n]
                    arc_data = G_x.get_edge_data(u_idx, v_idx)
                    current_train_orig_obj += arc_data.get('original_profit', 0.0)
                    if isinstance(v_n, NodeInfo):
                        z_local_new[(train_id, v_n)] = 1  # This train now occupies v_n
            total_objective_from_x += current_train_orig_obj

        return all_paths_arcs, z_local_new, total_objective_from_x

    def _z_update(self,
                  current_z_local: dict[tuple[str, NodeInfo], int],
                  current_mu: dict[tuple[str, NodeInfo], float]) -> dict[tuple[str, NodeInfo], int]:
        """Solves the z-subproblem using DP for headway constraints."""
        z_global_new: dict[tuple[str, NodeInfo], int] = defaultdict(int)

        # Calculate d_jv costs
        d_jv: dict[tuple[str, NodeInfo], float] = {}
        for train_id in self.train_ids:
            for node_v in self._all_occupiable_nodes:
                key_jv = (train_id, node_v)
                d_jv[key_jv] = -current_mu.get(key_jv, 0.0) + \
                    (self.rho / 2.0) * (1 - 2 * current_z_local.get(key_jv, 0.0))

        # For each (station, type), solve a 1D DP
        for s_idx in range(self.num_stations):
            for node_type in ['U', 'W']:
                # Aggregate costs: d*_v = min_j d_j,v
                d_star_v: dict[int, float] = {}  # time -> min_cost_d_jv
                j_star_v: dict[int, Optional[str]] = {}  # time -> best_train_id_for_this_v

                for t in self.time_points:
                    current_node = NodeInfo(node_type, s_idx, t)
                    min_d_for_node = float('inf')
                    best_j_for_node = None
                    for train_id in self.train_ids:
                        # Use inf if no train can occupy
                        cost = d_jv.get((train_id, current_node), float('inf'))
                        if cost < min_d_for_node:
                            min_d_for_node = cost
                            best_j_for_node = train_id
                    if best_j_for_node is not None:  # Only consider if a train can occupy this node
                        d_star_v[t] = min_d_for_node
                        j_star_v[t] = best_j_for_node

                # Decision: occupy node at time t or not.
                dp_costs: list[float] = [0.0] * (self.t_max + 1)
                dp_choices: list[int] = [0] * (self.t_max + 1)  # 1 if node at time t is chosen

                for t in self.time_points:  # t from 0 to t_max
                    cost_if_not_occupy_t = dp_costs[t - 1] if t > 0 else 0.0

                    # Cost of occupying node at time t
                    cost_if_occupy_t = d_star_v.get(t, float('inf'))
                    if cost_if_occupy_t == float('inf'):  # Cannot occupy this node
                        dp_costs[t] = cost_if_not_occupy_t
                        dp_choices[t] = 0
                        continue

                    prev_dp_cost_for_occupy = 0.0
                    if t >= self.headway:
                        prev_dp_cost_for_occupy = dp_costs[t - self.headway]

                    cost_if_occupy_t += prev_dp_cost_for_occupy

                    if cost_if_occupy_t < cost_if_not_occupy_t and d_star_v.get(t, float('inf')) < float('inf'):
                        dp_costs[t] = cost_if_occupy_t
                        dp_choices[t] = 1
                    else:
                        dp_costs[t] = cost_if_not_occupy_t
                        dp_choices[t] = 0

                # Backtrack to set z_global_new
                current_t = self.t_max
                while current_t >= 0:
                    if dp_choices[current_t] == 1:  # Node (node_type, s_idx, current_t) is chosen
                        chosen_node = NodeInfo(node_type, s_idx, current_t)
                        chosen_train_id = j_star_v.get(current_t)
                        if chosen_train_id:
                            z_global_new[(chosen_train_id, chosen_node)] = 1
                        # All other z for this chosen_node are 0 (handled by defaultdict)
                        current_t -= self.headway  # Move back by headway window
                    else:
                        current_t -= 1
        return z_global_new

    def _mu_update(
            self,
            current_mu: dict[tuple[str, NodeInfo], float],
            current_z_local: dict[tuple[str, NodeInfo], int],
            current_z_global: dict[tuple[str, NodeInfo], int]
    ) -> dict[tuple[str, NodeInfo], float]:
        """Updates the Lagrange multipliers."""
        mu_new = current_mu.copy()
        keys_to_update = set(current_mu.keys()) | set(
            current_z_local.keys()) | set(current_z_global.keys())

        for key_jv in keys_to_update:
            if not (isinstance(key_jv, tuple) and len(key_jv) == 2 and isinstance(key_jv[1], NodeInfo)):
                continue

            residual = current_z_local.get(key_jv, 0) - current_z_global.get(key_jv, 0)
            mu_new[key_jv] = current_mu.get(key_jv, 0.0) + self.rho * residual
        return mu_new

    def solve(self) -> dict[ArcKey, float]:
        start_solve_time = time.time()

        # Initialize z_global and mu (often to zero)
        self.z_global = defaultdict(int)
        self.mu = defaultdict(float)

        best_feas = float('inf')
        best_objective = float('-inf')
        best_x_arcs = None

        for k_iter in range(self.max_iterations):
            iter_start_time = time.time()

            # 1. x-update
            current_x_arcs, current_z_local, current_objective = self._x_update(
                self.z_global, self.mu)

            # 2. z-update
            self.z_global = self._z_update(current_z_local, self.mu)

            # 3. mu-update
            self.mu = self._mu_update(self.mu, current_z_local, self.z_global)

            # Calculate residuals for convergence check (optional but good practice)
            primal_residual_sq = 0.0
            all_relevant_keys = set(current_z_local.keys()) | set(self.z_global.keys())
            for key_jv in all_relevant_keys:
                primal_residual_sq += (current_z_local.get(key_jv, 0) -
                                       self.z_global.get(key_jv, 0)) ** 2

            iter_time = time.time() - iter_start_time
            current_feas = self._calculate_feasibility_metric(current_x_arcs)
            self.feassiability.append(current_feas)

            if self.verbose or (k_iter % 100 == 0):
                print(
                    f"Iter {k_iter:3d}: Obj(x)={current_objective:8.2f}, Feas(x)={current_feas:6.2f}, PrimalRes={primal_residual_sq ** 0.5:8.4f}, Time={iter_time:5.2f}s, rho={self.rho:.4f}")

            self.rho *= 1.01

            if primal_residual_sq ** 0.5 < 1e-3 and current_feas < 0.001:  # Adjust tolerance
                if self.verbose:
                    print("Stopping: Residuals and feasibility tolerance met.")
                best_feas = current_feas
                best_objective = current_objective
                best_x_arcs = current_x_arcs.copy()
                break

            if k_iter == self.max_iterations - 1 and self.verbose:
                print("Stopping: Max iterations reached.")

            if current_feas < best_feas or (current_feas == best_feas and current_objective > best_objective):
                best_feas = current_feas
                best_objective = current_objective
                best_x_arcs = current_x_arcs.copy()

        total_cpu_time = time.time() - start_solve_time
        print(f"\nADMM Finished. Total CPU Time: {total_cpu_time:.2f} seconds.")

        print(f"Final reported solution's objective value: {best_objective:.2f}")
        print(f"Final reported solution's feasibility (sum of violations): {best_feas:.2f}")

        if best_feas > 0.001:
            print("Warning: ADMM final solution is not feasible regarding headway constraints.")

        return best_x_arcs


if __name__ == "__main__":
    from data import STATIONS_DATA, STATION_MILEAGES_DATA, TRAIN_DATA, BLOCK_SECTION_TIMES_DATA

    solver = ADMMSolver(
        stations=STATIONS_DATA,
        station_mileages=STATION_MILEAGES_DATA,
        train_specs=TRAIN_DATA,
        block_section_times=BLOCK_SECTION_TIMES_DATA,
        t_max=160,
        headway=5,
        min_stop_dwell=2,
        max_stop_dwell=15,
        pass_dwell=0,
        rho=10000.0,
        max_iterations=10000,
        verbose=True,
        profit_setting="min_total_runtime"
    )
    solution = solver.solve()
    solver.plot_timetable(solution)
    solver.plot_feassibility()
