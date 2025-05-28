import time
from collections import defaultdict
from typing import Optional, Literal

import gurobipy as gp
from gurobipy import GRB

from base_solver import RailwayTimetablingSolver
from data import NodeInfo, RuntimeKey, ArcKey, Node, TrainData, BlockSectionTime


class GurobiSolver(RailwayTimetablingSolver):
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
                 objective_type: Literal["min_makespan", "min_sum_arrival", "feasibility"] = "feasibility"):
        super().__init__(stations, station_mileages, train_specs, block_section_times,
                         t_max, headway, min_stop_dwell, max_stop_dwell, pass_dwell)
        self.objective_type = objective_type
        self.model = gp.Model("TrainTimetabling")

    def _gen_model(self, var_type: str) -> dict[ArcKey, gp.Var]:
        x: dict[ArcKey, gp.Var] = {}  # Store variables with ArcKey

        arcs_for_train = {j: [] for j in self.train_ids}
        arcs_entering_node = defaultdict(lambda: defaultdict(list))
        arcs_leaving_node = defaultdict(lambda: defaultdict(list))

        # Keep track of potential final arrival arcs for objective functions
        final_arrival_arcs_info = []  # List of (train_id, arrival_time_at_dest, variable)

        for train_id in self.train_ids:
            train_spec = self.train_specs_map[train_id]
            speed_val = train_spec.speed

            # 1. Start Arcs
            for t_dep in self.time_points:
                from_n: Node = self.node_sigma
                to_n: Node = NodeInfo('W', 0, t_dep)  # W = departure node type
                var = self.model.addVar(vtype=var_type, name=f"x_{train_id}_{from_n}_to_{to_n}")
                x[train_id, from_n, to_n] = var
                arcs_for_train[train_id].append((from_n, to_n))
                arcs_leaving_node[from_n][train_id].append(var)
                arcs_entering_node[to_n][train_id].append(var)

            # 2. Segment Arcs & Station Arcs
            for s_idx in range(self.num_stations):
                # Station Arcs: (arrival at U, dwell, departure from W)
                if s_idx > 0:  # No station arcs at the very first station from sigma
                    for t_arr in self.time_points:
                        node_U: Node = NodeInfo('U', s_idx, t_arr)  # U = arrival node type
                        dwell_times = self.get_dwell_times(train_id, s_idx)
                        for dt in dwell_times:
                            t_dep_new: int = t_arr + dt
                            if t_dep_new in self.time_points:
                                node_W: Node = NodeInfo('W', s_idx, t_dep_new)
                                from_n, to_n = node_U, node_W
                                var = self.model.addVar(
                                    vtype=var_type, name=f"x_{train_id}_from_{from_n}_to_{to_n}")
                                x[train_id, from_n, to_n] = var
                                arcs_for_train[train_id].append((from_n, to_n))
                                arcs_leaving_node[from_n][train_id].append(var)
                                arcs_entering_node[to_n][train_id].append(var)

                # Segment Arcs (departure from W, run, arrival at U_next)
                if s_idx < self.num_stations - 1:  # No segment arcs from the last station
                    for t_dep in self.time_points:
                        node_W: Node = NodeInfo('W', s_idx, t_dep)
                        run_time_key = RuntimeKey(s_idx, s_idx + 1, speed_val)
                        run_time = self.runtimes.get(run_time_key, None)
                        if run_time is None:
                            # This case should ideally not happen if data is consistent
                            # or train_spec ensures valid segments for the train.
                            continue

                        t_arr_new: int = t_dep + run_time
                        if t_arr_new in self.time_points:
                            node_U_next: Node = NodeInfo('U', s_idx + 1, t_arr_new)
                            from_n, to_n = node_W, node_U_next
                            var = self.model.addVar(
                                vtype=var_type, name=f"x_{train_id}_from_{from_n}_to_{to_n}")
                            x[train_id, from_n, to_n] = var
                            arcs_for_train[train_id].append((from_n, to_n))
                            arcs_leaving_node[from_n][train_id].append(var)
                            arcs_entering_node[to_n][train_id].append(var)

            # 3. End Arcs (from final U node to tau)
            # These represent the arrival at the final destination for objective calculation
            dest_station_idx = self.num_stations - 1
            for t_arr_final in self.time_points:
                from_n: Node = NodeInfo('U', dest_station_idx, t_arr_final)
                to_n: Node = self.node_tau
                var = self.model.addVar(
                    vtype=var_type, name=f"x_{train_id}_from_{from_n}_to_{to_n}")
                x[train_id, from_n, to_n] = var
                arcs_for_train[train_id].append((from_n, to_n))
                arcs_leaving_node[from_n][train_id].append(var)
                arcs_entering_node[to_n][train_id].append(var)
                # Store info for objective function
                if isinstance(from_n, NodeInfo):  # Should always be true here
                    final_arrival_arcs_info.append(
                        {'train_id': train_id, 'arrival_time': from_n.time, 'var': var})

        self.model.update()

        z = self.model.addVars([(tr, nt, st, t)
                                for tr in self.train_ids
                                for nt in ['U', 'W']
                                for st in range(self.num_stations)
                                for t in self.time_points],
                               vtype=var_type, name="z")

        y = self.model.addVars([(nt, st, t)
                                for nt in ['U', 'W']
                                for st in range(self.num_stations)
                                for t in self.time_points],
                               vtype=var_type, name="y")

        # Path Start: Each train must start exactly once
        for train_id in self.train_ids:
            self.model.addConstr(gp.quicksum(arcs_leaving_node[self.node_sigma][train_id]) == 1,
                                 name=f"PathStart_{train_id}")

        # Path End: Each train must end exactly once
        for train_id in self.train_ids:
            self.model.addConstr(gp.quicksum(arcs_entering_node[self.node_tau][train_id]) == 1,
                                 name=f"PathEnd_{train_id}")

        # Flow Conservation
        physical_nodes: set[NodeInfo] = set()
        for train_id in self.train_ids:
            for arc_tuple in arcs_for_train[train_id]:
                n1, n2 = arc_tuple
                if isinstance(n1, NodeInfo):
                    physical_nodes.add(n1)
                if isinstance(n2, NodeInfo):
                    physical_nodes.add(n2)

        for train_id in self.train_ids:
            for node_v in physical_nodes:
                sum_in = gp.quicksum(arcs_entering_node[node_v].get(train_id, []))
                sum_out = gp.quicksum(arcs_leaving_node[node_v].get(train_id, []))
                self.model.addConstr(sum_in == sum_out, name=f"FlowCons_{train_id}_{node_v}")

        # Node Occupation by Train (z_jv)
        for train_id in self.train_ids:
            for node_v_key in physical_nodes:
                sum_incoming_x_train_node = gp.quicksum(
                    arcs_entering_node[node_v_key].get(train_id, []))
                self.model.addConstr(
                    z[train_id, node_v_key.type, node_v_key.station_idx,
                        node_v_key.time] == sum_incoming_x_train_node,
                    name=f"ZLink_{train_id}_{node_v_key}")

        # Node Occupation: Any Train (y_v)
        for node_v_key in physical_nodes:
            sum_z_for_node = gp.quicksum(
                z[tr, node_v_key.type, node_v_key.station_idx, node_v_key.time] for tr in self.train_ids)
            self.model.addConstr(y[node_v_key.type, node_v_key.station_idx, node_v_key.time] == sum_z_for_node,
                                 name=f"YLink_{node_v_key}")

        # Headway Constraints
        for st_idx in range(self.num_stations):
            for node_type_str in ['U', 'W']:
                for t0 in range(self.t_max - self.headway + 2):
                    nodes_in_headway_window = []
                    for t_offset in range(self.headway):
                        t_current: int = t0 + t_offset
                        if t_current in self.time_points:
                            nodes_in_headway_window.append(y[node_type_str, st_idx, t_current])

                    if nodes_in_headway_window:
                        self.model.addConstr(gp.quicksum(nodes_in_headway_window) <= 1,
                                             name=f"Headway_{node_type_str}_{st_idx}_{t0}")

        if self.objective_type == "min_makespan":
            C_max = self.model.addVar(name="C_max", vtype=GRB.CONTINUOUS, lb=0.0)
            for item in final_arrival_arcs_info:
                self.model.addConstr(C_max >= item['arrival_time'] * item['var'],
                                     name=f"Makespan_train_{item['train_id']}_t{item['arrival_time']}")
            self.model.setObjective(C_max, GRB.MINIMIZE)
            print("Objective set to: Minimize Makespan")

        elif self.objective_type == "min_sum_arrival":
            sum_arrival_times_expr = gp.LinExpr()
            for item in final_arrival_arcs_info:
                sum_arrival_times_expr += item['arrival_time'] * item['var']
            self.model.setObjective(sum_arrival_times_expr, GRB.MINIMIZE)
            print("Objective set to: Minimize Sum of Arrival Times")
        else:
            self.model.setObjective(0, GRB.MINIMIZE)
            print("Objective set to: Feasibility (0)")

        self.model.update()
        return x

    def solve(self) -> Optional[dict[ArcKey, float]]:
        x = self._gen_model(GRB.BINARY)

        start_time = time.time()
        self.model.optimize()
        cpu_time = time.time() - start_time

        print(f"CPU Time: {cpu_time:.2f} seconds")

        if self.model.Status == GRB.OPTIMAL or (self.model.Status != GRB.INFEASIBLE and self.model.SolCount > 0):
            solution_arcs = {}
            for arc_key, var in x.items():
                if var.X > 0.5:
                    solution_arcs[arc_key] = var.X
            return solution_arcs
        else:
            return None


class RelaxGurobiSolver(GurobiSolver):

    def solve(self) -> Optional[dict[ArcKey, float]]:
        x = self._gen_model(GRB.CONTINUOUS)

        start_time = time.time()
        self.model.optimize()
        cpu_time = time.time() - start_time

        print(f"CPU Time: {cpu_time:.2f} seconds")

        if self.model.Status == GRB.OPTIMAL or (self.model.Status != GRB.INFEASIBLE and self.model.SolCount > 0):
            solution_arcs = {}
            for arc_key, var in x.items():
                if var.X > 0.5:
                    solution_arcs[arc_key] = var.X
            return solution_arcs
        else:
            return None


if __name__ == "__main__":
    from data import STATIONS_DATA, STATION_MILEAGES_DATA, TRAIN_DATA, BLOCK_SECTION_TIMES_DATA

    solver = GurobiSolver(
        stations=STATIONS_DATA,
        station_mileages=STATION_MILEAGES_DATA,
        train_specs=TRAIN_DATA,
        block_section_times=BLOCK_SECTION_TIMES_DATA,
        t_max=160,
        headway=5,
        min_stop_dwell=2,
        max_stop_dwell=15,
        pass_dwell=0
    )
    solution = solver.solve()

    if solution:
        print(f"Solution obtained with {len(solution)} active arcs.")
        solver.plot_timetable(solution)
    else:
        print("No solution was found or an error occurred.")
