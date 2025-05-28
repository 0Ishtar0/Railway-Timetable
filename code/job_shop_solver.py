import time
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
from typing import Optional, Literal

from base_solver import RailwayTimetablingSolver
from data import RuntimeKey, TrainData, BlockSectionTime


class JobShopGurobiSolver(RailwayTimetablingSolver):
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
        self.model = gp.Model("TrainTimetabling_JobShop")
        self.objective_type = objective_type

    def solve(self) -> Optional[dict[str, list[dict]]]:
        a = {}
        d = {}

        for train_id in self.train_ids:
            for s_idx in range(self.num_stations):
                # Arrival variables (not for origin station 0)
                if s_idx > 0:
                    a[train_id, s_idx] = self.model.addVar(lb=0.0, vtype=GRB.CONTINUOUS,
                                                           name=f"a_{train_id}_s{s_idx}")
                # Departure variables (not for destination station num_stations-1)
                if s_idx < self.num_stations - 1:
                    d[train_id, s_idx] = self.model.addVar(lb=0.0, vtype=GRB.CONTINUOUS,
                                                           name=f"d_{train_id}_s{s_idx}")
        self.model.update()

        for train_id in self.train_ids:
            train_spec = self.train_specs_map[train_id]
            speed_val = train_spec.speed
            for s_idx in range(self.num_stations - 1):  # From station s_idx to s_idx+1
                run_time_key = RuntimeKey(s_idx, s_idx + 1, speed_val)
                run_time = self.runtimes.get(run_time_key)
                if run_time is None:
                    continue

                self.model.addConstr(a[train_id, s_idx + 1] == d[train_id, s_idx] + run_time,
                                     name=f"TravelTime_{train_id}_s{s_idx}_s{s_idx+1}")

        for train_id in self.train_ids:
            train_spec = self.train_specs_map[train_id]
            for s_idx in range(1, self.num_stations - 1):  # Intermediate stations
                is_stop = train_spec.stops[s_idx] == 1
                if is_stop:
                    self.model.addConstr(d[train_id, s_idx] >= a[train_id, s_idx] + self.min_stop_dwell,
                                         name=f"MinDwell_{train_id}_s{s_idx}")
                    self.model.addConstr(d[train_id, s_idx] <= a[train_id, s_idx] + self.max_stop_dwell,
                                         name=f"MaxDwell_{train_id}_s{s_idx}")
                else:  # Pass
                    self.model.addConstr(d[train_id, s_idx] == a[train_id, s_idx] + self.pass_dwell,
                                         name=f"PassDwell_{train_id}_s{s_idx}")

        delta_arr = {}
        delta_dep = {}
        pi_block_order = {}

        M = self.t_max * 5
        epsilon = 0.001

        for i1 in range(len(self.train_ids)):
            for i2 in range(i1 + 1, len(self.train_ids)):
                j1_id = self.train_ids[i1]
                j2_id = self.train_ids[i2]

                for s_idx in range(self.num_stations):
                    # Arrival Headway (not for origin s_idx=0)
                    if s_idx > 0:
                        delta_arr[j1_id, j2_id, s_idx] = self.model.addVar(
                            vtype=GRB.BINARY, name=f"delta_arr_{j1_id}_{j2_id}_s{s_idx}")
                        # If delta_arr = 1, j1 before j2 (or at same time for this constraint logic, then headway applies)
                        self.model.addConstr(a[j2_id, s_idx] >= a[j1_id, s_idx] + self.headway - M * (1 - delta_arr[j1_id, j2_id, s_idx]),
                                             name=f"ArrHeadway1_{j1_id}_{j2_id}_s{s_idx}")
                        self.model.addConstr(a[j1_id, s_idx] >= a[j2_id, s_idx] + self.headway - M * delta_arr[j1_id, j2_id, s_idx],
                                             name=f"ArrHeadway2_{j1_id}_{j2_id}_s{s_idx}")

                    # Departure Headway (not for destination s_idx = num_stations-1)
                    if s_idx < self.num_stations - 1:
                        delta_dep[j1_id, j2_id, s_idx] = self.model.addVar(
                            vtype=GRB.BINARY, name=f"delta_dep_{j1_id}_{j2_id}_s{s_idx}")
                        # If delta_dep = 1, j1 before j2
                        self.model.addConstr(d[j2_id, s_idx] >= d[j1_id, s_idx] + self.headway - M * (1 - delta_dep[j1_id, j2_id, s_idx]),
                                             name=f"DepHeadway1_{j1_id}_{j2_id}_s{s_idx}")
                        self.model.addConstr(d[j1_id, s_idx] >= d[j2_id, s_idx] + self.headway - M * delta_dep[j1_id, j2_id, s_idx],
                                             name=f"DepHeadway2_{j1_id}_{j2_id}_s{s_idx}")

                for s_idx in range(self.num_stations - 1):  # For block (s_idx, s_idx+1)
                    pi_block_order[j1_id, j2_id, s_idx] = self.model.addVar(
                        vtype=GRB.BINARY, name=f"pi_order_{j1_id}_{j2_id}_s{s_idx}")
                    # If pi_block_order = 1, j1 departs s_idx for s_idx+1 before or at same time as j2
                    self.model.addConstr(d[j1_id, s_idx] <= d[j2_id, s_idx] + M * (1 - pi_block_order[j1_id, j2_id, s_idx]),
                                         name=f"FIFO_Dep1_Order_{j1_id}_{j2_id}_s{s_idx}")
                    self.model.addConstr(a[j1_id, s_idx+1] <= a[j2_id, s_idx+1] + M * (1 - pi_block_order[j1_id, j2_id, s_idx]),
                                         name=f"FIFO_Arr1_Order_{j1_id}_{j2_id}_s{s_idx}")

                    # If pi_block_order = 0, j2 departs s_idx for s_idx+1 before j1
                    self.model.addConstr(d[j2_id, s_idx] <= d[j1_id, s_idx] - epsilon + M * pi_block_order[j1_id, j2_id, s_idx],  # Added -epsilon to enforce strict ordering for departure
                                         name=f"FIFO_Dep2_Order_{j1_id}_{j2_id}_s{s_idx}")
                    self.model.addConstr(a[j2_id, s_idx+1] <= a[j1_id, s_idx+1] + M * pi_block_order[j1_id, j2_id, s_idx],
                                         name=f"FIFO_Arr2_Order_{j1_id}_{j2_id}_s{s_idx}")

        for train_id in self.train_ids:
            # Arrival at destination within T_max
            self.model.addConstr(a[train_id, self.num_stations - 1] <= self.t_max,
                                 name=f"MaxTime_Dest_{train_id}")
            # All departures also <= t_max
            for s_idx in range(self.num_stations - 1):
                self.model.addConstr(d[train_id, s_idx] <= self.t_max,
                                     name=f"MaxTime_Dep_{train_id}_s{s_idx}")

        if self.objective_type == "min_sum_arrival":
            obj_expr = gp.quicksum(a[train_id, self.num_stations - 1]
                                   for train_id in self.train_ids)
            self.model.setObjective(obj_expr, GRB.MINIMIZE)
        elif self.objective_type == "min_makespan":
            c_max = self.model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="C_max")
            for train_id in self.train_ids:
                self.model.addConstr(c_max >= a[train_id, self.num_stations - 1],
                                     name=f"Makespan_Constr_{train_id}")
            self.model.setObjective(c_max, GRB.MINIMIZE)
        else:
            self.model.setObjective(0, GRB.MINIMIZE)

        self.model.update()

        solve_start_time = time.time()
        self.model.optimize()
        solve_cpu_time = time.time() - solve_start_time
        print(f"JobShop Model - CPU Time for Gurobi optimize(): {solve_cpu_time:.2f} seconds")
        print(f"JobShop Model - Number of variables: {self.model.NumVars}")
        print(f"JobShop Model - Number of constraints: {self.model.NumConstrs}")

        if self.model.Status == GRB.OPTIMAL or \
           (self.model.Status in [GRB.SUBOPTIMAL, GRB.INTERRUPTED] and self.model.SolCount > 0) or \
           (self.model.Status not in [GRB.INF_OR_UNBD, GRB.INFEASIBLE, GRB.UNBOUNDED] and self.model.SolCount > 0):
            print(
                f"Solution found. Status: {self.model.Status}, Objective: {self.model.ObjVal if self.model.SolCount > 0 else 'N/A'}")

            solution_schedule: dict[str, list[dict]] = {}
            for train_id in self.train_ids:
                solution_schedule[train_id] = []
                # Departure from origin
                if (train_id, 0) in d and d[train_id, 0].X is not None:
                    solution_schedule[train_id].append({'type': 'dep', 'station_idx': 0,
                                                        'time': d[train_id, 0].X,
                                                        'station_name': self.stations[0]})

                for s_idx in range(1, self.num_stations):
                    if (train_id, s_idx) in a and a[train_id, s_idx].X is not None:
                        solution_schedule[train_id].append({'type': 'arr', 'station_idx': s_idx,
                                                            'time': a[train_id, s_idx].X,
                                                            'station_name': self.stations[s_idx]})
                    # Departure if not destination
                    if s_idx < self.num_stations - 1 and (train_id, s_idx) in d and d[train_id, s_idx].X is not None:
                        solution_schedule[train_id].append({'type': 'dep', 'station_idx': s_idx,
                                                            'time': d[train_id, s_idx].X,
                                                            'station_name': self.stations[s_idx]})

                solution_schedule[train_id].sort(key=lambda x: (
                    x['time'], 0 if x['type'] == 'arr' else 1))

            return solution_schedule
        else:
            print(f"No solution was found or an error occurred. Status: {self.model.Status}")
            return None

    def plot_timetable_from_events(self, schedule: dict[str, list[dict]]):
        plt.figure(figsize=(18, 12))
        cmap = plt.colormaps.get_cmap('tab20')

        for i, train_id in enumerate(self.train_ids):
            if train_id not in schedule or not schedule[train_id]:
                continue

            color = cmap(i % cmap.N)
            events = schedule[train_id]

            plot_points_x = []
            plot_points_y = []

            for event in events:
                station_name = self.stations[event['station_idx']]
                mileage = self.station_mileages[station_name]
                time_val = event['time']

                plot_points_x.append(time_val)
                plot_points_y.append(mileage)

            if plot_points_x:
                plt.plot(plot_points_x, plot_points_y, color=color, marker='o', markersize=3,
                         linestyle='-', label=train_id if i < 20 else None)

                first_dep_event = next(
                    (ev for ev in events if ev['type'] == 'dep' and ev['station_idx'] == 0), None)
                if not first_dep_event and plot_points_x:
                    first_dep_event = events[0]

                if first_dep_event:
                    plt.text(first_dep_event['time'] + 0.5,
                             self.station_mileages[self.stations[first_dep_event['station_idx']]],
                             train_id, fontsize=7, color=color, va='center')

        plt.xlabel("Time (minutes)")
        plt.ylabel("Station / Mileage (km)")
        plt.title(f"Train Timetable (Job-Shop Model - Objective: {self.objective_type})")
        station_ticks_loc = [self.station_mileages[s_name] for s_name in self.stations]
        plt.yticks(station_ticks_loc, self.stations)

        max_event_time = 0
        for train_id in schedule:
            for event in schedule[train_id]:
                if event['time'] > max_event_time:
                    max_event_time = event['time']
        plot_t_max = max(self.t_max, max_event_time)

        plt.xticks(range(0, int(plot_t_max) + 10, 10 if plot_t_max < 300 else 20))
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.gca().invert_yaxis()
        plt.tight_layout(rect=[0, 0, 0.85, 1] if len(self.train_ids) <= 20 else None)

        try:
            filename = f"train_timetable_plot_jobshop_{self.objective_type}.pdf"
            plt.savefig(filename)
            print(f"Timetable plot saved to {filename}")
            plt.show()
        except Exception as e:
            print(f"Error saving or showing plot: {e}")


if __name__ == "__main__":
    from data import STATIONS_DATA, STATION_MILEAGES_DATA, TRAIN_DATA, BLOCK_SECTION_TIMES_DATA

    jobshop_solver = JobShopGurobiSolver(
        stations=STATIONS_DATA,
        station_mileages=STATION_MILEAGES_DATA,
        train_specs=TRAIN_DATA,
        block_section_times=BLOCK_SECTION_TIMES_DATA,
        t_max=160,
        headway=5,
        min_stop_dwell=2,
        max_stop_dwell=15,
        pass_dwell=0,
        objective_type="feasibility"
    )
    jobshop_solution = jobshop_solver.solve()

    if jobshop_solution:
        print("Job-Shop Solver: Solution obtained.")
        jobshop_solver.plot_timetable_from_events(jobshop_solution)
    else:
        print("Job-Shop Solver: No solution was found or an error occurred.")
