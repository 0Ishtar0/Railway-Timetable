We need to create an operational plan, or timetable, for a set of trains running on a railway network. The primary aim
is to design this timetable so that the total 'profit' generated from all scheduled train services is as high as
possible. The scheduling must consider specific, discrete moments in time for all train activities.
**Core Elements of the Timetable:**

* **Train Activities:** The timetable will be composed of various train activities. These include:
* **Starting a Journey:** Each train's overall journey begins with a conceptual 'starting activity' that links to its
  first physical departure from a station at a scheduled time.
* **Running Between Stations:** Trains perform 'running activities' when they travel from one station to another. This
  involves departing the origin station at a specific time and arriving at the destination station at a later specific
  time.
* **Waiting at Stations (Dwelling):** Trains may perform 'waiting activities' at stations. This means a train arrives at
  a station at one time and departs from the same station at a later scheduled time.
* **Ending a Journey:** Each train's overall journey concludes with a conceptual 'ending activity' following its last
  physical arrival at a station at a scheduled time.
* **Train-Specific Options:**
* Each individual train has a predefined set of allowed activities. This includes which stations it can visit, the
  specific station-to-station segments it can run on, and where it might wait.
* A complete, valid schedule for any single train must be a continuous sequence of these permitted activities, forming a
  coherent journey from its conceptual start to its conceptual end.
  **Objective of the Timetable:**
* The main goal is to maximize the total 'profit' accumulated from all chosen train activities in the timetable. Every
  potential train activity (starting, running, waiting, ending) has an associated profit value.
* The meaning of this 'profit' can be adapted to different operational goals:
* For instance, to maximize the number of trains running, a profit of 1 could be assigned to each train's starting
  activity, with all other activities having zero profit.
* To minimize the total travel time for all trains, the profit for running activities could be set to the negative value
  of their duration.
* Profit values can also be adjusted to discourage activities that might lead to operational issues like congestion.
  **Rules and Constraints for Scheduling:**

1. **Journey Integrity for Each Train:**

* Each train can commence its overall journey at most one time.
* Each train can conclude its overall journey at most one time.
* Train schedules must be logical and continuous. If a train is scheduled to arrive at a station at a particular time (
  as part of an activity), it must also be scheduled for a subsequent departing activity from that station (either
  waiting at the same station or running to the next). This ensures there are no "dangling" arrivals or departures in a
  train's sequence of activities, except for the very first start and very final end of its entire journey.

2. **Tracking Station Usage at Specific Times:**

* The timetable must keep track of whether a specific station at a specific discrete time point (an 'event point') is
  being used by a particular train. An event point is considered used by a train if one of that train's scheduled
  activities (like an arrival or a departure) occurs there.
* The timetable must also track if an event point is being used by *any* train. An event point is considered generally
  in use if one or more trains are scheduled to perform an activity there.

3. **Conflict Avoidance and Separation (Headways):**

* To ensure safe operations and prevent trains from being too close to each other, minimum separation standards must be
  maintained.
* Certain combinations of station/time event points are defined as 'conflicting'. This means that if one event point in
  a conflicting set is used by a train, other event points that clash with it cannot be used by any train in the
  timetable. For any given event point, there is a pre-defined group of other event points (which may include the
  original one) that are mutually exclusive. From any such group of mutually exclusive event points, at most one can be
  included in the final schedule.
  Based on this description of requirements, please first propose a mathematical optimization model. After that, please
  generate executable Gurobi Python code to find a timetable that meets these requirements and achieves the stated
  objective of maximizing total profit.