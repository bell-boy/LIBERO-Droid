"""Episode task progress from live LIBERO predicates, contacts, and joint travel."""
import numpy as np


class TaskProgress:
    def __init__(self, wrapper, task):
        self.env = wrapper.env
        self.sim = wrapper.sim
        self.values = {}
        self.placements = []
        self.joints = []
        self.push = None
        self.push_only = task.startswith("push_the_plate")
        robot = wrapper.robots[0]
        self.pads = [
            {self.sim.model.geom_name2id(n) for n in robot.gripper.important_geoms[key]}
            for key in ("left_fingerpad", "right_fingerpad")
        ]
        goals = list(self.env.parsed_problem["goal_state"])
        if task == "open_the_top_drawer_and_put_the_bowl_inside":
            goals.append(("open", "wooden_cabinet_1_top_region"))
        for pred, *args in goals:
            pred = pred.lower()
            if pred in ("on", "in"):
                item, dest = args
                if self.push_only:
                    self.push = (item, dest, self.region_distance(item, dest))
                    self.values["push"] = 0.
                    continue
                obj = self.env.get_object(item)
                geoms = {self.sim.model.geom_name2id(n) for n in obj.contact_geoms}
                self.placements.append((item, dest, geoms))
                self.values["grasp " + item] = self.values["deliver " + item] = 0.
            elif pred in ("open", "close", "turnon"):
                dest, = args
                if [pred, dest] in self.env.parsed_problem["initial_state"]:
                    continue
                site = self.env.object_sites_dict.get(dest)
                obj = self.env.get_object(site.parent_name if site else dest)
                joint, = site.joints if site else obj.joints
                addr = self.sim.model.get_joint_qpos_addr(joint)
                start = float(self.sim.data.qpos[addr])
                bounds = obj.object_properties["articulation"]["default_" + pred + "_ranges"]
                end = min(bounds, key=lambda x: abs(start - x))
                self.joints.append((pred, dest, addr, start, end))
                self.values[pred + " " + dest] = 0.
            else:
                raise ValueError(f"Unsupported goal: {pred} {args}")
        if self.placements:
            self.values["success"] = 0.

    def local_position(self, item, dest):
        data = self.sim.data
        point = data.body_xpos[self.env.obj_body_id[item]]
        return data.get_site_xmat(dest).T @ (point - data.get_site_xpos(dest))

    def region_distance(self, item, dest):
        size = np.asarray(self.env.object_sites_dict[dest].size)
        xy = self.local_position(item, dest)[:2]
        return float(np.linalg.norm(np.maximum(abs(xy) - size[:2], 0.)))

    def delivered(self, item, dest):
        site = self.env.object_sites_dict.get(dest)
        if site is None:
            return self.env.check_contact(self.env.get_object(item), self.env.get_object(dest))
        # Distance to the specified compartment / surface, including a 2 cm
        # margin for a rim contact before the object's center enters the region.
        p = self.local_position(item, dest)
        size = np.asarray(site.size)
        return bool(np.all(abs(p[:2]) <= size[:2] + .02) and -size[2] - .02 <= p[2] <= size[2] + .03)

    def update(self, success):
        data = self.sim.data
        pairs = [(c.geom1, c.geom2) for c in data.contact[:data.ncon]]
        for item, dest, geoms in self.placements:
            touched = {b for a,b in pairs if a in geoms} | {a for a,b in pairs if b in geoms}
            if all(touched & pad for pad in self.pads):
                self.values["grasp " + item] = 1.
            if self.values["grasp " + item] and self.delivered(item, dest):
                self.values["deliver " + item] = 1.
        for pred, dest, addr, start, end in self.joints:
            if pred == "close" and not all(self.values["deliver " + item] for item, _, _ in self.placements):
                continue
            key = pred + " " + dest
            travel = (data.qpos[addr] - start) / (end - start) if end != start else 0.
            self.values[key] = max(self.values[key], float(np.clip(travel, 0, 1)))
        if self.push:
            item, dest, initial = self.push
            travel = 1 - self.region_distance(item, dest) / initial if initial > 0 else 0.
            self.values["push"] = max(self.values["push"], float(np.clip(travel, 0, 1)))
        if success and self.placements:
            self.values["success"] = 1.
        return 1. if success else sum(self.values.values()) / len(self.values)
