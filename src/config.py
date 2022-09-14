import functools
import importlib
import operator
import random
import re

import numpy as np
import toml

import abstract_model


class ConfigUtils:
    @staticmethod
    def to_class(module, reference):
        m = importlib.import_module(module)
        return getattr(m, reference)

    @staticmethod
    def to_space(intervals):
        components = intervals.split(",")

        if ":" not in intervals:
            return [float(comp) for comp in components]
        else:
            dimensions = []
            for comp in components:
                try:
                    start_stop, steps = comp.split(";")
                    start, stop = start_stop.split(":")
                    dimension = np.linspace(float(start), float(stop), int(steps))
                    dimensions.append(dimension)
                except ValueError as e:
                    return None

            mesh = np.meshgrid(*dimensions, indexing="ij")
            datapoints = np.stack(mesh).reshape(len(dimensions), -1).T
            return datapoints

    @staticmethod
    def to_dt_time(dt, timestring):
        # Supports minutes (m), seconds (s) and milliseconds (ms)
        regex = re.compile(r"^(\d+)(m?s?)$")
        matches = regex.match(timestring)
        if matches is not None:
            amount = int(matches[1])
            unit = matches[2]

            if unit == "m":
                return int((60 * amount) / dt)
            elif unit == "s":
                return int(amount / dt)
            elif unit == "ms":
                return int(amount / (dt * 1000))


class ExperimentalConfiguration:
    def __init__(self, config_filename, seed=None):
        self._filename = config_filename

        with open(config_filename) as f:
            self._config = toml.load(f)

        if seed is not None:
            self._config["seed"] = seed
        elif self._config["seed"] is None:
            self._config["seed"] = random.randint(0, 10000)

        self._preprocess()

    def _preprocess(self):
        # Convert class references into Python objects
        self._config["environment"]["object"] = ConfigUtils.to_class(
            self._config["model"], self._config["environment"]["object"]
        )
        for name in self.agent_names:
            self._config["agents"][name]["object"] = ConfigUtils.to_class(
                self._config["model"], self._config["agents"][name]["object"]
            )

        # Convert the string into the numpy representation of the space
        for name in self.agent_names:
            for sub_agent_config in ["training", "testing"]:
                sub_config = self._config["agents"][name][sub_agent_config]
                init_keys = [k for k in sub_config.keys() if k.startswith("init_")]
                for k in init_keys:
                    sub_config[k] = ConfigUtils.to_space(sub_config[k])

        # Convert all the timing into dt times
        self._config["training"]["simulation_horizon"] = ConfigUtils.to_dt_time(
            self._config["simulator"]["dt"],
            self._config["training"]["simulation_horizon"],
        )
        self._config["testing"]["simulation_horizon"] = ConfigUtils.to_dt_time(
            self._config["simulator"]["dt"],
            self._config["testing"]["simulation_horizon"],
        )

    @property
    def agent_names(self):
        return tuple(self._config["agents"].keys())

    def __getitem__(self, item):
        keys = item.split(".")
        try:
            return functools.reduce(operator.getitem, keys, self._config)
        except KeyError as e:
            return None
