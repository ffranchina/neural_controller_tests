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

    @staticmethod
    def get_init_values(config, stage):
        items = {}

        # Get agents init values
        for name in config.agent_names:
            sub_config = config[f"agents.{name}.{stage}"]
            init_keys = [k for k in sub_config if k.startswith("init_")]
            for k in init_keys:
                item_label = k[len("init_") :]
                items[f"{name}_{item_label}"] = config[f"agents.{name}.{stage}"][k]

        return items


class ExperimentalConfiguration:
    def __init__(self, config_filename, seed=None):
        self._filename = config_filename

        with open(config_filename) as f:
            self._config = toml.load(f)

        if seed is not None:
            self._config["seed"] = seed
        elif "seed" not in self._config:
            self._config["seed"] = random.randint(0, 10000)

        self._preprocess()
        self._validate()

    def _preprocess(self):
        # Convert class references into Python objects
        self._config["simulator"]["object"] = ConfigUtils.to_class(
            self._config["model"], self._config["simulator"]["object"]
        )
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
                init_keys = [k for k in sub_config if k.startswith("init_")]
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

    def _validate(self):
        assert type(self._config["seed"]) == int
        assert type(self._config["label"]) == str
        assert type(self._config["model"]) == str
        assert type(self._config["workdir"]) == str

        assert type(self._config["simulator"]["dt"]) == float
        assert type(self._config["simulator"]["sigma_sampling"]) == float
        assert issubclass(self._config["simulator"]["object"], abstract_model.World)

        assert issubclass(
            self._config["environment"]["object"], abstract_model.Environment
        )

        for name in self.agent_names:
            assert issubclass(
                self._config["agents"][name]["object"], abstract_model.Agent
            )
            assert type(self._config["agents"][name]["target"]) == str

        assert type(self._config["agents"][name]["nn"]["n_layers"]) == int
        assert type(self._config["agents"][name]["nn"]["layer_size"]) == int

        if self._config["agents"][name]["nn"]["input_noise_size"] is not None:
            assert type(self._config["agents"][name]["nn"]["input_noise_size"]) == int

        if self._config["agents"][name]["nn"]["n_coefficients"] is not None:
            assert type(self._config["agents"][name]["nn"]["n_coefficients"]) == int

        assert type(self._config["agents"][name]["training"]["replay"]) == int

        for stage in ["training", "testing"]:
            init_keys = [
                k for k in self._config["agents"][name][stage] if k.startswith("init_")
            ]
            for k in init_keys:
                assert self._config["agents"][name][stage][k] is not None

        assert type(self._config["training"]["simulation_horizon"]) == int
        assert type(self._config["training"]["epochs"]) == int
        assert type(self._config["training"]["episodes"]) == int

        assert type(self._config["testing"]["simulation_horizon"]) == int
        assert type(self._config["testing"]["episodes"]) == int

    @property
    def agent_names(self):
        return tuple(self._config["agents"])

    def __getitem__(self, config_path):
        keys = config_path.split(".")
        try:
            return functools.reduce(operator.getitem, keys, self._config)
        except KeyError as e:
            return None
