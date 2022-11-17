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

        try:
            if ":" not in intervals:
                return [float(comp) for comp in components]
            elif ";" not in intervals:
                dimensions = []
                for comp in components:
                    start, stop = comp.split(":")
                    # Makes use of integers spaced by 1
                    dimension = np.arange(int(start), int(stop))
                    dimensions.append(dimension)
            else:
                dimensions = []
                for comp in components:
                    start_stop, steps = comp.split(";")
                    start, stop = start_stop.split(":")
                    # Makes use of float values equally spaced
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

        # Get environment init values
        if stage in config["environment"]:
            sub_config = config[f"environment.{stage}"]
            init_keys = [k for k in sub_config if k.startswith("init_")]
            for k in init_keys:
                item_label = k[len("init_") :]
                items[f"environment_{item_label}"] = config[f"environment.{stage}"][k]

        # Get agents init values
        for name in config.agent_names:
            sub_config = config[f"agents.{name}.{stage}"]
            init_keys = [k for k in sub_config if k.startswith("init_")]
            for k in init_keys:
                item_label = k[len("init_") :]
                items[f"{name}_{item_label}"] = config[f"agents.{name}.{stage}"][k]

        return items

    @staticmethod
    def get_const_values(config, target):
        items = {}

        if target not in config:
            return items
        else:
            const_keys = [k for k in config[target] if k.startswith("const_")]
            for k in const_keys:
                item_label = k[len("const_") :]
                items[item_label] = config[target][k]

            return items


class ExperimentalConfiguration:
    def __init__(self, config_filepath, label=None, seed=None):
        self._filepath = config_filepath

        with open(config_filepath) as f:
            self._config = toml.load(f)

        if label is not None:
            self._config["label"] = label

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
        for stage in self.training_stages:
            if stage in self._config["environment"]:
                sub_config = self._config["environment"][stage]
                init_keys = [k for k in sub_config if k.startswith("init_")]
                for k in init_keys:
                    sub_config[k] = ConfigUtils.to_space(sub_config[k])

            for name in self.agent_names:
                sub_config = self._config["agents"][name][stage]
                init_keys = [k for k in sub_config if k.startswith("init_")]
                for k in init_keys:
                    sub_config[k] = ConfigUtils.to_space(sub_config[k])

        # Convert all the timing into dt times
        self._config["training"]["simulation_horizon"] = ConfigUtils.to_dt_time(
            self._config["simulator"]["dt"],
            self._config["training"]["simulation_horizon"],
        )
        if "testing" in self._config:
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

        for stage in self.training_stages:
            if stage in self._config["environment"]:
                init_keys = [
                    k
                    for k in self._config["environment"][stage]
                    if k.startswith("init_")
                ]
                for k in init_keys:
                    assert self._config["environment"][stage][k] is not None

        for name in self.agent_names:
            assert issubclass(
                self._config["agents"][name]["object"], abstract_model.Agent
            )
            assert type(self._config["agents"][name]["target"]) == str

            assert type(self._config["agents"][name]["nn"]["n_layers"]) == int
            assert type(self._config["agents"][name]["nn"]["layer_size"]) == int

            if self._config["agents"][name]["nn"]["input_noise_size"] is not None:
                assert (
                    type(self._config["agents"][name]["nn"]["input_noise_size"]) == int
                )

            if self._config["agents"][name]["nn"]["n_coefficients"] is not None:
                assert type(self._config["agents"][name]["nn"]["n_coefficients"]) == int

            assert type(self._config["agents"][name]["training"]["replay"]) == int

            for stage in self.training_stages:
                init_keys = [
                    k
                    for k in self._config["agents"][name][stage]
                    if k.startswith("init_")
                ]
                for k in init_keys:
                    assert self._config["agents"][name][stage][k] is not None

        assert type(self._config["training"]["simulation_horizon"]) == int
        assert type(self._config["training"]["epochs"]) == int
        assert type(self._config["training"]["episodes"]) == int

        if "testing" in self._config:
            assert type(self._config["testing"]["simulation_horizon"]) == int
            assert type(self._config["testing"]["episodes"]) == int

    @property
    def training_stages(self):
        return ["training", "testing"] if "testing" in self._config else ["training"]

    @property
    def agent_names(self):
        return tuple(self._config["agents"])

    def __contains__(self, config_path):
        return self.__getitem__(config_path) != None

    def __getitem__(self, config_path):
        keys = config_path.split(".")
        try:
            return functools.reduce(operator.getitem, keys, self._config)
        except KeyError as e:
            return None
