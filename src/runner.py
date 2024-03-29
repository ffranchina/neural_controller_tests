import os
import shutil

import fire
import torch
import tqdm

import architecture
import config
import misc


def run_experiment(config_filepath, label=None, seed=None, stdout=True):
    experiment = config.ExperimentalConfiguration(config_filepath, label, seed)

    # Initialize the seeds of the stochastic generators
    seed = experiment["seed"]
    torch.manual_seed(seed)

    log_dest = f"{experiment['workdir']}/{experiment['label']}_{experiment['seed']:04}"

    # Initialize the environment object
    environment_class = experiment["environment.object"]
    environment_params = config.ConfigUtils.get_const_values(experiment, "environment")
    environment = environment_class(**environment_params)

    # Initialize the agents' objects
    agents = []
    for name in experiment.agent_names:
        agent_class = experiment[f"agents.{name}.object"]
        agent_params = config.ConfigUtils.get_const_values(experiment, f"agents.{name}")
        nn = architecture.NeuralAgent(
            agent_class.sensors,
            agent_class.actuators,
            experiment[f"agents.{name}.nn.n_layers"],
            experiment[f"agents.{name}.nn.layer_size"],
            input_noise_size=experiment[f"agents.{name}.nn.input_noise_size"],
            n_coeff=experiment[f"agents.{name}.nn.n_coefficients"],
        )
        agent = agent_class(
            name, nn, target_formula=experiment[f"agents.{name}.target"], **agent_params
        )
        agents.append(agent)

    # Instantiate the world model
    world_class = experiment["simulator.object"]
    world_model = world_class(environment, *agents, dt=experiment["simulator.dt"])

    # Instantiate simulator for the training
    training_values = config.ConfigUtils.get_init_values(experiment, "training")
    training_hyperspace = misc.ParametersHyperspace(
        **training_values,
        seed=experiment["seed"],
        sigma=experiment["simulator.sigma_sampling"],
    )
    training_simulator = misc.Simulator(world_model, training_hyperspace)
    trainer = architecture.Trainer(training_simulator, log_dest)

    if "testing" in experiment:
        # Instantiate simulator for the testing
        testing_values = config.ConfigUtils.get_init_values(experiment, "testing")
        testing_hyperspace = misc.ParametersHyperspace(
            **testing_values,
            seed=experiment["seed"],
            sigma=experiment["simulator.sigma_sampling"],
        )
        testing_simulator = misc.Simulator(world_model, testing_hyperspace)
        tester = architecture.Tester(testing_simulator, log_dest)

    with tqdm.trange(
        experiment["training.epochs"],
        disable=not stdout,
        bar_format="{l_bar}{bar}| Epoch {n_fmt}/{total_fmt}, {elapsed}<{remaining}",
    ) as bar:

        for epoch in bar:
            replays = {
                name: experiment[f"agents.{name}.training.replay"]
                for name in experiment.agent_names
            }

            bar.set_description("TRAIN")
            trainer.run(
                experiment["training.episodes"],
                replays,
                experiment["training.simulation_horizon"],
                epoch=epoch,
            )

            if "testing" in experiment:
                bar.set_description(" TEST")
                tester.run(
                    experiment["testing.episodes"],
                    experiment["testing.simulation_horizon"],
                    epoch=epoch,
                )

    agents_nn = {agent.label: agent.nn for agent in agents}

    # Saves the trained models
    misc.save_models(log_dest, **agents_nn)

    # Copies the experiment .toml file in the log directory
    config_filename = os.path.basename(config_filepath)
    dest_config_filepath = os.path.join(log_dest, config_filename)
    shutil.copyfile(config_filepath, dest_config_filepath)


if __name__ == "__main__":
    fire.Fire(run_experiment)
