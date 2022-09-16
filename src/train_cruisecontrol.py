import random

import numpy as np
import torch

import architecture
import misc
import model_cruisecontrol

seed = random.randint(0, 10000)
torch.manual_seed(seed)

# Specifies the initial conditions of the setup
initial_conditions_ranges = {
    "environment_seed": np.arange(0, 1_000_000),
    "car_position": 0,
    "car_velocity": np.linspace(-12, 12, 25),
}
# Initializes the generator of initial states
pg = misc.ParametersHyperparallelepiped(**initial_conditions_ranges, seed=seed)

# Specifies the STL formula to compute the robustness
agent_target = "G(vel >= 4.75 & vel <= 5.25)"

# Instantiates the NN architectures
nn_agent = architecture.NeuralAgent(
    model_cruisecontrol.Agent.sensors, model_cruisecontrol.Agent.actuators, 2, 10
)

dt = 0.05  # timestep

# Build the whole setting for the experiment
env = model_cruisecontrol.Environment()
agent = model_cruisecontrol.Agent("car", nn_agent, agent_target)
world_model = model_cruisecontrol.World(env, agent, dt=dt)

# Instantiates the world's model
simulator = misc.Simulator(world_model, pg.sample(sigma=0.05))

working_dir = "/tmp/experiments/" + f"cruise_{seed:04}"

# Instantiates the traning environment
trainer = architecture.Trainer(simulator, working_dir)

epochs = 10  # number of train/test iterations

training_steps = 10  # number of episodes for training
train_simulation_horizon = int(0.5 / dt)  # 5 seconds

for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}:")
    # Starts the training
    trainer.run(
        training_steps,
        {"car": 10},
        train_simulation_horizon,
        epoch=epoch,
    )

# Saves the trained models
misc.save_models(working_dir, car=nn_agent)
