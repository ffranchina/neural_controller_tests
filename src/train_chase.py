import random

import numpy as np
import torch

import architecture
import misc
import model_chase

seed = random.randint(0, 10000)
torch.manual_seed(seed)

# Specifies the initial conditions of the setup
follower_position = [0.0, 0.0]
follower_velocity = np.array(
    np.meshgrid(np.linspace(0, 20, 40), np.linspace(0, 20, 40))
).T.reshape(-1, 2)
leader_position = np.array(
    np.meshgrid(np.linspace(1, 12, 15), np.linspace(1, 12, 15))
).T.reshape(-1, 2)
leader_velocity = np.array(
    np.meshgrid(np.linspace(0, 20, 40), np.linspace(0, 20, 40))
).T.reshape(-1, 2)
initial_conditions_ranges = [
    leader_position,
    leader_velocity,
    follower_position,
    follower_velocity,
]
# Initializes the generator of initial states
pg = misc.ParametersHyperparallelepiped(*initial_conditions_ranges, seed=seed)

# Specifies the STL formula to compute the robustness
# attacker_target = "G(!(dist <= 10 & dist >= 2))"
leader_target = "!(G(dist <= 10 & dist >= 2))"
follower_target = "G(dist <= 10 & dist >= 2)"

# Instantiates the NN architectures
nn_leader = architecture.NeuralAgent(
    model_chase.Leader.sensors, model_chase.Leader.actuators, 2, 10, 2
)
nn_follower = architecture.NeuralAgent(
    model_chase.Agent.sensors, model_chase.Agent.actuators, 2, 10
)

dt = 0.05  # timestep

# Build the whole setting for the experiment
env = model_chase.Environment()
leader = model_chase.Leader("leader", nn_leader, leader_target)
follower = model_chase.Agent("follower", nn_follower, follower_target)
world_model = model_chase.Model(env, leader, follower, dt=dt)

# Instantiates the world's model
simulator = misc.Simulator(world_model, pg.sample(sigma=0.05))

working_dir = "/tmp/experiments/" + f"chase_{seed:04}"

# Instantiates the traning and test environments
trainer = architecture.Trainer(simulator, working_dir)
tester = architecture.Tester(simulator, working_dir)

epochs = 10  # number of train/test iterations

training_steps = 100  # number of episodes for training
train_simulation_horizon = int(5 / dt)  # 5 seconds

test_steps = 10  # number of episodes for testing
test_simulation_horizon = int(60 / dt)  # 60 seconds

for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}:")
    # Starts the training
    trainer.run(
        training_steps,
        {"leader": 3, "follower": 10},
        train_simulation_horizon,
        epoch=epoch,
    )
    # Starts the testing
    # tester.run(test_steps, test_simulation_horizon, epoch=epoch)

# Saves the trained models
misc.save_models(working_dir, leader=nn_leader, follower=nn_follower)
