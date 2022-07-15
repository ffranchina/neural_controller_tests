import misc
import architecture
import model_platooning

import torch
import random
import numpy as np

seed = random.randint(0, 10000)
torch.manual_seed(seed)

# Specifies the initial conditions of the setup
follower_position = 0
follower_velocity = np.linspace(0, 20, 40)
leader_position = np.linspace(1, 12, 15)
leader_velocity = np.linspace(0, 20, 40)
initial_conditions_ranges = [
    leader_position,
    leader_velocity,
    follower_position,
    follower_velocity,
]
# Initializes the generator of initial states
pg = misc.ParametersHyperparallelepiped(*initial_conditions_ranges, seed=seed)

# Specifies the STL formula to compute the robustness
leader_target = "!(G(dist <= 10 & dist >= 2))"
follower_target = "G(dist <= 10 & dist >= 2)"

# Instantiates the NN architectures
nn_leader = architecture.NeuralAgent(
    model_platooning.Agent.sensors, model_platooning.Agent.actuators, 2, 10, 2
)
nn_follower = architecture.NeuralAgent(
    model_platooning.Agent.sensors, model_platooning.Agent.actuators, 2, 10
)

dt = 0.05  # timestep

# Build the whole setting for the experiment
env = model_platooning.Environment()
leader = model_platooning.Agent("leader", nn_leader, leader_target)
follower = model_platooning.Agent("follower", nn_follower, follower_target)
world_model = model_platooning.Model(env, leader, follower, dt=dt)

# Instantiates the world's model
simulator = misc.Simulator(world_model, pg.sample(sigma=0.05))

working_dir = "/tmp/experiments/" + f"platooning_{seed:04}"

# Instantiates the traning and test environments
trainer = architecture.Trainer(simulator, working_dir)
tester = architecture.Tester(simulator, working_dir)

epochs = 10  # number of train/test iterations

training_steps = 15  # number of episodes for training
train_simulation_horizon = int(5 / dt)  # 5 seconds

test_steps = 2  # number of episodes for testing
test_simulation_horizon = int(60 / dt)  # 60 seconds

for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}:")
    # Starts the training
    trainer.run(
        training_steps,
        {"leader": 3, "follower": 5},
        train_simulation_horizon,
        epoch=epoch,
    )
    # Starts the testing
    tester.run(test_steps, test_simulation_horizon, epoch=epoch)

# Saves the trained models
misc.save_models(nn_leader, nn_follower, working_dir)
