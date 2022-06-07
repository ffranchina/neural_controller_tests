import misc
import architecture
import model_platooning

import torch
import random
import numpy as np

seed = random.randint(0, 10000)
torch.manual_seed(seed)

# Specifies the initial conditions of the setup
agent_position = 0
agent_velocity = np.linspace(0, 20, 40)
leader_position = np.linspace(1, 12, 15)
leader_velocity = np.linspace(0, 20, 40)
# Initializes the generator of initial states
pg = misc.ParametersHyperparallelepiped(
    agent_position, agent_velocity, leader_position, leader_velocity, seed=seed
)

# Instantiates the world's model
simulator = misc.Simulator(model_platooning.Model, pg.sample(sigma=0.05))

# Specifies the STL formula to compute the robustness
robustness_formula = "G(dist <= 10 & dist >= 2)"
robustness_computer = model_platooning.RobustnessComputer(robustness_formula)

# Instantiates the NN architectures
attacker = architecture.Attacker(simulator, 2, 10, 2)
defender = architecture.Defender(simulator, 2, 10)

working_dir = "/tmp/experiments/" + f"platooning_{seed:04}"

# Instantiates the traning and test environments
trainer = architecture.Trainer(
    simulator, robustness_computer, attacker, defender, working_dir
)
tester = architecture.Tester(
    simulator, robustness_computer, attacker, defender, working_dir
)

dt = 0.05  # timestep
epochs = 10  # number of train/test iterations

training_steps = 5  # number of episodes for training
train_simulation_horizon = int(5 / dt)  # 5 seconds

test_steps = 10  # number of episodes for testing
test_simulation_horizon = int(60 / dt)  # 60 seconds

for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}:")
    # Starts the training
    trainer.run(
        training_steps,
        train_simulation_horizon,
        dt,
        atk_steps=3,
        def_steps=5,
        epoch=epoch,
    )
    # Starts the testing
    tester.run(test_steps, test_simulation_horizon, dt, epoch=epoch)

# Saves the trained models
misc.save_models(attacker, defender, working_dir)
