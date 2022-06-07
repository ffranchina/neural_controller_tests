import misc
import architecture
import model_cruisecontrol

import torch
import random
import numpy as np

seed = random.randint(0, 10000)
torch.manual_seed(seed)

# Specifies the initial conditions of the setup
agent_position = np.arange(model_cruisecontrol.ROAD_LENGTH)
agent_velocity = np.linspace(-12, 12, 25)
# Initializes the generator of initial states
pg = misc.ParametersHyperparallelepiped(agent_position, agent_velocity, seed=seed)

# Instantiates the world's model
simulator = misc.Simulator(model_cruisecontrol.Model, pg.sample(sigma=0.05))

# Specifies the STL formula to compute the robustness
robustness_formula = "G(v >= 4.75 & v <= 5.25)"
robustness_computer = model_cruisecontrol.RobustnessComputer(robustness_formula)

# Instantiates the NN architectures
attacker = architecture.Attacker(simulator, 1, 10, 5, n_coeff=1)
defender = architecture.Defender(simulator, 2, 10)

working_dir = "/tmp/experiments/" + f"cruise_{seed:04}"

# Instantiates the traning environment
trainer = architecture.Trainer(
    simulator, robustness_computer, attacker, defender, working_dir
)

dt = 0.05  # timestep
epochs = 10  # number of train/test iterations

training_steps = 30  # number of episodes for training
train_simulation_horizon = int(0.5 / dt)  # 5 seconds

for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}:")
    # Starts the training
    trainer.run(
        training_steps,
        train_simulation_horizon,
        dt,
        atk_steps=1,
        def_steps=10,
        atk_static=True,
        epoch=epoch,
    )

# Saves the trained models
misc.save_models(attacker, defender, working_dir)
