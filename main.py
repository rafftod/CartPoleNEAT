import gym
import neat
import os
import numpy as np

# checkpoints duration for NEAT checkpointer
POPULATION_SAVE_PERIOD = 1
# number of generations to train for
NUM_GENERATIONS = 100
# number of steps to take per episode, which is the maximum defined by Gym
CARTPOLE_MAX_STEPS = 500

# create gym environment
env = gym.make("CartPole-v1")


def run_episode(net: neat.nn.FeedForwardNetwork) -> float:
    """Evaluates the performance of a genome on an episode.
    A single evaluation episode is sufficient as the environment is deterministic.

    Parameters
    ----------
    net : neat.nn.FeedForwardNetwork
        Genome neural network.

    Returns
    -------
    float
        Number of steps the pole stayed up.
    """
    observation = env.reset()
    total_reward = 0
    for _ in range(CARTPOLE_MAX_STEPS):
        env.render()
        # get action from network and convert it to discrete {0,1} action space
        action = np.around(net.activate(observation)).astype(int)[0]
        observation, reward, done, _ = env.step(action)
        total_reward += reward
        if done:
            break
    return total_reward


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        genome.fitness = run_episode(net)


def run(config: neat.Config) -> None:
    # create population
    population = neat.Population(config)
    # add console monitoring for training
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
    # add checkpoints to save progress
    population.add_reporter(neat.Checkpointer(POPULATION_SAVE_PERIOD))
    # run training
    winner = population.run(eval_genomes, NUM_GENERATIONS)


if __name__ == "__main__":
    # Load the config file
    config_filepath = os.path.join(os.path.dirname(__file__), "config.txt")
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_filepath,
    )
    run(config)
