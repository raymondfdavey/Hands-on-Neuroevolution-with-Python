import gymnasium as gym
import neat
import os
import visualize
import math
env = gym.make('CartPole-v1')
# env = gym.make('CartPole-v1', render_mode="human")

local_dir = os.path.dirname(__file__)

out_dir = os.path.join(local_dir, 'out')

config_file = os.path.join(local_dir, 'single_pole_config.ini')

config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

p = neat.Population(config)

# Add a stdout reporter to show progress in the terminal.
p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)
p.add_reporter(neat.Checkpointer(5, filename_prefix='out/neat-checkpoint-'))


max_time_steps = 50000
test_runs = 50


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = 0.0
        total_fitness = 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)

        for run in range(test_runs):
            # print(f"genome {genome_id}, run {run}")
            observation = env.reset()[0]
            for no_time_steps in range(max_time_steps):
                inputs = observation
                output = net.activate(inputs)
                if (output[0] >= 0.5):
                    observation, reward, done, info, _ = env.step(1)
                else:
                    observation, reward, done, info, _ = env.step(0)
                if done: 
                    env.reset()  
                    break
            total_fitness += no_time_steps
        genome.fitness = math.log10(total_fitness)




best_genome = p.run(eval_genomes,100)
# Display the best genome among generations.
print('\nBest genome:\n{!s}'.format(best_genome))

def test_best(net, runs):
    # env.render()
    reward_total = 0
    perfect_runs = 0 
    for run in range(runs):
        observation = env.reset()[0]

        for no_time_steps in range(10000):
            inputs = observation
            output = net.activate(inputs)
            if (output[0] >= 0.5):
                observation, reward, done, info, _ = env.step(1)
            else:
                observation, reward, done, info, _ = env.step(0)
            reward_total += reward
            if done: 
                break
        print(f"run: {run}, steps: {no_time_steps} from 10000")
        if no_time_steps == 10000:
            perfect_runs += 1
    print(f"perfect runs: {perfect_runs}")
        # print("id", genome_id, "fitness", no_time_steps)
    

# Check if the best genome is a winning Single-Pole balancing controller 
net = neat.nn.FeedForwardNetwork.create(best_genome, config)
print("\n\nEvaluating the best genome in random runs")
test_best(net, 100)
# Visualize the experiment results
node_names = {-1:'x', -2:'dot_x', -3:'θ', -4:'dot_θ', 0:'action'}
visualize.draw_net(config, best_genome, True, node_names=node_names, directory=out_dir, fmt='svg')
visualize.plot_stats(stats, ylog=False, view=True, filename=os.path.join(out_dir, 'avg_fitness.svg'))
visualize.plot_species(stats, view=True, filename=os.path.join(out_dir, 'speciation.svg'))

