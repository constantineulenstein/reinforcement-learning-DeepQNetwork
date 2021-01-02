# reinforcement-learning-DeepQNetwork

## Description:

This project contains the implementation of a Deep Q-learning agent which finds a path to the goal in a randomly generated grid world. Following techniques were used:
* Double Deep Q-network (created in PyTorch)
* Models with discrete actions including diagonal actions (agent.py)
* Prioritised experience replay buffer
* Epsilon greedy policy
* Target network
* Early stopping


## Folder structure:
* **part1**: Individual python files enhance the agent to solve the gridworld with one static obstacle. Files improve from an agent with replay buffer to an agent that uses a target network and learns the environment through the bellman equation. 
* **part2**: agent.py entails all the logic behind the agent. 
* All the specifications of the exercise are inside task.pdf and the answers to the exercises as well as a detailed description of the implemented code for part2 can be found in report.pdf

## Requirements
reinforcement-learning-DeepQNetwork requires the following to run: 
* ```Python3```
* ```numpy ```
* ```cv2```
* ```torch```
* ```collections```
* ```time```

## Run instructions:

* This project can be run via the command line using the command ```python3 part2/train_and_test.py``` . In order to visualize the grid world set the boolean display_on to True.
    * The code then builds the environment and runs the agent. This is done for 10 minutes and the aim for the agent is to reach the goal 
  
