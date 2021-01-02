# Import some modules from other libraries
import numpy as np
import torch
import time
import matplotlib.pyplot as plt


# Import the environment module
from environment import Environment


# The Agent class allows the agent to interact with the environment.
class Agent:

    # The class initialisation function.
    def __init__(self, environment):
        # Set the agent's environment.
        self.environment = environment
        # Create the agent's current state
        self.state = None
        # Create the agent's total reward for the current episode.
        self.total_reward = None
        # Reset the agent.
        self.reset()

    # Function to reset the environment, and set the agent to its initial state. This should be done at the start of every episode.
    def reset(self):
        # Reset the environment for the start of the new episode, and set the agent's state to the initial state as defined by the environment.
        self.state = self.environment.reset()
        # Set the agent's total reward for this episode to zero.
        self.total_reward = 0.0

    # Function to make the agent take one step in the environment.
    def step(self):
        # Choose the next action.
        discrete_action = self._choose_next_action()
        # Convert the discrete action into a continuous action.
        continuous_action = self._discrete_action_to_continuous(discrete_action)
        # Take one step in the environment, using this continuous action, based on the agent's current state. This returns the next state, and the new distance to the goal from this new state. It also draws the environment, if display=True was set when creating the environment object..
        next_state, distance_to_goal = self.environment.step(self.state, continuous_action)
        # Compute the reward for this paction.
        reward = self._compute_reward(distance_to_goal)
        # Create a transition tuple for this step.
        transition = (self.state, discrete_action, reward, next_state)
        # Set the agent's state for the next step, as the next state from this step
        self.state = next_state
        # Update the agent's reward for this episode
        self.total_reward += reward
        # Return the transition
        return transition

    # Function for the agent to choose its next action
    def _choose_next_action(self):
        # Return discrete action 0
        # 0 -> E, 1 -> N, 2 -> W, 3 -> S
        return np.random.choice([0,1,2,3])

    # Function to convert discrete action (as used by a DQN) to a continuous action (as used by the environment).
    def _discrete_action_to_continuous(self, discrete_action):
        if discrete_action == 0:
            # Move 0.1 to the right, and 0 upwards
            continuous_action = np.array([0.1, 0], dtype=np.float32)
        elif discrete_action == 1:
            # Move 0 to the right, and 0.1 upwards
            continuous_action = np.array([0, 0.1], dtype=np.float32)
        elif discrete_action == 2:
            # Move 0.1 to the left, and 0 upwards
            continuous_action = np.array([-0.1, 0], dtype=np.float32)
        elif discrete_action == 3:
            # Move 0 to the right, and 0.1 downwards
            continuous_action = np.array([0, -0.1], dtype=np.float32)
        return continuous_action

    # Function for the agent to compute its reward. In this example, the reward is based on the agent's distance to the goal after the agent takes an action.
    def _compute_reward(self, distance_to_goal):
        reward = float(0.1*(1 - distance_to_goal))
        return reward


# The Network class inherits the torch.nn.Module class, which represents a neural network.
class Network(torch.nn.Module):

    # The class initialisation function. This takes as arguments the dimension of the network's input (i.e. the dimension of the state), and the dimension of the network's output (i.e. the dimension of the action).
    def __init__(self, input_dimension, output_dimension):
        # Call the initialisation function of the parent class.
        super(Network, self).__init__()
        # Define the network layers. This example network has two hidden layers, each with 100 units.
        self.layer_1 = torch.nn.Linear(in_features=input_dimension, out_features=100)
        self.layer_2 = torch.nn.Linear(in_features=100, out_features=100)
        self.output_layer = torch.nn.Linear(in_features=100, out_features=output_dimension)

    # Function which sends some input data through the network and returns the network's output. In this example, a ReLU activation function is used for both hidden layers, but the output layer has no activation function (it is just a linear layer).
    def forward(self, input):
        layer_1_output = torch.nn.functional.relu(self.layer_1(input))
        layer_2_output = torch.nn.functional.relu(self.layer_2(layer_1_output))
        output = self.output_layer(layer_2_output)
        return output


# The DQN class determines how to train the above neural network.
class DQN:

    # The class initialisation function.
    def __init__(self):
        # Create a Q-network, which predicts the q-value for a particular state.
        self.q_network = Network(input_dimension=2, output_dimension=4)
        # Define the optimiser which is used when updating the Q-network. The learning rate determines how big each gradient step is during backpropagation.
        self.optimiser = torch.optim.Adam(self.q_network.parameters(), lr=0.001)

    # Function that is called whenever we want to train the Q-network. Each call to this function takes in a transition tuple containing the data we use to update the Q-network.
    def train_q_network(self, transition):
        # Set all the gradients stored in the optimiser to zero.
        self.optimiser.zero_grad()
        # Calculate the loss for this transition.
        loss = self._calculate_loss(transition)
        # Compute the gradients based on this loss, i.e. the gradients of the loss with respect to the Q-network parameters.
        loss.backward()
        # Take one gradient step to update the Q-network.
        self.optimiser.step()
        # Return the loss as a scalar
        return loss.item()

    # Function to calculate the loss for a particular transition.
    def _calculate_loss(self, transition):
        # NOTE: when just training on a single example on each iteration, the NumPy array (and Torch tensor) still needs to have two dimensions: the mini-batch dimension, and the data dimension. And in this case, the mini-batch dimension would be 1, instead of 5. This can be done by using the torch.unsqueeze() function.
        # Convert the NumPy array into a Torch tensor
        #input is state position
        input_tensor = torch.tensor(transition[0], dtype = torch.float32)
        input_tensor = torch.unsqueeze(input_tensor, 0)
        # label is reward
        label_tensor = torch.tensor(transition[2], dtype = torch.float32)
        label_tensor = torch.unsqueeze(label_tensor, 0)
        #print(label_tensor)
        #print(label_tensor.size())
        # Do a forward pass of the network using the inputs batch # NOTE: when training a Q-network, you will need to find the prediction for a particular action. This can be done using the "torch.gather()" function.
        network_prediction = self.q_network.forward(input_tensor)
        #print(network_prediction)
        #print(transition[1])
        network_prediction = torch.gather(network_prediction,1,torch.tensor([[transition[1]]]))[0]
        #print(network_prediction)
        # Compute the loss based on the label's batch
        loss = torch.nn.MSELoss()(label_tensor, network_prediction)
        return loss


# Main entry point
if __name__ == "__main__":

    # Create an environment.
    # If display is True, then the environment will be displayed after every agent step. This can be set to False to speed up training time. The evaluation in part 2 of the coursework will be done based on the time with display=False.
    # Magnification determines how big the window will be when displaying the environment on your monitor. For desktop monitors, a value of 1000 should be about right. For laptops, a value of 500 should be about right. Note that this value does not affect the underlying state space or the learning, just the visualisation of the environment.
    environment = Environment(display=True, magnification=500)
    # Create an agent
    agent = Agent(environment)
    # Create a DQN (Deep Q-Network)
    dqn = DQN()

    losses = []
    iterations = []
    episode_length = 20

    #fig, ax = plt.subplots()
    #ax.set(xlabel='Iteration', ylabel='Loss', title='Loss Curve')
    plt.ion()
    training_iteration = 0
    # Loop over episodes
    for epoch in range(100):
        # Reset the environment for the start of the episode.
        agent.reset()
        training_iteration += 1
        # Loop over steps within this episode. The episode length here is 20.
        avg_loss = np.zeros(episode_length)
        loss = 0
        for step_num in range(episode_length):
            # Step the agent once, and get the transition tuple for this step
            transition = agent.step()
            # Calculate new loss function for this step
            loss += dqn.train_q_network(transition)

            # Sleep, so that you can observe the agent moving. Note: this line should be removed when you want to speed up training
            #time.sleep(0.2)
        iterations.append(training_iteration)
        losses.append(loss/episode_length)
        #uncomment for real time plotting
        #if training_iteration % 10 == 0:
        #    plt.clf()
        #    plt.plot(iterations, losses, color='blue')
        #    plt.yscale('log')
        #    plt.draw()

    plt.clf()
    plt.plot(iterations, losses, color='blue')
    plt.yscale('log')
    plt.ylabel('Loss')
    plt.xlabel('Episodes')
    plt.title('Online Learning - Average Loss per Episode')
    plt.savefig('loss_online')


