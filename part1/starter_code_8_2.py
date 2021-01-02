# Import some modules from other libraries
import numpy as np
import torch
import time
from matplotlib import pyplot as plt
import collections
import random
import cv2

# Import the environment module
from environment import Environment
from q_value_visualiser import QValueVisualiser     


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
        return np.random.choice([0,1,2,3])

    # Function to convert discrete action (as used by a DQN) to a continuous action (as used by the environment).
    def _discrete_action_to_continuous(self, discrete_action):
        #if discrete_action == 0:
            # Move 0.1 to the right, and 0 upwards
            #continuous_action = np.array([0.1, 0], dtype=np.float32)
        
        directions = np.array([[0,0.1],[0.1,0],[0,-0.1],[-0.1,0]])
        continuous_action = directions[discrete_action]
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
        self.q_network_target = Network(input_dimension = 2, output_dimension = 4)
        # Define the optimiser which is used when updating the Q-network. The learning rate determines how big each gradient step is during backpropagation.
        self.optimiser = torch.optim.Adam(self.q_network.parameters(), lr=0.001)
        self.optimiser_target = torch.optim.Adam(self.q_network_target.parameters(),lr = 0.001)
        
        self.q_network_target.load_state_dict(self.q_network.state_dict())

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
    
    def _calculate_loss(self,minibatch):
        
        states,actions,rewards,next_states = minibatch
        
        #Create tensors
        states_tensor = torch.tensor(states,dtype = torch.float32)
        actions_tensor = torch.tensor(actions)
        rewards_tensor = torch.tensor(rewards,dtype = torch.float32)
        next_states_tensor = torch.tensor(next_states, dtype = torch.float32)
        
        #Compute Q(s,a)
        q_states_prediction = self.q_network.forward(states_tensor)
        #Compute Q(s',a)
        q_next_states_prediction = self.q_network_target.forward(next_states_tensor).detach()
        
        #Get max over actions a
        q_next_states_maxes = torch.max(q_next_states_prediction,dim = -1)[0]
        
        #Get max across batch according to specification of CW
        state_action_q_values = q_states_prediction.gather(dim = 1,index = actions_tensor.unsqueeze(-1)).squeeze(-1)
        
        #R + gamma * Q(s',a)
        network_prediction = torch.add(rewards_tensor,q_next_states_maxes,alpha = 0.9)
        #
        loss = torch.nn.MSELoss()(network_prediction,state_action_q_values)
        return loss
    
    def get_q_values(self):
        
        #Creating the states 
        states = np.zeros([100,2])
        i = 0
        for col in range(10):
            for row in range(10):
                states[i,0] = col/10 + 0.05
                states[i,1] = row/10 + 0.05
                i += 1
        #Create tensor from states array   
        input_tensor = torch.tensor(states,dtype = torch.float32)
        #Exclude this from gradient computation and turn into numpy
        qs = self.q_network.forward(input_tensor).detach().numpy()
        return qs
    
    def update_target(self):
        
        self.q_network_target.load_state_dict(self.q_network.state_dict())

    
    

class ReplayBuffer:
    
    def __init__(self):
        
        self.buffer = collections.deque(maxlen = 5000)
    
    #Appends an element to the buffer
    def append(self,element):
        
        self.buffer.append(element)
    
    #Samples amount elements from buffer and returns states,action,rewards & stateprimes each in seperate list
    def sample(self,amount):
        
        if(amount > len(self.buffer)):
            amount = len(self.buffer)
        
        samples = random.sample(self.buffer,amount)
        return zip(*samples)
    
    #returns number of elements in buffer
    def len(self):
        return len(self.buffer)
        
    
    
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
    
    #Create a Replay Buffer
    replbuff = ReplayBuffer()
    
    #fig, ax = plt.subplots()
    #ax.set(xlabel='Iteration', ylabel='Loss', title='First Q-Learning')
    plt.ion()    
           
    # Create lists to store the losses and epochs
    losses = []
    iterations = []
    target_update_frequency = 10
    

    # Loop over episodes
    for episode in range(100):
        # Reset the environment for the start of the episode.
        agent.reset()
        # Loop over steps within this episode. The episode length here is 20.
        total_loss = 0
        save = False
        for step_num in range(20):
            # Step the agent once, and get the transition tuple for this step
            transition = agent.step()
            replbuff.append(transition)
            
            if (replbuff.len() >= 100):
            #Get minibatch of size 100
                save = True
                minibatch = replbuff.sample(100)
                loss = dqn.train_q_network(minibatch)
                total_loss += loss
                
            # Sleep, so that you can observe the agent moving. Note: this line should be removed when you want to speed up training
            #time.sleep(0.2)
            
            
          
        if save:
           #For plotting
            if(episode % target_update_frequency == 0) & (episode != 0 ):
                    dqn.update_target()
            losses.append(total_loss/20)
            iterations.append(episode + 1)
    qs = dqn.get_q_values()
    q_visualizer = QValueVisualiser(environment,magnification = 500)
    q_visualizer.draw_q_values(qs.reshape(10,10,4),'8_2')
    env2 = Environment(display=True, magnification=500)
    env2.draw_greedy_policy(qs.reshape(10,10,4),10,'8_2')
    
        
    plt.clf()
    plt.plot(iterations,losses, color='blue')
    plt.yscale('log')
    plt.savefig("loss_exercise82.png")
    plt.show()
    #cv2.destroyAllWindows()