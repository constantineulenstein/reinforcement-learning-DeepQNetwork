import numpy as np
import torch
import collections
import time


class Agent:

    # Function to initialise the agent
    def __init__(self):
        # Set the episode length
        self.episode_length = 1000
        # Reset the total number of steps which the agent has taken
        self.num_steps_taken = 0
        # The state variable stores the latest state of the agent in the environment
        self.state = None
        # The action variable stores the latest action which the agent has applied to the environment
        self.action = None
        # Define discrete action space to E, NE, N, W, S, SW
        self.action_space = [0,1,2,3,4,5]
        # Initialiuze DQN()
        self.dqn = DQN()
        #set exploration steps. those are the steps that the agent just explores without updating Q in the beginning
        self.exploration_steps = 1000
        #epsilon starts at 1 and decays to 0.1
        self.epsilon = 1
        self.epsilon_min = 0.1
        # föag for checking whether we should try greedy policy
        self.check_greedy = False
        # flag for checking whether we are right now in round which uses greedy policy
        self.new_greedy_round = False
        # flag for early stopping training because we reached goal in under 100 steps and greedy
        self.stop = False
        # flag that tells agent to temporarily switch to exploring mode
        self.exploring = False
        # counter to count the number of steps in current episode
        self.episode_counter = 0
        # time used to decay epsilon and for episode reduction
        self.start_time = time.time()
        self.reduce_episode_time = self.start_time



    # Function to check whether the agent has reached the end of an episode
    def has_finished_episode(self):
        if (self.num_steps_taken % self.episode_length == 0):
            # reset current episode counter
            self.episode_counter = 0
            #when 5 mins passed, ensure agent is no longer in explore mode when new episode begins, otherwise agent will stay in explore mode for first 15 % of episode
            if (self.exploring == True) & ((time.time()-self.start_time)>5*60):
                self.exploring = False
            #check whether the greedy round just ended and reset epsilon in case goal was not reached
            if (self.new_greedy_round) & (self.stop == False):
                self.new_greedy_round = False
                self.check_greedy = False
            #check whether greedy round should start
            if self.check_greedy == True:
                self.epsilon = 0
                self.new_greedy_round = True
            return True
        else:
            #increment steps counter for current episode
            self.episode_counter += 1
            return False


    # Function to get the next action, using whatever method you like
    def get_next_action(self, state):
        #in the beginning,when still exploring first 1000 steps, choose random action
        if self.num_steps_taken < self.exploration_steps:
            discrete_action = np.random.choice(self.action_space)
        # otherwise do epsilon greedy policy
        else:
            #calculate highest q value for state and get action according to epsilon greedy
            best_action = self.get_highest_q_value_and_action(state)[1]
            discrete_action = self.get_epsilon_greedy_action(best_action)
        continuous_action = self._discrete_action_to_continuous(discrete_action)
        # Update the number of steps which the agent has taken
        self.num_steps_taken += 1
        #print(self.num_steps_taken)
        # Store the state; this will be used later, when storing the transition
        self.state = state
        # Store the action; this will be used later, when storing the transition
        self.action = discrete_action
        return continuous_action


    # Function to set the next state and distance, which resulted from applying action self.action at state self.state
    def set_next_state_and_distance(self, next_state, distance_to_goal):
        # Convert the distance to a reward
        reward = self.get_reward(distance_to_goal, next_state)
        # Create a transition
        transition = (self.state, self.action, reward, next_state)
        #initialize weights for prioritized experience replay buffer
        w_init = self.dqn.replay_buffer.initialize_w()
        # add transition and initialized weights to buffer if not in greedy round
        if not self.new_greedy_round:
            self.dqn.replay_buffer.buffer_append(transition, w_init)
        # check whether exploration time of first 1000 episodes is over and start training network
        if self.num_steps_taken >= self.exploration_steps:
            #check whether we should actually train or are currently trying out a greedy policy (or already found the working greedy policy)
            if not self.new_greedy_round:
                self.dqn.train_q_network()
            # update target network if specific num_steps_taken (each 50 steps)
            if (self.num_steps_taken % self.dqn.target_update_freq) == 0:
                #check whether we should actually update or are currently trying out a greedy policy (or already found the working greedy policy)
                if not self.new_greedy_round:
                    self.dqn.update_target_network()
        #calculate time elapsed since beginning
        time_elapsed = time.time()
        # reduce episode length to ensure convergence towards 100; reduction starts after 4 mins and reduces 50 steps roughly every 14s
        if ((time_elapsed-self.start_time) > 4*60) & (int(time_elapsed-self.reduce_episode_time) > 14):
            self.reduce_episode_length(50)
            self.reduce_episode_time = time_elapsed
        #if in first 5 mins, self.exploring will be TRUE in the beginning of each episode; set false after 15% of episode length
        if (self.episode_counter/self.episode_length == 0.15) & (self.exploring == True):
            self.exploring = False
        #if number of episodes is > 100 and we are in our last 10% of the episode, go in exploring mode (epsilon = 0.9)
        if (self.episode_counter/self.episode_length == 0.9) & (self.exploring == False) & (self.episode_length > 100) & (self.epsilon < 0.9):
            self.exploring = True
            self.epsilon = 0.9
        #try out early stopping every 3 episodes as soon es episode length is 100; every 2 episodes in last minute
        if ((time_elapsed-self.start_time) > 9*60) & (self.episode_length == 100) & (self.num_steps_taken % 200 == 0):
            self.check_greedy = True
        elif (self.episode_length == 100) & (self.num_steps_taken % 300 == 0):
            self.check_greedy = True
        #check whether agent reached goal in greedy round -> stop training
        if (distance_to_goal<0.03) & (self.new_greedy_round):
            self.stop = True


    # Function to get the greedy action for a particular state based on highes Q-value
    def get_greedy_action(self, state):
        discrete_action = self.get_highest_q_value_and_action(state)[1]
        return self._discrete_action_to_continuous(discrete_action)


    # Convert discrete action into continuous
    def _discrete_action_to_continuous(self, discrete_action):
        action_steps = np.array([[0.02,0.],[0.014,0.014],[0,0.02],[-0.02,0],[0.,-0.02],[0.014,-0.014]])
        continuous_action = action_steps[discrete_action]
        return continuous_action


    #Compute current best action and q-value for specific state
    def get_highest_q_value_and_action(self,state):
        state_tensor = torch.tensor(state, dtype=torch.float32)
        q_values_state = self.dqn.q_network.forward(torch.unsqueeze(state_tensor, 0)).detach().numpy()
        best_q = np.max(q_values_state[0])
        best_action = np.argmax(q_values_state[0])
        return (best_q, best_action)


    # Choose action based on epsilon greedy policy; also perform epsilon decay
    def get_epsilon_greedy_action(self, best_action):
        # decay epsilon linearly and ensure its never less than epsilon_min; reached epsilon_min after 8 minutes
        if (not self.exploring) & (not self.new_greedy_round):
            self.epsilon = 1 - 0.9*((time.time()-self.start_time)/(60*8))
            self.epsilon = max(self.epsilon,self.epsilon_min)
        # build probabilities to choose actions according to epsilon
        prob_array = np.full(len(self.action_space), self.epsilon/len(self.action_space))
        prob_array[best_action] = 1 - self.epsilon + self.epsilon/len(self.action_space)
        return np.random.choice(self.action_space, 1, p = prob_array)[0]


    # reduce episode lenght by subtract. Never reduce to less than 100
    def reduce_episode_length(self, subtract):
        self.episode_length = max(self.episode_length-subtract, 100)


    #calculate reward function. Give higher reward for closer states -> cubic reward function
    def get_reward(self,distance_to_goal, next_state):
        reward = (1 - distance_to_goal)**3
        #extra incentive to reach goal
        if distance_to_goal < 0.03:
            reward *= 3
        #penalize reward, when agent moves against wall
        if (next_state == self.state).all():
            reward = 0
        return reward




# The Network class inherits the torch.nn.Module class, which represents a neural network.
class Network(torch.nn.Module):
    # The class initialisation function. This takes as arguments the dimension of the network's input (i.e. the dimension of the state), and the dimension of the network's output (i.e. the dimension of the action).
    def __init__(self, input_dimension, output_dimension):
        # Call the initialisation function of the parent class.
        super(Network, self).__init__()
        neurons = 128
        # Define the network layers.
        self.layer_1 = torch.nn.Linear(in_features=input_dimension, out_features=256)
        self.layer_2 = torch.nn.Linear(in_features=256, out_features=neurons)
        self.layer_3 = torch.nn.Linear(in_features=neurons, out_features=neurons)
        self.layer_4 = torch.nn.Linear(in_features=neurons, out_features=neurons)
        self.layer_5 = torch.nn.Linear(in_features=neurons, out_features=neurons)
        self.layer_6 = torch.nn.Linear(in_features=neurons, out_features=neurons)
        self.layer_7 = torch.nn.Linear(in_features=neurons, out_features=neurons)
        self.output_layer = torch.nn.Linear(in_features=neurons, out_features=output_dimension)

    # Function which sends some input data through the network and returns the network's output.
    def forward(self, input):
        layer_1_output = torch.nn.functional.relu(self.layer_1(input))
        layer_2_output = torch.nn.functional.relu(self.layer_2(layer_1_output))
        layer_3_output = torch.nn.functional.relu(self.layer_3(layer_2_output))
        layer_4_output = torch.nn.functional.relu(self.layer_4(layer_3_output))
        layer_5_output = torch.nn.functional.relu(self.layer_5(layer_4_output))
        layer_6_output = torch.nn.functional.relu(self.layer_6(layer_5_output))
        layer_7_output = torch.nn.functional.relu(self.layer_7(layer_6_output))
        output = self.output_layer(layer_7_output)
        return output





# The DQN class determines how to train the above neural network.
class DQN:
    # The class initialisation function.
    def __init__(self):
        # Create a Q-network, which predicts the q-value for a particular state.
        self.q_network = Network(input_dimension=2, output_dimension=6)
        self.q_network_target = Network(input_dimension=2, output_dimension=6)
        # set target update frequency to 50
        self.target_update_freq = 50
        #Create replay buffer for network with size 10000
        self.replay_buffer = ReplayBuffer(10000)
        # Define the optimiser which is used when updating the Q-network. The learning rate determines how big each gradient step is during backpropagation.
        self.optimiser = torch.optim.Adam(self.q_network.parameters(), lr=0.001)
        # gamma, sidcount factor of 0.95
        self.discount_factor = 0.95
        #ensure that q network and target network have same initialized weights in the beginning
        self.q_network_target.load_state_dict(self.q_network.state_dict())
        #set true for double dqn
        self.ddqn = True


    # Function that is called whenever we want to train the Q-network. Each call to this function takes in a transition tuple containing the data we use to update the Q-network.
    def train_q_network(self):
        # Set all the gradients stored in the optimiser to zero.
        self.optimiser.zero_grad()
        #get batch sample from buffer
        sample_batch = self.replay_buffer.get_batch()
        # Calculate the loss for this batch.
        loss = self._calculate_loss(sample_batch)
        # Compute the gradients based on this loss, i.e. the gradients of the loss with respect to the Q-network parameters.
        loss.backward()
        # Take one gradient step to update the Q-network.
        self.optimiser.step()
        # Return the loss as a scalar
        return loss.item()


    # Function to calculate the loss for a particular batch.
    def _calculate_loss(self, batch):
        #convert batch to tensors
        states_tensor = torch.tensor(batch[0], dtype=torch.float32)
        actions_tensor = torch.tensor(batch[1], dtype=torch.int64)
        rewards_tensor = torch.tensor(batch[2], dtype=torch.float32)
        nextstates_tensor = torch.tensor(batch[3], dtype=torch.float32)
        #get q network predictions for current state and next state
        states_q_values = self.q_network.forward(states_tensor)
        nextstates_q_values = self.q_network.forward(nextstates_tensor)
        #get target network prediction of next_states and detach to ensure no training on them
        nextstates_q_values_target = self.q_network_target.forward(nextstates_tensor)
        nextstates_q_values_target = nextstates_q_values_target.detach()
        #check whether double deep q network
        if self.ddqn:
            #get action that gives maximum q_value for target network
            nextstates_maxq_action = torch.max(nextstates_q_values,1)[1]
            #gather the q_values of next states that correspond to the optimal action for target network
            nextstates_maxq_values = torch.gather(nextstates_q_values_target.detach(),1,nextstates_maxq_action.unsqueeze(-1)).squeeze(-1)
        else:
            #get maximum q_value of each four actions of target network for next state
            nextstates_maxq_values = torch.max(nextstates_q_values_target,1)[0]
        #calculate target tensor R + gamma*Q_next_state_target)
        target_tensor = torch.add(rewards_tensor, nextstates_maxq_values, alpha = self.discount_factor)
        #gather the q_values that correspond to the actions taken
        state_action_q_values = torch.gather(states_q_values,1,actions_tensor.unsqueeze(-1)).squeeze(-1)
        #calculate new weights delta (for prioritized experience replay buffer) and update them
        weights = torch.abs(torch.sub(target_tensor, state_action_q_values))
        weights = weights.detach().numpy()
        # update weights in reply buffer according to new deltas
        self.replay_buffer.update_w(weights,batch[4])
        # Compute the loss
        loss = torch.nn.MSELoss()(target_tensor, state_action_q_values)
        return loss


    #load weights of q network (trained network) into the target network
    def update_target_network(self):
        self.q_network_target.load_state_dict(self.q_network.state_dict())




#Experience Replay buffer
class ReplayBuffer:
    def __init__(self, maxlength):
        #init buffer
        self.buffer = collections.deque(maxlen=maxlength)
        self.maxlength = maxlength
        self.batch_size = 256
        #init weights of priotitized buffer
        self.w = collections.deque(maxlen=maxlength)
        #will later be updated according to max weight; now just initialized
        self.epsilon_buffer = 0.0001
        #set extent of prioritization and decay rate
        self.alpha = 0.6
        self.alpha_decay = 0.99995

    #get current length of buffer
    def buffer_length(self):
        return len(self.buffer)

    #when called, add transition and weights to buffer; if buffer is full, pop old values
    def buffer_append(self,transition, weight):
        if len(self.buffer) == self.maxlength:
            self.buffer.popleft()
            self.w.popleft()
        self.buffer.append(transition)
        self.w.append(weight)

    #get random batch from buffer according to prioritization
    def get_batch(self):
        #slowly let alpha decay from 0.6 to 0.3 to ensure less prioritization later when network is stable
        self.alpha *= self.alpha_decay
        self.alpha = max(self.alpha, 0.3)
        #update probabilities to draw to be updated indices
        probabilities = np.array(self.w)**self.alpha
        probabilities /= np.sum(probabilities)
        #choose batch indices according to probabilities; and return the batch
        indices = np.random.choice(len(self.buffer), self.batch_size, replace=False, p = probabilities)
        states, actions, rewards, next_states = zip(*[self.buffer[idx] for idx in indices])
        return (np.array(states,dtype=np.float32), np.array(actions), np.array(rewards,dtype=np.float32), np.array(next_states,dtype=np.float32), indices)

    #initialize weights to always the maximum of all weights to ensure high prioritization in the beginning
    def initialize_w(self):
        if len(self.w) == 0:
            return self.epsilon_buffer
        else:
            return max(self.w)

    #update weights when trained this transition and new delta was found
    def update_w(self, weights, sample_indices):
        #ensure epsilon in buffer is always 20% of the maximum weight
        self.epsilon_buffer = 0.2 * max(self.w)
        for idx, sample_idx in enumerate(sample_indices):
            self.w[sample_idx] = weights[idx] + self.epsilon_buffer

