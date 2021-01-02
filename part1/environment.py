import numpy as np
import cv2


# The Environment class defines the "world" within which the agent is acting
class Environment:

    # Function to initialise an Environment object
    def __init__(self, display, magnification=500):
        # Set whether the environment should be displayed after every step
        self.display = display
        # Set the magnification factor of the display
        self.magnification = magnification
        # Set the initial state of the agent
        self.init_state = np.array([0.35, 0.15], dtype=np.float32)
        # Set the initial state of the goal
        self.goal_state = np.array([0.35, 0.85], dtype=np.float32)
        # Set the space which the obstacle occupies
        self.obstacle_space = np.array([[0.0, 0.7], [0.4, 0.5]], dtype=np.float32)
        # Create an image which will be used to display the environment
        self.image = np.zeros([int(self.magnification), int(self.magnification), 3], dtype=np.uint8)

    # Function to reset the environment, which is done at the start of each episode
    def reset(self):
        return self.init_state

    # Function to execute an agent's step within this environment, returning the next state and the distance to the goal
    def step(self, state, action):
        # Determine what the new state would be if the agent could move there
        next_state = state + action
        # If this state is outside the environment's perimeters, then the agent stays still
        if next_state[0] < 0.0 or next_state[0] > 1.0 or next_state[1] < 0.0 or next_state[1] > 1.0:
            next_state = state
        # If this state is inside the obstacle, then the agent stays still
        if self.obstacle_space[0, 0] <= next_state[0] < self.obstacle_space[0, 1] and self.obstacle_space[1, 0] <= next_state[1] < self.obstacle_space[1, 1]:
            next_state = state
        # Compute the distance to the goal
        distance_to_goal = np.linalg.norm(next_state - self.goal_state)
        # Draw and show the environment, if required
        if self.display:
            self.draw(next_state)
        # Return the next state and the distance to the goal
        return next_state, distance_to_goal

    # Function to draw the environment and display it on the screen, if required
    def draw(self, agent_state):
        # Create the background image
        window_top_left = (0, 0)
        window_bottom_right = (self.magnification * 1, self.magnification * 1)
        cv2.rectangle(self.image, window_top_left, window_bottom_right, (246, 238, 229), thickness=cv2.FILLED)
        # Draw the obstacle
        obstacle_left = int(self.magnification * self.obstacle_space[0, 0])
        obstacle_top = int(self.magnification * (1 - self.obstacle_space[1, 1]))
        obstacle_width = int(self.magnification * (self.obstacle_space[0, 1] - self.obstacle_space[0, 0]))
        obstacle_height = int(self.magnification * (self.obstacle_space[1, 1] - self.obstacle_space[1, 0]))
        obstacle_top_left = (obstacle_left, obstacle_top)
        obstacle_bottom_right = (obstacle_left + obstacle_width, obstacle_top + obstacle_height)
        cv2.rectangle(self.image, obstacle_top_left, obstacle_bottom_right, (0, 0, 150), thickness=cv2.FILLED)
        # Create the border
        border_top_left = (0, 0)
        border_bottom_right = (self.magnification * 1, self.magnification * 1)
        cv2.rectangle(self.image, border_top_left, border_bottom_right, (0, 0, 0), thickness=int(self.magnification * 0.02))
        # Draw the agent
        agent_centre = (int(agent_state[0] * self.magnification), int((1 - agent_state[1]) * self.magnification))
        agent_radius = int(0.02 * self.magnification)
        agent_colour = (100, 199, 246)
        cv2.circle(self.image, agent_centre, agent_radius, agent_colour, cv2.FILLED)
        # Draw the goal
        goal_centre = (int(self.goal_state[0] * self.magnification), int((1 - self.goal_state[1]) * self.magnification))
        goal_radius = int(0.02 * self.magnification)
        goal_colour = (227, 158, 71)
        cv2.circle(self.image, goal_centre, goal_radius, goal_colour, cv2.FILLED)
        # Show the image
        cv2.imshow("Environment", self.image)
        # This line is necessary to give time for the image to be rendered on the screen
        cv2.waitKey(1)

    def draw_greedy_line(self, q_values):
        # Create the background image
        window_top_left = (0, 0)
        window_bottom_right = (self.magnification * 1, self.magnification * 1)
        cv2.rectangle(self.image, window_top_left, window_bottom_right, (246, 238, 229), thickness=cv2.FILLED)
        # Draw the obstacle
        obstacle_left = int(self.magnification * self.obstacle_space[0, 0])
        obstacle_top = int(self.magnification * (1 - self.obstacle_space[1, 1]))
        obstacle_width = int(self.magnification * (self.obstacle_space[0, 1] - self.obstacle_space[0, 0]))
        obstacle_height = int(self.magnification * (self.obstacle_space[1, 1] - self.obstacle_space[1, 0]))
        obstacle_top_left = (obstacle_left, obstacle_top)
        obstacle_bottom_right = (obstacle_left + obstacle_width, obstacle_top + obstacle_height)
        cv2.rectangle(self.image, obstacle_top_left, obstacle_bottom_right, (0, 0, 150), thickness=cv2.FILLED)
        # Create the border
        border_top_left = (0, 0)
        border_bottom_right = (self.magnification * 1, self.magnification * 1)
        cv2.rectangle(self.image, border_top_left, border_bottom_right, (0, 0, 0), thickness=int(self.magnification * 0.02))
        # Draw the agent
        agent_centre = (int(self.init_state[0] * self.magnification), int((1 - self.init_state[1]) * self.magnification))
        agent_radius = int(0.02 * self.magnification)
        agent_colour = (100, 199, 246)
        cv2.circle(self.image, agent_centre, agent_radius, agent_colour, cv2.FILLED)
        # Draw the goal
        goal_centre = (int(self.goal_state[0] * self.magnification), int((1 - self.goal_state[1]) * self.magnification))
        goal_radius = int(0.02 * self.magnification)
        goal_colour = (227, 158, 71)
        cv2.circle(self.image, goal_centre, goal_radius, goal_colour, cv2.FILLED)



        state = self.init_state
        for step in range(20):
            col = round((state[0]-0.05)*10)
            row = round((state[1]-0.05)*10)
            print(col,row)
            greedy_action_idx = np.argmax(q_values[col,row])
            if greedy_action_idx == 0:
                next_state = state + np.array([0.1, 0], dtype=np.float32)
                if next_state[0] < 0.0 or next_state[0] > 1.0 or next_state[1] < 0.0 or next_state[1] > 1.0:
                    next_state = state
                if self.obstacle_space[0, 0] <= next_state[0] < self.obstacle_space[0, 1] and self.obstacle_space[1, 0] <= next_state[1] < self.obstacle_space[1, 1]:
                    next_state = state
                green_top_left = (int(self.magnification*(state[0]-0.005)), int(self.magnification*(1-state[1]-0.005)))
                green_bottom_right = (int(self.magnification*(next_state[0]+0.005)), int(self.magnification*(1-next_state[1]+0.005)))
                cv2.rectangle(self.image, green_top_left, green_bottom_right, (0, 250, 0), cv2.FILLED)
            elif greedy_action_idx == 1:
                next_state = state + np.array([0, 0.1], dtype=np.float32)
                if next_state[0] < 0.0 or next_state[0] > 1.0 or next_state[1] < 0.0 or next_state[1] > 1.0:
                    next_state = state
                if self.obstacle_space[0, 0] <= next_state[0] < self.obstacle_space[0, 1] and self.obstacle_space[1, 0] <= next_state[1] < self.obstacle_space[1, 1]:
                    next_state = state
                green_top_left = (int(self.magnification*(next_state[0]-0.005)), int(self.magnification*(1-next_state[1]-0.005)))
                green_bottom_right = (int(self.magnification*(state[0]+0.005)), int(self.magnification*(1-state[1]+0.005)))
                cv2.rectangle(self.image, green_top_left, green_bottom_right, (0, 250, 0), cv2.FILLED)
            elif greedy_action_idx == 2:
                next_state = state + np.array([-0.1, 0], dtype=np.float32)
                if next_state[0] < 0.0 or next_state[0] > 1.0 or next_state[1] < 0.0 or next_state[1] > 1.0:
                    next_state = state
                if self.obstacle_space[0, 0] <= next_state[0] < self.obstacle_space[0, 1] and self.obstacle_space[1, 0] <= next_state[1] < self.obstacle_space[1, 1]:
                    next_state = state
                green_top_left = (int(self.magnification*(next_state[0]-0.005)), int(self.magnification*(1-next_state[1]-0.005)))
                green_bottom_right = (int(self.magnification*(state[0]+0.005)), int(self.magnification*(1-state[1]+0.005)))
                cv2.rectangle(self.image, green_top_left, green_bottom_right, (0, 250, 0), cv2.FILLED)
            elif greedy_action_idx == 3:
                next_state = state + np.array([0, -0.1], dtype=np.float32)
                if next_state[0] < 0.0 or next_state[0] > 1.0 or next_state[1] < 0.0 or next_state[1] > 1.0:
                    next_state = state
                if self.obstacle_space[0, 0] <= next_state[0] < self.obstacle_space[0, 1] and self.obstacle_space[1, 0] <= next_state[1] < self.obstacle_space[1, 1]:
                    next_state = state
                green_top_left = (int(self.magnification*(state[0]-0.005)), int(self.magnification*(1-state[1]-0.005)))
                green_bottom_right = (int(self.magnification*(next_state[0]+0.005)), int(self.magnification*(1-next_state[1]+0.005)))
                cv2.rectangle(self.image, green_top_left, green_bottom_right, (0, 250, 0), cv2.FILLED)

            state = next_state
        # Show the image
        cv2.imwrite('greedy_image_epsilon.png', self.image)
        cv2.imshow("Environment", self.image)
        # This line is necessary to give time for the image to be rendered on the screen
        cv2.waitKey(1)




