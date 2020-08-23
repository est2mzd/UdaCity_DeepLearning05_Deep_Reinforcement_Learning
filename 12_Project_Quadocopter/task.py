import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self,
                 init_pose = np.array([0.0,0.0,10.0,0.0,0.0,0.0]),
                 init_velocities = np.array([0.0,0.0,0.1]),
                 init_angle_velocities = np.array([0.0,0.0,0.0]),
                 runtime=5.,
                 target_pos=np.array([0.0,0.0,50.0])):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3
        self.state_size    = self.action_repeat * 6
        
        self.action_low  = 10
        self.action_high = 900
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.])

        # to calc reward
        self.pos_diff_init = None

    def get_reward(self, done):
        """Uses current pose of sim to return reward."""
        reward = 0
        self.calc_pos_diff_ratio()
        reward = self.calc_base_reward_2(reward)

        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward   = 0
        pose_all = []
        #
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            # done = self.check_episode_end(done)
            reward += self.get_reward(done)
            pose_all.append(self.sim.pose)
            #
            if done:
                missing = self.action_repeat - len(pose_all)
                pose_all.extend([pose_all[-1]] * missing)
                break
        #
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state

    #============================================================================#
    def calc_pos_diff_ratio(self):
        if self.pos_diff_init is None:
            self.pos_diff_init = self.sim.init_pose[:3] - self.target_pos
            self.pos_diff_init = (sum(self.pos_diff_init ** 2)) ** 0.5
        pos_diff = self.sim.pose[:3] - self.target_pos
        pos_diff = (sum(pos_diff ** 2)) ** 0.5
        # Normalized distance
        self.pos_diff_ratio = pos_diff / (self.pos_diff_init + 0.001)

    def calc_base_reward_2(self, reward):
        # reward += 1 - self.pos_diff_ratio * 1.0
        reward += 1 - self.pos_diff_ratio * (1.0 / 50.0)
        return reward

    def calc_episode_end_reward(self, reward, done):
        if done and self.check_near_goal():
            reward += 100
        elif done and self.sim.pose[2] <= 0.:
            reward -= 50
        elif done and self.check_out_of_range():
            reward -= 30
        elif done and not self.check_out_of_range() and self.sim.runtime > self.sim.time:
            reward -= 20

    def check_episode_end(self, done):
        if self.check_near_goal():
            done = True
        if self.check_near_goal():
            done = True
        return done

    def check_near_goal(self):
        return np.abs(self.target_pos[2] - self.sim.pose[2]) < 1.0

    def check_out_of_range(self):
        return self.pos_diff_ratio > 2.0
