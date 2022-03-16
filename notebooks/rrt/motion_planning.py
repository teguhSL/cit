import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output
import networkx as nx
from pb_utils import set_q
import pybullet as p
import time

class RRT():
    def __init__(self, D, sampler, col_checker, interpolator):
        self.D = D
        self.sampler = sampler
        self.col_checker = col_checker
        self.interpolator = interpolator
        self.samples = [] 
        
    def check_collision(self, sample):
        return self.col_checker.check_collision(sample)
    
    def sample(self, get_valid = True):
        status = True
        if get_valid:
            while status is True:
                sample =  self.sampler.sample()
                status = self.check_collision(sample.flatten())
        else:
                sample =  self.sampler.sample()            
        return sample
    
    def interpolate(self, sample1, sample2, N):
        return self.interpolator.interpolate(sample1, sample2, N)
               
    def extend(self, cur_index, sample1, sample2, step_length = 0.1):
        #state1, state2 = self.correct_manifold(sample1, sample2)
        state1, state2 = sample1.copy(), sample2.copy()
        dist = np.linalg.norm(state2-state1)
        N = int(dist/step_length) + 2
        state_list = self.interpolate(state1, state2, N)
        next_states = []
        for state in state_list:
            if self.check_collision(state):
                #print('collision')
                #print(state)
                break
            next_states += [(state)]
            
        #add the nodes to the graph
        for next_state in next_states[1:]:
            next_index = self.G.number_of_nodes()
            self.G.add_nodes_from([next_index])
            self.G.add_edge(cur_index, next_index)
            self.samples += [next_state]
            cur_index = next_index
            
        return next_states

    def find_nearest(self, sample1, sample_set, n = 1):
        #find a nearest node
        if len(sample_set) > 1:
            index = np.argpartition(np.linalg.norm(sample1-sample_set,axis=1),n)[0:n]
            return index,sample_set[index]
        else:
            return [0], sample_set[0]  
    
    def init_plan(self, init_state, goal_state):
        self.G = nx.Graph()
        self.nodes = [0]
        self.edges = []
        self.samples = [init_state]
        self.G.add_node(0)
        self.init_state = init_state.copy()
        self.goal_state = goal_state.copy()
    
    def step_plan(self):
        success = False
        #sample random state
        self.random_sample = self.sample()

        #find a nearest node
        nearest_index, nearest_sample = self.find_nearest(self.random_sample, np.array(self.samples))

        #extend to the random state
        self.next_states = self.extend(nearest_index[0], nearest_sample.flatten(), self.random_sample.flatten() )
        
        #check distance with goal state
        nearest_index, nearest_sample = self.find_nearest(self.goal_state, np.array(self.samples))        
        clear_output()
        print('Trying to reach a random state...')
        print(nearest_sample)

        #extend to the goal state
        
        self.next_states_goal = self.extend(nearest_index[0], nearest_sample.flatten(), self.goal_state.flatten())
        #print(next_states)
        if len(self.next_states_goal) == 0: return False
        clear_output()
        print('Trying to reach the goal state...')
        print(self.next_states_goal[-2:])
        print(self.goal_state)
        
        if np.linalg.norm(self.next_states_goal[-1] - self.goal_state)< 0.001:
            print('Solution is found!')
            success = True
        return success,0
        
    def plan(self, init_state, goal_state):
        self.init_plan(init_state, goal_state)
        success = False
        self.num_plan = 0
        self.nfevs = 0
        while success is False:
            success, nfev = self.step_plan()
            self.nfevs += nfev
            self.num_plan += 1
            print('Planning...')
        #clear_output()
        #find the path
        path = nx.dijkstra_path(self.G, 0, self.G.number_of_nodes()-1)
        
        #Get the traj
        traj = []
        for i in path:
            traj.append(self.samples[i])
        traj = np.array(traj)
        return traj
                
    def shortcut_path(self, path_in, step_length=0.1):
        #simple shortcutting algo trying to iteratively remove nodes from the path
        path = path_in.copy()
        while len(path) > 2:
            idx_remove = -1
            for idx in range(len(path)-2):
                #try to remove node(idx+1) from path: Is interpolation between node(idx) and node(idx+2) free?
                node1 = self.samples[path[idx]]
                node2 = self.samples[path[idx+2]]
                dist = np.linalg.norm(node2-node1)
                N = int(dist/step_length) + 2
                state_list = self.interpolate(node1, node2, N)
                free = True
                for state in state_list:
                    if self.check_collision(state):
                        free = False
                        break
                        
                if free is True:
                    idx_remove = idx+1
                    break
                
            if idx_remove == -1:
                break
            del path[idx_remove]
        return path

class sampler():
    """
    General sampler, given the joint limits
    """

    def __init__(self, joint_limits=None):
        self.dof = joint_limits.shape[1]
        self.joint_limits = joint_limits

    def sample(self, N=1):
        samples = np.random.rand(N, self.dof)
        samples = self.joint_limits[0] + samples * (self.joint_limits[1] - self.joint_limits[0])
        return samples


def lin_interpolate(state1, state2, n=1.):
    state_list = []
    for i in range(n+1):
        state_list.append(state1 + 1.*i*(state2-state1)/n)
    return state_list

class interpolator():
    def __init__(self):
        pass

    def interpolate(self, state1, state2, N):
        states = lin_interpolate(state1, state2, N)
        return states

def check_collision(robot_id, object_ids, omit_indices=[-1]):
    col_info = []
    is_col = False
    for object_id in object_ids:
        ress = p.getClosestPoints(robot_id, object_id, distance=2)
        for res in ress:
            if res[3] in omit_indices:
                continue
            if res[8] < 0:
                is_col = True
                col_info += [(res[3], res[4], res[7], res[8])]  # linkA, linkB, normal, distance
    return is_col, col_info

class col_checker():
    def __init__(self, robot_id, joint_indices, object_ids, omit_indices=[-1], floating_base = False):
        self.robot_id = robot_id
        self.joint_indices = joint_indices
        self.object_ids = object_ids
        self.omit_indices = omit_indices
        self.floating_base = floating_base

    def check_collision(self, q):
        if self.floating_base:
            set_q(q, self.robot_id, self.joint_indices, True)
        else:
            set_q(q, self.robot_id, self.joint_indices)
        return check_collision(self.robot_id, self.object_ids, omit_indices=self.omit_indices)[0]
