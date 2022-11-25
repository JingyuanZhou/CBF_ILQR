import numpy as np 
import math
import pdb

class Obstacle:
    def __init__(self, args, track_id, bb):
        self.args = args
        self.car_length = bb[0]
        self.car_width = bb[1]
        self.track_id = track_id

    def get_obstacle_cost_derivatives(self, npc_traj, i, ego_state):

        a = self.car_length + np.abs(npc_traj[2, i]*math.cos(npc_traj[3, i]))*self.args.t_safe + self.args.s_safe_a + self.args.ego_rad
        b = self.car_width + np.abs(npc_traj[2, i]*math.sin(npc_traj[3, i]))*self.args.t_safe + self.args.s_safe_b + self.args.ego_rad
        
        P1 = np.diag([1/a**2, 1/b**2, 0, 0])

        theta = npc_traj[3, i]
        theta_ego = ego_state[3]

        transformation_matrix = np.array([[ math.cos(theta), math.sin(theta), 0, 0],
                                          [-math.sin(theta), math.cos(theta), 0, 0],
                                          [               0,               0, 0, 0],
                                          [               0,               0, 0, 0]])
        
        ego_front = ego_state + np.array([math.cos(theta_ego)*self.args.ego_lf, math.sin(theta_ego)*self.args.ego_lf, 0, 0])
        diff = (transformation_matrix @ (ego_front - npc_traj[:, i])).reshape(-1, 1) # (x- xo)
        c = 1 - diff.T @ P1 @ diff # Transform into a constraint function
        c_dot = -2 * P1 @ diff
        b_f, b_dot_f, b_ddot_f = self.barrier_function(self.args.q1_front, self.args.q2_front, c, c_dot)

        ego_rear = ego_state - np.array([math.cos(theta_ego)*self.args.ego_lr, math.sin(theta_ego)*self.args.ego_lr, 0, 0])
        diff = (transformation_matrix @ (ego_rear - npc_traj[:, i])).reshape(-1, 1)
        c = 1 - diff.T @ P1 @ diff
        c_dot = -2 * P1 @ diff
        b_r, b_dot_r, b_ddot_r = self.barrier_function(self.args.q1_rear, self.args.q2_rear, c, c_dot)

        return b_dot_f + b_dot_r, b_ddot_f + b_ddot_r

    def get_obstacle_cost(self, npc_traj, i, ego_state_nominal, ego_state):
        a = self.car_length + np.abs(npc_traj[2, i]*math.cos(npc_traj[3, i]))*self.args.t_safe + self.args.s_safe_a + self.args.ego_rad
        b = self.car_width + np.abs(npc_traj[2, i]*math.sin(npc_traj[3, i]))*self.args.t_safe + self.args.s_safe_b + self.args.ego_rad
        
        P1 = np.diag([1/a**2, 1/b**2, 0, 0])

        theta = npc_traj[3, i]
        theta_ego = ego_state[3]
        theta_ego_nominal = ego_state_nominal[3]


        transformation_matrix = np.array([[ math.cos(theta), math.sin(theta), 0, 0],
                                          [-math.sin(theta), math.cos(theta), 0, 0],
                                          [               0,               0, 0, 0],
                                          [               0,               0, 0, 0]])
        
        # front circle
        ego_front_nominal = ego_state_nominal + np.array([math.cos(theta_ego)*self.args.ego_lf, math.sin(theta_ego)*self.args.ego_lf, 0, 0])
        ego_front = ego_state + np.array([math.cos(theta_ego_nominal)*self.args.ego_lf, math.sin(theta_ego_nominal)*self.args.ego_lf, 0, 0])

        x_del = ego_front - ego_front_nominal

        diff = (transformation_matrix @ (ego_front_nominal - npc_traj[:, i])).reshape(-1, 1)
        c = 1 - diff.T @ P1 @ diff
        c_dot = -2 * P1 @ diff
        b_f, b_dot_f, b_ddot_f = self.barrier_function(self.args.q1_front, self.args.q2_front, c, c_dot)

        cost = b_f + x_del.T @ b_dot_f + x_del.T @ b_ddot_f @ x_del  

        # rear circle
        ego_rear_nominal = ego_state_nominal - np.array([math.cos(theta_ego)*self.args.ego_lr, math.sin(theta_ego)*self.args.ego_lr, 0, 0])
        ego_rear = ego_state - np.array([math.cos(theta_ego_nominal)*self.args.ego_lr, math.sin(theta_ego_nominal)*self.args.ego_lr, 0, 0])

        x_del = ego_rear - ego_rear_nominal

        diff = (transformation_matrix @ (ego_rear_normalized - npc_traj[:, i])).reshape(-1, 1)
        c = 1 - diff.T @ P1 @ diff
        c_dot = -2 * P1 @ diff
        b_r, b_dot_r, b_ddot_r = self.barrier_function(self.args.q1_rear, self.args.q2_rear, c, c_dot)

        cost += b_r + x_del.T @ b_dot_r + x_del.T @ b_ddot_r @ x_del  

        return cost

    def barrier_function(self, q1, q2, c, c_dot):
        b = q1*np.exp(q2*c)
        b_dot = q1*q2*np.exp(q2*c)*c_dot
        b_ddot = q1*(q2**2)*np.exp(q2*c)*np.matmul(c_dot, c_dot.T)

        return b, b_dot, b_ddot

    def CBF_filter(self, state, npc_traj, i):
        pos_x = state[0]
        pos_y = state[1]
        v = state[2]
        theta = state[3]
        v_x = v*np.cos(theta)
        v_y = v*np.sin(theta)

        a = self.car_length + np.abs(npc_traj[2, i]*math.cos(npc_traj[3, i]))*self.args.t_safe + self.args.s_safe_a + self.args.ego_rad
        b = self.car_width + np.abs(npc_traj[2, i]*math.sin(npc_traj[3, i]))*self.args.t_safe + self.args.s_safe_b + self.args.ego_rad

        state_obs = npc_traj[:,i]
        obs_pos_x = state_obs[0]
        obs_pos_y = state_obs[1]
        obs_v = state_obs[2]
        obs_theta = state_obs[3]

        k = np.array([0.4,0.8])

        coef_1 = 2*(np.cos(obs_theta)**2/(a**2)+np.sin(obs_theta**2)/(b**2))
        coef_2 = 2*(np.sin(obs_theta)**2/(a**2)+np.cos(obs_theta**2)/(b**2))
        coef_3 = 2*np.cos(obs_theta)*np.sin(obs_theta)*(1/(a**2)-1/(b**2))

        delta_h2 = np.array([coef_1*v_x+coef_3*v_y+k[0]*(coef_1*(pos_x-obs_pos_x)+coef_3*(pos_y-obs_pos_y)),
							coef_2*v_y+coef_3*v_x+k[0]*(coef_2*(pos_y-obs_pos_y)+coef_3*(pos_x-obs_pos_x)),
							coef_1*(pos_x-obs_pos_x)+coef_3*(pos_y-obs_pos_y),
							coef_2*(pos_y-obs_pos_y)+coef_3*(pos_x-obs_pos_x)])

        f = np.array([[v_x],[v_y],[0],[0]])
        g = np.array([[0,0],[0,0],[-(v**2)*np.sin(theta),np.cos(theta)],[(v**2)*np.sin(theta),np.cos(theta)]])

        Lfh2 = np.dot(delta_h2.reshape(1,4),f)
        Lgh2 = np.dot(delta_h2.reshape(1,4),g)
        h1 = 0.5*coef_1*((pos_x-obs_pos_x)**2)+coef_3*(pos_x-obs_pos_x)*(pos_y-obs_pos_y)+0.5*coef_2*((pos_y-obs_pos_y)**2)-1
        h2 = coef_1*(pos_x-obs_pos_x)*v_x + coef_3*(pos_y-obs_pos_y)*v_x+coef_3*(pos_x-obs_pos_x)*v_y+coef_1*(pos_y-obs_pos_y)*v_y+k[0]*h1

        return Lgh2,-Lfh2-k[1]*h2