import numpy as  np
import math
class MountainCar(object):
	def __init__(self, start, goal, Xrange, Vrange):
		self.start = start
		self.goal = goal
		self.Xrange = Xrange
		self.Vrange = Vrange
		self.num_actions = 3

	def _DoAction(self, action):
		#MountainCarDoAction: executes the action (a) into the mountain car
		# a: is the force to be applied to the car
		# x: is the vector containning the position and speed of the car
		# xp: is the vector containing the new position and velocity of the car

		position = self.state[0]
		speed    = self.state[1]

		# bounds for position
		bpleft=self.Xrange[0] 

		# bounds for speed
		bsleft=self.Vrange[0]
		bsright=self.Vrange[1]

		speedt1= speed + (0.001*action) + (-0.0025 * math.cos( 3.0*position) )	 
		speedt1= speedt1 * 0.999 # thermodynamic law, for a more real system with friction.

		if speedt1<bsleft: 
			speedt1=bsleft 
		elif speedt1>bsright:
			speedt1=bsright    

		post1 = position + speedt1 

		if post1<=bpleft:
			post1=bpleft
			speedt1=0.0

		xp = np.array([post1,speedt1])
		self.state = xp

	def _GetReward(self):
		# MountainCarGetReward returns the reward at the current state
		# x: a vector of position and velocity of the car
		# r: the returned reward.
		# f: true if the car reached the goal, otherwise f is false

		position = self.state[0]
		# bound for position; the goal is to reach position = 0.45
		bpright  = self.goal

		r = -1
		f = False

		if  position >= bpright:
			r = 100
			f = True

		return r, f

	def act(self, action):
		self._DoAction(action)
		reward, game_over = self._GetReward()
		return self.observe(), reward, game_over

	def observe(self):
		return self.state.reshape((1,-1))

	def reset(self):
		self.state = np.array([self.start[0], self.start[1]])

    # def BuildActionList(self):
    #     return np.array([-1.0 , 0.0 , 1.0])
    
    # def BuildSateList(self):    
    #     # state discretization for the mountain car problem
    #     xdiv  = (0.55-(-1.5))   / 10.0
    #     xpdiv = (0.07-(-0.07)) / 5.0
        
    #     x = np.arange(-1.5,0.5+xdiv,xdiv)
    #     xp= np.arange(-0.07,0.07+xpdiv,xpdiv)

    #     # N=size(x)
    #     N = x.size
    #     # M=size(xp)
    #     M = xp.size

    #     states=[] #zeros((N*M,2)).astype(Float32)
    #     index=0
    #     for i in range(N):    
    #         for j in range(M):
    #             states.append([x[i], xp[j]])
                
    #     return np.array(states)

    # def BuildQTable(self):
    #     nstates     = self.statelist.shape[0]
    #     nactions    = self.actionlist.shape[0]
    #     if self.tQ == 0:
    #         Q = [[0.0 for i in range(nactions)] for i in range(nstates)]
    #         # print(len(Q), len(Q[0]))
    #     else:
    #         Q = deepcopy(self.tQ)
    #     return Q

    # def retQTable(self):
    #     return self.Q

    # def GetInitialState(self):
    #     initial_position = self.start[0]
    #     initial_speed    =  self.start[1]
    #     return  np.array([initial_position,initial_speed])