from colorPlot import *
import numpy as np
import matplotlib.pyplot as plt
import random

### ENVIRONMENT ### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
n = 20 #city grid size
actions = 26 #number of actions 

City = np.zeros([n,n]) #initialize city
Q = np.zeros((n,n,11,actions)) #initialize Q with n x n x 11 environment and 26 actions

### CITY CREATION ### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
building_heights = np.arange(0,550,50) #array of possible building heights 
weights=[15,20,15,10,10,5,5,5,5,5,5] #probabilities of each building height

random.seed(0)
for i in range(n):
    for j in range(n):
        #City[i,j] = 50*random.randint(0,10)
        height = random.choices(building_heights,weights)
        City[i,j] = height[0]

#street locations with height 0   
r1,r2,r3,r4 = 3,7,12,16 #rows
c1,c2,c3,c4 = 3,7,12,16 #columns      
City[r1,:] = 0
City[r2,:] = 0
City[r3,:] = 0
City[r4,:] = 0
City[:,c1] = 0
City[:,c2] = 0
City[:,c3] = 0
City[:,c4] = 0

fig, ax = plt.subplots()
im = ax.imshow(City)
plt.colorbar(im)
plt.ylim(-0.5,19.5)
plt.xlim(-0.5,19.5)
plt.yticks(np.arange(0,20,1))
plt.xticks(np.arange(0,20,1))

#plt.grid()

### ACTIONS ### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Forward = F, Backward = B, Right = R, Left = left, Up = U, Down = D 
#actions correspond to 0 through 25, respectively 

Actions = ['F','B','R','L','U','D','FR','FL','FU','FD','BR','BL','BU','BD',
           'RU','RD','LU','LD','FRU','FRD','FLU','FLD','BRU','BRD','BLU','BLD']

### FUNCTIONS ### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#gives random start location and delivery location with the elevation
def start_delivery_locations():
  #get a random row and column index
  start_row = np.random.randint(n)
  start_column = np.random.randint(n)
  delivery_row = np.random.randint(n)
  delivery_column = np.random.randint(n)
  
  start_height = City[start_row,start_column]
  delivery_height = City[delivery_row,delivery_column]

  while start_row in [r1,r2,r3,r4] or start_column in [c1,c2,c3,c4]\
      or start_height >= 400:
      start_row = np.random.randint(n)
      start_column = np.random.randint(n)
      start_height = City[start_row,start_column]
  while delivery_row in [r1,r2,r3,r4] or delivery_column in [c1,c2,c3,c4]\
      or delivery_height >= 400:
      delivery_row = np.random.randint(n)
      delivery_column = np.random.randint(n)
      delivery_height = City[delivery_row,delivery_column]

  return start_row,start_column,start_height,\
         delivery_row,delivery_column,delivery_height
         
#compares drone height to building height, works as a terminal state function
def height_reward(row_index,column_index,height,time):
    if height == d_h and row_index == d_r and column_index == d_c: # delivery reward
        return 1000
    elif row_index == s_r and column_index == s_c and \
        height >= s_h and height <= 400 and time <= 5:
        return -1
    elif (row_index == d_r and column_index == d_c and \
        (height == 100 and height >= City[row_index,column_index])): # hovering over start or end location
        return 1000
    elif height >= 400 or height <= 100 or height <= City[row_index,column_index]:
        #rewards[row_index,column_index] = -100
        #return False
        return -100
    else:
        #rewards[row_index,column_index] = -1
        #return True
        return -1

#factors in effect of wind 
def wind(row_index,column_index,height,ep):
    #right = 0, left = 1, up = 2, down = 3, no wind = 4
    direction = np.random.randint(5)
    if direction == 0: #wind coming from right
        #wind blows drone into building
        if height_reward(row_index,column_index-1,height) == -100 \
            or column_index == 0:
            return False,3 #'L'
        #adjacent building blocks wind 
        elif height < City[row_index,column_index+1]:
            return True, new_action(row_index,column_index,height,ep)
        #wind blows drone in direction of wind without hitting building
        else:
            return True, 3 #'L'
    if direction == 1: #wind coming from left
        if height_reward(row_index,column_index+1,height) == -100 \
            or column_index == 19:
            return False, 2 #'R'
        elif height < City[row_index,column_index-1]:
            return True, new_action(row_index,column_index,height,ep)
        else:
            return True, 2 #'R'
    if direction == 2: #wind coming from above/up
        if height_reward(row_index-1,column_index,height) == -100 \
            or row_index == 0:
            return False, 5 #'D'
        elif height < City[row_index+1,column_index]:
            return True, new_action(row_index,column_index,height,ep)
        else:
            return True, 5 #'D'
    if direction == 3: #wind coming from below/down
        if height_reward(row_index+1,column_index,height) == -100 \
            or row_index == 19:
            return False, 4 #'U'
        elif height < City[row_index-1,column_index]:
            return True, new_action(row_index,column_index,height,ep)
        else:
            return True, 4 #'U'
    if direction == 4: #no wind
        return True, new_action(row_index,column_index,height,ep)

#next action using epsilon greedy
def new_action(row_index,column_index,height_index,ep):
  if np.random.random() < ep:
    return np.argmax(Q[row_index,column_index,height_index]) #standard Q-learning
  else: 
    return np.random.randint(actions) #exploration 

#next location based on action 
def new_location(row_index,column_index,height,action):
  new_row_index = row_index
  new_column_index = column_index
  new_height = height 
  
  if Actions[action] == 'F' and new_row_index < n-1:
      new_row_index += 1
  elif Actions[action] == 'B' and new_row_index > 0:
      new_row_index -= 1
  elif Actions[action] == 'R' and new_column_index < n-1:
      new_column_index += 1
  elif Actions[action] == 'L' and new_column_index > 0:
      new_column_index -= 1
  elif Actions[action] == 'U':
      new_height += 50 
  elif Actions[action] == 'D':
      new_height -= 50
  elif Actions[action] == 'FR' and new_row_index < n-1 and new_column_index < n-1:
      new_row_index += 1
      new_column_index += 1
  elif Actions[action] == 'FL' and new_row_index < n-1 and new_column_index > 0:
      new_row_index += 1
      new_column_index -= 1
  elif Actions[action] == 'FU' and new_row_index < n-1:
      new_row_index += 1
      new_height += 50 
  elif Actions[action] == 'FD' and new_row_index < n-1:
      new_row_index += 1
      new_height -= 50
  elif Actions[action] == 'BR' and new_row_index > 0 and new_column_index < n-1:
      new_row_index -= 1
      new_column_index += 1
  elif Actions[action] == 'BL' and new_row_index > 0 and new_column_index > 0:
      new_row_index -= 1
      new_column_index -= 1
  elif Actions[action] == 'BU' and new_row_index > 0:
      new_row_index -= 1
      new_height += 50
  elif Actions[action] == 'BD' and new_row_index > 0:
      new_row_index -= 1
      new_height -= 50
  elif Actions[action] == 'RU' and new_column_index < n-1:
      new_column_index += 1
      new_height += 50
  elif Actions[action] == 'RD' and new_column_index < n-1:
      new_column_index += 1
      new_height -= 50
  elif Actions[action] == 'LU' and new_column_index > 0:
      new_column_index -= 1
      new_height += 50
  elif Actions[action] == 'LD' and new_column_index > 0:
      new_column_index -= 1
      new_height -= 50
  elif Actions[action] == 'FRU' and new_row_index < n-1 and new_column_index < n-1:
      new_row_index += 1
      new_column_index += 1
      new_height += 50
  elif Actions[action] == 'FRD' and new_row_index < n-1 and new_column_index < n-1:
      new_row_index += 1
      new_column_index += 1
      new_height -= 50
  elif Actions[action] == 'FLU' and new_row_index < n-1 and new_column_index > 0:
      new_row_index += 1
      new_column_index -= 1
      new_height += 50
  elif Actions[action] == 'FLD' and new_row_index < n-1 and new_column_index > 0:
      new_row_index += 1
      new_column_index -= 1
      new_height -= 50
  elif Actions[action] == 'BRU' and new_row_index > 0 and new_column_index < n-1:
      new_row_index -= 1
      new_column_index += 1
      new_height += 50
  elif Actions[action] == 'BRD' and new_row_index > 0 and new_column_index < n-1:
      new_row_index -= 1
      new_column_index += 1
      new_height -= 50
  elif Actions[action] == 'BLU' and new_row_index > 0 and new_column_index > 0:
      new_row_index -= 1
      new_column_index -= 1
      new_height += 50
  elif Actions[action] == 'BLD' and new_row_index > 0 and new_column_index > 0:
      new_row_index -= 1
      new_column_index -= 1
      new_height -= 50
  return new_row_index, new_column_index, new_height
  
def terminalState(row_index,column_index,height,time):
    if height_reward(row_index,column_index,height,time) == -1:
        return False
    else:
        return True

#Plotting start and end points     
#getting start and delivery locations
s_r,s_c,s_h,d_r,d_c,d_h = start_delivery_locations()
 
# For testing:
s_r,s_c,s_h = 10,11,City[10,11]
d_r,d_c,d_h = 9,13,City[9,13]

plt.scatter(s_c, s_r, s=20, c='red', marker='o')
plt.scatter(d_c, d_r, s=50, c='red', marker='x')
plt.show()
#%%

### REWARDS ### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#rewards for each state
rewards = -np.ones((n,n)) #set penalty to -1 by default
rewards[d_r, d_c] = 100 #successful delivery reward

rewards = np.flip(rewards.T,0)

### Q-LEARNING ### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
ep = 0.8 #epsilon - decrease for exploration  
gamma = 0.95 #discount factor 
alpha = 0.05 #learning rate
episodes = 50000

for i in range(episodes):
    print(i)
    row_index,column_index,height = s_r,s_c,s_h
    time = 0
    while not terminalState(row_index,column_index,height,time) and time < 100:
        height_index = int(np.where(building_heights == height)[0])
        action = new_action(row_index,column_index,height_index,ep)
        nextRow,nextCol,nextHeight = new_location(row_index,column_index,height,action)
        reward = height_reward(nextRow,nextCol,nextHeight,time)
        oldQ = Q[row_index,column_index,height_index,action]
        TD = reward + gamma * max(Q[row_index,column_index,height_index,:]) - oldQ
        newQ = oldQ + alpha * TD
        Q[row_index,column_index,height_index,action] = newQ
        row_index,column_index,height = nextRow,nextCol,nextHeight
        time += 1

policy = np.zeros([n,n,11])
for i in range(n):
    for j in range(n):
        for k in range(11):
            policy[i,j,k] = np.argmax(Q[i,j,k,:])

#%% Run simulation
path = []
path.append(np.array([s_r,s_c,s_h]))
row_index,column_index,height = s_r,s_c,s_h
time = 0
if np.max(Q) != 0:
    while not terminalState(row_index,column_index,height,time) and time < 1000:
        height_index = int(np.where(building_heights == height)[0])
        action = new_action(row_index,column_index,height_index,1)
        nextRow,nextCol,nextHeight = new_location(row_index,column_index,height,action)
        path.append([nextRow,nextCol,nextHeight])
        row_index,column_index,height = nextRow,nextCol,nextHeight
        time += 1

#%%
# x = np.array(range(n))
# y = np.array(range(n))
# z = np.array(range(11))

# fig = plt.figure(figsize=(10,4))
# ax = fig.add_axes([0.1, 0.1, 0.7, 0.8], projection='3d')

# plotMatrix(ax, x, y, z, policy/13+1, cmap="jet")
