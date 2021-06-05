from colorPlot import *
import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib.ticker import FormatStrFormatter

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~ CHOOSE YOUR CASE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 1 = no wind, 2 = wind (all directions, moves the drone),
# 3 = wind (all directions, penalty only), 4 = wind (one direction, moves the drone)
# 5 = wind (one direction, penalty only)

case = 1

# Choose wind direction for windy cases. Value doesn't matter if not in 
# unidirectional case (right = 0, left = 1, up = 2, down = 3)
windDir = 0

# Choose the probablilities of wind affecting the drone
oneDirProb = 0.25 # out of 1, chance drone gets blown in direction in unidirectional case
# NOTE: Do not let allDirProb exceed 0.25
allDirProb = 0.1 # out of 1, chance drone gets blown in any of the four directions

# Cost for fighting wind at each step (cases 3 and 5 only) (default value: -2)
windCost = -2 # make sure this is negative

# Maximum timesteps for drone path (for learning algorithm) (suggested: 100)
# This can be decreased if the start and end points are closer together or if not converging
maxSteps = 100

# Number of Q-learning iterations (suggested: 50000 to start)
numIterations = 50000

# If you don't like the iteration printing when running, that can be commented out.
# That statement is located somewhere around Line 375 (it might've moved a bit).

# Pick start and end locations (optional)
pickStartEnd = False # True or False
start = [7,9]
end = [1,2]

# Q-learning parameters can be found in the Q-learning section of the code

### ENVIRONMENT ### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
n = 10 #city grid size
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
r1,r2,r3,r4 = 2,4,6,8 #rows
c1,c2,c3,c4 = 2,4,6,8 #columns      
City[r1,:] = 0
City[r2,:] = 0
City[r3,:] = 0
City[r4,:] = 0
City[:,c1] = 0
City[:,c2] = 0
City[:,c3] = 0
City[:,c4] = 0

# Plot city as heatmap
fig, ax = plt.subplots()
im = ax.imshow(City)
plt.colorbar(im,label='Building Height')
plt.ylim(-0.5,n-0.5)
plt.xlim(-0.5,n-0.5)
plt.yticks(np.arange(0,n,1))
plt.xticks(np.arange(0,n,1))
#plt.grid()

### CASE CONDITIONS ### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if case == 1:
    windy = False # wind flag
    oneDir = False # unidirectional wind flag
    penalty = False # penalty (instead of movement) flag
elif case == 2:
    windy = True
    oneDir = False
    penalty = False
elif case == 3:
    windy = True
    oneDir = False
    penalty = True
elif case == 4:
    windy = True
    oneDir = True
    penalty = False
elif case == 5:
    windy = True
    oneDir = True
    penalty = True

### ACTIONS ### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Forward = F, Backward = B, Right = R, Left = left, Up = U, Down = D 
#actions correspond to 0 through 25, respectively 

Actions = ['F','B','R','L','U','D','FR','FL','FU','FD','BR','BL','BU','BD',
            'RU','RD','LU','LD','FRU','FRD','FLU','FLD','BRU','BRD','BLU','BLD']
# Actions = ['F','B','R','L','U','D']

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
      or start_height >= 400 or start_height <= 50:
      start_row = np.random.randint(n)
      start_column = np.random.randint(n)
      start_height = City[start_row,start_column]
  while delivery_row in [r1,r2,r3,r4] or delivery_column in [c1,c2,c3,c4]\
      or delivery_height >= 400 or start_height <= 50:
      delivery_row = np.random.randint(n)
      delivery_column = np.random.randint(n)
      delivery_height = City[delivery_row,delivery_column]

  return start_row,start_column,start_height,\
         delivery_row,delivery_column,delivery_height
         
#compares drone height to building height, works as a terminal state function
def height_reward(row_index,column_index,height,time,path,penalty,blown):
    if penalty and blown:
        cost = windCost # cost of counteracting wind
    else:
        cost = 0
    if any((np.array([row_index,column_index,height]) == x).all() for x in path):
        return -100 + cost
    elif height == d_h and row_index == d_r and column_index == d_c: # delivery reward
        return 10000 + cost
    elif row_index == s_r and column_index == s_c and \
        height >= s_h and height <= 400 and time <= 5:
        return -1 + cost
    elif (row_index == d_r and column_index == d_c and \
        (height == 100 and height >= City[row_index,column_index])): # hovering over start or end location
        return 100 + cost
    elif (row_index == d_r and column_index == d_c and \
        (height == 50 and height >= City[row_index,column_index])): # hovering over start or end location
        return 100 + cost
    elif height >= 400 or height <= 100 or height <= City[row_index,column_index]:
        #rewards[row_index,column_index] = -100
        #return False
        return -100 + cost
    else:
        #rewards[row_index,column_index] = -1
        #return True
        return -1 + cost

#factors in effect of wind 
def wind(row_index,column_index,height,ep,windy,oneDir,penalty):
    # Outputs: [blown (yes/no), action]
    # right = 0, left = 1, up = 2, down = 3, no wind = 4
    randWind = random.uniform(0,1)
    if not windy: # case 1
        direction = 4
    elif windy and oneDir: # case 5
        if randWind <= oneDirProb:
            direction = windDir
        else:
            direction = 4
    else:
        if randWind <= allDirProb:
            direction = np.random.randint(4)
        # elif randWind < 2*allDirProb:
        #     direction = 1
        # elif randWind < 3*allDirProb:
        #     direction = 2
        # elif randWind < 4*allDirProb:
        #     direction = 3
        else:
            direction = 4
    if direction == 0: #wind coming from right
        if column_index == n-1:
            if penalty:
                return True, new_action(row_index,column_index,height,ep)
            else:
                return True, 3
        #adjacent building blocks wind 
        elif height < City[row_index,column_index+1] or column_index == 0:
            return False, new_action(row_index,column_index,height,ep)
        #wind blows drone in direction of wind
        else:
            if penalty:
                return True, new_action(row_index,column_index,height,ep)
            else:
                return True, 3 #'L'
    if direction == 1: #wind coming from left
        if column_index == 0:
            if penalty:
                return True, new_action(row_index,column_index,height,ep)
            else:
                return True, 2
        if height < City[row_index,column_index-1] or column_index == n-1:
            return False, new_action(row_index,column_index,height,ep)
        else:
            if penalty:
                return True, new_action(row_index,column_index,height,ep)
            else:
                return True, 2 #'R'
    if direction == 2: #wind coming from above/up
        if row_index == n-1:
            if penalty:
                return True, new_action(row_index,column_index,height,ep)
            else:
                return True, 1 #'D'
        elif height < City[row_index+1,column_index] or row_index == 0:
            return False, new_action(row_index,column_index,height,ep)
        #wind blows drone in direction of wind
        else:
            if penalty:
                return True, new_action(row_index,column_index,height,ep)
            else:
                return True, 1 #'B'
    if direction == 3: #wind coming from below/down
        if row_index == 0:
            if penalty:
                return True, new_action(row_index,column_index,height,ep)
            else:
                return True, 0
        if height < City[row_index-1,column_index] or row_index == n-1:
            return False, new_action(row_index,column_index,height,ep)
        else:
            if penalty:
                return True, new_action(row_index,column_index,height,ep)
            else:
                return True, 0 #'F'
    if direction == 4: #no wind
        return False, new_action(row_index,column_index,height,ep)

#next action using epsilon greedy
def new_action(row_index,column_index,height_index,ep):
  if np.random.random() < ep:
    return np.random.choice(np.where(Q==np.max(Q[row_index,column_index,height_index]))[3])
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
  
def terminalState(row_index,column_index,height,time,path,penalty,blown):
    if height_reward(row_index,column_index,height,time,path,penalty,blown)== -1 or \
        height_reward(row_index,column_index,height,time,path,penalty,blown)== (-1 + windCost) or \
        height_reward(row_index,column_index,height,time,path,penalty,blown)== 100 or \
        height_reward(row_index,column_index,height,time,path,penalty,blown)== (100 + windCost):
        return False
    else:
        return True
    
def removearray(L,arr):
    ind = 0
    size = len(L)
    while ind != size and not np.array_equal(L[ind],arr):
        ind += 1
    if ind != size:
        L.pop(ind)
    # else:
    #     raise ValueError('array not found in list.')

#Plotting start and end points     
#getting start and delivery locations
s_r,s_c,s_h,d_r,d_c,d_h = start_delivery_locations()
 
# For testing:
if pickStartEnd:
    s_r,s_c,s_h = start[0],start[1],City[start[0],start[1]]
    d_r,d_c,d_h = end[0],end[1],City[end[0],end[1]]

plt.scatter(s_r, s_c, s=20, c='red', marker='o',label='Start')
plt.scatter(d_r, d_c, s=50, c='red', marker='x',label='End')
plt.legend(); plt.show()
#%%

### REWARDS ### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#rewards for each state
# rewards = -np.ones((n,n)) #set penalty to -1 by default
# rewards[d_r, d_c] = 100 #successful delivery reward

# rewards = np.flip(rewards.T,0)

### Q-LEARNING ### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
ep = 0.8 #epsilon - decrease for exploration  
gamma = 0.99 #discount factor 
alpha = 0.05 #learning rate
episodes = numIterations # number of Q-learning iterations
blown = False # initialize windblown value

for i in range(episodes):
    if i % 1000 == 0:
        print(i)
    row_index,column_index,height = s_r,s_c,s_h
    time = 0
    path = []
    while not terminalState(row_index,column_index,height,time,path,penalty,blown) and \
            time < maxSteps:
        path.append(np.array([row_index,column_index,height]))
        height_index = int(np.where(building_heights == height)[0])
        blown,action = wind(row_index,column_index,height_index,ep,\
                                    windy,oneDir,penalty) # action choice
        nextRow,nextCol,nextHeight = new_location(row_index,column_index,height,action)
        if blown == True and not penalty and\
            any((np.array([nextRow,nextCol,nextHeight]) == x).all() for x in path):
                # Remove value drone was blown to from list
                removearray(path,np.array([nextRow,nextCol,nextHeight]))
                # Remove last value from list (in case drone wants to return)
                removearray(path,np.array([row_index,column_index,height]))
        reward = height_reward(nextRow,nextCol,nextHeight,time,path,penalty,blown)
        oldQ = Q[row_index,column_index,height_index,action]
        nextHeight_index = int(np.where(building_heights == nextHeight)[0])
        TD = reward + gamma * max(Q[nextRow,nextCol,nextHeight_index,:]) - oldQ
        newQ = oldQ + alpha * TD
        Q[row_index,column_index,height_index,action] = newQ
        row_index,column_index,height = nextRow,nextCol,nextHeight
        time += 1

policy = np.zeros([n,n,11])
for i in range(n):
    for j in range(n):
        for k in range(11):
            policy[i,j,k] = np.argmax(Q[i,j,k,:])

#%% Plot City
plt.figure()
ax= plt.axes(projection='3d')
plotCity(ax,City)

#%% Run simulation
if windy and not penalty:
    # ax = plt.axes(projection='3d')
    for i in range(10): # try 10 paths
        path = []
        row_index,column_index,height = s_r,s_c,s_h
        time = 0
        while not terminalState(row_index,column_index,height,time,path,penalty,blown)\
                and time < maxSteps:
            path.append(np.array([row_index,column_index,height]))
            height_index = int(np.where(building_heights == height)[0])
            blown,action = wind(row_index,column_index,height_index,ep,\
                                        windy,oneDir,penalty) # action choice
            nextRow,nextCol,nextHeight = new_location(row_index,column_index,height,action)
            if blown == True and not penalty and\
                any((np.array([nextRow,nextCol,nextHeight]) == x).all() for x in path):
                    # Remove value drone was blown to from list
                    removearray(path,np.array([nextRow,nextCol,nextHeight]))
                    # Remove last value from list (in case drone wants to return)
                    removearray(path,np.array([row_index,column_index,height]))
            row_index,column_index,height = nextRow,nextCol,nextHeight
            time += 1
        path.append(np.array([row_index,column_index,height]))
        x = np.zeros(len(path))
        y = np.zeros(len(path))
        z = np.zeros(len(path))
        for i in range(len(path)):
            x[i] = path[i][0]
            y[i] = path[i][1]
            z[i] = path[i][2]
        ax.plot3D(x[:], y[:], z[:], 'red')
    ax.scatter3D(s_r,s_c,s_h,'o')
    ax.scatter3D(d_r,d_c,d_h,'o')
else:
    path = []
    row_index,column_index,height = s_r,s_c,s_h
    time = 0
    while not terminalState(row_index,column_index,height,time,path,penalty,blown)\
            and time < maxSteps:
        path.append(np.array([row_index,column_index,height]))
        height_index = int(np.where(building_heights == height)[0])
        action = new_action(row_index,column_index,height_index,1)
        nextRow,nextCol,nextHeight = new_location(row_index,column_index,height,action)
        row_index,column_index,height = nextRow,nextCol,nextHeight
        time += 1
    path.append(np.array([row_index,column_index,height]))
    x = np.zeros(len(path))
    y = np.zeros(len(path))
    z = np.zeros(len(path))
    for i in range(len(path)):
        x[i] = path[i][0]
        y[i] = path[i][1]
        z[i] = path[i][2]
    
    # ax = plt.axes(projection='3d')
    ax.plot3D(x, y, z, 'red')
    ax.scatter3D(s_r,s_c,s_h,'o')
    ax.scatter3D(d_r,d_c,d_h,'o')

    
#%% Policy Map
x = np.array(range(n))
y = np.array(range(n))
z = np.array(range(11))

fig = plt.figure()
ax = fig.add_axes([0.1, 0.1, 0.7, 0.8], projection='3d')

plotMatrix(ax, x, y, z, policy/13+1, cmap="jet")
ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
ax.set_zticks([z[0],z[2],z[4],z[6],z[8], z[10]])
ax.set_zticklabels(['0','100','200','300','400','500'])