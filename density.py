from qutip import *
import numpy as np
from math import *
from scipy import *
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import pyplot
import itertools
from matplotlib import cm


#prints full matrix instead of partial parts
np.set_printoptions(threshold=np.inf)
#bloch sphere diagram, not bloch3d because of vtk version clash with mayavi
c=Bloch()

#|0> and |1> in 2D Hilbert space
a = basis (2,0)
b = basis(2,1)       

#logical zero after application of 16 generators pf [[5,1,3]] stabilizer code
zero = 1/4 * (tensor(a,a,a,a,a) + tensor(b,a,a,b,a) + tensor(a,b,a,a,b) + tensor(b,a,b,a,a) + tensor(a,b,a,b,a)  + tensor(a,a,b,a,b) - (tensor(b,b,b,b,a) + tensor(a,b,b,b,b) + tensor(b,a,b,b,b) + tensor(b,b,a,b,b) + tensor(b,b,b,a,b)  )- ( tensor(a,b,b,a,a) + tensor(a,a,b,b,a) + tensor(a,a,a,b,b) + tensor(b,a,a,a,b) + tensor(b,b,a,a,a) ))

#-------------square root of 1/16 to normalize-------------#
#logical one
one = 1/4 *(tensor(b,b,b,b,b) + tensor(a,b,b,a,b) + tensor(b,a,b,b,a) + tensor(a,b,a,b,b) + tensor(b,a,b,a,b)  + tensor(b,b,a,b,a) - (tensor(a,a,a,a,b) + tensor(b,a,a,a,a) + tensor(a,b,a,a,a) + tensor(a,a,b,a,a) + tensor(a,a,a,b,a)  )- ( tensor(b,a,a,b,b) + tensor(b,b,a,a,b) + tensor(b,b,b,a,a) + tensor(a,b,b,b,a) + tensor(a,a,b,b,b) ))

projector = (zero*zero.dag() +  one*one.dag()) #not normalized, trace is 32, thus divided by 32

''' DISTILLATION'''

def distill(list):
    #-------------ARRAY---------------------#
    x = list[0]
    y = list[1]
    z = list[2]
     #-----------ENCODE---------------------#
    rho  =  0.5*(qeye(2) + x*sigmax() + y*sigmay() + z*sigmaz() )
    rho_five = tensor(rho,rho,rho,rho,rho)
    encoder = projector*rho_five*projector #step that needs to be normalized
    encoder = encoder*(1/encoder.tr())

    #------------DECODE-------------------#
    alpha1 = np.array((zero.dag()*encoder*zero).full())
    beta1  = np.array((one.dag()*encoder*one).full())
    gamma1 = np.array((zero.dag()*encoder*one).full())
    delta1 = np.array((one.dag()*encoder*zero).full())

    alpha = alpha1[0][0]       #------COEFFICIENTS----------------------------------#
    beta  = beta1[0][0]        #------OF RHO'---------------------------------------#      
    gamma = gamma1[0][0]       #------INDEXING TWICE TO GET THE VALUES--------------#
    delta = delta1[0][0]

    rho_p = Qobj([[alpha, delta],[gamma, beta]]) #-------CONVERTING INTO 2X2 MATRIX WHICH IS RHO'----# 
                                                 #-------AND THEN INTO QUTIP OBJECT------------------#

    x_p = (rho_p*sigmax()).tr()      #----------SOLVING FOR X',Y',Z'---------------#
    y_p = (rho_p*sigmay()).tr()      #-----------------IN RHO'---------------------#
    z_p = (rho_p*sigmaz()).tr()

    return [x_p,y_p,z_p]            #---------RETURNING A DISTILLED ARRAY----------#
 

''' PATH CREATION ''' 
#-----------------FUNCTION TO MAKE THE PATH OF DISTILLATION FOR EACH POINT--------------------------#
def makepath(init,iters):#---------INIT IS INITIAL POINT WHILE ITERS IS NUMBER OF ITERATIONS------#
    point = init
    path = [point]
    for i in range(iters):
        point = distill(point)
        path  = np.append(path, [point], axis=0)
    return path

''' NUMBER OF STEPS TO DISTILL TO A CERTAIN PURITY '''
#function that calculates distance to T state
def distancetoT(point):
    return sqrt((1/sqrt(3)-point[0])**2 +(1/sqrt(3)-point[1])**2+ (1/sqrt(3)-point[2])**2)

#print (distancetoT([(1/sqrt(3)),(1/sqrt(3)),(1/sqrt(3))]))

#function that calculates number of iterations required to reach a certain purity = dist
def pathlength(point,dist):
    steps = 0
    x = point[0]
    y = point[1]
    z = point[2]
    d = distancetoT(point)

    while d > dist and steps < 20:
        point = distill(point)
        d = distancetoT(point)
        steps += 1
    return steps

#print (pathlength([0.5,0.5,0.1],0.0001))     

''' COMPARING WITH RALL'S FRACTAL PLOT '''   
#-----------CREATE A GRID OF POINTS ON X AND Y WITH Z = 0.1----------#
def pointgrid(n,a,b): #-------NUMBER OF POINTS ON X AND Y AXIS-----------#
    xx = np.linspace(0, a, num=n)
    yy = np.linspace(0, b,num=n)
    xxyy = np.array([xx,yy])
    pnt = [] 

    for i in itertools.product(*xxyy):
        pnt.append([i]) #arrays within array created    
    #pnt = np.array(pnt) #converted to numpy

    point = []
    for i in pnt:
        i = np.append(i,0.4)  #Z    #Append each array from [x,y] to [x,y,0.1]
        point.append([i])         #Append empty list point with [x,y,0.1]
    for inner_array in point:
        inner_array.append(0)

    points = np.array(point)            
    return point

#grid  = pointgrid(4,1,1)
#print (grid)
'''Check if within sphere'''
def ifwithinsphere(point):
    return sqrt( (point[0])**2 +(point[1])**2+ (point[2])**2 )

'''Assign a colourvalue based on number of iterations'''
def colvalue(num,a,b):
    grid = pointgrid(num,a,b)
    for i in range(len(grid)):
        if ifwithinsphere(grid[i][0]) < 1:
            col_val = pathlength(grid[i][0],0.001)
            grid[i][1] = (col_val)
    return grid    

density = (colvalue(300,1,1))

''' PLOTTING '''

'''Extract points and colours'''
points1 = []
colours  = []
for i in range(len(density)):
    point = density[i][0]
    colour = density[i][1]
    points1.append(point)
    colours.append(colour)
points = np.array(points1)  

x = points[:,0]
y = points[:,1]

print (x,y,colours)

'''Colormap plotting using Matplotlib'''
cmap = plt.scatter(x,y,c=colours,cmap=cm.jet,vmin=0.,vmax=20.)
plt.colorbar(cmap,boundaries=np.arange(0,21,1))
plt.show()

