# -*- coding: mbcs -*-
#
# Abaqus/CAE 2021
# Run by Liu on Thu Mar 21 14:12:35 2024
# from driverUtils import executeOnCaeGraphicsStartup
# executeOnCaeGraphicsStartup()
#: Executing "onCaeGraphicsStartup()" in the site directory ...
############################
# Script algorithm in the literature:
##  An improved random sequential absorption method for thermal conductivity study of diamond-reinforced composites
# This program introduces close-packed points based on previous work to further increase the packing volume fraction of the algorithm and improve the iterative rearrangement efficiency.
# For more details, interested readers can refer to the article link.
##  Dio:https://doi.org/10.1016/j.diamond.2025.112324
############################

from abaqus import *
from abaqusConstants import *
# Open a window
session.Viewport(name='Viewport: 1', origin=(0.0, 0.0), width=171.000503540039, 
    height=114.288887023926)
session.viewports['Viewport: 1'].makeCurrent()
session.viewports['Viewport: 1'].maximize()
from caeModules import *
from driverUtils import executeOnCaeStartup
import numpy as np
import random
# Import equation solving function
from scipy.optimize import fsolve
# Import the math library for trigonometric function operations
import math
from math import sin,cos,floor
import sys
sys.path.append(r"C:\TEMP")

##### 
# Periodic boundary condition enforcement algorithm is under development; 
# the relevant PBC library will use a face-face matching approach to improve node matching efficiency.
#import PBC
#######

import zipfile
import os
import zlib
import time
import csv



########################
executeOnCaeStartup()
# Generate model data object mdb
Mdb()
#: A new model database has been created.
#: The model "Model-1" has been created.
# Set the working directory
# # import os
# os.chdir(r"C:\Users\Liu\Desktop\diamond")
# if os.path.exists('diamond_data.txt')==False:
#     with open("diamond_data.txt", "a") as f:
#                 f.write('Matrix-K'+' '+'Diamond-lambda'+' '+'ITC'+' '+'V'+' '+'D_eq'+' '+'Matrix-E'+' '+
#                         'Matrix-JC-A'+' '+'Matrix-JC-B'+' '+'Matrix-JC-m'+' '+'Interface-E'+' '+'Interface-eta'+' '+'Interface-t'+' '+
#                         'effective-K'+' '+'effective-a'+' '+'effective-a-list'+' '+'effective-E'+' '+'effective-v'+' '+'effective-yiled_0_2'+'\n')
if os.path.exists('diamond_data.csv')==False:
    with open('diamond_data.csv', mode='w') as outfile:
        writer = csv.writer(outfile)
        data=['Matrix-K','matrix_rho','matrix_alpha','matrix_c','matrix_v','Diamond-lambda','ITC','V','D_eq','Matrix-E','Matrix-JC-A','Matrix-JC-B','Matrix-JC-m','Interface-E',
                'Interface-eta','Interface-t','effective-K','effective-a','effective-E','effective-v','effective-yiled_0_2','CTE_list','epsilon_list','sigma_list']
        writer.writerow(data)
if os.path.exists('diamond_data_parameter.csv')==False:
    with open('diamond_data_parameter.csv', mode='w') as outfile:
        writer = csv.writer(outfile)
        data=['Matrix-K','matrix_rho','matrix_alpha','matrix_c','matrix_v','diamond_K','ITC','V','D_eq','Matrix-E','Matrix-JC-A','Matrix-JC-B','Matrix-JC-m','Interface-Knn',
                'Interface-Kss','effective-K','effective-a','effective-E','effective-v','effective-yiled_0_2','CTE_list','epsilon_list','sigma_list']
        writer.writerow(data)




mesh_size=0.1

# Define a small space object:

class space_of_diamond:
    # Define the spatial extent
    def __init__(self):
        self.xyzmax=[]
        self.xyzmin=[]
        # Define whether there are particles present and record the number of particles
        self.diamond_lable=0
        # Define whether the current space is an edge space identifier
        self.boundary_lable=False
        # Save the positions and Euler angles of particles in the current space
        self.diamond_points=[]
        # Save the information of diamond particles in the current space
        self.diamond_datas=[]
        # Define the virtual auxiliary space encoding corresponding to the current space and record the count.
        self.auxiliary_space=[]
        self.auxiliary_space_num=0
        # Record the densely packed diamond points and their count in the current space
        self.pack_diapoint=[]
        self.pack_diapoint_num=0
        # Record the densely packed sphere points and their count in the current space. For simplification, only ABAB packing is considered here
        self.pack_spherepoint=[]
        self.pack_spherepoint_num=0

# Define diamond point data:
class data_of_diamond:
    # Define the spatial range
    def __init__(self):
        # Define diamond a, l data
        self.a=0.2
        self.cut_ratio=sqrt(2)/3
        self.l=self.cut_ratio*self.a
        # Define the equivalent particle size, envelope sphere radius, volume of diamond
        self.dengxiaoD=0.2
        self.shpereR=0.2
        self.V=0.002

        # Define diamond material properties
        self.thermal=0.0
        self.Ei=0.0
        self.v=0.0
        self.sigmas=0.0
        self.sigmat=0.0

        # Define point data in the local coordinate system
        self.linelength=[2*(self.a/sqrt(2)-self.l)*(self.a/sqrt(2)-self.l),
                         2*(self.a/sqrt(2)-2*self.l)*(self.a/sqrt(2)-2*self.l)]
        self.partname='part-200'
        self.xi=self.l
        self.xj=self.a/sqrt(2)-self.l
        self.xk=0.0
        self.points=[[self.xk,self.xi,self.xj,1.0],[self.xk,self.xi,-self.xj,1.0],[self.xk,-self.xi,self.xj,1.0],[self.xk,-self.xi,-self.xj,1.0],
                     [self.xk,self.xj,self.xi,1.0],[self.xk,self.xj,-self.xi,1.0],[self.xk,-self.xj,self.xi,1.0],[self.xk,-self.xj,-self.xi,1.0],
                     [self.xi,self.xk,self.xj,1.0],[self.xi,self.xk,-self.xj,1.0],[-self.xi,self.xk,self.xj,1.0],[-self.xi,self.xk,-self.xj,1.0],
                     [self.xj,self.xk,self.xi,1.0],[self.xj,self.xk,-self.xi,1.0],[-self.xj,self.xk,self.xi,1.0],[-self.xj,self.xk,-self.xi,1.0],
                     [self.xi,self.xj,self.xk,1.0],[self.xi,-self.xj,self.xk,1.0],[-self.xi,self.xj,self.xk,1.0],[-self.xi,-self.xj,self.xk,1.0],
                     [self.xj,self.xi,self.xk,1.0],[self.xj,-self.xi,self.xk,1.0],[-self.xj,self.xi,self.xk,1.0],[-self.xj,-self.xi,self.xk,1.0]]
        self.lines=[[0, 4], [0, 20], [0, 22], [1, 5], [1, 20], [1, 22], [2, 6], [2, 21], [2, 23], [3, 7], [3, 21], [3, 23], [4, 12], [4, 14], 
                    [5, 13], [5, 15], [6, 12], [6, 14], [7, 13], [7, 15], [8, 12], [8, 16], [8, 17], [9, 13], [9, 16], [9, 17], [10, 14], 
                    [10, 18], [10, 19], [11, 15], [11, 18], [11, 19], [16, 20], [17, 21], [18, 22], [19, 23]]

    
    def diamond_refresh(self):
        
        self.l=self.cut_ratio*(self.a)
        self.V=(sqrt(2)/3)*(self.a**3)-2*((self.a-sqrt(2)*self.l)**2)*(self.a/sqrt(2)-self.l) # Volume of diamond particle
        self.dengxiaoD=1.2407009817988*(self.V)**(1.0/3) # Equivalent particle size of diamond
        self.shpereR=sqrt(self.l**2+(self.a/sqrt(2)-self.l)**2) # Envelope sphere radius of diamond particle
        self.xi=self.l
        self.xj=(self.a)/sqrt(2)-self.l
        self.xk=0.0
        self.points=[[self.xk,self.xi,self.xj,1.0],[self.xk,self.xi,-self.xj,1.0],[self.xk,-self.xi,self.xj,1.0],[self.xk,-self.xi,-self.xj,1.0],
                     [self.xk,self.xj,self.xi,1.0],[self.xk,self.xj,-self.xi,1.0],[self.xk,-self.xj,self.xi,1.0],[self.xk,-self.xj,-self.xi,1.0],
                     [self.xi,self.xk,self.xj,1.0],[self.xi,self.xk,-self.xj,1.0],[-self.xi,self.xk,self.xj,1.0],[-self.xi,self.xk,-self.xj,1.0],
                     [self.xj,self.xk,self.xi,1.0],[self.xj,self.xk,-self.xi,1.0],[-self.xj,self.xk,self.xi,1.0],[-self.xj,self.xk,-self.xi,1.0],
                     [self.xi,self.xj,self.xk,1.0],[self.xi,-self.xj,self.xk,1.0],[-self.xi,self.xj,self.xk,1.0],[-self.xi,-self.xj,self.xk,1.0],
                     [self.xj,self.xi,self.xk,1.0],[self.xj,-self.xi,self.xk,1.0],[-self.xj,self.xi,self.xk,1.0],[-self.xj,-self.xi,self.xk,1.0]]
            
    # Used to find two points whose line length equals linelength; initially used to determine whether two points are edges of a hexoctahedron,
    # establishing point-line connections. This function is no longer needed subsequently.   
    # def points_lines(self,points,lines,linelength):
    #     for i in range(0,len(points)):
    #         x1=points[i][0]
    #         y1=points[i][1]
    #         z1=points[i][2]
    #         for j in range(0,len(points)):
    #             if i==j:
    #                 continue
    #             x2=points[j][0]
    #             y2=points[j][1]
    #             z2=points[j][2]
    #             dis=(x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)+(z2-z1)*(z2-z1)
    #             if abs(dis-linelength)<0.00001:
    #                 cunzai=False
    #                 for line in lines:
    #                     if line[1]==i and line[0]==j:
                            
    #                         cunzai=True
    #                         break
    #                 if cunzai:
    #                     print(cunzai)
    #                     continue
    #                 lines.append([i,j])
            
def zip_odbfile(file_list,ZIPfile2):
    
    # Create a ZipFile object for operating compressed files
    with zipfile.ZipFile(ZIPfile2, "a", allowZip64=True,compression=zipfile.ZIP_DEFLATED) as zipf:
        for filename1 in file_list:
            zipf.write(filename1)     
# Define file reading function
def readTxt(filepath='diamond_data.txt'):
    data=[]
    data_line = []
    with open(filepath,"r") as f:
        for line in f.readlines():
            line = line.strip("\n")
            line = line.split()
            data_line.append(line)
            print(data_line)
    return data
   
############################
# Generally, the higher the volume fraction, the closer the arrangement tends to be close-packed.
# Therefore, initial particles are generated here in the densest arrangement.
# The distance between random particle positions and close-packed points is controlled by a
# volume-fraction-based adjustment function, achieving more random distribution at low volume fractions,
# while approaching close-packed distribution at high volume fractions, thus improving the algorithm's
# maximum filling density.
#############################

# Close-packing mode identifier
dia_or_sphere_lable=0

# Particle shape identifier. Currently only two shapes are defined: sphere and hexoctahedron.
# Corresponding collision detection methods need to be developed for each geometry.
# Future developers can extend to ellipses, cylinders, etc. Currently: 0-hexoctahedron, 1-sphere.
shape_lable=0

# Periodic boundary condition loading identifier
Period_boundary_lable=0
# Thermal conductivity calculation identifier
CTC_lable=1
# Coefficient of thermal expansion calculation identifier
CTE_lable=1
# Young's modulus calculation identifier, 0.2% strain as yield strength
E_lable=1
# Yield/compressive strength calculation identifier, yield at 0.2% strain, compression at 10% strain (under development)
Ys_Cs_lable=0


# Variable for storing simulation data and results
simulation_data_list=[]
simulation_data_list1=[]

# Define function to take random values from N equally divided intervals in the range ValueMin-ValueMax, and return a shuffled list
def list_from_range(ValueMin,ValueMax,N):
    # Parameter validation
    if not isinstance(N, int) or N <= 0:
        raise ValueError("N must be int")
    if ValueMax <= ValueMin:
        raise ValueError("value_max must be greater than value_min")
    # Calculate interval step size
    interval = (ValueMax - ValueMin) / N
    
    # Generate random values for each interval
    result = []
    for i in range(N):
        # Calculate current interval boundaries
        lower = ValueMin + i * interval
        upper = lower + interval
        
        # Use exact upper limit for the last interval
        if i == N - 1:
            upper = ValueMax
        
        # Generate a random value within the interval
        rand_val = random.uniform(lower, upper)
        result.append(rand_val)
    # Shuffle the result order
    random.shuffle(result)   
    return result
# Function to create spherical diamond particles
def sphere_generator(r,modelname,partname):
    sphere_name=partname
    # Create sketch
    s = mdb.models[modelname].ConstrainedSketch(name='__profile__', sheetSize=1.0)
    g, v, d, c = s.geometry, s.vertices, s.dimensions, s.constraints
    s.setPrimaryObject(option=STANDALONE)
    # Create square
    #s.rectangle(point1=(itemi*0.0005, itemi*0.0005), point2=(-itemi*0.0005, -itemi*0.0005))
    # Create arc
    s.ConstructionLine(point1=(0.0, -100.0), point2=(0.0, 100.0))
    s.FixedConstraint(entity=g[2])
    s.ArcByCenterEnds(center=(0.0, 0.0), point1=(0.0, r), point2=(0.0, -r), 
    direction=CLOCKWISE)
    # Connect the two ends of the arc with a line
    s.Line(point1=(0.0, r), point2=(0.0, -r))
    # Generate sphere
    p = mdb.models[modelname].Part(name=sphere_name, dimensionality=THREE_D, 
    type=DEFORMABLE_BODY)
    p = mdb.models[modelname].parts[sphere_name]
    p.BaseSolidRevolve(sketch= s, angle=360.0, flipRevolveDirection=OFF)
    s.unsetPrimaryObject()
    p = mdb.models[modelname].parts[sphere_name]
    #session.viewports['Viewport: 1'].setValues(displayedObject=p)
    del mdb.models[modelname].sketches['__profile__']

# Used to create hexoctahedral diamond particles. Parameters a and l control diamond generation.
# (For specific definitions, please refer to the paper (https://doi.org/10.1016/j.diamond.2025.112324),
# modelname is the name of the model where the part should be created, partname is the name of the new part
def diamond_generator(a,l,modelname,partname):
    diamond_name=partname
    # Create sketch
    s = mdb.models[modelname].ConstrainedSketch(name='__profile__', sheetSize=1.0)
    g, v, d, c = s.geometry, s.vertices, s.dimensions, s.constraints
    #s.setPrimaryObject(option=STANDALONE)
    # Create square
    s.rectangle(point1=(a*0.5, a*0.5), point2=(-a*0.5, -a*0.5))
    # Rotate 45 degrees
    s.rotate(centerPoint=(0.0, 0.0), angle=45.0, objectList=(g[2], g[3], g[4], 
    g[5]))
    # Create quadrangular pyramid
    p = mdb.models[modelname].Part(name=diamond_name, dimensionality=THREE_D, type=DEFORMABLE_BODY)
    p = mdb.models[modelname].parts[diamond_name]
    p.BaseSolidExtrude(sketch=s, depth=a, draftAngle=-(90-54.7356103172))
    #s.unsetPrimaryObject()
    # Display model; if commented, the drawn model will not be shown in the interface
    #session.viewports['Viewport: 1'].setValues(displayedObject=p)
    p = mdb.models[modelname].parts[diamond_name]
    del mdb.models[modelname].sketches['__profile__']
    # Mirror quadrangular pyramid to obtain octahedron
    p = mdb.models[modelname].parts[diamond_name]
    f = p.faces
    p.Mirror(mirrorPlane=f[4], keepOriginal=ON)
    ##############
    ##############
    ###########
    ## Create three points
    p = mdb.models[modelname].parts[diamond_name]
    p.DatumPointByCoordinate(coords=(1.0, 1.0, 1.0))
    p = mdb.models[modelname].parts[diamond_name]
    p.DatumPointByCoordinate(coords=(1.0, -1.0, 1.0))
    p = mdb.models[modelname].parts[diamond_name]
    p.DatumPointByCoordinate(coords=(-1.0, 1.0, 1.0))
    p = mdb.models[modelname].parts[diamond_name]
    d2 = p.datums
    p.DatumPlaneByThreePoints(point1=d2[3], point2=d2[4], point3=d2[5])
    # Create X, Y, Z coordinate axes
    p = mdb.models[modelname].parts[diamond_name] #datums[7]
    p.DatumAxisByPrincipalAxis(principalAxis=XAXIS)
    p = mdb.models[modelname].parts[diamond_name] #datums[8]
    p.DatumAxisByPrincipalAxis(principalAxis=ZAXIS)
    p = mdb.models[modelname].parts[diamond_name] #datums[9]
    p.DatumAxisByPrincipalAxis(principalAxis=YAXIS)
    # Create sketch
    p = mdb.models[modelname].parts[diamond_name]
    f1, e1, d1 = p.faces, p.edges, p.datums
    t = p.MakeSketchTransform(sketchPlane=d1[6], sketchUpEdge=d1[9], 
    sketchPlaneSide=SIDE1, sketchOrientation=BOTTOM, origin=(0.0, 0.0, 1.0))
    s = mdb.models[modelname].ConstrainedSketch(name='__profile__', sheetSize=6.41, 
    gridSpacing=0.16, transform=t)
    g, v, d, c = s.geometry, s.vertices, s.dimensions, s.constraints
    #s.setPrimaryObject(option=SUPERIMPOSE)
    p = mdb.models[modelname].parts[diamond_name]
    p.projectReferencesOntoSketch(sketch=s, filter=COPLANAR_EDGES)
    s.rectangle(point1=(-l, -l), point2=(l, l))
    s.rectangle(point1=(-100.0, -100.0), point2=(100.0, 100.0))
    p = mdb.models[modelname].parts[diamond_name]
    f, e, d2 = p.faces, p.edges, p.datums
    p.CutExtrude(sketchPlane=d2[6], sketchUpEdge=d2[9], sketchPlaneSide=SIDE1, 
    sketchOrientation=BOTTOM, sketch=s, flipExtrudeDirection=ON)
    #s.unsetPrimaryObject()
    del mdb.models[modelname].sketches['__profile__']
    ###
    p = mdb.models[modelname].parts[diamond_name]
    p.DatumPointByCoordinate(coords=(1.0, 1.0, -1.0))
    p = mdb.models[modelname].parts[diamond_name]
    d1 = p.datums
    p.DatumPlaneByThreePoints(point1=d1[3], point2=d1[4], point3=d1[11])
    p = mdb.models[modelname].parts[diamond_name]
    f1, e1, d2 = p.faces, p.edges, p.datums
    t = p.MakeSketchTransform(sketchPlane=d2[12], sketchUpEdge=d2[8], 
    sketchPlaneSide=SIDE1, sketchOrientation=TOP, origin=(1.0, 0.0, 0.0))
    s1 = mdb.models[modelname].ConstrainedSketch(name='__profile__', 
    sheetSize=6.92, gridSpacing=0.17, transform=t)
    g, v, d, c = s1.geometry, s1.vertices, s1.dimensions, s1.constraints
    #s1.setPrimaryObject(option=SUPERIMPOSE)
    p = mdb.models[modelname].parts[diamond_name]
    p.projectReferencesOntoSketch(sketch=s1, filter=COPLANAR_EDGES)
    s1.rectangle(point1=(-l, -l), point2=(l, l))
    s1.rectangle(point1=(100.0, 100.0), point2=(-100.0, -100.0))
    p = mdb.models[modelname].parts[diamond_name]
    f, e, d1 = p.faces, p.edges, p.datums
    p.CutExtrude(sketchPlane=d1[12], sketchUpEdge=d1[8], sketchPlaneSide=SIDE1, 
    sketchOrientation=TOP, sketch=s1, flipExtrudeDirection=OFF)
    #s1.unsetPrimaryObject()
    session.viewports['Viewport: 1'].setValues(displayedObject=p)
    del mdb.models[modelname].sketches['__profile__']
    # # Assign diamond section
    # p = mdb.models[modelname].parts[diamond_name]
    # c = p.cells
    # region = regionToolset.Region(cells=c)
    # p.SectionAssignment(region=region, sectionName='Section-diamond', offset=0.0, 
    # offsetType=MIDDLE_SURFACE, offsetField='',
    # thicknessAssignment=FROM_SECTION)


for xunhuan in range(0,10):

    with open('diamond_data0.csv', mode='w') as outfile:
        writer = csv.writer(outfile)
        data=['Matrix-K','matrix_rho','matrix_alpha','matrix_c','matrix_v','Diamond-lambda','ITC','V','D_eq','Matrix-E','Matrix-JC-A','Matrix-JC-B','Matrix-JC-m','Interface-E',
                'Interface-eta','Interface-t','effective-K','effective-a','effective-E','effective-v','effective-yiled_0_2','CTE_list','epsilon_list','sigma_list']
        writer.writerow(data)

    with open('diamond_data_parameter0.csv', mode='w') as outfile:
        writer = csv.writer(outfile)
        data=['Matrix-K','matrix_rho','matrix_alpha','matrix_c','matrix_v','diamond_K','ITC','V','D_eq','Matrix-E','Matrix-JC-A','Matrix-JC-B','Matrix-JC-m','Interface-Knn',
                'Interface-Kss','effective-K','effective-a','effective-E','effective-v','effective-yiled_0_2','CTE_list','epsilon_list','sigma_list']
        writer.writerow(data)

    # Latin hypercube sampling to generate initial samples
    # Define sample size
    Latin_num=30
    # Thermophysical parameters
    
    #Enhancer_thermal_list=list_from_range(1300,2000,10) # Reinforcement thermal conductivity

    

    # Constant parameters
    
    diamond_rho=3.52E-09 # Diamond density
    diamond_alpha=1E-06 # Diamond thermal expansion coefficient
    diamond_c=512000000 # Diamond specific heat capacity
    # matrix_rho=2.7E-09 # Matrix density
    # matrix_alpha=2.34E-05 # Matrix thermal expansion coefficient
    # matrix_c=895000000 # Matrix specific heat capacity
    # matrix_v=0.33
    #matrix_E=69000 # Matrix elastic modulus #69-71
    # matrix_B=117.5 #100-140
    # matrix_m=0.5142 #0.2-0.8
    diamond_E=1130000
    diamond_v=0.085
    # diamond_thermal=2000

    ## Variable parameters
    # Conventional parameters
    matrix_thermal_list=list_from_range(130.0,500.0,Latin_num) # Matrix thermal conductivity
    matrix_A_list=list_from_range(30.0,300.0,Latin_num) # Matrix yield strength
    diamond_zeta_list=list_from_range(1.0,300.0,Latin_num) # Diamond nitrogen content
    matrix_E_list=list_from_range(50000.0,300000.0,Latin_num) # Matrix elastic modulus #50-300
    matrix_B_list=list_from_range(100.0,300.0,Latin_num) #100-140
    matrix_m_list=list_from_range(0.2,0.8,Latin_num) #0.2-0.8
    matrix_rho_list=list_from_range(2.7E-09,3.2E-09,Latin_num) # Matrix density
    matrix_alpha_list=list_from_range(9.34E-06,2.34E-05,Latin_num) # Matrix thermal expansion coefficient
    matrix_c_list=list_from_range(700000000,895000000,Latin_num) # Matrix specific heat capacity
    matrix_v_list=list_from_range(0.2,0.33,Latin_num) # Matrix Poisson's ratio

    # Variable parameters in uncertainty modeling
    diamond_ratio_list=list_from_range(10,50.0,Latin_num) # Diamond volume fraction
    diamond_dxD_list=list_from_range(0.01,1.0,Latin_num) # Diamond equivalent particle size
    diamond_lambda=0.8
    ###
    # Diamond thermal conductivity, elastic modulus, Poisson's ratio, and fracture strength
    # will be defined in the diamond point data class
    ###
    # Interface parameters
    jiemian_hindex_list=list_from_range(1.0,8.0,Latin_num) # Interface thermal conductance
    jiemian_hi_list=list_from_range(1.0,10.0,Latin_num) # Interface thermal conductance



    jiemian_EC_list=list_from_range(200E3,500E3,Latin_num)# Interface material elastic modulus
    jiemian_eta_list=list_from_range(1.6,2.7,Latin_num)# Ratio of interface material elastic modulus to shear modulus
    jiemian_t_list=list_from_range(100E-6,1000E-6,Latin_num)# Interface layer thickness
    

    # Enhancer_E_list=list_from_range(1050000,1210000,10) # Diamond elastic modulus
    # Enhancer_v_list=list_from_range(0.07,0.1,10) # Diamond Poisson's ratio


    odb_file=[]
    for list_i in range(0,Latin_num):
        # Conventional parameters
        matrix_thermal=matrix_thermal_list[list_i] # Matrix thermal conductivity
        matrix_A=matrix_A_list[list_i] # Matrix yield strength
        diamond_zeta=diamond_zeta_list[list_i] # Diamond nitrogen content
        matrix_E=matrix_E_list[list_i] # Matrix elastic modulus#69-71
        matrix_B=matrix_B_list[list_i] #100-140
        matrix_m=matrix_m_list[list_i] #0.2-0.8

        matrix_rho=matrix_rho_list[list_i] # Matrix density
        matrix_alpha=matrix_alpha_list[list_i] # Matrix thermal expansion coefficient
        matrix_c=matrix_c_list[list_i] # Matrix specific heat
        matrix_v=matrix_v_list[list_i] # Matrix Poisson's ratio

        # Variable parameters in uncertainty modeling
        diamond_ratio=diamond_ratio_list[list_i] # Diamond volume fraction
       
        diamond_lambda=0.8

        jiemian_h=jiemian_hi_list[list_i]*(10**int(jiemian_hindex_list[list_i])) # Interface thermal conductance
        jiemian_EC=jiemian_EC_list[list_i] # Interface material elastic modulus
        jiemian_eta=jiemian_eta_list[list_i] # Ratio of interface material elastic modulus to shear modulus
        jiemian_t=jiemian_t_list[list_i] # Interface layer thickness

        j_GC=jiemian_EC/jiemian_eta
        Knn=jiemian_EC/jiemian_t
        Kss=j_GC/jiemian_t

        # diamond_thermal=2000
        # diamond_ratio=50
        diamond_dxD=diamond_dxD_list[list_i]
        # model_jiemian=1000
        # The relationship between a and D can be estimated as a=1.0770911741143*D
        acenter=1.0770911741143*diamond_dxD
        a_sigma=acenter*0.2
        diamond_dxD=acenter/1.0770911741143#diamond_dxD_list[list_i] # Diamond equivalent particle size

        
        # Original parameters
        
        simulation_data=[matrix_thermal,matrix_rho,matrix_alpha,matrix_c,matrix_v,diamond_zeta,jiemian_h,diamond_ratio,diamond_dxD,matrix_E,matrix_A,matrix_B,matrix_m,jiemian_EC,jiemian_eta,jiemian_t]
        # Performance parameters
        f_zeta=diamond_lambda*(1042.6*exp(-diamond_zeta/9573.4)+834.9*exp(-diamond_zeta/882.9)+550.5*exp(-diamond_zeta/192.2))

        diamond_K=f_zeta-f_zeta*exp(-diamond_dxD*1000/110.0)

        simulation_data1=[matrix_thermal,matrix_rho,matrix_alpha,matrix_c,matrix_v,diamond_K,jiemian_h,diamond_ratio,diamond_dxD,matrix_E,matrix_A,matrix_B,matrix_m,Knn,Kss]
        
        # Calculate diamond particle related data
        cut_ratio_d=sqrt(2)/3 # l/a ratio
        Dia_l_center=cut_ratio_d*acenter  # Control parameter l
        Dia_V_center=(sqrt(2)/3)*(acenter**3)-2*((acenter-sqrt(2)*Dia_l_center)**2)*(acenter/sqrt(2)-Dia_l_center) # Diamond particle volume
        Dia_center_Bounding_R=sqrt(Dia_l_center**2+(acenter/sqrt(2)-Dia_l_center)**2)  # Diamond particle envelope sphere radius

        # Calculate sphere particle volume related data
        sphere_V_center=pi*(diamond_dxD**3)/6

        # Parameters for diamond close-packed point generation, using median a value
        dia_l=Dia_l_center#+0.15*acenter*(1-diamond_ratio*0.01)
        dia_b=dia_l*cut_ratio_d*2
        
        # Parameters for sphere close-packed point generation, using median a value
        if shape_lable==0:
            sphere_r=Dia_center_Bounding_R
            sphere_h=2*sqrt(6)*sphere_r/3
            sphere_d=sphere_r*2
            sphere_y=sqrt(3)*sphere_r
        elif shape_lable==1:
            sphere_r=diamond_dxD/2
            sphere_h=2*sqrt(6)*sphere_r/3
            sphere_d=sphere_r*2
            sphere_y=sqrt(3)*sphere_r

        

          
        #model_sim_data=[]
        model_name='model_0'
        # Create model, define absolute zero
        mdb.Model(name=model_name, absoluteZero=-273.15, modelType=STANDARD_EXPLICIT)

        # Calculate RVE model dimensions
        if shape_lable==0:

            aldim=6.49*1.05*(Dia_V_center**(1.0/3.0))
            # Determine which close-packed point generation method to use based on identifier
            if dia_or_sphere_lable==0:
                aldim=(int(aldim/dia_b)+1)*dia_b
            else:
                aldim=(int(0.5*aldim/Dia_center_Bounding_R)+1)*(2*Dia_center_Bounding_R)
            # Calculate number of diamond particles using median a value
            Dia_num=int(aldim*aldim*aldim*diamond_ratio*0.01/Dia_V_center)
            if Dia_num==0:
                Dia_num=Dia_num+1
            Dia_a_size=int(Dia_num*3)
            # Uncertainty modeling of particle size
            # Generate normally distributed diamond particles, controllable via acenter and a_sigma
            Dia_a=np.random.normal(loc=acenter, scale=a_sigma,size=Dia_a_size)
            random.shuffle(Dia_a)
            Dia_dxD=Dia_a/1.0770911741143

        elif shape_lable==1:
            aldim=6.49*1.05*(sphere_V_center**(1.0/3.0))
            aldim=(int(aldim/diamond_dxD)+1)*(diamond_dxD)
            # Calculate number of diamond particles using median a value
            Dia_num=int(aldim*aldim*aldim*diamond_ratio*0.01/sphere_V_center)
            Dia_a_size=int(Dia_num*1.2)
            # Generate normally distributed diamond particles, controllable via acenter and a_sigma
            a_sigma=0.0*diamond_dxD
            Dia_a=np.random.normal(loc=diamond_dxD, scale=a_sigma,size=Dia_a_size)
            random.shuffle(Dia_a)
        #aldim=1.1
        Enhancer_volume_design=aldim*aldim*aldim*diamond_ratio*0.01
        # Generate aluminum matrix block
        s = mdb.models[model_name].ConstrainedSketch(name='__profile__', sheetSize=10.0)
        g, v, d, c = s.geometry, s.vertices, s.dimensions, s.constraints
        s.rectangle(point1=(-aldim/2, -aldim/2), point2=(aldim/2, aldim/2))
        p = mdb.models[model_name].Part(name='Part-Al', dimensionality=THREE_D, 
            type=DEFORMABLE_BODY)
        p = mdb.models[model_name].parts['Part-Al']
        p.BaseSolidExtrude(sketch=s, depth=aldim)
        p = mdb.models[model_name].parts['Part-Al']
        del mdb.models[model_name].sketches['__profile__']
        

        diamond_data_list=[]

        # Record start time
        
        start=time.time()

        # Add diamond particles into assembly
        Dia_position=[[Dia_a[0]]]
        Dia_ass_name=[]
        blank_list_dia_data=[]
        Dia_ass_name_data=[]
        i=0
        Dia_temp=Dia_a[i]
        Dia_tempd=cut_ratio_d*Dia_a[i]
    
        tempn=1  
        # Determine the number of cube subdivisions
        if shape_lable==0:
            tempn=int(aldim/(2*Dia_center_Bounding_R))
            Al_cut=tempn
            tempn0=tempn
        elif shape_lable==1:
            tempn=int(aldim/(diamond_dxD))
            Al_cut=tempn
            tempn0=tempn


        

        # Generate empty 3D lists
        blank_list_dia=[]
        blank_list_dia1=[]
        blank_list_dia2=[]

        for blanki in range(0,Al_cut+4):
            blank_list_dia.append([])
            blank_list_dia1.append([])
            blank_list_dia2.append([])

            for blankj in range(0,Al_cut+4):
                blank_list_dia[blanki].append([])
                blank_list_dia1[blanki].append([])
                blank_list_dia2[blanki].append([])
                # Initialize linked lists
                for blankk in range(0,Al_cut+4):
                    #spacei=space_of_diamond()
                    blank_list_dia[blanki][blankj].append(space_of_diamond())
                    blank_list_dia1[blanki][blankj].append(space_of_diamond())
                    blank_list_dia2[blanki][blankj].append(space_of_diamond())

        
        # Calculate the probability of particle existence in each space
        # Probability function improvement: first determine whether the current space has close-packed positions;
        # if close-packed positions exist, add a probability value related to the volume fraction.
        # From the close-packing perspective, the larger the volume fraction, the closer the distribution is to close-packed.
        # tempratio=float(Dia_N[i])/(tempn*tempn*tempn)
        # First generate the current particle close-packed lattice, then generate Gaussian-distributed random numbers
        # to perturb the lattice, and finally place diamond particles according to the perturbed positions.
        # The method of generating movement coordinates needs adjustment; direct random numbers easily lead
        # to large computational loads and make it difficult to achieve non-penetration between particles.
        # Currently, a spatially uniform partitioning approach is adopted, where particles only undergo
        # local perturbations within small spaces to approximate the actual distribution.
        list_daluan1=range(1,Al_cut+1)
        list_daluan2=range(1,Al_cut+1)
        list_daluan3=range(1,Al_cut+1)
        random.shuffle(list_daluan1)
        random.shuffle(list_daluan2)
        random.shuffle(list_daluan3)
        # First, randomly generate a point within the range of (+-a/2, +-a/2, +-a/2)
        # x_ref=np.random.uniform(-Dia_a[0]*0.0005,Dia_a[0]*0.0005)
        # y_ref=np.random.uniform(-Dia_a[0]*0.0005,Dia_a[0]*0.0005)
        # z_ref=np.random.uniform(-Dia_a[0]*0.0005,Dia_a[0]*0.0005)

        # Bind virtual auxiliary spaces
        # Labels 0 and Al_cut+1 are virtual space indices
        for tempx in range(0,Al_cut+2):
            for tempy in range(0,Al_cut+2):
                for tempz in range(0,Al_cut+2):
                    if tempx==0 or tempy==0 or tempz==0 or tempx==Al_cut+1 or tempy==Al_cut+1 or tempz==Al_cut+1:
                        if tempx==0:
                            tempx_x=Al_cut
                        elif tempx==Al_cut+1:
                            tempx_x=1
                        else:
                            tempx_x=tempx
                        if tempy==0:
                            tempy_y=Al_cut
                        elif tempy==Al_cut+1:
                            tempy_y=1
                        else:
                            tempy_y=tempy
                        if tempz==0:
                            tempz_z=Al_cut
                        elif tempz==Al_cut+1:
                            tempz_z=1
                        else:
                            tempz_z=tempz
                        space_xmax=-aldim/2+(aldim/tempn)*tempx
                        space_xmin=-aldim/2+(aldim/tempn)*(tempx-1.0)
                        space_ymax=-aldim/2+(aldim/tempn)*tempy
                        space_ymin=-aldim/2+(aldim/tempn)*(tempy-1.0)
                        space_zmax=(aldim/tempn)*(tempz)
                        space_zmin=(aldim/tempn)*(tempz-1.0)

                        blank_list_dia[tempx][tempy][tempz].xyzmax=[space_xmax,space_ymax,space_zmax]
                        blank_list_dia[tempx][tempy][tempz].xyzmin=[space_xmin,space_ymin,space_zmin]

                        blank_list_dia[tempx][tempy][tempz].boundary_lable=True
                        blank_list_dia[tempx_x][tempy_y][tempz_z].auxiliary_space_num=blank_list_dia[tempx_x][tempy_y][tempz_z].auxiliary_space_num+1
                        blank_list_dia[tempx_x][tempy_y][tempz_z].auxiliary_space.append([tempx,tempy,tempz])


##################################################
# Generate close-packed points and bind with divided small spaces
##################################################

        # First generate close-packed points, the close-packed point space size is 2*aldim
        # Hexa-octahedron close-packed points
        xrange=range(0,int(2*aldim/dia_b))
        yrange=range(0,int(2*aldim/dia_b))
        zrange=range(0,int(2*aldim/dia_l))
        # Define local coordinate system
        x_ref=0#np.random.uniform(-dia_b,+dia_b)
        y_ref=0#np.random.uniform(-dia_b,+dia_b)
        z_ref=aldim/2#np.random.uniform(aldim/2-dia_b,aldim/2+dia_b)
        x_roa=np.random.uniform(-45.0,+45.0)
        y_roa=np.random.uniform(-45.0,+45.0)
        z_roa=np.random.uniform(-45.0,+45.0)
        # Consider that the overall close-packed points are established relative to the local coordinate system of a given particle
        # After Euler rotation and translation of the particle, the expression of close-packed points in the world coordinate system can be obtained via the rotation-translation matrix

        coord_point=np.array([[x_ref,y_ref,z_ref,1.0]])
        degree1=math.radians(x_roa)
        degree2=math.radians(y_roa)
        degree3=math.radians(z_roa)
        rx=np.array([[1.0,0.0,0.0],[0.0,cos(degree1),-sin(degree1)],[0.0,sin(degree1),cos(degree1)]])
        ry=np.array([[cos(degree2),0.0,sin(degree2)],[0.0,1.0,0.0],[-sin(degree2),0.0,cos(degree2)]])
        rz=np.array([[cos(degree3),-sin(degree3),0.0],[sin(degree3),cos(degree3),0.0],[0.0,0.0,1.0]])
        # Euler rotation matrix
        roa1=rx.dot(ry).dot(rz)
        # Euler rotation-translation matrix, extend the Euler rotation matrix by one row and combine with translation data
        roa_tran=np.concatenate( [roa1,[[0.0,0.0,0.0]]],axis=0 )
        roa_tran=np.concatenate( [roa_tran,coord_point.T],axis=1 )
        




        dia_pack_point_list=[]

        for tempx in xrange:
            for tempy in yrange:
                for tempz in zrange:
                    # Determine if on an odd or even layer
                    if tempz%2==1:
                        # Determine if there is a close-packed point in the x,y directions
                        x_pack=tempx*dia_b-aldim
                        y_pack=tempy*dia_b-aldim
                        z_pack=tempz*dia_l-aldim
                        # Expression of particle vertex in world coordinate system after Euler rotation-translation
                        vertex1=np.array([x_pack,y_pack,z_pack,1.0])
                        vertex=roa_tran.dot(vertex1.T)

                        dia_pack_point_list.append([vertex[0],vertex[1],vertex[2]])
                                
                    else:
                        x_pack=tempx*dia_b-aldim+dia_l
                        y_pack=tempy*dia_b-aldim+dia_l
                        z_pack=tempz*dia_l-aldim
                        vertex1=np.array([x_pack,y_pack,z_pack,1.0])
                        vertex=roa_tran.dot(vertex1.T)

                        dia_pack_point_list.append([vertex[0],vertex[1],vertex[2]])
        # Sphere close-packed points
        xrange=range(0,int(2*aldim/sphere_d))
        yrange=range(0,int(2*aldim/sphere_y))
        zrange=range(0,int(2*aldim/sphere_h))

        sphere_pack_point_list=[]

        for tempx in xrange:
            for tempy in yrange:
                for tempz in zrange:
                    # Determine if on an odd or even layer
                    if tempz%2==1:
                        # Determine if on an odd or even row
                        if tempy%2==1:

                            x_pack=tempx*sphere_d-aldim
                            y_pack=tempy*sphere_y-aldim
                            z_pack=tempz*sphere_h-aldim

                            vertex1=np.array([x_pack,y_pack,z_pack,1.0])
                            vertex=roa_tran.dot(vertex1.T)

                            sphere_pack_point_list.append([vertex[0],vertex[1],vertex[2]])

                        else:
                            x_pack=tempx*sphere_d-aldim+sphere_r
                            y_pack=tempy*sphere_y-aldim
                            z_pack=tempz*sphere_h-aldim

                            vertex1=np.array([x_pack,y_pack,z_pack,1.0])
                            vertex=roa_tran.dot(vertex1.T)

                            sphere_pack_point_list.append([vertex[0],vertex[1],vertex[2]])
                                
                    else:
                        if tempy%2==1:
                            x_pack=tempx*sphere_d-aldim+sphere_r
                            y_pack=tempy*sphere_y-aldim+sphere_y/3
                            z_pack=tempz*sphere_h-aldim

                            vertex1=np.array([x_pack,y_pack,z_pack,1.0])
                            vertex=roa_tran.dot(vertex1.T)

                            sphere_pack_point_list.append([vertex[0],vertex[1],vertex[2]])

                        else:
                            x_pack=tempx*sphere_d-aldim+sphere_d
                            y_pack=tempy*sphere_y-aldim+sphere_y/3
                            z_pack=tempz*sphere_h-aldim

                            vertex1=np.array([x_pack,y_pack,z_pack,1.0])
                            vertex=roa_tran.dot(vertex1.T)

                            sphere_pack_point_list.append([vertex[0],vertex[1],vertex[2]])
        # Bind dense packing points to sub-spaces
        num_pack=0
        num_pack_sphere=0
        for tempx in list_daluan1:
            for tempy in list_daluan2:
                for tempz in list_daluan3:
                    
                    # Calculate bounding box data for the current space 
                    space_xmax=-aldim/2+(aldim/tempn)*tempx
                    space_xmin=-aldim/2+(aldim/tempn)*(tempx-1)
                    space_ymax=-aldim/2+(aldim/tempn)*tempy
                    space_ymin=-aldim/2+(aldim/tempn)*(tempy-1.0)
                    space_zmax=(aldim/tempn)*(tempz)
                    space_zmin=(aldim/tempn)*(tempz-1.0)

                    blank_list_dia[tempx][tempy][tempz].xyzmax=[space_xmax,space_ymax,space_zmax]
                    blank_list_dia[tempx][tempy][tempz].xyzmin=[space_xmin,space_ymin,space_zmin]

                    for pack_point in dia_pack_point_list:
                        x=pack_point[0]
                        y=pack_point[1]
                        z=pack_point[2]
                        if space_xmin<x<=space_xmax and space_ymin<y<=space_ymax and space_zmin<z<=space_zmax:
                            num_pack+=1
                            blank_list_dia[tempx][tempy][tempz].pack_diapoint_num+=1
                            blank_list_dia[tempx][tempy][tempz].pack_diapoint.append(pack_point)
                    for pack_point in sphere_pack_point_list:
                        x=pack_point[0]
                        y=pack_point[1]
                        z=pack_point[2]
                        if space_xmin<x<=space_xmax and space_ymin<y<=space_ymax and space_zmin<z<=space_zmax:
                            num_pack_sphere+=1
                            blank_list_dia[tempx][tempy][tempz].pack_spherepoint_num+=1
                            blank_list_dia[tempx][tempy][tempz].pack_spherepoint.append(pack_point)

############################
#Place particles
############################
         
        if dia_or_sphere_lable==0:
            point_pack_num=num_pack
        else:
            point_pack_num=num_pack_sphere

        tempratio=float(Dia_num)/point_pack_num
        Dia_num_i=0
        Dia_volume_sum=0
        for tempx in list_daluan1:
            
            if Dia_volume_sum>Enhancer_volume_design:
                break

            for tempy in list_daluan2:
                
                for tempz in list_daluan3:

                    # First determine whether it is hexoctahedral close-packing or spherical close-packing
                    if dia_or_sphere_lable==0:
                        packpoint_blank=blank_list_dia[tempx][tempy][tempz].pack_diapoint
                        packpoint_blank_num=blank_list_dia[tempx][tempy][tempz].pack_diapoint_num
                    else:
                        packpoint_blank=blank_list_dia[tempx][tempy][tempz].pack_spherepoint
                        packpoint_blank_num=blank_list_dia[tempx][tempy][tempz].pack_spherepoint_num
                    
                    # Iterate through close-packed points for particle placement
                    space_xmax=blank_list_dia[tempx][tempy][tempz].xyzmax[0]
                    space_xmin=blank_list_dia[tempx][tempy][tempz].xyzmin[0]
                    space_ymax=blank_list_dia[tempx][tempy][tempz].xyzmax[1]
                    space_ymin=blank_list_dia[tempx][tempy][tempz].xyzmin[1]
                    space_zmax=blank_list_dia[tempx][tempy][tempz].xyzmax[2]
                    space_zmin=blank_list_dia[tempx][tempy][tempz].xyzmin[2]

                    for tempi in range(0,packpoint_blank_num):

                        tempratio1=np.random.uniform(0,1)  #Generate a random number to determine whether a particle should be placed in the current sub-space
                        if tempratio1>tempratio:
                            continue              
                        #Move Dia_tran_X, Dia_tran_Y, Dia_tran_Z distances along X, Y, Z axes
                        # Dia_tran_X=np.random.uniform(space_xmin,space_xmax)
                        # Dia_tran_Y=np.random.uniform(space_ymin,space_ymax)
                        # Dia_tran_Z=np.random.uniform(space_zmin,space_zmax)

                        #Uncertainty modeling: generate particle sizes following Gaussian distribution
                        diamond_name='part-'+str(tempx)+str(tempy)+str(tempz)+str(tempi)
                        
                        ##diamond_generator(itemi,dia_l,model_name,diamond_name)
                        #Create diamond object, calculate equivalent particle size, enveloping sphere radius and volume, and save
                        diamonddata=data_of_diamond()
                        diamonddata.a=Dia_a[Dia_num_i]
                        diamonddata.partname=diamond_name

                        #Update object data based on a value
                        diamonddata.diamond_refresh()
                        ###
                        #Calculate current particle thermal conductivity based on nitrogen content, lambda and particle size, and generate mechanical property data saved to data_of_diamond object
                        ###
                        lambdai=np.random.normal(loc=diamond_lambda, scale=0.05)
                        f_zetai=lambdai*(1042.6*exp(-diamond_zeta/9573.4)+834.9*exp(-diamond_zeta/882.9)+550.5*exp(-diamond_zeta/192.2))
                        if shape_lable==0:
                            diamond_Ki=f_zetai-f_zetai*exp(-Dia_dxD[Dia_num_i]*1000/110.0)
                        elif shape_lable==1:
                            diamond_Ki=f_zetai-f_zetai*exp(-Dia_a[Dia_num_i]*1000/110.0)

                        diamonddata.thermal=diamond_Ki
                        diamonddata.Ei=np.random.uniform(1050E3,1210E3)
                        diamonddata.v=np.random.uniform(0.07,0.1)
                        diamonddata.sigmas=np.random.uniform(6E3,8E3)
                        diamonddata.sigmat=np.random.uniform(30E3,36E3)

                        #Due to the introduction of uncertainty modeling, the particle volume may easily exceed the set limit
                        #Calculate the added particle volume sum here
                        Dia_volume_sum=Dia_volume_sum+diamonddata.V
                        
                        #Generate random numbers as rotation and translation data; as volume fraction increases, positions approach close-packing points and rotation angles approach roa_x, roa_y, roa_z
                        Dia_tran_X=(diamond_ratio*0.01+0.2)*packpoint_blank[tempi][0]+(1-diamond_ratio*0.01-0.2)*np.random.uniform(space_xmin,space_xmax)

                        Dia_roa_X=(diamond_ratio*0.01+0.3)*x_roa+(1-diamond_ratio*0.01-0.3)*np.random.uniform(0,1)*45

                        Dia_tran_Y=(diamond_ratio*0.01+0.2)*packpoint_blank[tempi][1]+(1-diamond_ratio*0.01-0.2)*np.random.uniform(space_ymin,space_ymax)

                        Dia_roa_Y=(diamond_ratio*0.01+0.3)*y_roa+(1-diamond_ratio*0.01-0.3)*np.random.uniform(0,1)*45

                        Dia_tran_Z=(diamond_ratio*0.01+0.2)*packpoint_blank[tempi][2]+(1-diamond_ratio*0.01-0.2)*np.random.uniform(space_zmin,space_zmax)

                        Dia_roa_Z=(diamond_ratio*0.01+0.3)*z_roa+(1-diamond_ratio*0.01-0.3)*np.random.uniform(0,1)*45
                        
                        # Dia_tran_X=packpoint_blank[tempi][0]

                        # Dia_roa_X=x_roa

                        # Dia_tran_Y=packpoint_blank[tempi][1]

                        # Dia_roa_Y=y_roa

                        # Dia_tran_Z=packpoint_blank[tempi][2]

                        # Dia_roa_Z=z_roa

                        # Dia_tran_X=packpoint_blank[tempi][0]
                        # Dia_tran_Y=packpoint_blank[tempi][1]
                        # Dia_tran_Z=packpoint_blank[tempi][2]


                        
                        # Dia_roa_X=(diamond_ratio*0.01+0.3)*x_roa+(1-diamond_ratio*0.01-0.3)*np.random.uniform(-1,1)*45
                        # Dia_roa_Y=(diamond_ratio*0.01+0.3)*y_roa+(1-diamond_ratio*0.01-0.3)*np.random.uniform(-1,1)*45
                        # Dia_roa_Z=(diamond_ratio*0.01+0.3)*z_roa+(1-diamond_ratio*0.01-0.3)*np.random.uniform(-1,1)*45
                        # Dia_roa_X=x_roa
                        # Dia_roa_Y=y_roa
                        # Dia_roa_Z=z_roa
                        
                        
                        blank_list_dia[tempx][tempy][tempz].diamond_points.append([Dia_tran_X,Dia_tran_Y,Dia_tran_Z,Dia_roa_X,Dia_roa_Y,Dia_roa_Z])
                        blank_list_dia[tempx][tempy][tempz].diamond_datas.append(diamonddata)
                        blank_list_dia[tempx][tempy][tempz].diamond_lable+=1
                        Dia_num_i+=1
                    
                    
                    
                    # Generate twin space points for corresponding auxiliary spaces
                    # First extract the minimum xyz data of the current space
                    xmin=blank_list_dia[tempx][tempy][tempz].xyzmin[0]
                    ymin=blank_list_dia[tempx][tempy][tempz].xyzmin[1]
                    zmin=blank_list_dia[tempx][tempy][tempz].xyzmin[2]
                    auxiliary_space_list=blank_list_dia[tempx][tempy][tempz].auxiliary_space
                    diamond_points_list=blank_list_dia[tempx][tempy][tempz].diamond_points
                    
                    # Traverse virtual spaces to generate twin particles
                    for auxiliary_space in auxiliary_space_list:
                        tempx_x=auxiliary_space[0]
                        tempy_y=auxiliary_space[1]
                        tempz_z=auxiliary_space[2]
                        # Minimum xyz data of the virtual space
                        x_xmin=blank_list_dia[tempx_x][tempy_y][tempz_z].xyzmin[0]
                        y_ymin=blank_list_dia[tempx_x][tempy_y][tempz_z].xyzmin[1]
                        z_zmin=blank_list_dia[tempx_x][tempy_y][tempz_z].xyzmin[2]

                        point_list=[]
                        for temp_point in diamond_points_list:
                            x=temp_point[0]
                            y=temp_point[1]
                            z=temp_point[2]
                            xr=temp_point[3]
                            yr=temp_point[4]
                            zr=temp_point[5]

                            x_x=x-xmin+x_xmin
                            y_y=y-ymin+y_ymin
                            z_z=z-zmin+z_zmin

                            point_list.append([x_x,y_y,z_z,xr,yr,zr])

                        blank_list_dia[tempx_x][tempy_y][tempz_z].diamond_points=point_list[:]
                        blank_list_dia[tempx_x][tempy_y][tempz_z].diamond_lable=blank_list_dia[tempx][tempy][tempz].diamond_lable
                        blank_list_dia[tempx_x][tempy_y][tempz_z].diamond_datas=blank_list_dia[tempx][tempy][tempz].diamond_datas[:]
        if Dia_volume_sum==0:
            # If no diamond particle inside, place one at a random position
            # Uncertainty modeling: generate particle size following a Gaussian distribution
            diamond_name='part-'+str(tempx)+str(tempy)+str(tempz)+str(tempi)
            
            ##diamond_generator(itemi,dia_l,model_name,diamond_name)
            # Create diamond object, compute equivalent diameter, enveloping sphere radius and volume, then save
            diamonddata=data_of_diamond()
            diamonddata.a=Dia_a[Dia_num_i]
            diamonddata.partname=diamond_name

            # Update object data according to the a value
            diamonddata.diamond_refresh()
            ###
            # Compute current particle thermal conductivity based on nitrogen content, lambda and particle size,
            # and generate mechanical property data saved into the data_of_diamond object
            ###
            lambdai=np.random.normal(loc=diamond_lambda, scale=0.05)
            f_zetai=lambdai*(1042.6*exp(-diamond_zeta/9573.4)+834.9*exp(-diamond_zeta/882.9)+550.5*exp(-diamond_zeta/192.2))
            if shape_lable==0:
                diamond_Ki=f_zetai-f_zetai*exp(-Dia_dxD[Dia_num_i]*1000/110.0)
            elif shape_lable==1:
                diamond_Ki=f_zetai-f_zetai*exp(-Dia_a[Dia_num_i]*1000/110.0)

            diamonddata.thermal=diamond_Ki
            diamonddata.Ei=np.random.uniform(1050E3,1210E3)
            diamonddata.v=np.random.uniform(0.07,0.1)
            diamonddata.sigmas=np.random.uniform(6E3,8E3)
            diamonddata.sigmat=np.random.uniform(30E3,36E3)

            # With uncertainty modeling, particle volume may easily exceed the target too much;
            # accumulate the volume of the added particle here
            Dia_volume_sum=Dia_volume_sum+diamonddata.V
            
            # Generate random numbers for rotation and translation; as volume fraction increases,
            # positions approach close-packing points, and rotation angles approach roa_x, roa_y, roa_z
            Dia_tran_X=(diamond_ratio*0.01+0.2)*packpoint_blank[tempi][0]+(1-diamond_ratio*0.01-0.2)*np.random.uniform(space_xmin,space_xmax)

            Dia_roa_X=(diamond_ratio*0.01+0.3)*x_roa+(1-diamond_ratio*0.01-0.3)*np.random.uniform(0,1)*45

            Dia_tran_Y=(diamond_ratio*0.01+0.2)*packpoint_blank[tempi][1]+(1-diamond_ratio*0.01-0.2)*np.random.uniform(space_ymin,space_ymax)

            Dia_roa_Y=(diamond_ratio*0.01+0.3)*y_roa+(1-diamond_ratio*0.01-0.3)*np.random.uniform(0,1)*45

            Dia_tran_Z=(diamond_ratio*0.01+0.2)*packpoint_blank[tempi][2]+(1-diamond_ratio*0.01-0.2)*np.random.uniform(space_zmin,space_zmax)

            Dia_roa_Z=(diamond_ratio*0.01+0.3)*z_roa+(1-diamond_ratio*0.01-0.3)*np.random.uniform(0,1)*45
            
            # Dia_tran_X=packpoint_blank[tempi][0]

            # Dia_roa_X=x_roa

            # Dia_tran_Y=packpoint_blank[tempi][1]

            # Dia_roa_Y=y_roa

            # Dia_tran_Z=packpoint_blank[tempi][2]

            # Dia_roa_Z=z_roa

            # Dia_tran_X=packpoint_blank[tempi][0]
            # Dia_tran_Y=packpoint_blank[tempi][1]
            # Dia_tran_Z=packpoint_blank[tempi][2]


            
            # Dia_roa_X=(diamond_ratio*0.01+0.3)*x_roa+(1-diamond_ratio*0.01-0.3)*np.random.uniform(-1,1)*45
            # Dia_roa_Y=(diamond_ratio*0.01+0.3)*y_roa+(1-diamond_ratio*0.01-0.3)*np.random.uniform(-1,1)*45
            # Dia_roa_Z=(diamond_ratio*0.01+0.3)*z_roa+(1-diamond_ratio*0.01-0.3)*np.random.uniform(-1,1)*45
            # Dia_roa_X=x_roa
            # Dia_roa_Y=y_roa
            # Dia_roa_Z=z_roa
            
            
            blank_list_dia[tempx][tempy][tempz].diamond_points.append([Dia_tran_X,Dia_tran_Y,Dia_tran_Z,Dia_roa_X,Dia_roa_Y,Dia_roa_Z])
            blank_list_dia[tempx][tempy][tempz].diamond_datas.append(diamonddata)
            blank_list_dia[tempx][tempy][tempz].diamond_lable+=1
            Dia_num_i+=1
        
        
        
            # Generate twin space points corresponding to the auxiliary space
            # First, extract the minimum xyz data of the current space
            xmin=blank_list_dia[tempx][tempy][tempz].xyzmin[0]
            ymin=blank_list_dia[tempx][tempy][tempz].xyzmin[1]
            zmin=blank_list_dia[tempx][tempy][tempz].xyzmin[2]
            auxiliary_space_list=blank_list_dia[tempx][tempy][tempz].auxiliary_space
            diamond_points_list=blank_list_dia[tempx][tempy][tempz].diamond_points
            
            # Traverse virtual space and generate twin particles
            for auxiliary_space in auxiliary_space_list:
                tempx_x=auxiliary_space[0]
                tempy_y=auxiliary_space[1]
                tempz_z=auxiliary_space[2]
                # Minimum xyz data of the virtual space
                x_xmin=blank_list_dia[tempx_x][tempy_y][tempz_z].xyzmin[0]
                y_ymin=blank_list_dia[tempx_x][tempy_y][tempz_z].xyzmin[1]
                z_zmin=blank_list_dia[tempx_x][tempy_y][tempz_z].xyzmin[2]

                point_list=[]
                for temp_point in diamond_points_list:
                    x=temp_point[0]
                    y=temp_point[1]
                    z=temp_point[2]
                    xr=temp_point[3]
                    yr=temp_point[4]
                    zr=temp_point[5]

                    x_x=x-xmin+x_xmin
                    y_y=y-ymin+y_ymin
                    z_z=z-zmin+z_zmin

                    point_list.append([x_x,y_y,z_z,xr,yr,zr])

                blank_list_dia[tempx_x][tempy_y][tempz_z].diamond_points=point_list[:]
                blank_list_dia[tempx_x][tempy_y][tempz_z].diamond_lable=blank_list_dia[tempx][tempy][tempz].diamond_lable
                blank_list_dia[tempx_x][tempy_y][tempz_z].diamond_datas=blank_list_dia[tempx][tempy][tempz].diamond_datas[:]
                   

###########################################################
# Iterative displacement adjustment
###########################################################       
        diedai_biaoshi=True
        diedia_list1=range(1,Al_cut+1)
        diedia_list2=range(1,Al_cut+1)
        diedia_list3=range(1,Al_cut+1)
        # Iteration flag, used to control iteration stop
        diedailable=True
        loop_num=0
        while diedailable and loop_num<1000:
        # for diedai in range(1,diedai_num+1):
            diedailable=False
            random.shuffle(diedia_list1)
            random.shuffle(diedia_list2)
            random.shuffle(diedia_list3)
            loop_num=loop_num+1
            for tempx in diedia_list1:
                for tempy in diedia_list2:
                    for tempz in diedia_list3:     
                        if blank_list_dia[tempx][tempy][tempz].diamond_lable==0:
                            continue
                        diamond_points_list=blank_list_dia[tempx][tempy][tempz].diamond_points
                        diamond_data_list=blank_list_dia[tempx][tempy][tempz].diamond_datas
                        space_xmax=blank_list_dia[tempx][tempy][tempz].xyzmax[0]
                        space_xmin=blank_list_dia[tempx][tempy][tempz].xyzmin[0]
                        space_ymax=blank_list_dia[tempx][tempy][tempz].xyzmax[1]
                        space_ymin=blank_list_dia[tempx][tempy][tempz].xyzmin[1]
                        space_zmax=blank_list_dia[tempx][tempy][tempz].xyzmax[2]
                        space_zmin=blank_list_dia[tempx][tempy][tempz].xyzmin[2]

                        
                        for point1i in range(0,len(diamond_points_list)):
                            point1=diamond_points_list[point1i]
                            diamond1= diamond_data_list[point1i]
                            Dia_tran_X=point1[0]
                            Dia_tran_Y=point1[1]
                            Dia_tran_Z=point1[2]
                            Dia_roa_X=point1[3]
                            Dia_roa_Y=point1[4]
                            Dia_roa_Z=point1[5]

                            # Check for interference with diamond particles in the current space and surrounding small spaces

                            # Bounding sphere radius
                            if shape_lable==0:
                                dia_Bounding_R1=diamond1.shpereR
                            elif shape_lable==1:
                                dia_Bounding_R1=diamond1.a/2                            
                            # Set minimum spacing
                            dia_min_distance=0.00001

                            # Intercept
                            tempc=diamond1.a/sqrt(2)
                            dia_l0=diamond1.l
                            
                            # Traverse surrounding 27 small spaces to check for interference; if exists, adjust particle position via weighted displacement
                            #blank_list_dia[tempxx+1][tempyy+1][tempzz+1]=[0,0,0,0]
                            # Initialize empty lists to store interference particle displacement vectors and magnitudes
                            Dia_tran0=[]# Store info of spaces without particles
                            Dia_tran1=[]# Store info of spaces with particles and interference
                            Dia_tran2=[]## Store info of spaces with particles but without interference
                            dia_distance_sum0=0
                            dia_distance_sum1=0
                            dia_distance_sum2=0
                            
                            
                            # Establish spatial coordinate transformation to determine if diamond points interfere with diamond particles
                            # First, calculate the three-axis unit vectors and origin coordinates of the particle in world coordinate system after Euler rotation and translation
                            coord_xl=np.array([[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]])
                            coord_tran=np.array([[1.0,0.0,0.0,-Dia_tran_X],[0.0,1.0,0.0,-Dia_tran_Y],[0.0,0.0,1.0,-Dia_tran_Z],[0.0,0.0,0.0,1.0]])
                            degree1=math.radians(Dia_roa_X)
                            degree2=math.radians(Dia_roa_Y)
                            degree3=math.radians(Dia_roa_Z)
                            rx=np.array([[1.0,0.0,0.0],[0.0,cos(degree1),-sin(degree1)],[0.0,sin(degree1),cos(degree1)]])
                            ry=np.array([[cos(degree2),0.0,sin(degree2)],[0.0,1.0,0.0],[-sin(degree2),0.0,cos(degree2)]])
                            rz=np.array([[cos(degree3),-sin(degree3),0.0],[sin(degree3),cos(degree3),0.0],[0.0,0.0,1.0]])
                            # Euler rotation matrix
                            roa=rx.dot(ry).dot(rz)
                            # Transformation matrix from world coordinate system to particle local coordinate system (4x4 homogeneous matrix)
                            coord_roa=(roa.dot(coord_xl.T)).T
                            coord_roa=np.concatenate( [coord_roa,[[0.0,0.0,0.0]]],axis=0 )
                            coord_roa=np.concatenate( [coord_roa,[[0.0],[0.0],[0.0],[1.0]]],axis=1 )

                            # Due to uncertain diamond particle size modeling, the discrimination space may exceed the 3x3x3 spatial range; bypass it here
                            for tempi in range(tempx-2,tempx+3):
                                for tempj in range(tempy-2,tempy+3):
                                    for tempk in range(tempz-2,tempz+3):
                            # for tempi in range(0,Al_cut+2):
                            #     for tempj in range(0,Al_cut+2):
                            #         for tempk in range(0,Al_cut+2):
                                        
                                        diamond_points_list1=blank_list_dia[tempi][tempj][tempk].diamond_points
                                        diamond_data_list1=blank_list_dia[tempi][tempj][tempk].diamond_datas

                                        


                                        for point2i in range(0,len(diamond_points_list1)):
                                            ganshebiaoshifu=0
                                            point2=diamond_points_list1[point2i]
                                            diamond2= diamond_data_list1[point2i]
                                            Dia_tran_X1=point2[0]
                                            Dia_tran_Y1=point2[1]
                                            Dia_tran_Z1=point2[2]
                                            Dia_roa_X1=point2[3]
                                            Dia_roa_Y1=point2[4]
                                            Dia_roa_Z1=point2[5]

                                            if Dia_tran_X1==Dia_tran_X and Dia_tran_Y1==Dia_tran_Y and Dia_tran_Z1==Dia_tran_Z:
                                                continue

                                            # Check if interference exists
                                            dia_distance=sqrt((Dia_tran_X1-Dia_tran_X)*(Dia_tran_X1-Dia_tran_X)+(Dia_tran_Y1-Dia_tran_Y)*(Dia_tran_Y1-Dia_tran_Y)+
                                                                (Dia_tran_Z1-Dia_tran_Z)*(Dia_tran_Z1-Dia_tran_Z))

                                            if shape_lable==0:
                                                dia_Bounding_R2=diamond2.shpereR
                                                diamond_vertex=np.array(diamond2.points)
                                                dia_Bounding_R=dia_Bounding_R1+dia_Bounding_R2+dia_min_distance

                                                if dia_distance<dia_Bounding_R:
                                                    # Calculate the world coordinates of the 24 vertices of the judgment particle after Euler rotation and translation
                                                    coord_point=np.array([[Dia_tran_X1,Dia_tran_Y1,Dia_tran_Z1,1.0]])
                                                    degree1=math.radians(Dia_roa_X1)
                                                    degree2=math.radians(Dia_roa_Y1)
                                                    degree3=math.radians(Dia_roa_Z1)
                                                    rx=np.array([[1.0,0.0,0.0],[0.0,cos(degree1),-sin(degree1)],[0.0,sin(degree1),cos(degree1)]])
                                                    ry=np.array([[cos(degree2),0.0,sin(degree2)],[0.0,1.0,0.0],[-sin(degree2),0.0,cos(degree2)]])
                                                    rz=np.array([[cos(degree3),-sin(degree3),0.0],[sin(degree3),cos(degree3),0.0],[0.0,0.0,1.0]])
                                                    # Euler rotation matrix
                                                    roa1=rx.dot(ry).dot(rz)
                                                    # Euler rotation-translation matrix
                                                    roa_tran=np.concatenate( [roa1,[[0.0,0.0,0.0]]],axis=0 )
                                                    roa_tran=np.concatenate( [roa_tran,coord_point.T],axis=1 )
                                                    # Expression of particle vertices in world coordinate system after Euler rotation and translation
                                                    diamond_vertex1=roa_tran.dot(diamond_vertex.T)

                                                    # Transform to particle local coordinate system
                                                    diamond_vertex2=coord_roa.dot(coord_tran.dot(diamond_vertex1))
                                                    diamond_vertex2=diamond_vertex2.T
                                                    
                                                    for vertexi in diamond_vertex2:
                                                        # Check interference by whether within bounding box and x1/(±a/√2)+x2/(±a/√2)+x3/(±a/√2) < 1
                                                        if -dia_l0<vertexi[0]<dia_l0 and -dia_l0<vertexi[1]<dia_l0 and -dia_l0<vertexi[2]<dia_l0:
                                                            
                                                            if (vertexi[0]/tempc+vertexi[1]/tempc+vertexi[2]/tempc<=1 and vertexi[0]/tempc+vertexi[1]/tempc-vertexi[2]/tempc<=1 and
                                                                vertexi[0]/tempc-vertexi[1]/tempc+vertexi[2]/tempc<=1 and vertexi[0]/tempc-vertexi[1]/tempc-vertexi[2]/tempc<=1 and
                                                                -vertexi[0]/tempc+vertexi[1]/tempc+vertexi[2]/tempc<=1 and -vertexi[0]/tempc+vertexi[1]/tempc-vertexi[2]/tempc<=1 and
                                                                -vertexi[0]/tempc-vertexi[1]/tempc+vertexi[2]/tempc<=1 and -vertexi[0]/tempc-vertexi[1]/tempc-vertexi[2]/tempc<=1):
                                                                #vert=vertexi[:]/0
                                                                ganshebiaoshifu=1
                                                                break
                                                    # Indicates the point is not inside the other diamond; check if the line segment between the two nearest points to the origin intersects the diamond
                                                    point_list=[]
                                                    if ganshebiaoshifu==0:
                                                        # Calculate intersection of diamond edges with pyramid planes, then check if the intersection is within the bounding box
                                                        XYZ=[tempc,-tempc]
                                                        diamond2_line=diamond2.lines

                                                        for line in diamond2_line:
                                                            vert1=diamond_vertex2[line[0]]
                                                            vert2=diamond_vertex2[line[1]]
                                                            x1=vert1[0]
                                                            y1=vert1[1]
                                                            z1=vert1[2]
                                                            x2=vert2[0]
                                                            y2=vert2[1]
                                                            z2=vert2[2]
                                                            u1=x1-x2
                                                            v1=y1-y2
                                                            w1=z1-z2
                                                            if x1**2+y1**2+z1**2>1.5*dia_Bounding_R1**2 and x2**2+y2**2+z2**2>1.5*dia_Bounding_R1**2:
                                                                continue
                                                            for a in XYZ:
                                                                for b in XYZ:
                                                                    for c in XYZ:
                                                                        if (u1/a+v1/b+w1/c)==0:
                                                                            continue
                                                                        t=(1-(x2/a+y2/b+z2/c))/(u1/a+v1/b+w1/c)
                                                                        x=x2+u1*t
                                                                        y=y2+v1*t
                                                                        z=z2+w1*t
                                                                        # Check if in the same quadrant
                                                                        if (x*a>=0 and y*b>=0 and z*c>=0):
                                                                            l1=sqrt((x1-x)*(x1-x)+(y1-y)*(y1-y)+(z1-z)*(z1-z))
                                                                            l2=sqrt((x2-x)*(x2-x)+(y2-y)*(y2-y)+(z2-z)*(z2-z))
                                                                            l3=sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2)+(z1-z2)*(z1-z2))
                                                                            # Check if the intersection point is within the line segment
                                                                            
                                                                            if abs(l1+l2-l3)<0.000000001:
                                                                                point=[x,y,z]
                                                                                point_list.append(point)
                                                                                
                                                    
                                                        for vertexi in point_list:
                                                            if -dia_l0<vertexi[0]<dia_l0 and -dia_l0<vertexi[1]<dia_l0 and -dia_l0<vertexi[2]<dia_l0:
                                                                ganshebiaoshifu=1
                                                                
                                                                break
                                                    if ganshebiaoshifu==1:
                                                        temptran=[(Dia_tran_X-Dia_tran_X1)/dia_distance,
                                                                  (Dia_tran_Y-Dia_tran_Y1)/dia_distance,
                                                                  (Dia_tran_Z-Dia_tran_Z1)/dia_distance,
                                                                  dia_Bounding_R-dia_distance]
                                                
                                                        # temptran=[(Dia_tran_X-blank_list_dia[tempi][tempj][tempk][0])/dia_distance,
                                                        #         (Dia_tran_Y-blank_list_dia[tempi][tempj][tempk][1])/dia_distance,
                                                        #         (Dia_tran_Z-blank_list_dia[tempi][tempj][tempk][2])/dia_distance,
                                                        #         dia_Bounding_R-dia_distance]
                                                        dia_distance_sum1=dia_distance_sum1+temptran[3]
                                                        Dia_tran1.append(temptran)

                                            elif shape_lable==1:
                                                dia_Bounding_R2=diamond2.a/2   
                                                dia_Bounding_R=dia_Bounding_R1+dia_Bounding_R2+dia_min_distance

                                                if dia_distance<dia_Bounding_R:
                                                    ganshebiaoshifu=1
                                                    temptran=[(Dia_tran_X-Dia_tran_X1)/dia_distance,
                                                              (Dia_tran_Y-Dia_tran_Y1)/dia_distance,
                                                              (Dia_tran_Z-Dia_tran_Z1)/dia_distance,
                                                              dia_Bounding_R-dia_distance]
                                                    dia_distance_sum1=dia_distance_sum1+temptran[3]
                                                    Dia_tran1.append(temptran)
                                                    
                            # Initialize displacement direction vector
                            Dia_tranvector_X=0
                            Dia_tranvector_Y=0
                            Dia_tranvector_Z=0
                            # Calculate displacement vector
                            
                            for tempi in range(len(Dia_tran1)):
                                Dia_tranvector_X=Dia_tranvector_X+Dia_tran1[tempi][0]*Dia_tran1[tempi][3]/(dia_distance_sum1+dia_distance_sum0)
                                Dia_tranvector_Y=Dia_tranvector_Y+Dia_tran1[tempi][1]*Dia_tran1[tempi][3]/(dia_distance_sum1+dia_distance_sum0)
                                Dia_tranvector_Z=Dia_tranvector_Z+Dia_tran1[tempi][2]*Dia_tran1[tempi][3]/(dia_distance_sum1+dia_distance_sum0)
                            if len(Dia_tran1)>0:
                                diedailable=True
                            #for tempi in range(len(Dia_tran0)):
                                #Dia_tranvector_X=Dia_tranvector_X-Dia_tran0[tempi][0]*Dia_tran0[tempi][3]/(dia_distance_sum1+dia_distance_sum0)
                                #Dia_tranvector_Y=Dia_tranvector_Y-Dia_tran0[tempi][1]*Dia_tran0[tempi][3]/(dia_distance_sum1+dia_distance_sum0)
                                #Dia_tranvector_Z=Dia_tranvector_Z-Dia_tran0[tempi][2]*Dia_tran0[tempi][3]/(dia_distance_sum1+dia_distance_sum0)
                            # Determine if displacement exceeds boundary; if so, stop iteration
                            #Dia_tranvector_X=0
                            #Dia_tranvector_Y=0
                            #Dia_tranvector_Z=0
                            #print(str(Dia_tran_X+Dia_tranvector_X*dia_distance_sum1)+','+str(Dia_tran_Y+Dia_tranvector_Y*dia_distance_sum1)+','+str(Dia_tran_Z+Dia_tranvector_Z*dia_distance_sum1))
                            #dia_distance_sum1=dia_distance_sum1+dia_distance_sum0
                            if shape_lable==0:
                                xishu=0.38
                                # Calculate updated coordinates after particle displacement
                                xnew=Dia_tran_X+Dia_tranvector_X*(dia_distance_sum1+0.0002)*xishu
                                ynew=Dia_tran_Y+Dia_tranvector_Y*(dia_distance_sum1+0.0002)*xishu
                                znew=Dia_tran_Z+Dia_tranvector_Z*(dia_distance_sum1+0.0002)*xishu
                            elif shape_lable==1:
                                xishu=0.5
                                # Calculate updated coordinates after particle displacement
                                xnew=Dia_tran_X+Dia_tranvector_X*(dia_distance_sum1)*xishu
                                ynew=Dia_tran_Y+Dia_tranvector_Y*(dia_distance_sum1)*xishu
                                znew=Dia_tran_Z+Dia_tranvector_Z*(dia_distance_sum1)*xishu

                            # Check if displacement exceeds boundary; if so, add compensation to prevent edge particles from moving out of bounds
                            # Also to avoid extremely small particles from cutting, add constraints to prevent being too far from the boundary
                            if xnew>aldim/2:
                                xnew=aldim/2-0.0000001
                            
                            elif xnew<-aldim/2:
                                xnew=-aldim/2+0.0000001
                            
                            if ynew>aldim/2:
                                ynew=aldim/2-0.0000001
                            
                            elif ynew<-aldim/2:
                                ynew=-aldim/2+0.0000001
                            
                            if znew>aldim:# or (aldim-dia_Bounding_R1)<znew<(aldim-dia_Bounding_R1/2):
                                znew=aldim-0.0000001

                            elif znew<0:# or (dia_Bounding_R1/2)<znew<(dia_Bounding_R1):
                                znew=0.0000001
                            
                            # if (aldim/2-dia_Bounding_R1)<xnew<(aldim/2-5*dia_Bounding_R1/6):
                            #     xnew=aldim/2-dia_Bounding_R1
                            # elif (-aldim/2+5*dia_Bounding_R1/6)<xnew<(-aldim/2+dia_Bounding_R1):
                            #     xnew=-aldim/2+dia_Bounding_R1
                            # if (aldim/2-dia_Bounding_R1)<ynew<(aldim/2-5*dia_Bounding_R1/6):
                            #     ynew=aldim/2-dia_Bounding_R1
                            # elif (-aldim/2+5*dia_Bounding_R1/6)<ynew<(-aldim/2+dia_Bounding_R1):
                            #     ynew=-aldim/2+dia_Bounding_R1
                            # if (aldim-dia_Bounding_R1)<znew<(aldim-5*dia_Bounding_R1/6):
                            #     znew=aldim-dia_Bounding_R1
                            # elif (5*dia_Bounding_R1/6)<znew<(dia_Bounding_R1):
                            #     znew=dia_Bounding_R1
                            # When periodic boundary conditions are used, particles located between
                            # the radius distance and 2/3 radius position may become extremely small
                            # after trimming, which can cause meshing failures when setting up periodic boundaries.
                            if Period_boundary_lable==1:

                                if (aldim/2-dia_Bounding_R1)<xnew<(aldim/2-3*dia_Bounding_R1/4):
                                    xnew=aldim/2-dia_Bounding_R1
                                elif (-aldim/2+3*dia_Bounding_R1/4)<xnew<(-aldim/2+dia_Bounding_R1):
                                    xnew=-aldim/2+dia_Bounding_R1
                                if (aldim/2-dia_Bounding_R1)<ynew<(aldim/2-3*dia_Bounding_R1/4):
                                    ynew=aldim/2-dia_Bounding_R1
                                elif (-aldim/2+3*dia_Bounding_R1/4)<ynew<(-aldim/2+dia_Bounding_R1):
                                    ynew=-aldim/2+dia_Bounding_R1
                                if (aldim-dia_Bounding_R1)<znew<(aldim-3*dia_Bounding_R1/4):
                                    znew=aldim-dia_Bounding_R1
                                elif (3*dia_Bounding_R1/4)<znew<(dia_Bounding_R1):
                                    znew=dia_Bounding_R1

                            tempxx=int((xnew+0.5*aldim)/(aldim/tempn))+1
                            tempyy=int((ynew+0.5*aldim)/(aldim/tempn))+1
                            tempzz=int(znew/(aldim/tempn))+1
                            # Save particle displacement data
                            blank_list_dia1[tempxx][tempyy][tempzz].diamond_points.append([xnew,ynew,znew,Dia_roa_X,Dia_roa_Y,Dia_roa_Z])
                            blank_list_dia1[tempxx][tempyy][tempzz].diamond_lable+=1
                            blank_list_dia1[tempxx][tempyy][tempzz].diamond_datas.append(diamond1)

                            
            for tempx in range(1,Al_cut+1):
                for tempy in range(1,Al_cut+1):
                    for tempz in range(1,Al_cut+1):
                        blank_list_dia[tempx][tempy][tempz].diamond_points=blank_list_dia1[tempx][tempy][tempz].diamond_points[:]
                        blank_list_dia[tempx][tempy][tempz].diamond_lable=blank_list_dia1[tempx][tempy][tempz].diamond_lable
                        blank_list_dia[tempx][tempy][tempz].diamond_datas=blank_list_dia1[tempx][tempy][tempz].diamond_datas[:]
                        blank_list_dia1[tempx][tempy][tempz].diamond_points=[]
                        blank_list_dia1[tempx][tempy][tempz].diamond_lable=0
                        blank_list_dia1[tempx][tempy][tempz].diamond_datas=[]


                        # Generate twin space points for the corresponding auxiliary space
                        # First, extract the minimum xyz data of the current space
                        xmin=blank_list_dia[tempx][tempy][tempz].xyzmin[0]
                        ymin=blank_list_dia[tempx][tempy][tempz].xyzmin[1]
                        zmin=blank_list_dia[tempx][tempy][tempz].xyzmin[2]
                        auxiliary_space_list=blank_list_dia[tempx][tempy][tempz].auxiliary_space
                        diamond_points_list=blank_list_dia[tempx][tempy][tempz].diamond_points
                        
                        # Traverse virtual space and generate twin particles
                        for auxiliary_space in auxiliary_space_list:
                            tempx_x=auxiliary_space[0]
                            tempy_y=auxiliary_space[1]
                            tempz_z=auxiliary_space[2]
                            # Minimum xyz data of the virtual space
                            x_xmin=blank_list_dia[tempx_x][tempy_y][tempz_z].xyzmin[0]
                            y_ymin=blank_list_dia[tempx_x][tempy_y][tempz_z].xyzmin[1]
                            z_zmin=blank_list_dia[tempx_x][tempy_y][tempz_z].xyzmin[2]

                            point_list=[]
                            for temp_point in diamond_points_list:
                                x=temp_point[0]
                                y=temp_point[1]
                                z=temp_point[2]
                                xr=temp_point[3]
                                yr=temp_point[4]
                                zr=temp_point[5]

                                x_x=x-xmin+x_xmin
                                y_y=y-ymin+y_ymin
                                z_z=z-zmin+z_zmin

                                point_list.append([x_x,y_y,z_z,xr,yr,zr])

                            blank_list_dia[tempx_x][tempy_y][tempz_z].diamond_points=point_list[:]
                            blank_list_dia[tempx_x][tempy_y][tempz_z].diamond_lable=blank_list_dia[tempx][tempy][tempz].diamond_lable
                            blank_list_dia[tempx_x][tempy_y][tempz_z].diamond_datas=blank_list_dia[tempx][tempy][tempz].diamond_datas[:]    
            

# # #############################################
# Resolve the issue of interfering particles remaining after reaching the maximum iteration count
# # ##############################################
        random.shuffle(diedia_list1)
        random.shuffle(diedia_list2)
        random.shuffle(diedia_list3)
        for tempx in diedia_list1:
            for tempy in diedia_list2:
                for tempz in diedia_list3:     
                    if blank_list_dia[tempx][tempy][tempz].diamond_lable==0:
                        continue
                    diamond_points_list=blank_list_dia[tempx][tempy][tempz].diamond_points[:]
                    diamond_data_list=blank_list_dia[tempx][tempy][tempz].diamond_datas[:]
                    space_xmax=blank_list_dia[tempx][tempy][tempz].xyzmax[0]
                    space_xmin=blank_list_dia[tempx][tempy][tempz].xyzmin[0]
                    space_ymax=blank_list_dia[tempx][tempy][tempz].xyzmax[1]
                    space_ymin=blank_list_dia[tempx][tempy][tempz].xyzmin[1]
                    space_zmax=blank_list_dia[tempx][tempy][tempz].xyzmax[2]
                    space_zmin=blank_list_dia[tempx][tempy][tempz].xyzmin[2]

                    pointi=0

                    for point1i in range(0,len(diamond_points_list)):
                        point1=diamond_points_list[point1i]
                        diamond1= diamond_data_list[point1i]
                        Dia_tran_X=point1[0]
                        Dia_tran_Y=point1[1]
                        Dia_tran_Z=point1[2]
                        Dia_roa_X=point1[3]
                        Dia_roa_Y=point1[4]
                        Dia_roa_Z=point1[5]

                        # Check for interference with diamond particles in the current space and surrounding 9 small spaces
                        # Bounding sphere radius
                        if shape_lable==0:
                            dia_Bounding_R1=diamond1.shpereR
                        elif shape_lable==1:
                            dia_Bounding_R1=diamond1.a/2                            
                        # Set minimum clearance distance
                        dia_min_distance=0.00001

                        # Half-diagonal (intercept)
                        tempc=diamond1.a/sqrt(2)
                        dia_l0=diamond1.l
                        
                        # Iterate through 27 neighboring small spaces, check for interference;
                        # if interference exists, adjust particle position via weighted displacement
                        #blank_list_dia[tempxx+1][tempyy+1][tempzz+1]=[0,0,0,0]
                        # Initialize empty lists to store displacement vectors and magnitudes for interfering particles
                        Dia_tran0=[]  # Store info of spaces with no particles
                        Dia_tran1=[]  # Store info of spaces with particles that interfere
                        Dia_tran2=[]  # Store info of spaces with particles but no interference
                        dia_distance_sum0=0
                        dia_distance_sum1=0
                        dia_distance_sum2=0
                        
                        
                        # Build spatial coordinate transform to determine if diamond points interfere with diamond particles
                        # First compute the three-axis unit vectors and origin coordinates of the particle
                        # in world coordinate system after Euler rotation and translation
                        coord_xl=np.array([[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]])
                        coord_tran=np.array([[1.0,0.0,0.0,-Dia_tran_X],[0.0,1.0,0.0,-Dia_tran_Y],[0.0,0.0,1.0,-Dia_tran_Z],[0.0,0.0,0.0,1.0]])
                        degree1=math.radians(Dia_roa_X)
                        degree2=math.radians(Dia_roa_Y)
                        degree3=math.radians(Dia_roa_Z)
                        rx=np.array([[1.0,0.0,0.0],[0.0,cos(degree1),-sin(degree1)],[0.0,sin(degree1),cos(degree1)]])
                        ry=np.array([[cos(degree2),0.0,sin(degree2)],[0.0,1.0,0.0],[-sin(degree2),0.0,cos(degree2)]])
                        rz=np.array([[cos(degree3),-sin(degree3),0.0],[sin(degree3),cos(degree3),0.0],[0.0,0.0,1.0]])
                        # Euler rotation matrix
                        roa=rx.dot(ry).dot(rz)
                        # Transformation matrix for converting a point in world coordinate system to the particle local coordinate system, which is a 4x4 homogeneous matrix
                        coord_roa=(roa.dot(coord_xl.T)).T
                        coord_roa=np.concatenate( [coord_roa,[[0.0,0.0,0.0]]],axis=0 )
                        coord_roa=np.concatenate( [coord_roa,[[0.0],[0.0],[0.0],[1.0]]],axis=1 )


                        for tempi in range(tempx-2,tempx+3):
                            for tempj in range(tempy-2,tempy+3):
                                for tempk in range(tempz-2,tempz+3):
                        # for tempi in range(0,Al_cut+2):
                        #     for tempj in range(0,Al_cut+2):
                        #         for tempk in range(0,Al_cut+2):
                                    
                                    diamond_points_list1=blank_list_dia[tempi][tempj][tempk].diamond_points
                                    diamond_data_list1=blank_list_dia[tempi][tempj][tempk].diamond_datas

                                    


                                    for point2i in range(0,len(diamond_points_list1)):
                                        ganshebiaoshifu=0
                                        point2=diamond_points_list1[point2i]
                                        diamond2= diamond_data_list1[point2i]
                                        Dia_tran_X1=point2[0]
                                        Dia_tran_Y1=point2[1]
                                        Dia_tran_Z1=point2[2]
                                        Dia_roa_X1=point2[3]
                                        Dia_roa_Y1=point2[4]
                                        Dia_roa_Z1=point2[5]

                                        if Dia_tran_X1==Dia_tran_X and Dia_tran_Y1==Dia_tran_Y and Dia_tran_Z1==Dia_tran_Z:
                                            continue

                                        # Check if there is interference
                                        dia_distance=sqrt((Dia_tran_X1-Dia_tran_X)*(Dia_tran_X1-Dia_tran_X)+(Dia_tran_Y1-Dia_tran_Y)*(Dia_tran_Y1-Dia_tran_Y)+
                                                            (Dia_tran_Z1-Dia_tran_Z)*(Dia_tran_Z1-Dia_tran_Z))

                                        if shape_lable==0:
                                            dia_Bounding_R2=diamond2.shpereR
                                            diamond_vertex=np.array(diamond2.points)
                                            dia_Bounding_R=dia_Bounding_R1+dia_Bounding_R2+dia_min_distance

                                            if dia_distance<dia_Bounding_R:
                                                # Calculate the coordinates of the 24 vertices of the judgment particle in the world coordinate system after Euler rotation and translation
                                                coord_point=np.array([[Dia_tran_X1,Dia_tran_Y1,Dia_tran_Z1,1.0]])
                                                degree1=math.radians(Dia_roa_X1)
                                                degree2=math.radians(Dia_roa_Y1)
                                                degree3=math.radians(Dia_roa_Z1)
                                                rx=np.array([[1.0,0.0,0.0],[0.0,cos(degree1),-sin(degree1)],[0.0,sin(degree1),cos(degree1)]])
                                                ry=np.array([[cos(degree2),0.0,sin(degree2)],[0.0,1.0,0.0],[-sin(degree2),0.0,cos(degree2)]])
                                                rz=np.array([[cos(degree3),-sin(degree3),0.0],[sin(degree3),cos(degree3),0.0],[0.0,0.0,1.0]])
                                                # Euler rotation matrix
                                                roa1=rx.dot(ry).dot(rz)
                                                # Euler rotation and translation matrix
                                                roa_tran=np.concatenate( [roa1,[[0.0,0.0,0.0]]],axis=0 )
                                                roa_tran=np.concatenate( [roa_tran,coord_point.T],axis=1 )
                                                # Expression of particle vertices in world coordinate system after Euler rotation and translation
                                                diamond_vertex1=roa_tran.dot(diamond_vertex.T)

                                                # Transform to particle local coordinate system
                                                diamond_vertex2=coord_roa.dot(coord_tran.dot(diamond_vertex1))
                                                diamond_vertex2=diamond_vertex2.T
                                                
                                                for vertexi in diamond_vertex2:
                                                    # Check interference by whether within bounding box and x1/((+-)a/sqrt(2)+x2/((+-)a/sqrt(2)+x2/((+-)a/sqrt(2)<1
                                                    if -dia_l0<vertexi[0]<dia_l0 and -dia_l0<vertexi[1]<dia_l0 and -dia_l0<vertexi[2]<dia_l0:
                                                        
                                                        if (vertexi[0]/tempc+vertexi[1]/tempc+vertexi[2]/tempc<=1 and vertexi[0]/tempc+vertexi[1]/tempc-vertexi[2]/tempc<=1 and
                                                            vertexi[0]/tempc-vertexi[1]/tempc+vertexi[2]/tempc<=1 and vertexi[0]/tempc-vertexi[1]/tempc-vertexi[2]/tempc<=1 and
                                                            -vertexi[0]/tempc+vertexi[1]/tempc+vertexi[2]/tempc<=1 and -vertexi[0]/tempc+vertexi[1]/tempc-vertexi[2]/tempc<=1 and
                                                            -vertexi[0]/tempc-vertexi[1]/tempc+vertexi[2]/tempc<=1 and -vertexi[0]/tempc-vertexi[1]/tempc-vertexi[2]/tempc<=1):
                                                            #vert=vertexi[:]/0
                                                            ganshebiaoshifu=1
                                                            break
                                                # Point is not inside another diamond, check if the line segment formed by the two points closest to the origin intersects with the diamond
                                                point_list=[]
                                                if ganshebiaoshifu==0:
                                                    # Calculate intersection points of diamond edges with pyramid planes, then check if intersection points are within the bounding box
                                                    XYZ=[tempc,-tempc]
                                                    diamond2_line=diamond2.lines

                                                    for line in diamond2_line:
                                                        vert1=diamond_vertex2[line[0]]
                                                        vert2=diamond_vertex2[line[1]]
                                                        x1=vert1[0]
                                                        y1=vert1[1]
                                                        z1=vert1[2]
                                                        x2=vert2[0]
                                                        y2=vert2[1]
                                                        z2=vert2[2]
                                                        u1=x1-x2
                                                        v1=y1-y2
                                                        w1=z1-z2
                                                        if x1**2+y1**2+z1**2>1.5*dia_Bounding_R1**2 and x2**2+y2**2+z2**2>1.5*dia_Bounding_R1**2:
                                                            continue
                                                        for a in XYZ:
                                                            for b in XYZ:
                                                                for c in XYZ:
                                                                    if (u1/a+v1/b+w1/c)==0:
                                                                        continue
                                                                    t=(1-(x2/a+y2/b+z2/c))/(u1/a+v1/b+w1/c)
                                                                    x=x2+u1*t
                                                                    y=y2+v1*t
                                                                    z=z2+w1*t
                                                                    # Check if in the same quadrant
                                                                    if (x*a>=0 and y*b>=0 and z*c>=0):
                                                                        l1=sqrt((x1-x)*(x1-x)+(y1-y)*(y1-y)+(z1-z)*(z1-z))
                                                                        l2=sqrt((x2-x)*(x2-x)+(y2-y)*(y2-y)+(z2-z)*(z2-z))
                                                                        l3=sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2)+(z1-z2)*(z1-z2))
                                                                        # Check if the intersection point is within the line segment
                                                                        
                                                                        if abs(l1+l2-l3)<0.000000001:
                                                                            point=[x,y,z]
                                                                            point_list.append(point)
                                                                            
                                                
                                                    for vertexi in point_list:
                                                        if -dia_l0<vertexi[0]<dia_l0 and -dia_l0<vertexi[1]<dia_l0 and -dia_l0<vertexi[2]<dia_l0:
                                                            ganshebiaoshifu=1
                                                            
                                                            break
                                                if ganshebiaoshifu==1:
                                                    temptran=[(Dia_tran_X-Dia_tran_X1)/dia_distance,
                                                                (Dia_tran_Y-Dia_tran_Y1)/dia_distance,
                                                                (Dia_tran_Z-Dia_tran_Z1)/dia_distance,
                                                                dia_Bounding_R-dia_distance]
                                            
                                                    # temptran=[(Dia_tran_X-blank_list_dia[tempi][tempj][tempk][0])/dia_distance,
                                                    #         (Dia_tran_Y-blank_list_dia[tempi][tempj][tempk][1])/dia_distance,
                                                    #         (Dia_tran_Z-blank_list_dia[tempi][tempj][tempk][2])/dia_distance,
                                                    #         dia_Bounding_R-dia_distance]
                                                    dia_distance_sum1=dia_distance_sum1+temptran[3]
                                                    Dia_tran1.append(temptran)

                                        elif shape_lable==1:
                                            dia_Bounding_R2=diamond2.a/2   
                                            dia_Bounding_R=dia_Bounding_R1+dia_Bounding_R2+dia_min_distance

                                            if dia_distance<dia_Bounding_R:
                                                ganshebiaoshifu=1
                                                temptran=[(Dia_tran_X-Dia_tran_X1)/dia_distance,
                                                            (Dia_tran_Y-Dia_tran_Y1)/dia_distance,
                                                            (Dia_tran_Z-Dia_tran_Z1)/dia_distance,
                                                            dia_Bounding_R-dia_distance]
                                                dia_distance_sum1=dia_distance_sum1+temptran[3]
                                                Dia_tran1.append(temptran)
                        # len(Dia_tran1)>0 indicates interference with surrounding particles, delete the particle
                        if len(Dia_tran1)>0:
                            
                            del blank_list_dia[tempx][tempy][tempz].diamond_points[point1i-pointi]
                            del blank_list_dia[tempx][tempy][tempz].diamond_datas[point1i-pointi]
                            blank_list_dia[tempx][tempy][tempz].diamond_lable-=1
                            
                            # Traverse auxiliary space and delete the corresponding particle
                            auxiliary_space_list=blank_list_dia[tempx][tempy][tempz].auxiliary_space
                            for auxiliary_space in auxiliary_space_list:
                                tempx_x=auxiliary_space[0]
                                tempy_y=auxiliary_space[1]
                                tempz_z=auxiliary_space[2]
                                del blank_list_dia[tempx_x][tempy_y][tempz_z].diamond_points[point1i-pointi]
                                del blank_list_dia[tempx_x][tempy_y][tempz_z].diamond_datas[point1i-pointi]
                                blank_list_dia[tempx_x][tempy_y][tempz_z].diamond_lable-=1
                            
                            pointi=pointi+1

# # #############################################
# Add particles into model
# # ##############################################       
            
        j=0
        testroa=[]
        for tempxx in range(0,Al_cut+2):
            for tempyy in range(0,Al_cut+2):
                for tempzz in range(0,Al_cut+2):
                    diamond_points_list=blank_list_dia[tempxx][tempyy][tempzz].diamond_points
                    diamond_datas_list=blank_list_dia[tempxx][tempyy][tempzz].diamond_datas

                    for tempi in range(0,len(diamond_points_list)):
                        temp_point=diamond_points_list[tempi]
                        Dia_tran_X=temp_point[0]
                        Dia_tran_Y=temp_point[1]
                        Dia_tran_Z=temp_point[2]
                        Dia_roa_X=temp_point[3]
                        Dia_roa_Y=temp_point[4]
                        Dia_roa_Z=temp_point[5]
                        # Dia_roa_X=x_roa
                        # Dia_roa_Y=y_roa
                        # Dia_roa_Z=z_roa
                        testroa.append([Dia_roa_X,Dia_roa_Y,Dia_roa_Z])

                        # Generate diamond particle
                        diamonddata=diamond_datas_list[tempi]
                        diamond_name='D-'+str(j)
                        diamond_a=diamonddata.a
                        diamond_l=diamonddata.l
                        if shape_lable==0:
                            diamond_generator(diamond_a,diamond_l,model_name,diamond_name)
                            
                            diamond_ass_name=diamond_name
                            p = mdb.models[model_name].parts[diamond_name]
                            a1 = mdb.models[model_name].rootAssembly
                            #a1.DatumCsysByDefault(CARTESIAN)
                            a1.Instance(name=diamond_ass_name, part=p, dependent=ON)
                            Dia_ass_name_data.append(a1.instances[diamond_ass_name])
                            Dia_ass_name.append(a1.instances[diamond_ass_name])
                            a1 = mdb.models[model_name].rootAssembly

                            # Rotate around X, Y, Z axes by Dia_roa_X, Dia_roa_Y, Dia_roa_Z degrees
                            a1.rotate(instanceList=(diamond_ass_name, ), axisPoint=(0.0, 0.0, 0.0), 
                                axisDirection=a1.instances[diamond_ass_name].datums[7].direction, angle=Dia_roa_X)
                            a1.rotate(instanceList=(diamond_ass_name, ), axisPoint=(0.0, 0.0, 0.0), 
                                axisDirection=a1.instances[diamond_ass_name].datums[9].direction, angle=Dia_roa_Y)
                            a1.rotate(instanceList=(diamond_ass_name, ), axisPoint=(0.0, 0.0, 0.0), 
                                axisDirection=a1.instances[diamond_ass_name].datums[8].direction, angle=Dia_roa_Z)     
                            a1.translate(instanceList=(diamond_ass_name, ), vector=(Dia_tran_X, Dia_tran_Y, Dia_tran_Z))
                            j+=1                    
                        elif shape_lable==1:
                            sphere_generator(diamond_a/2,model_name,diamond_name)
                            diamond_ass_name=diamond_name
                            p = mdb.models[model_name].parts[diamond_name]
                            a1 = mdb.models[model_name].rootAssembly
                            #a1.DatumCsysByDefault(CARTESIAN)
                            a1.Instance(name=diamond_ass_name, part=p, dependent=ON)
                            Dia_ass_name_data.append(a1.instances[diamond_ass_name])
                            Dia_ass_name.append(a1.instances[diamond_ass_name])
                            a1 = mdb.models[model_name].rootAssembly

                            # Rotate around X, Y, Z axes by Dia_roa_X, Dia_roa_Y, Dia_roa_Z degrees
                            a1.translate(instanceList=(diamond_ass_name, ), vector=(Dia_tran_X, Dia_tran_Y, Dia_tran_Z))
                            j+=1                    


                        

        session.viewports['Viewport: 1'].assemblyDisplay.geometryOptions.setValues(datumPoints=OFF, datumAxes=OFF, datumPlanes=OFF, datumCoordSystems=OFF)

# ####################################
# Boolean operation, create diamond and aluminum matrix assembly
# ####################################                        
        # Record end time
        end=time.time()
        # Get running time
        yunxingshijian=end-start

        # Add Al block into assembly, named CubeProfile and CubeProfile1 respectively
        a1 = mdb.models[model_name].rootAssembly
        p = mdb.models[model_name].parts['Part-Al']
        a1.Instance(name='CubeProfile', part=p, dependent=ON)
        a1.Instance(name='CubeProfile1', part=p, dependent=ON)


        try:
            # Merge diamond particles into a single body, named dianmond_temp
            #a1.InstanceFromBooleanMerge(name='dianmond_temp', instances=Dia_ass_name, keepIntersections=ON,originalInstances=SUPPRESS, domain=GEOMETRY)
            a1.InstanceFromBooleanMerge(name='dianmond_temp', instances=Dia_ass_name_data, keepIntersections=OFF,originalInstances=DELETE, domain=GEOMETRY)

            # Subtract diamond particles from matrix via Boolean operation, get perforated Al, named Al_body
            a1.InstanceFromBooleanCut(name='Al_body', instanceToBeCut=mdb.models[model_name].rootAssembly.instances['CubeProfile'], 
            cuttingInstances=(a1.instances['dianmond_temp-1'], ), originalInstances=DELETE)

            # Cut perforated Al from matrix via Boolean operation, get edge-trimmed diamond particles, named dianmond_ass
            
            a1.InstanceFromBooleanCut(name='dianmond_ass', instanceToBeCut=mdb.models[model_name].rootAssembly.instances['CubeProfile1'], 
            cuttingInstances=(a1.instances['Al_body-1'], ), originalInstances=SUPPRESS)
            del a1.features['CubeProfile1']
        except Exception as e:
            print('Found exceptions, failed Bool operation')
            continue


        # Resume perforated Al
        a1.resumeFeatures(('Al_body-1', 'dianmond_temp0-1', ))


################################
# Uncertainty modeling of material properties
################################
        ####################
        ######################
        # # Define material - Diamond
        diamond_E=1100000
        diamond_v=0.085
        diamond_thermal=2000
       
        mdb.models[model_name].Material(name='diamond')
        mdb.models[model_name].materials['diamond'].Density(table=((diamond_rho, ), ))
        mdb.models[model_name].materials['diamond'].Elastic(table=((diamond_E, diamond_v), ))
        mdb.models[model_name].materials['diamond'].Conductivity(table=((diamond_thermal, ), ))
        mdb.models[model_name].materials['diamond'].SpecificHeat(table=((diamond_c, ), ))
        mdb.models[model_name].materials['diamond'].Expansion(table=((diamond_alpha, ), ))
        
        # Define material - Aluminum
        mdb.models[model_name].Material(name='Al')
        mdb.models[model_name].materials['Al'].Density(table=((matrix_rho, ), ))
        mdb.models[model_name].materials['Al'].Elastic(table=((matrix_E, matrix_v), 
            ))
        mdb.models[model_name].materials['Al'].Conductivity(table=((matrix_thermal, ), ))
        mdb.models[model_name].materials['Al'].SpecificHeat(table=((matrix_c, ), ))
        # The expansion rate defined in Abaqus is the linear expansion coefficient
        mdb.models[model_name].materials['Al'].Expansion(table=((matrix_alpha, ), ))
        #JC-model
        mdb.models[model_name].materials['Al'].Plastic(hardening=JOHNSON_COOK, table=((matrix_A, matrix_B, matrix_m, 0.0, 0.0, 0.0), ))

       

    

        # Define sections
        mdb.models[model_name].HomogeneousSolidSection(name='Section-diamond', 
            material='diamond', thickness=None)
        mdb.models[model_name].HomogeneousSolidSection(name='Section-Al', 
            material='Al', thickness=None)
        ############################
        ############################
        
        # Assign materials
        # For diamond_ass, materials need to be assigned per cell to achieve uncertain modeling
        p = mdb.models[model_name].parts['dianmond_ass']
        c = p.cells
        # region = regionToolset.Region(cells=c)
        # p.SectionAssignment(region=region, sectionName='Section-diamond', offset=0.0, offsetType=MIDDLE_SURFACE, offsetField='',thicknessAssignment=FROM_SECTION)
        

        # Since diamond particles form a single part after boolean operations, cell matching must be done based on placement data
        # Restore the diamond particle information corresponding to each cell
        i=0
        for tempxx in range(0,Al_cut+2):
            for tempyy in range(0,Al_cut+2):
                for tempzz in range(0,Al_cut+2):
                    diamond_points_list=blank_list_dia[tempxx][tempyy][tempzz].diamond_points
                    diamond_datas_list=blank_list_dia[tempxx][tempyy][tempzz].diamond_datas

                    for tempi in range(0,len(diamond_points_list)):
                        temp_point=diamond_points_list[tempi]
                        Dia_tran_X=temp_point[0]
                        Dia_tran_Y=temp_point[1]
                        Dia_tran_Z=temp_point[2]
                        
                        diamonddata=diamond_datas_list[tempi]
                        B_sphere_R=diamonddata.shpereR
                        celllist=c.getByBoundingSphere(center=(Dia_tran_X,Dia_tran_Y,Dia_tran_Z),radius=1.1*B_sphere_R)
                        diamond_E=diamonddata.Ei
                        diamond_v=diamonddata.v
                        diamond_thermal=diamonddata.thermal
                        diamond_sigmas=diamonddata.sigmas
                        diamond_sigmat=diamonddata.sigmat

                        # If there is a corresponding cell, create material
                        if len(celllist)>0:
                            # Create diamond material model
                            Dia_materialname='dia-'+str(i)
                            i=i+1
                            mdb.models[model_name].Material(name=Dia_materialname)
                            mdb.models[model_name].materials[Dia_materialname].Density(table=((diamond_rho, ), ))
                            mdb.models[model_name].materials[Dia_materialname].Elastic(table=((diamond_E, diamond_v), ))
                            mdb.models[model_name].materials[Dia_materialname].Conductivity(table=((diamond_thermal, ), ))
                            mdb.models[model_name].materials[Dia_materialname].SpecificHeat(table=((diamond_c, ), ))
                            mdb.models[model_name].materials[Dia_materialname].Expansion(table=((diamond_alpha, ), ))
                            
                            # Define brittle failure for diamond
                            mdb.models[model_name].materials[Dia_materialname].BrittleCracking(table=((diamond_sigmas, 0.0), (0.0, 1e-05)))
                            mdb.models[model_name].materials[Dia_materialname].brittleCracking.BrittleShear(type=POWER_LAW, table=((1e-05, 1.0), ))
                            mdb.models[model_name].materials[Dia_materialname].brittleCracking.BrittleFailure(table=((1e-05, ), ))
                            
                            # Define section
                            mdb.models[model_name].HomogeneousSolidSection(name=Dia_materialname, material=Dia_materialname, thickness=None)
                            # Assign material section
                            region = regionToolset.Region(cells=celllist)
                            p.SectionAssignment(region=region, sectionName=Dia_materialname, offset=0.0, offsetType=MIDDLE_SURFACE, offsetField='',thicknessAssignment=FROM_SECTION)
             

       
        # Aluminum matrix material section assignment
        p = mdb.models[model_name].parts['Al_body']
        c = p.cells
        region = regionToolset.Region(cells=c)
        p.SectionAssignment(region=region, sectionName='Section-Al', offset=0.0, offsetType=MIDDLE_SURFACE, offsetField='',thicknessAssignment=FROM_SECTION)

        # Calculate volume of diamond_ass for volume fraction calculation
        p = mdb.models[model_name].parts['dianmond_ass']
        tempv=p.getVolume()
        dia_ratio=tempv/(aldim*aldim*aldim)
        print(dia_ratio)
        dia_ratio1=dia_ratio
        dia_ratio=np.random.normal(loc=dia_ratio,scale=0.01)
        simulation_data[7]=dia_ratio
        simulation_data1[7]=dia_ratio



# #############################################
# Create contact properties and build face-face-contact property pair list,
# for subsequent individual face-to-face contact settings for different
# thermal-mechanical performance parameter calculations
# #############################################        
        # Create contact
        j_hc=jiemian_h##np.random.uniform(0.8*jiemian_h,1.2*jiemian_h)
        j_EC=jiemian_EC#np.random.uniform(0.9*jiemian_EC,1.1*jiemian_EC)
        j_eta=jiemian_eta##np.random.uniform(0.9*jiemian_eta,1.1*jiemian_eta)
        j_t=jiemian_t#np.random.uniform(0.8*jiemian_t,1.2*jiemian_t)
        j_GC=j_EC/j_eta
        Knn=j_EC/j_t
        Kss=j_GC/j_t
        Knn_Dam=1000
        Kss_Dam=500
        Knn_Dam_eng=100
        Kss_Dam_eng=50

        # Create contact property IntProp-1
        mdb.models[model_name].ContactProperty('IntProp-1')
        # Set IntProp-1 properties -- ThermalConductance
        mdb.models[model_name].interactionProperties['IntProp-1'].ThermalConductance(
            definition=TABULAR, clearanceDependency=ON, pressureDependency=OFF, 
            temperatureDependencyC=OFF, massFlowRateDependencyC=OFF, dependenciesC=0, 
            clearanceDepTable=((jiemian_h, 0.0), (0.0, 0.01)))
        # Normal contact behavior
        mdb.models[model_name].interactionProperties['IntProp-1'].NormalBehavior(
            pressureOverclosure=HARD, allowSeparation=ON, 
            constraintEnforcementMethod=DEFAULT)
        # Cohesive behavior settings
        mdb.models[model_name].interactionProperties['IntProp-1'].CohesiveBehavior(
            defaultPenalties=OFF, table=((Knn, Kss, Kss), ))




        
        # Create tie constraint
        a2 = mdb.models[model_name].rootAssembly    # Assign the rootAssembly instance from Model-1 to a2
        s1 = a2.instances['dianmond_ass-1'].faces       # Assign the face list of dianmond_ass-1 in a2 to s1
        s2 = a2.instances['Al_body-1'].faces       # Assign the face list of Al_body-1 in a2 to s2

        face_face_intpro_list=[]

        i=0
        for Diaface in s1:   
            cen_point=Diaface.getCentroid()  # Get the centroid coordinates of face Diaface
            if s1.findAt(cen_point):
                # The findAt function can be used to check whether a point lies within a body, for collision detection
                # First, check if the point is on an outer boundary face
                if (abs(cen_point[0][0]-0.5*aldim)<0.000001 or abs(cen_point[0][0]+0.5*aldim)<0.000001 
                    or abs(cen_point[0][1]-0.5*aldim)<0.000001 or abs(cen_point[0][1]+0.5*aldim)<0.000001 
                    or abs(cen_point[0][2])<0.000001 or abs(cen_point[0][2]-aldim)<0.000001):
                    continue
                side1Faces1 = s1.findAt(cen_point)
                side1Faces2 = s2.findAt(cen_point)   # Find the face in s2 passing through cen_point   

                dia_face_name='dia_face_'+str(i)   # Face name variable on the diamond side
                Al_face_name='Al_face_'+str(i)     # Face name variable on the aluminum matrix side
                dai_Al_tie_name='dai_Al_tie_'+str(i)    # Name variable for diamond-Al tie constraint
                dai_Al_contact_name='dai_Al_contact_'+str(i)    # Name variable for diamond-Al contact
                a2.Surface(side1Faces=side1Faces1, name=dia_face_name)  # Create a Surface from face Diaface, named dia_face_name
                a2.Surface(side1Faces=side1Faces2, name=Al_face_name)  # Create a Surface from found face side1Faces1, named Al_face_name
                region1=a2.surfaces[dia_face_name]
                region2=a2.surfaces[Al_face_name]
                # Create tie constraint
                #mdb.models[model_name].Tie(name=dai_Al_tie_name, master=region1, slave=region2, 
                #positionToleranceMethod=COMPUTED, adjust=ON, tieRotations=ON, thickness=ON)   # Tie constraint settings
                
                # Create contact
                ######
                # Add distribution function to simulate uncertainty issues in actual preparation process
                j_hc=np.random.uniform(0.5*jiemian_h,1.5*jiemian_h)
                j_EC=np.random.uniform(0.9*jiemian_EC,1.1*jiemian_EC)
                j_eta=np.random.uniform(0.9*jiemian_eta,1.1*jiemian_eta)
                j_t=np.random.uniform(0.7*jiemian_t,1.3*jiemian_t)
                j_GC=j_EC/j_eta
                Knn=j_EC/j_t
                Kss=j_GC/j_t
                cover_ratio=np.random.uniform(0.5,1.0)
                Knn=Knn*cover_ratio
                Kss=Kss*cover_ratio
                Knn_Dam=1000
                Kss_Dam=500
                Knn_Dam_eng=100
                Kss_Dam_eng=50

                ######
                # Create contact property IntProp-i
                intpropname='IntProp-'+str(i)
               
                mdb.models[model_name].ContactProperty(intpropname)
                # Set IntProp-i property--ThermalConductance
                mdb.models[model_name].interactionProperties[intpropname].ThermalConductance(
                    definition=TABULAR, clearanceDependency=ON, pressureDependency=OFF, 
                    temperatureDependencyC=OFF, massFlowRateDependencyC=OFF, dependenciesC=0, 
                    clearanceDepTable=((j_hc, 0.0), (0.0, 0.0001)))
                # #Tangential behavior
                mdb.models[model_name].interactionProperties[intpropname].TangentialBehavior(
                    formulation=PENALTY, directionality=ISOTROPIC, slipRateDependency=OFF, 
                    pressureDependency=OFF, temperatureDependency=OFF, dependencies=0, table=((
                    0.05, ), ), shearStressLimit=None, maximumElasticSlip=FRACTION, 
                    fraction=0.005, elasticSlipStiffness=None)
                # Normal behavior
                mdb.models[model_name].interactionProperties[intpropname].NormalBehavior(
                    pressureOverclosure=HARD, allowSeparation=ON, 
                    constraintEnforcementMethod=DEFAULT)
                # Cohesive settings
                mdb.models[model_name].interactionProperties[intpropname].CohesiveBehavior(
                    defaultPenalties=OFF, table=((Knn, Kss, Kss), ))
                mdb.models[model_name].interactionProperties[intpropname].Damage(
                    initTable=((Knn_Dam, Kss_Dam, Kss_Dam), ), useEvolution=ON, evolutionType=ENERGY, 
                    useMixedMode=ON, mixedModeType=BK, evolTable=((Knn_Dam_eng, Kss_Dam_eng, Kss_Dam_eng), ))
                                
                # Save face-to-face contact to list
                face_face_intpro_list.append([dia_face_name,Al_face_name,intpropname])
                # #Create surface-to-surface contact
                # #When using quadratic elements, improved elements are required to use NODE_TO_SURFACE surface-to-surface contact
                # mdb.models[model_name].SurfaceToSurfaceContactStd(name=dai_Al_contact_name, 
                # createStepName='Step-1', master=region1, slave=region2, sliding=FINITE, 
                # enforcement=NODE_TO_SURFACE, thickness=OFF, 
                # interactionProperty='IntProp-1', surfaceSmoothing=NONE, adjustMethod=NONE, 
                # smooth=0.2, initialClearance=OMIT, datumAxis=None, clearanceRegion=None)
                
                # mdb.models[model_name].SurfaceToSurfaceContactStd(name=dai_Al_contact_name, 
                # createStepName='Step-1', master=region1, slave=region2, sliding=FINITE, 
                # thickness=ON, interactionProperty=intpropname, surfaceSmoothing=AUTOMATIC, 
                # adjustMethod=NONE, initialClearance=OMIT, datumAxis=None, 
                # clearanceRegion=None)
                i+=1

        ################ Meshing ###############
        elemType1 = mesh.ElemType(elemCode=DC3D20, elemLibrary=STANDARD)
        elemType2 = mesh.ElemType(elemCode=DC3D15, elemLibrary=STANDARD)
        elemType3 = mesh.ElemType(elemCode=DC3D10, elemLibrary=STANDARD,secondOrderAccuracy=OFF, distortionControl=DEFAULT)
        p = mdb.models[model_name].parts['dianmond_ass']
        c = p.cells
        pickedRegions = c.getByBoundingBox(xMin=-2*aldim,xMax=2*aldim,yMin=-aldim*2,yMax=aldim*2,zMin=-aldim*2,zMax=aldim*2)
        p.setMeshControls(regions=pickedRegions, elemShape=TET, technique=FREE, algorithm=DEFAULT, sizeGrowthRate=1.0, allowMapped=False)
        pickedRegions =(pickedRegions, )
        p.setElementType(regions=pickedRegions, elemTypes=(elemType3,))
        p.seedPart(size=mesh_size*acenter, deviationFactor=0.001, minSizeFactor=0.1)
        p.generateMesh()
        
        p1 = mdb.models[model_name].parts['Al_body']
        c = p1.cells
        pickedRegions = c.getByBoundingBox(xMin=-2*aldim,xMax=2*aldim,yMin=-aldim*2,yMax=aldim*2,zMin=-aldim*2,zMax=aldim*2)
        p1.setMeshControls(regions=pickedRegions, elemShape=TET, technique=FREE, algorithm=DEFAULT, sizeGrowthRate=1.0, allowMapped=False)
        pickedRegions =(pickedRegions, )
        p1.setElementType(regions=pickedRegions, elemTypes=(elemType3,))
        p1.seedPart(size=mesh_size*acenter, deviationFactor=0.001, minSizeFactor=0.1)
        p1.generateMesh()
        elementsnum=len(p.elements)+len(p1.elements)
        # Control mesh density to prevent excessive memory usage from too many elements causing interruption
        while elementsnum>1500000:
            mesh_size=1.05*mesh_size

            elemType1 = mesh.ElemType(elemCode=DC3D20, elemLibrary=STANDARD)
            elemType2 = mesh.ElemType(elemCode=DC3D15, elemLibrary=STANDARD)
            elemType3 = mesh.ElemType(elemCode=DC3D10, elemLibrary=STANDARD,secondOrderAccuracy=OFF, distortionControl=DEFAULT)
            p = mdb.models[model_name].parts['dianmond_ass']
            c = p.cells
            pickedRegions = c.getByBoundingBox(xMin=-2*aldim,xMax=2*aldim,yMin=-aldim*2,yMax=aldim*2,zMin=-aldim*2,zMax=aldim*2)
            p.setMeshControls(regions=pickedRegions, elemShape=TET, technique=FREE, algorithm=DEFAULT, sizeGrowthRate=1.0, allowMapped=False)
            pickedRegions =(pickedRegions, )
            p.setElementType(regions=pickedRegions, elemTypes=(elemType3,))
            p.seedPart(size=mesh_size*acenter, deviationFactor=0.001, minSizeFactor=0.1)
            p.generateMesh()
            
            p1 = mdb.models[model_name].parts['Al_body']
            c = p1.cells
            pickedRegions = c.getByBoundingBox(xMin=-2*aldim,xMax=2*aldim,yMin=-aldim*2,yMax=aldim*2,zMin=-aldim*2,zMax=aldim*2)
            p1.setMeshControls(regions=pickedRegions, elemShape=TET, technique=FREE, algorithm=DEFAULT, sizeGrowthRate=1.0, allowMapped=False)
            pickedRegions =(pickedRegions, )
            p1.setElementType(regions=pickedRegions, elemTypes=(elemType3,))
            p1.seedPart(size=mesh_size*acenter, deviationFactor=0.001, minSizeFactor=0.1)
            p1.generateMesh()
            elementsnum=len(p.elements)+len(p1.elements)


######################
# Copy models for thermal conductivity, CTE, elastic modulus, yield strength and tensile strength calculations
########################
        model_name_CTC=model_name+'-CTC'
        model_name_CTE=model_name+'-CTE'
        model_name_E=model_name+'-E'
        #model_name_Explict=model_name+'-Explict'
        mdb.Model(name=model_name_CTC, objectToCopy=mdb.models[model_name])
        mdb.Model(name=model_name_CTE, objectToCopy=mdb.models[model_name])
        mdb.Model(name=model_name_E, objectToCopy=mdb.models[model_name])
        #mdb.Model(name=model_name_Explict, objectToCopy=mdb.models[model_name])

##########################
# Thermal conductivity calculation setup
##########################
        if CTC_lable==1:
            model_name=model_name_CTC

            ######## Remove fracture parameters and plastic parameters from materials ################
            for i in  mdb.models[model_name].materials.keys():
                del mdb.models[model_name].materials[i].brittleCracking
            del mdb.models[model_name].materials['Al'].plastic
            del mdb.models[model_name].materials['Al'].johnsonCookDamageInitiation

            ############ Create step: steady-state heat transfer analysis ##############
            mdb.models[model_name].HeatTransferStep(name='Step-1', previous='Initial', 
                response=STEADY_STATE, initialInc=0.1, amplitude=RAMP)
            # Create field output
            mdb.models[model_name].fieldOutputRequests['F-Output-1'].setValues(variables=(
                'NT', 'HFL', 'RFL', 'EVOL', 'IVOL'),frequency=LAST_INCREMENT)
            # Create history output
            mdb.models[model_name].HistoryOutputRequest(name='H-Output-1', 
                createStepName='Step-1', variables=('FTEMP', 'HFLA'),frequency=LAST_INCREMENT)

            ############ Remove mechanical settings from contact properties #############
            for i in mdb.models[model_name].interactionProperties.keys():
                del mdb.models[model_name].interactionProperties[i].tangentialBehavior
                del mdb.models[model_name].interactionProperties[i].normalBehavior
                del mdb.models[model_name].interactionProperties[i].cohesiveBehavior
                del mdb.models[model_name].interactionProperties[i].damage
            # Create contacts
            a = mdb.models[model_name].rootAssembly    # Assign rootAssembly of instance part object in Model-1 to a
            j=0
            for i in face_face_intpro_list:
                surface1=i[0]
                surface2=i[1]
                intpro1=i[2]
                region1=a.surfaces[surface1]
                region2=a.surfaces[surface2]
                dai_Al_contact_name='CTC_contact'+str(j)
                j+=1

                mdb.models[model_name].SurfaceToSurfaceContactStd(name=dai_Al_contact_name, 
                    createStepName='Step-1', master=region1, slave=region2, sliding=FINITE, 
                    thickness=ON, interactionProperty=intpro1, surfaceSmoothing=AUTOMATIC, 
                    adjustMethod=NONE, initialClearance=OMIT, datumAxis=None, 
                    clearanceRegion=None)
            
            ######## Thermal load setup ###############
            a = mdb.models[model_name].rootAssembly    # Assign rootAssembly of instance part object in Model-1 to a
            s1 = a.instances['dianmond_ass-1'].faces       # Assign face list of dianmond_ass-1 in a2 to s1
            s2 = a.instances['Al_body-1'].faces       # Assign face list of Al_body-1 in a2 to s2

            
            Xpos_face_list=[]# Edge faces in +X direction
            Xneg_face_list=[]# Edge faces in -X direction
            Ypos_face_list=[]# Edge faces in +Y direction
            Yneg_face_list=[]# Edge faces in -Y direction
            Zpos_face_list=[]# Edge faces in +Z direction
            Zneg_face_list=[]# Edge faces in -Z direction

            # Create thermal loads for heat conduction analysis
            Xpos_face_list=(s1+s2).getByBoundingBox(xMin=aldim/2-0.01,xMax=aldim/2+0.01,yMin=-aldim*2,yMax=aldim*2,zMin=-aldim*2,zMax=aldim*2)
            Xneg_face_list=(s1+s2).getByBoundingBox(xMin=-aldim/2-0.01,xMax=-aldim/2+0.01,yMin=-aldim*2,yMax=aldim*2,zMin=-aldim*2,zMax=aldim*2)
            Ypos_face_list=(s1+s2).getByBoundingBox(xMin=-aldim*2,xMax=aldim*2,yMin=aldim/2-0.01,yMax=aldim/2+0.01,zMin=-aldim*2,zMax=aldim*2)
            Yneg_face_list=(s1+s2).getByBoundingBox(xMin=-aldim*2,xMax=aldim*2,yMin=-aldim/2-0.01,yMax=-aldim/2+0.01,zMin=-aldim*2,zMax=aldim*2)
            Zpos_face_list=(s1+s2).getByBoundingBox(xMin=-aldim*2,xMax=aldim*2,yMin=-aldim*2,yMax=aldim*2,zMin=aldim-0.01,zMax=aldim+0.01)
            Zneg_face_list=(s1+s2).getByBoundingBox(xMin=-aldim*2,xMax=aldim*2,yMin=-aldim*2,yMax=aldim*2,zMin=-0.01,zMax=0.01)
            # Create sets for subsequent load application; faces keyword takes a geometric Sequence, not a list
            a.Set(faces=Xpos_face_list, name='Xpos_faces')
            a.Set(faces=Xneg_face_list, name='Xneg_faces')
            a.Set(faces=Ypos_face_list, name='Ypos_faces')
            a.Set(faces=Yneg_face_list, name='Yneg_faces')
            a.Set(faces=Zpos_face_list, name='Zpos_faces')
            a.Set(faces=Zneg_face_list, name='Zneg_faces')

            # Create thermal loads
            a = mdb.models[model_name].rootAssembly
            region = a.sets['Zneg_faces']
            mdb.models[model_name].TemperatureBC(name='BC-1', createStepName='Step-1', 
                region=region, fixed=OFF, distributionType=UNIFORM, fieldName='', 
                magnitude=26.0, amplitude=UNSET)
            a = mdb.models[model_name].rootAssembly
            region = a.sets['Zpos_faces']
            mdb.models[model_name].TemperatureBC(name='BC-2', createStepName='Step-1', 
                region=region, fixed=OFF, distributionType=UNIFORM, fieldName='', 
                magnitude=27.0, amplitude=UNSET)
            
            ################ Mesh generation ###############

            elemType1 = mesh.ElemType(elemCode=DC3D20, elemLibrary=STANDARD)
            elemType2 = mesh.ElemType(elemCode=DC3D15, elemLibrary=STANDARD)
            elemType3 = mesh.ElemType(elemCode=DC3D10, elemLibrary=STANDARD,secondOrderAccuracy=OFF, distortionControl=DEFAULT)
            p = mdb.models[model_name].parts['dianmond_ass']
            c = p.cells
            pickedRegions = c.getByBoundingBox(xMin=-2*aldim,xMax=2*aldim,yMin=-aldim*2,yMax=aldim*2,zMin=-aldim*2,zMax=aldim*2)
            p.setMeshControls(regions=pickedRegions, elemShape=TET, technique=FREE, algorithm=DEFAULT, sizeGrowthRate=1.0, allowMapped=False)
            pickedRegions =(pickedRegions, )
            p.setElementType(regions=pickedRegions, elemTypes=(elemType3,))
            p.seedPart(size=mesh_size*acenter, deviationFactor=0.001, minSizeFactor=0.1)
            p.generateMesh()
            if type(p.getUnmeshedRegions())!=NoneType:
                continue
            p1 = mdb.models[model_name].parts['Al_body']
            c = p1.cells
            pickedRegions = c.getByBoundingBox(xMin=-2*aldim,xMax=2*aldim,yMin=-aldim*2,yMax=aldim*2,zMin=-aldim*2,zMax=aldim*2)
            p1.setMeshControls(regions=pickedRegions, elemShape=TET, technique=FREE, algorithm=DEFAULT, sizeGrowthRate=1.0, allowMapped=False)
            pickedRegions =(pickedRegions, )
            p1.setElementType(regions=pickedRegions, elemTypes=(elemType3,))
            p1.seedPart(size=mesh_size*acenter, deviationFactor=0.001, minSizeFactor=0.1)
            p1.generateMesh()
            if type(p1.getUnmeshedRegions())!=NoneType:
                continue

            ############# Create job and submit ################
            jobname=model_name#+str(list_i)
            mdb.Job(name=jobname, model=model_name, description='', type=ANALYSIS, 
                atTime=None, waitMinutes=0, waitHours=0, queue=None, memory=90, 
                memoryUnits=PERCENTAGE, getMemoryFromAnalysis=True, 
                explicitPrecision=SINGLE, nodalOutputPrecision=SINGLE, echoPrint=OFF, 
                modelPrint=OFF, contactPrint=OFF, historyPrint=OFF, userSubroutine='', 
                scratch='', resultsFormat=ODB, multiprocessingMode=DEFAULT, numCpus=2, 
                numDomains=2, numGPUs=1)
            mdb.jobs[jobname].submit(consistencyChecking=OFF)
            mdb.jobs[jobname].waitForCompletion()

            if mdb.jobs[jobname].messages[-1].type!=JOB_COMPLETED:
                continue 



            ############## Extract results and compute thermal conductivity ################
            model_jobname=jobname+'.odb'
            odb_file.append(model_jobname)
            o=session.openOdb(name=model_jobname, readOnly=True)
            conducticity=0 # Thermal conductivity
            average_heatv=0  # Average heat flux
            average_heatv1=0
            frames=o.steps['Step-1'].frames
            f1=frames[-1]
            fop=f1.fieldOutputs
            fopEVOL=fop['EVOL']
            fopIVOL=fop['IVOL']
            fopHFL=fop['HFL']
            for i in range(0,len(fopEVOL.values)):
                temp0=0
                temp1=0
                for j in range(0,15):  # 4 integration points for DC3D4 element
                    #temp0=temp0+fopHFL.values[i*15+j].data[2] # Get heat flux at integration point
                    temp1=temp1+fopIVOL.values[i*15+j].data*fopHFL.values[i*15+j].data[2]
                #tempevol=fopEVOL.values[i].data   # Get element volume
                #average_heatv=average_heatv+temp0*tempevol/15
                average_heatv1=average_heatv1+temp1
            #average_heatv=average_heatv*aldim/(aldim*aldim*aldim)
            average_heatv1=-average_heatv1*aldim/(aldim*aldim*aldim)
            # Print volume ratio
            print ('The volume ratio of diamonds is:')
            print(dia_ratio)
            print 'Thermal Conductivity:'+str(average_heatv1)

            o.close()
            simulation_data.append(average_heatv1)
            simulation_data1.append(average_heatv1)


##########################
# Thermal expansion coefficient related calculation settings
##########################

        if CTE_lable==1:
            model_name=model_name_CTE

            ######## Delete brittle fracture parameters of diamond and damage parameters of aluminum matrix, but retain plasticity parameters of aluminum matrix ################
            for i in  mdb.models[model_name].materials.keys():
                del mdb.models[model_name].materials[i].brittleCracking
            del mdb.models[model_name].materials['Al'].johnsonCookDamageInitiation


            ############ Create analysis step, coupled thermal-mechanical analysis ##############
            
            mdb.models[model_name].CoupledTempDisplacementStep(name='Step-1', 
                previous='Initial', response=STEADY_STATE, initialInc=0.05, maxInc=0.5, 
                deltmx=None, cetol=None, creepIntegration=None, amplitude=RAMP, 
                matrixStorage=SOLVER_DEFAULT, solutionTechnique=SEPARATED, nlgeom=ON)
            # Create field output
            mdb.models[model_name].fieldOutputRequests['F-Output-1'].setValues(variables=( 'S', 'E', 'PE', 'PEEQ', 'PEMAG', 'LE', 'U', 'RF', 'CF', 'CSTRESS', 
                'CDISP', 'NT', 'HFL', 'RFL', 'EVOL'),frequency=LAST_INCREMENT)
            mdb.models[model_name].fieldOutputRequests['F-Output-1'].setValues(timeInterval=0.1)
            # Create history output
            mdb.models[model_name].HistoryOutputRequest(name='H-Output-1', 
                createStepName='Step-1', variables=('CSTRESS', 'CFNM', 'CFN1', 'CFN2', 'CFN3', 'CFSM', 'CFS1', 
                'CFS2', 'CFS3', 'CFTM', 'CFT1', 'CFT2', 'CFT3', 'CAREA', 'ALLAE', 'ALLCD', 
                'ALLDMD', 'ALLEE', 'ALLFD', 'ALLIE', 'ALLJD', 'ALLKE', 'ALLKL', 'ALLPD', 
                'ALLQB', 'ALLSE', 'ALLSD', 'ALLVD', 'ALLWK', 'ETOTAL'),frequency=LAST_INCREMENT)


            ############ Delete mechanical-related settings in contact properties, assuming no interfacial failure during thermal expansion #############
            for i in mdb.models[model_name].interactionProperties.keys():
                del mdb.models[model_name].interactionProperties[i].tangentialBehavior
                del mdb.models[model_name].interactionProperties[i].normalBehavior
                #del mdb.models[model_name].interactionProperties[i].cohesiveBehavior
                del mdb.models[model_name].interactionProperties[i].damage
            # Create contact
            a = mdb.models[model_name].rootAssembly    # Assign the rootAssembly object of instance part in Model-1 to a
            j=0
            for i in face_face_intpro_list:
                surface1=i[0]
                surface2=i[1]
                intpro1=i[2]
                region1=a.surfaces[surface1]
                region2=a.surfaces[surface2]
                dai_Al_contact_name='CTE_contact'+str(j)
                j+=1

                # When using second-order elements, improved elements are required for NODE_TO_SURFACE surface-to-surface contact
                mdb.models[model_name].SurfaceToSurfaceContactStd(name=dai_Al_contact_name, 
                    createStepName='Step-1', master=region1, slave=region2, sliding=FINITE, 
                    enforcement=NODE_TO_SURFACE, thickness=OFF, 
                    interactionProperty=intpro1, surfaceSmoothing=NONE, adjustMethod=NONE, 
                    smooth=0.2, initialClearance=OMIT, datumAxis=None, clearanceRegion=None)

            ######## Thermal-displacement load settings ###############
            # Create temperature field load
            a = mdb.models[model_name].rootAssembly
            instancename_list=a.allInstances.keys()
            setnum=0
            for instancename in instancename_list:
                #f = instance1.elements
                setnum=setnum+1
                c1 = a.instances[instancename].cells
                f1 = a.instances[instancename].faces
                e1 = a.instances[instancename].edges
                v1 = a.instances[instancename].vertices
                region = a.Set(vertices=v1, edges=e1, faces=f1, cells=c1, name='Set-Tempfield%s' % (setnum))
                mdb.models[model_name].TemperatureBC(name='BC-Temp%s' % (setnum), createStepName='Initial', 
                    region=region, distributionType=UNIFORM, fieldName='', magnitude=0.0)
                mdb.models[model_name].boundaryConditions['BC-Temp%s' % (setnum)].setValuesInStep(
                    stepName='Step-1', magnitude=5.0)

            # Create corner constraints: P0 fixed, P1-P3 constrain x, y, z directions respectively      
            a = mdb.models[model_name].rootAssembly
            instancename_list=a.allInstances.keys()
            v1_total=a.vertices
            for instancename in instancename_list:
                #f = instance1.elements
                setnum=setnum+1
                v1_total = v1_total+a.instances[instancename].vertices
            vertices_P0=v1_total.getByBoundingSphere(center=(-aldim/2, -aldim/2,0),radius=0.00001)
            vertices_P1=v1_total.getByBoundingSphere(center=(aldim/2, -aldim/2,0),radius=0.00001)
            vertices_P2=v1_total.getByBoundingSphere(center=(-aldim/2, -aldim/2,aldim),radius=0.00001)
            vertices_P3=v1_total.getByBoundingSphere(center=(-aldim/2, aldim/2,0),radius=0.00001)
            # P0 constraint
            region = a.Set(vertices=vertices_P0, name='Set-P0')
            mdb.models[model_name].PinnedBC(name='BC-P0', createStepName='Initial', 
                region=region, localCsys=None)
            # P1 constraint
            region = a.Set(vertices=vertices_P1, name='Set-P1')
            mdb.models[model_name].DisplacementBC(name='BC-P1', createStepName='Initial', 
                region=region, u1=UNSET, u2=UNSET, u3=SET, ur1=UNSET, ur2=UNSET, ur3=UNSET, 
                amplitude=UNSET, distributionType=UNIFORM, fieldName='', localCsys=None)
            # P2 constraint
            region = a.Set(vertices=vertices_P2, name='Set-P2')
            mdb.models[model_name].DisplacementBC(name='BC-P2', createStepName='Initial', 
                region=region, u1=UNSET, u2=SET, u3=UNSET, ur1=UNSET, ur2=UNSET, ur3=UNSET, 
                amplitude=UNSET, distributionType=UNIFORM, fieldName='', localCsys=None)
            # P3 constraint
            region = a.Set(vertices=vertices_P3, name='Set-P3')
            mdb.models[model_name].DisplacementBC(name='BC-P3', createStepName='Initial', 
                region=region, u1=SET, u2=UNSET, u3=UNSET, ur1=UNSET, ur2=UNSET, ur3=UNSET, 
                amplitude=UNSET, distributionType=UNIFORM, fieldName='', localCsys=None)
            
            ################ Meshing ###############
            elemType1 = mesh.ElemType(elemCode=C3D8T, elemLibrary=STANDARD)
            elemType2 = mesh.ElemType(elemCode=C3D6T, elemLibrary=STANDARD)
            elemType3 = mesh.ElemType(elemCode=C3D4T, elemLibrary=STANDARD,secondOrderAccuracy=OFF, distortionControl=DEFAULT)
            p = mdb.models[model_name].parts['dianmond_ass']
            c = p.cells
            pickedRegions = c.getByBoundingBox(xMin=-2*aldim,xMax=2*aldim,yMin=-aldim*2,yMax=aldim*2,zMin=-aldim*2,zMax=aldim*2)
            p.setMeshControls(regions=pickedRegions, elemShape=TET, technique=FREE, algorithm=DEFAULT, sizeGrowthRate=1.0, allowMapped=False)
            pickedRegions =(pickedRegions, )
            p.setElementType(regions=pickedRegions, elemTypes=(elemType3,))
            p.seedPart(size=mesh_size*acenter, deviationFactor=0.001, minSizeFactor=0.1)
            p.generateMesh()
            if type(p.getUnmeshedRegions())!=NoneType:
                continue
            p1 = mdb.models[model_name].parts['Al_body']
            c = p1.cells
            pickedRegions = c.getByBoundingBox(xMin=-2*aldim,xMax=2*aldim,yMin=-aldim*2,yMax=aldim*2,zMin=-aldim*2,zMax=aldim*2)
            p1.setMeshControls(regions=pickedRegions, elemShape=TET, technique=FREE, algorithm=DEFAULT, sizeGrowthRate=1.0, allowMapped=False)
            pickedRegions =(pickedRegions, )
            p1.setElementType(regions=pickedRegions, elemTypes=(elemType3,))
            p1.seedPart(size=mesh_size*acenter, deviationFactor=0.001, minSizeFactor=0.1)
            p1.generateMesh()
            if type(p1.getUnmeshedRegions())!=NoneType:
                continue
            
            if Period_boundary_lable==1:
                PBC.PBC_condation(modelname=model_name,partnamelist=['Al_body','dianmond_ass'],X=True,Y=True,Z=True,Error_Tol=0.0000005)

            ############# Create job and submit ################
            jobname=model_name #+str(list_i)
            mdb.Job(name=jobname, model=model_name, description='', type=ANALYSIS, 
                atTime=None, waitMinutes=0, waitHours=0, queue=None, memory=90, 
                memoryUnits=PERCENTAGE, getMemoryFromAnalysis=True, 
                explicitPrecision=SINGLE, nodalOutputPrecision=SINGLE, echoPrint=OFF, 
                modelPrint=OFF, contactPrint=OFF, historyPrint=OFF, userSubroutine='', 
                scratch='', resultsFormat=ODB, multiprocessingMode=DEFAULT, numCpus=2, 
                numDomains=2, numGPUs=1)
            mdb.jobs[jobname].submit(consistencyChecking=OFF)
            # starttime=time.time()
            # timetotal=0

            # while timetotal<3600.0 and (mdb.jobs[jobname].messages[-1].type==SUBMITTED or mdb.jobs[jobname].messages[-1].type==RUNNING):
            #     endtime=time.time()
            #     timetotal=endtime-starttime

            mdb.jobs[jobname].waitForCompletion()
            # if mdb.jobs[jobname].messages[-1].type==RUNNING:
            #     mdb.jobs[jobname].kill()

            if mdb.jobs[jobname].messages[-1].type!=JOB_COMPLETED:

                continue 


            ############## Extract results, calculate thermal expansion coefficient ################
            model_jobname=jobname+'.odb'
            odb_file.append(model_jobname)
            o=session.openOdb(name=model_jobname, readOnly=True)
            # Define volume variable
            # Define volume variable
            volume_part=0
            vol1=aldim**3
            
            repengzhang_list=[]

            # Get element volume
            frames=o.steps['Step-1'].frames
            
            for f in frames:
            
                fop=f.fieldOutputs
                fopVol=fop['EVOL']
                vol2=0

                for i in range(0,len(fopVol.values)):
                    vol2=vol2+fopVol.values[i].data
                ti_repengzhang=(vol2-vol1)/(5*vol1)
                repengzhang=ti_repengzhang/3
                repengzhang_list.append(repengzhang)
                vol1=vol2
            
            ti_repengzhang=(vol2-aldim*aldim*aldim)/(aldim*aldim*aldim*50)
            repengzhang=ti_repengzhang/3
        

            print 'Expansionline,Expansionvol,Expansionline2_by_vol and time were:'
            print(ti_repengzhang,repengzhang)

            o.close()

            simulation_data.append(repengzhang)
            simulation_data1.append(repengzhang)
            #simulation_data.append(repengzhang_list)



##########################
# Implicit analysis - mechanical property calculation settings
##########################
        if E_lable==1:
            model_name=model_name_E

            ######## Remove diamond fracture parameters and aluminum damage parameters, but retain aluminum plasticity parameters ################
            for i in  mdb.models[model_name].materials.keys():
                del mdb.models[model_name].materials[i].brittleCracking
            del mdb.models[model_name].materials['Al'].johnsonCookDamageInitiation

            ############ Create analysis step, static mechanical analysis ##############

            # Create analysis step
            mdb.models[model_name].StaticStep(name='Step-1', previous='Initial', maxNumInc=10000, initialInc=0.05, nlgeom=ON)
            
            # Create field output
            mdb.models[model_name].fieldOutputRequests['F-Output-1'].setValues(variables=( 'S', 'E', 'PE', 'PEEQ', 'PEMAG', 'LE', 'U', 'RF', 'CF', 'CSTRESS', 
                'CDISP', 'EVOL'),frequency=LAST_INCREMENT)
            mdb.models[model_name].fieldOutputRequests['F-Output-1'].setValues(timeInterval=0.1)

            # Create history output
            mdb.models[model_name].HistoryOutputRequest(name='H-Output-1', 
                createStepName='Step-1', variables=('CSTRESS', 'CFNM', 'CFN1', 'CFN2', 'CFN3', 'CFSM', 'CFS1', 
                'CFS2', 'CFS3', 'CFTM', 'CFT1', 'CFT2', 'CFT3', 'CAREA', 'ALLAE', 'ALLCD', 
                'ALLDMD', 'ALLEE', 'ALLFD', 'ALLIE', 'ALLJD', 'ALLKE', 'ALLKL', 'ALLPD', 
                'ALLQB', 'ALLSE', 'ALLSD', 'ALLVD', 'ALLWK', 'ETOTAL'),frequency=LAST_INCREMENT)
            
            ############ Remove mechanical-related settings in contact properties; assume no interface failure in elastic stage, also delete thermal interface behavior #############
            for i in mdb.models[model_name].interactionProperties.keys():
                del mdb.models[model_name].interactionProperties[i].tangentialBehavior
                del mdb.models[model_name].interactionProperties[i].normalBehavior
                #del mdb.models[model_name].interactionProperties[i].cohesiveBehavior
                del mdb.models[model_name].interactionProperties[i].damage
                del mdb.models[model_name].interactionProperties[i].thermalConductance

            # Create contact
            a = mdb.models[model_name].rootAssembly    # Assign the instance part object rootAssembly in Model-1 to a
            j=0
            for i in face_face_intpro_list:
                surface1=i[0]
                surface2=i[1]
                intpro1=i[2]
                region1=a.surfaces[surface1]
                region2=a.surfaces[surface2]
                dai_Al_contact_name='E_contact'+str(j)
                j+=1

                # When using quadratic elements, improved elements are needed to use NODE_TO_SURFACE surface-to-surface contact
                mdb.models[model_name].SurfaceToSurfaceContactStd(name=dai_Al_contact_name, 
                    createStepName='Step-1', master=region1, slave=region2, sliding=FINITE, 
                    enforcement=NODE_TO_SURFACE, thickness=OFF, 
                    interactionProperty=intpro1, surfaceSmoothing=NONE, adjustMethod=NONE, 
                    smooth=0.2, initialClearance=OMIT, datumAxis=None, clearanceRegion=None)


            ################ Mesh generation ###############
                
            elemType1 = mesh.ElemType(elemCode=C3D8R, elemLibrary=STANDARD)
            elemType2 = mesh.ElemType(elemCode=C3D6, elemLibrary=STANDARD)
            elemType3 = mesh.ElemType(elemCode=C3D4, elemLibrary=STANDARD,secondOrderAccuracy=OFF, distortionControl=DEFAULT)
            p = mdb.models[model_name].parts['dianmond_ass']
            c = p.cells
            pickedRegions = c.getByBoundingBox(xMin=-2*aldim,xMax=2*aldim,yMin=-aldim*2,yMax=aldim*2,zMin=-aldim*2,zMax=aldim*2)
            p.setMeshControls(regions=pickedRegions, elemShape=TET, technique=FREE, algorithm=DEFAULT, sizeGrowthRate=1.0, allowMapped=False)
            pickedRegions =(pickedRegions, )
            p.setElementType(regions=pickedRegions, elemTypes=(elemType3,))
            p.seedPart(size=mesh_size*acenter, deviationFactor=0.001, minSizeFactor=0.1)
            p.generateMesh()
            if type(p.getUnmeshedRegions())!=NoneType:
                continue
            p1 = mdb.models[model_name].parts['Al_body']
            c = p1.cells
            pickedRegions = c.getByBoundingBox(xMin=-2*aldim,xMax=2*aldim,yMin=-aldim*2,yMax=aldim*2,zMin=-aldim*2,zMax=aldim*2)
            p1.setMeshControls(regions=pickedRegions, elemShape=TET, technique=FREE, algorithm=DEFAULT, sizeGrowthRate=1.0, allowMapped=False)
            pickedRegions =(pickedRegions, )
            p1.setElementType(regions=pickedRegions, elemTypes=(elemType3,))
            p1.seedPart(size=mesh_size*acenter, deviationFactor=0.001, minSizeFactor=0.1)
            p1.generateMesh()
            if type(p1.getUnmeshedRegions())!=NoneType:
                continue
            if Period_boundary_lable==1:
                PBC.PBC_condation(modelname=model_name,partnamelist=['Al_body','dianmond_ass'],X=True,Y=True,Z=False,Error_Tol=0.0000005)
            
            ######## Apply displacement loads and constraint settings ###############

            # Z+ face
            a = mdb.models[model_name].rootAssembly
            a.regenerate()
            # a.ReferencePoint(point=(0.0, 0.0, 20.0))
            instancename_list=a.allInstances.keys()
            n1_total=a.nodes
            setnum=0
            for instancename in instancename_list:
                #f = instance1.elements
                setnum=setnum+1
                n1_total = n1_total+a.instances[instancename].nodes
            Nodes_ZP_1=n1_total.getByBoundingBox(xMin=-aldim*2,xMax=aldim*2,yMin=-aldim*2,yMax=aldim*2,zMin=aldim-0.00001,zMax=aldim+0.00001)
            Nodes_ZN_1=n1_total.getByBoundingBox(xMin=-aldim*2,xMax=aldim*2,yMin=-aldim*2,yMax=aldim*2,zMin=-0.00001,zMax=0.00001)
            mdb.models[model_name].rootAssembly.Set(name='Nodes_ZP_1',nodes=Nodes_ZP_1)
            mdb.models[model_name].rootAssembly.Set(name='Nodes_ZN_1',nodes=Nodes_ZN_1)

            # Establish corner constraints: fix P0, constrain P1-P3 in x, y, z directions respectively      
            a = mdb.models[model_name].rootAssembly
            instancename_list=a.allInstances.keys()
            v1_total=a.vertices
            for instancename in instancename_list:
                #f = instance1.elements
                setnum=setnum+1
                v1_total = v1_total+a.instances[instancename].vertices
            vertices_P0=v1_total.getByBoundingSphere(center=(-aldim/2, -aldim/2,0),radius=0.00001)
            vertices_P1=v1_total.getByBoundingSphere(center=(aldim/2, -aldim/2,0),radius=0.00001)
            vertices_P2=v1_total.getByBoundingSphere(center=(-aldim/2, -aldim/2,aldim),radius=0.00001)
            vertices_P3=v1_total.getByBoundingSphere(center=(-aldim/2, aldim/2,0),radius=0.00001)
            #Constrain P0
            region = a.Set(vertices=vertices_P0, name='Set-P0')
            mdb.models[model_name].EncastreBC(name='BC-P0', createStepName='Initial', 
                region=region, localCsys=None)            
            #Constrain P3
            region = a.Set(vertices=vertices_P3, name='Set-P3')
            mdb.models[model_name].DisplacementBC(name='BC-P3', createStepName='Initial', 
                region=region, u1=SET, u2=UNSET, u3=UNSET, ur1=UNSET, ur2=UNSET, ur3=UNSET, 
                amplitude=UNSET, distributionType=UNIFORM, fieldName='', localCsys=None)
            
            #Constrain Z+ plane     
            region = a.sets['Nodes_ZP_1']       
            mdb.models[model_name].DisplacementBC(name='BC-ZP', createStepName='Initial', 
                region=region, u1=UNSET, u2=UNSET, u3=SET, ur1=UNSET, ur2=UNSET, ur3=UNSET, 
                amplitude=UNSET, distributionType=UNIFORM, fieldName='', localCsys=None)
            mdb.models[model_name].boundaryConditions['BC-ZP'].setValuesInStep(
                stepName='Step-1', u3=-0.002*aldim)
            #Constrain Z- plane   
            region = a.sets['Nodes_ZN_1']       
            mdb.models[model_name].DisplacementBC(name='BC-ZN', createStepName='Initial', 
                region=region, u1=UNSET, u2=UNSET, u3=SET, ur1=UNSET, ur2=UNSET, ur3=UNSET, 
                amplitude=UNSET, distributionType=UNIFORM, fieldName='', localCsys=None)
            
            #############Create job and submit################
            jobname=model_name #+str(list_i)
            mdb.Job(name=jobname, model=model_name, description='', type=ANALYSIS, 
                atTime=None, waitMinutes=0, waitHours=0, queue=None, memory=90, 
                memoryUnits=PERCENTAGE, getMemoryFromAnalysis=True, 
                explicitPrecision=SINGLE, nodalOutputPrecision=SINGLE, echoPrint=OFF, 
                modelPrint=OFF, contactPrint=OFF, historyPrint=OFF, userSubroutine='', 
                scratch='', resultsFormat=ODB, multiprocessingMode=DEFAULT, numCpus=2, 
                numDomains=2, numGPUs=1)
            mdb.jobs[jobname].submit(consistencyChecking=OFF)
            mdb.jobs[jobname].waitForCompletion()

            if mdb.jobs[jobname].messages[-1].type!=JOB_COMPLETED:
                continue 


            ############## Extract results: thermal Young's modulus, Poisson's ratio and yield stress ################
            model_jobname=jobname+'.odb'
            odb_file.append(model_jobname)
            o=session.openOdb(name=model_jobname, readOnly=True)

            # Define node force variables
            
            E_sigma_list=()
            E_eps_list=()



            Nodes_force1=0
            Nodes_U1=0
            Nodes_force2=0
            Nodes_U2=0
            volume1=0
            volume2=0

            # Get node reaction forces and element volume data from results
            f1=o.steps['Step-1'].getFrame(frameValue=0.1)
            f2=o.steps['Step-1'].getFrame(frameValue=1.0)
            fop1=f1.fieldOutputs
            fop2=f2.fieldOutputs
            # Get node coordinate data
            fopRF1=fop1['RF']
            fopU1=fop1['U']
            fopRF2=fop2['RF']
            fopU2=fop2['U']
            fopVol1=fop1['EVOL']
            fopVol2=fop2['EVOL']

            # Create odbset object
            set_NODES_F_U=o.rootAssembly.nodeSets['NODES_ZP_1']

            # Get node reaction forces and element volume data from results
            frames=o.steps['Step-1'].frames
            for f in frames:
                Nodes_force=0
                Nodes_U=0
                volume=0
                fop=f.fieldOutputs
                fopRF=fop['RF']
                fopU=fop['U']
                fopVol=fop['EVOL']
                temp_F=fopRF.getSubset(region=set_NODES_F_U)
                temp_U=fopU.getSubset(region=set_NODES_F_U)

                for i in range(0,len(temp_F.values)):
                    Nodes_force=temp_F.values[i].data[2]+Nodes_force
                    Nodes_U=temp_U.values[i].data[2]+Nodes_U
                   
                Nodes_U=abs(Nodes_U/(len(temp_F.values)))

                for i in range(0,len(fopVol.values)):
                    volume=volume+fopVol.values[i].data

                d_z=aldim-Nodes_U
                effective_sigma=-Nodes_force/(aldim*aldim)
                effective_eps=Nodes_U/aldim
                E_eps_list=E_eps_list+(effective_eps,)
                E_sigma_list=E_sigma_list+(effective_sigma,)


            temp_F1=fopRF1.getSubset(region=set_NODES_F_U)
            temp_U1=fopU1.getSubset(region=set_NODES_F_U)
            temp_F2=fopRF2.getSubset(region=set_NODES_F_U)
            temp_U2=fopU2.getSubset(region=set_NODES_F_U)

            for i in range(0,len(temp_F1.values)):
                Nodes_force1=temp_F1.values[i].data[2]+Nodes_force1
                Nodes_U1=temp_U1.values[i].data[2]+Nodes_U1
                Nodes_force2=temp_F2.values[i].data[2]+Nodes_force2
                Nodes_U2=temp_U2.values[i].data[2]+Nodes_U2
            
            Nodes_U1=abs(Nodes_U1/(len(temp_F1.values)))
            Nodes_U2=abs(Nodes_U2/(len(temp_F1.values)))

            for i in range(0,len(fopVol1.values)):
                volume1=volume1+fopVol1.values[i].data
                volume2=volume2+fopVol2.values[i].data
            
            # Calculate Young's modulus, Poisson's ratio, and yield strength

            d_z=aldim-Nodes_U1
            d_z2=aldim-Nodes_U2
            effective_eps_t=(sqrt(volume1/d_z)-aldim)/aldim

            effective_sigma=-Nodes_force1/(aldim*aldim)
            effective_eps=Nodes_U1/aldim

            effective_E=effective_sigma/effective_eps
            effective_v=abs(effective_eps_t/effective_eps)

            effective_sigma_0_2=-Nodes_force2/(aldim*aldim)

            print(effective_E, effective_v,effective_sigma_0_2)

            o.close()

            simulation_data.append(effective_E)
            simulation_data.append(effective_v)
            simulation_data.append(effective_sigma_0_2)
            simulation_data1.append(effective_E)
            simulation_data1.append(effective_v)
            simulation_data1.append(effective_sigma_0_2)
            # simulation_data.append(E_eps_list)
            # simulation_data.append(E_sigma_list)
            
        simulation_data.append(repengzhang_list)
        simulation_data.append(E_eps_list)
        simulation_data.append(E_sigma_list)
        simulation_data1.append(repengzhang_list)
        simulation_data1.append(E_eps_list)
        simulation_data1.append(E_sigma_list)
        with open('diamond_data.csv', mode='ab+') as outfile:
            writer = csv.writer(outfile)
            data=simulation_data
            writer.writerow(data)
        with open('diamond_data_parameter.csv', mode='ab+') as outfile:
            writer = csv.writer(outfile)
            data=simulation_data1
            writer.writerow(data)
        with open('diamond_data0.csv', mode='ab+') as outfile:
            writer = csv.writer(outfile)
            data=simulation_data
            writer.writerow(data)
        with open('diamond_data_parameter0.csv', mode='ab+') as outfile:
            writer = csv.writer(outfile)
            data=simulation_data1
            writer.writerow(data)
        

        
    #压缩文件保存的路径
    zip_save_path='file_'+str(xunhuan)+'.zip'
    zip_odbfile(odb_file,zip_save_path)










    

