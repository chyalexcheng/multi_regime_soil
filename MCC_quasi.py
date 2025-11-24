# encoding: utf-8
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root 
from matplotlib import rcParams
rcParams['mathtext.fontset'] = 'cm'
rcParams['font.family'] = 'serif'

# Set figure size for half-width of A4
a4_half_width = 6  # in inches
fig_aspect_ratio = 4 / 3  # width-to-height ratio (adjust as needed)
fig_height = a4_half_width / fig_aspect_ratio
figsize = (a4_half_width, fig_height)

print('Input Parameters for Modified Cam-Clay:');

cp=input('Enter the inital Consolidation pressure (kPa] (eg., 150 kPa]  = ');
cp=150 if cp == '' else float(cp)
p0=input('Enter the initial Confining pressure (kPa]    (eg., 150 kPa]  = ');
p0=150 if p0 == '' else float(p0)
M=input( 'Enter the value of Critical Friction Angle M  (eg., 0.95)     = ');
M=0.95 if M == '' else float[m]
l=input( 'Enter the value of Lamda                      (eg., 0.2)      = ');
l=0.2 if l == '' else float(l)
k=input( 'Enter the value of Kappa                      (eg., 0.04)     = ');
k=0.04 if k == '' else float(k)
N=input( 'Enter the value of N                          (eg., 2.5)      = ');
N=2.5 if N == '' else float(N)
nu=input('Enter the value of poissons ratio             (eg., 0.15)     = ');
nu=0.15 if nu == '' else float(nu)
analysis = input('Enter the type of Analysis: [1] Triaxial Drained [2] Triaxial Undrained = ');
analysis=1 if analysis == '' else float(analysis)

if analysis==1:
    print('Triaxial Drained Simulation is in progress ...');
elif analysis==2:
    print('Triaxial Undrained Simulation is in progress ...');
else: exit();

# Computation of Other Parameters (V,e0 and OCR)
pc=cp;
V=N-(l*np.log(pc))+(k*np.log(pc/p0))    # Specific Volume
e0=V-1;    # Initial Void Ratio
OCR=cp/p0    # Over Consolidation Ratio

# Strain Increament and Strain Matrix Definition
print('Strain increament and nIteration:');
nIter=10000; de=1e-4;

# Block Memory allocation
De=np.zeros([6,6]);      # Stiffness Matrix
dfds=np.zeros([6,1]);
dfdep=np.zeros([6,1]);
dStrain=np.zeros([6,1]); # Increamental Strain

u=np.zeros([nIter+1,1]);    # Pore Water Pressure
p=np.zeros([nIter+1,1]);    # Mean Effective Stress
q=np.zeros([nIter+1,1]);    # Deviatoric Stress
void=np.zeros([nIter+1,1]); # Void ratio
epsV=np.zeros([nIter+1,1]); # Volumetic Strain
epsD=np.zeros([nIter+1,1]); # Deviatoric Strain

# yieldSurf Surface and Conditions
p_ini_yield=np.linspace(0,pc,nIter);
q_ini_yield = M*p_ini_yield;
qyf_ini=(M**2*(pc*p_ini_yield-p_ini_yield**2))**0.5;

# Initialize
a=0;                                    # nIterator
S=np.array([p0+1.0e-10,p0,p0,0,0,0])    # Stress 
strain=np.array([0,0,0,0,0,0])            # Strain
p[a]=(S[0]+2*S[2])/3;                
q[a]=(S[0]-S[1]);
yieldSurf=(q[a]**2/M**2+p[a]**2)-p[a]*pc; # Defining the yieldSurf surface
void[a]=e0

# CamClay nIteration for normal and over consolidated clay& Inside/Outside yieldSurf
while a<nIter:
    K=V*p[a]/k;                       # Bulk Modulus
    G=(3*K*(1-2*nu))/(2*(1+nu));      # Shear Modulus
    if yieldSurf==0: pc=(q[a]**2/M**2+p[a]**2)/p[a]; 
    else: pc=cp

    # Elastic Stiffness and other Matrix 
    for m in range(6):
       for n in range(6):
           if m <= 2:
               if yieldSurf < 0:
                   dfds[m,0]=0;
                   dfdep[m,0]=0
               else:
                   dfds[m,0]=(2*p[a]-pc)/3 + 3*(S[m]-p[a])/M**2; # df/ds
                   dfdep[m,0]=(-p[a])*pc*(1+void[a])/(l-k)*1; # df/dep
               if m==n:
                   De[m,n]=K+4/3*G; # Elastic Stiffness
               elif n<=2:
                   De[m,n]=K-2/3*G;
           if m>2:
               dfds[m,0]=0;
               dfdep[m,0]=0;
               if m==n: De[m,n]= G;  # Elastic Stiffness
               else: De[m,n]=0;

    #Stiffness Matrix  
    if yieldSurf<0: D=De; #Elastic
    else: D=De-(De.dot(dfds).dot(dfds.T).dot(De))/(-(dfdep.T).dot(dfds)+(dfds.T).dot(De).dot(dfds)); #Plastic
    #~ if a%(nIter/20) == 0:
        #~ print '(dfdep.T).dot(dfds):%.f; -p[a]*V*pc/(l-k)*(2*p[a]-pc):%.f'%((dfdep.T).dot(dfds),-p[a]*V*pc/(l-k)*(2*p[a]-pc))

    #Stress and Strain Updates
    if analysis==1: #Triaxial Drained (?)
       dStrain = np.array([de,-1*D[1,0]/(D[1,1]+D[1,2])*de,-1*D[2,0]/(D[2,1]+D[2,2])*de,0.,0.,0.]);
    elif analysis==2: #Triaxial Undrained
       dStrain = np.array([de,-de/2.,-de/2.,0.,0.,0.]);

    dS=D.dot(dStrain);
    S=S+dS;
    strain=strain+dStrain;

    depsV = dStrain[0] + dStrain[1] + dStrain[2]; # Increamental Volumetric Strain
    depsD = 2./3. * (dStrain[0] - dStrain[2]);      # Increamental Deviatoric Strain

    #Update Specific Volume
    V=N-(l*np.log(pc))+(k*np.log(pc/p[a]));

    #Subsequent cycle update
    a=a+1;
    p[a]=(S[0]+S[1]+S[2])/3;
    q[a]=S[0]-S[2];
    u[a]=p0+q[a]/3-p[a];

    void[a] = V-1.0;
    epsV[a] = epsV[a-1] + depsV;
    epsD[a] = epsD[a-1] + depsD;

    if yieldSurf<0: yieldSurf=q[a]**2+M**2*p[a]**2-M**2*p[a]*pc;
    else: yieldSurf=0;
   
    # Normal Consolidation Line (NCL)
    pNCL = np.linspace(1,pc,nIter)
    qNCL = M * pNCL;
    eNCL = (N - l*np.log(pNCL)) - 1;
     
    # Critical State Line (CSL)
    pCSL = pNCL;
    Gamma = 1+void[a]+l*np.log(p[a]);
    eCSL = (Gamma - l*np.log(pCSL)) - 1;
     
    # Final Yield Surface
    # CSL in p-q space
    p_fyield = np.linspace(1,pc,nIter);
    q_fyield = M*p_fyield;
    # Plot the final yield locus
    qyf = (M**2*(pc*p_fyield-p_fyield**2))**0.5;    
    
    if a%(nIter/200) == 0:
        plt.figure(figsize=figsize)
#       plt.plot(np.array([0,2.0*p[a,0]]),np.array([0,2.0*q[a,0]]),'k--')
        plt.plot(p[p.nonzero()],q[q.nonzero()],'r',label='Stress path');
        plt.plot(p_fyield,q_fyield,'k');
        plt.plot(p_fyield,qyf,'b-',label='Current yield surface');
        plt.plot(p_ini_yield,qyf_ini,'g',label='Initial yield surface');
        plt.xlabel(r'$p$'); plt.ylabel(r'$q$');
        plt.legend()
        plt.show()
