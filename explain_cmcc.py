# encoding: utf-8
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root 
from matplotlib import rcParams
rcParams['mathtext.fontset'] = 'cm'
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 14

# Set figure size for half-width of A4
a4_half_width = 6  # in inches
fig_aspect_ratio = 4 / 3  # width-to-height ratio (adjust as needed)
fig_height = a4_half_width / fig_aspect_ratio
figsize = (a4_half_width, fig_height)

# Cam-clay critical state parameters
pc_0 = 1 * 150.0  # 'Initial consolidation pressure [kPa]'
M = 1.0  # 'Critical friction angle'
lambda_val = 0.2  # 'Lambda'
kappa = 0.04  # 'kappa'
Gamma = 1.39  # 'Intercept of the critical state line'
N = 2.5  # 'Intercept of the normal consolidation line'
nu = 0.15  # 'Poisson ratio'

#%% CSL
pCSL = np.linspace(0.001,110,110)
qCSL = M * pCSL;
eCSL = Gamma - lambda_val*np.log(pCSL)
p_yield = np.linspace(0,100,1000);
q_yield = (M**2*(100*p_yield-p_yield**2))**0.5;    

plt.figure(figsize=figsize)
plt.plot(pCSL, qCSL, 'r',label='Critical state line')
plt.plot(p_yield,q_yield,'k-',label='Yield surface');
plt.xlabel(r'Pressure $p$')
plt.ylabel(r'Deviatoric stress $q$')
plt.xlim(0,110)
plt.ylim(0)
plt.legend()
plt.savefig("p_q_CSL.png", dpi=300, bbox_inches="tight")

plt.figure(figsize=figsize)
pCSL = np.linspace(1,110,110)
eCSL = Gamma - lambda_val*np.log(pCSL);
plt.semilogx(pCSL, eCSL, 'r')
plt.xlabel(r'Pressure $p$ [$p_\mathrm{ref}$]')
plt.ylabel(r'Void ratio $e=\frac{1}{\phi}-1$')
plt.xlim(1,110)
plt.savefig("e_p_CSL.png", dpi=300, bbox_inches="tight")

#%% CSL_dynamic
pCSL = np.linspace(0.001,110,110)
qCSL = M * pCSL;
eCSL = Gamma - lambda_val*np.log(pCSL)
p_yield = np.linspace(0,100,1000);
q_yield = (M**2*(100*p_yield-p_yield**2))**0.5;    

plt.figure(figsize=figsize)
plt.plot(pCSL, qCSL, 'r',label='Critical state line')
plt.plot(p_yield,q_yield,'k-',label='Yield surface');
plt.plot(p_yield[500],q_yield[500],'ko');
plt.text(p_yield[500] - 4, q_yield[500] - 8, r'$(p_1^{\mathrm{q}}, q_1^{\mathrm{q}})$', fontsize=16)  # label near point A

p_yield = np.linspace(0,50,1000);
q_yield = (M**2*(50*p_yield-p_yield**2))**0.5;    
plt.plot(p_yield,q_yield,'k-');
plt.plot(p_yield[500],q_yield[500],'ko');
plt.text(p_yield[500] - 4, q_yield[500] - 8, r'$(p_2^{\mathrm{q}}, q_2^{\mathrm{q}})$', fontsize=16)  # label near point B

p_yield = np.linspace(0,75,1000);
q_yield = (M**2*(75*p_yield-p_yield**2))**0.5;    
plt.plot(p_yield,q_yield,'k--');

# Coordinates of A and B
A_x, A_y = 50, (M**2 * (100 * 50 - 50**2))**0.5
B_x, B_y = 25, (M**2 * (50 * 25 - 25**2))**0.5

# Draw arrow from A to B
plt.annotate(
    '', xy=(B_x, B_y), xytext=(A_x, A_y),
    arrowprops=dict(arrowstyle='->', color='blue', linestyle=(0, (5, 5)), linewidth=2)
)

plt.xlabel(r'Pressure $p$ [$p_\mathrm{ref}$]')
plt.ylabel(r'Deviatoric stress $q$ [$p_\mathrm{ref}$]')
plt.xlim(0,110)
plt.ylim(0)
plt.legend()
plt.savefig("p_q_c.png", dpi=300, bbox_inches="tight")

plt.figure(figsize=figsize)
pCSL = np.linspace(10,110,110)
eCSL = Gamma - lambda_val*np.log(pCSL);
plt.semilogx(pCSL, eCSL, 'r')
e_point_25 = Gamma - lambda_val * np.log(25)
e_point_50 = Gamma - lambda_val * np.log(50)
plt.semilogx(25, e_point_25,'ko');
plt.semilogx(50, e_point_50,'ko');
plt.text(25 * 1.06, e_point_25, r'$(e_2^{\mathrm{q}}, p_2^{\mathrm{q}})$', fontsize=16)
plt.text(50 * 1.06, e_point_50, r'$(e_1^{\mathrm{q}}, p_1^{\mathrm{q}})$', fontsize=16)

A_x, A_y = 50, (Gamma - lambda_val * np.log(50))
B_x, B_y = 25, (Gamma - lambda_val * np.log(25))
# Draw arrow from A to B
plt.annotate(
    '', xy=(B_x, B_y), xytext=(A_x, A_y),
    arrowprops=dict(arrowstyle='->', color='blue', linestyle=(0, (5, 5)), linewidth=2)
)

plt.xlabel(r'Pressure $p$ [$p_\mathrm{ref}$]')
plt.ylabel(r'Void ratio $e=\frac{1}{\phi}-1$')
plt.xlim(10,110)
ylim = plt.ylim()
plt.savefig("e_p_c.png", dpi=300, bbox_inches="tight")

#%% Triaxial stress path
pCSL = np.linspace(0.001,110,110)
qCSL = M * pCSL;
eCSL = (Gamma - lambda_val*np.log(pCSL));
p_yield = np.linspace(0,100,1000);
q_yield = (M**2*(100*p_yield-p_yield**2))**0.5  

plt.figure(figsize=figsize)
plt.plot(pCSL, qCSL, 'r', label='Critical state line')
plt.plot(p_yield,q_yield,'k-',label='Current yield surface');

p_yield = np.linspace(0,50,1000);
q_yield = (M**2*(50*p_yield-p_yield**2))**0.5;    
plt.plot(p_yield,q_yield,'k--',label='Initial yield surface')
plt.plot([50, 50],[0, 50],'b-', label='Stress path')
plt.annotate('', xy=(50, 50+0.5), xytext=(50, 0),
             arrowprops=dict(arrowstyle='-|>, head_width=0.4, head_length=0.7', color='b'))
plt.text(43, 56, r'$(\frac{p_\mathrm{c}}{2},\frac{\mu^\mathrm{cs} p_\mathrm{c}}{2})$', fontsize=16, va='center')

plt.xlim(0,110)
plt.ylim(0)
plt.legend()
plt.xlabel(r'Pressure $p$ [$p_\mathrm{ref}$]')
plt.ylabel(r'Deviatoric stress $q$ [$p_\mathrm{ref}$]')
plt.savefig("MCC_pq.png", dpi=300, bbox_inches="tight")

plt.figure(figsize=figsize)
eNCL = (N - lambda_val*np.log(pCSL)) - 1
plt.semilogx(pCSL, eCSL, 'r', label='Critical state line')
plt.semilogx(pCSL, eNCL, 'g', label='Normal consolidation line')
# draw a straight line acrossing pCSL[50], eCSL[50] in the semilogx plot with a slope of -kappa.
a_x = np.array([pCSL[50] / 10, pCSL[50] * 2])
a_y = eCSL[50] - kappa * np.log(a_x / pCSL[50])
plt.semilogx(a_x, a_y, 'g--', label='Swelling line')
plt.semilogx([pCSL[50], pCSL[50]],[eNCL[50], eCSL[50]],'b-', label='Stress path')
plt.annotate('', xy=(pCSL[50], eCSL[50]-0.005), xytext=(pCSL[50], eNCL[50]),
             arrowprops=dict(arrowstyle='-|>, head_width=0.4, head_length=0.7', color='b'))

plt.xlabel(r'Pressure $p$ [$p_\mathrm{ref}$]')
plt.ylabel(r'Void ratio $e=\frac{1}{\phi}-1$')
plt.xlim(10,110)
plt.ylim(ylim)
plt.legend(loc='upper right')
plt.savefig("MCC_ep.png", dpi=300, bbox_inches="tight")
plt.show()