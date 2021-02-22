#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 09:48:28 2020

Equation of motion for body and tail
(assuming no external forces or torques)
following Jusufi, 2010 (DOI: 10.1088/1748-3182/5/4/045001)

Moment of inertia calculations following Kwon3D: http://www.kwon3d.com/theory/bspeq/bspeq.html

@author: emilyhager
"""

import numpy as np

def get_initial_motion_params(s):
    ''' Return the initial parameters needed for tail use simulations: 
        Initial body and tail angles as per Jusufi, 2010
        Time sequence, sequence of gamma and eta angles.
        Options for s are: 
            1 = roll0
            2 = roll45
            3 = pitch
            4 = yaw
        '''

    if s in [1, 'roll0']:
        # SCENARIO 1:
        # Mouse starts upside down.
        # Tail is held maximally angled wrt body (gamma = 0)
        # Tail starts all the way to the left and is swung 180 degrees to the right in a half-cosine.
        psi0 = 0
        theta0 = 0
        phi0 = -np.pi # start upsidedown.
        
        gamma0 = 0 # tail upright at 90 degrees from body
        eta0 = np.pi/2 # tail all the way to the mouse's left.
        
        x = np.linspace(0,np.pi,111)
        c = np.cos(x)
        
        times = np.linspace(0,0.110, 111)
        
        # eta motion
        eta_profile = c*np.pi/2# goes like half-cosine with duration of 110 ms
        gamma_profile = np.zeros(eta_profile.shape) # does not change.
        
        
        body_init = np.array([psi0, theta0, phi0])
        tail_init = np.array([eta0, gamma0])
        
        body_index = 2
        tail_index = 0
    
    if s in [2, 'roll45']:
        
        # SCENARIO 2:
        # Mouse starts upside down.
        # Tail is held at 45 degrees wrt body (gamma = np.pi/4)
        # Tail starts all the way to the left and is swung 180 degrees to the right in a half-cosine.
        psi0 = 0
        theta0 = 0
        phi0 = -np.pi # start upsidedown.
        
        gamma0 = np.pi/4# tail upright at 90 degrees from body
        eta0 = np.pi/2 # tail all the way to the mouse's left.
        
        x = np.linspace(0,np.pi,111)
        c = np.cos(x)
        
        times = np.linspace(0,0.110, 111)
        
        # eta motion
        eta_profile = c*np.pi/2# goes like half-cosine with duration of 110 ms
        gamma_profile = np.zeros(eta_profile.shape) # does not change.
        
        
        body_init = np.array([psi0, theta0, phi0])
        tail_init = np.array([eta0, gamma0])
        
        body_index = 2
        tail_index = 0
        
    if s in [3, 'pitch']:
        
        # SCENARIO 3:
        # Mouse starts pitched up, rotated 90 degrees to the side.
        # Tail is held at 45 degrees wrt body (gamma = np.pi/4)
        # Tail starts all the way to the left and is swung 180 degrees to the right in a linear fashion.
        
        psi0 = np.pi/2
        theta0 = -15*np.pi/32 # matrix singularity when start w/ -pi/2
        phi0 = 0
        
        gamma0 = np.pi# tail 90 degrees ventral
        eta0 = 0 # tail at mouse midline
        
        
        x = np.linspace(0,np.pi,111)
        c = np.cos(x)
        
        times = np.linspace(0,0.110, 111)
        
        # eta motion
        gamma_profile = np.pi/2 * c + np.pi/2
        eta_profile = np.zeros(gamma_profile.shape) # does not change.

        
        body_init = np.array([psi0, theta0, phi0])
        tail_init = np.array([eta0, gamma0])

        
        body_index = 1
        tail_index = 1
        
    if s in [4, 'yaw']:
        
        # SCENARIO 4: Note, because of symmetries expect same result as 3/pitch
        # Mouse starts upright (0,0,0).
        # Tail is held at 45 degrees wrt body (eta = -np.pi/4)
        # Tail starts all the way to the left and is swung 180 degrees to the right in a linear fashion.
        psi0 = 0
        theta0 = 0
        phi0 = 0
        
        gamma0 = np.pi# tail 90 degrees ventral
        eta0 = -np.pi/2 # tail flat
        
        
        x = np.linspace(0,np.pi,111)
        c = np.cos(x)
        
        times = np.linspace(0,0.110, 111)
        
        # eta motion
        gamma_profile = np.pi/2 * c + np.pi/2
        eta_profile = np.zeros(gamma_profile.shape) # does not change.

        
        body_init = np.array([psi0, theta0, phi0])
        tail_init = np.array([eta0, gamma0])
        
        body_index = 0
        tail_index = 1
        
    if not(s in [1, 2, 3, 4, 'roll0', 'roll45', 'pitch', 'yaw']):
        
        body_init = None
        tail_init = None
        eta_profile = None
        gamma_profile = None
        times = None
        body_index = None
        tail_index = None
        print('Scenario must be one of: 1, 2, 3, 4, roll0, roll45, pitch, yaw; not {}'.format(s))
    
    return [body_init, tail_init, eta_profile, gamma_profile, times, body_index, tail_index]


def tail_motion_simulation(body_params, tail_params, motion_params):
    ''' 
    Run a single instance of the simulation and return degrees of body rotation
    per degree tail rotation on the relevant axis.'''
    body_init, tail_init, eta_profile, gamma_profile, times, body_index, tail_index = motion_params
    body_output, tail_output = run_sim(body_params, tail_params, body_init, tail_init, eta_profile, gamma_profile, times)
    deg_change_body = (np.max(body_output[:,body_index])-
                       np.min(body_output[:,body_index])) * 180/np.pi 
    deg_change_tail = (np.max(tail_output[:,tail_index])-
                       np.min(tail_output[:,tail_index])) * 180/np.pi # expect to be 180 deg. 
    output_ratio = deg_change_body/deg_change_tail
    return output_ratio

    

# Equation of motion: A ydot = f1 etadot + f2 gammadot.
def get_jusufi_components(body_angles, body_params, tail_angles, tail_params):

    # extract parameters
    body_MOI, body_L, body_M = body_params
    tail_MOI, tail_L, tail_M = tail_params
    Jt1, Jt2, Jt3 = tail_MOI # MOI around tail COM: x, y, z
    Jb1, Jb2, Jb3 = body_MOI

    psi, theta, phi = body_angles
    eta, gamma = tail_angles

    L1, L2 = body_L, tail_L

    # mass ratio
    m = (body_M * tail_M)/(body_M + tail_M)

    # needed trig functions
    S_eta = np.sin(eta)
    S_gamma = np.sin(gamma)
    S_phi = np.sin(phi)
    S_theta = np.sin(theta)

    C_eta = np.cos(eta)
    C_gamma = np.cos(gamma)
    C_phi = np.cos(phi)
    C_theta = np.cos(theta)

    S_2gamma = np.sin(2*gamma)
    C_2gamma = np.cos(2*gamma)

    C_phieta = np.cos(phi + eta)
    S_phieta = np.sin(phi + eta)

    S_2etaphi = np.sin(2*eta + phi)
    C_2etaphi = np.cos(2*eta + phi)

    # These all align with the paper.

    # A
    A11 = 0.25 * (2 * S_eta * (Jt2 - Jt3 + m * np.square(L2)) * (S_2gamma * S_theta - C_2gamma * C_phieta * C_theta) + 4 * m * L1 * L2 * (2*C_theta*S_gamma*S_phi - C_gamma * S_eta * S_theta) - C_theta * (4 * C_eta * S_phieta * Jt1 + 4 * S_phi * (Jb1 + m * np.square(L1)) - 2 * C_phieta * S_eta * (Jt2 + Jt3) + m * np.square(L2) * (S_2etaphi + 3 * S_phi) ))
    # In the paper, 4*m*L1*L2 is marked L1*L1 ^

    A12 = C_phi * Jb1 + C_eta * C_phieta * Jt1 + m * (C_eta * C_phieta * np.square(C_gamma * L2) + C_phi * np.square(L1 - L2 * S_gamma)) + S_eta * S_phieta * (Jt3 * np.square(C_gamma) + Jt2* np.square(S_gamma))

    A13 = C_gamma * S_eta * ((Jt2 - Jt3 + m * np.square(L2)) * S_gamma - m * L1 * L2)

    A21 = Jt2 * C_gamma * (C_gamma * S_theta + C_phieta*C_theta*S_gamma) + Jt3 * S_gamma * (S_gamma * S_theta - C_phieta * C_theta * C_gamma) + m * L2 * C_gamma * (L2 * (C_gamma * S_theta + C_phieta * C_theta * S_gamma) - C_phieta*C_theta*L1) + Jb2*S_theta

    A22 = C_gamma * S_phieta * ((Jt2 - Jt3 + m * np.square(L2)) * S_gamma - m * L1 * L2)

    A23 = Jb2 + (Jt2 + m*np.square(L2))*np.square(C_gamma) + Jt3 * np.square(S_gamma)

    A31 = 0.25 * ( 2* C_eta * (Jt2 - Jt3 + m * np.square(L2)) * (S_2gamma * S_theta - C_2gamma*C_phieta*C_theta) - 4 * m * L1 * L2 * (C_gamma * C_eta * S_theta + 2*C_theta * S_gamma * C_phi) + C_theta * (4 * S_eta * S_phieta * Jt1 + 4 * C_phi * (Jb3 + m* np.square(L1)) + 2*C_phieta*C_eta*(Jt2+Jt3) - m * np.square(L2)*(C_2etaphi - 3*C_phi)))
    # In the paper, 4*m*L1*L2 is marked L1*L1 ^

    A32 = S_phi * Jb3 - S_eta * C_phieta * Jt1 + m * (S_phi * np.square(L1 - L2* S_gamma) - S_eta * C_phieta * np.square(C_gamma * L2))+ C_eta * S_phieta * (Jt3 * np.square(C_gamma) + Jt2*np.square(S_gamma))

    A33 = C_gamma * C_eta * ((Jt2 - Jt3 + m * np.square(L2)) * S_gamma - m * L1 * L2)

    # f1
    f11 = -C_gamma * S_eta * ((Jt2 - Jt3 + m * np.square(L2)) * S_gamma - m * L1 * L2)

    f12 = -(Jt2 + m * np.square(L2)) * np.square(C_gamma) - Jt3 * np.square(S_gamma)

    f13 = -C_gamma * C_eta * ((Jt2 - Jt3 + m * np.square(L2)) * S_gamma - m * L1 * L2)

    # f2

    f21 = -C_eta * (Jt1 + m * L2 * (L2 - L1 * S_gamma))

    f22 = 0

    f23 = S_eta * (Jt1 + m * L2 * (L2 - L1 * S_gamma))


    A = np.array([[A11, A12, A13],[A21, A22, A23], [A31, A32, A33]])
    f1 = np.array([[f11],[f12],[f13]])
    f2 = np.array([[f21],[f22],[f23]])

    return A, f1, f2

def update_angles(body_angles, body_params, tail_angles, tail_params, delta_tail_angles, delta_time):
    deta, dgamma = delta_tail_angles
    etadot = deta/delta_time
    gammadot = dgamma/delta_time
    A, f1, f2 = get_jusufi_components(body_angles, body_params, tail_angles, tail_params)

    RHS = np.multiply(f1, etadot) + np.multiply(f2, gammadot)
    ydot = np.matmul(np.linalg.inv(A),RHS)
    delta_body_angles = ydot * delta_time

    return np.squeeze(delta_body_angles)+body_angles, np.squeeze(delta_tail_angles) + tail_angles

def run_sim(body_params, tail_params, body_init, tail_init, eta_profile, gamma_profile, times):
    d_eta_profile = np.diff(eta_profile)
    d_time_profile = np.diff(times)
    d_gamma_profile= np.diff(gamma_profile)

    # to hold output
    body_angles_out = np.zeros((eta_profile.shape[0],3))
    tail_angles_out = np.zeros((eta_profile.shape[0],2))
    body_angles_out[0,:] = body_init.copy()
    tail_angles_out[0,:] = tail_init.copy()

    # initiate
    body_angles = body_init.copy()
    tail_angles = tail_init.copy()

    # run sim
    for i in range(len(d_eta_profile)):
        delta_tail_angles = np.array([d_eta_profile[i], d_gamma_profile[i]])
        delta_time = d_time_profile[i]
        body_angles, tail_angles = update_angles(body_angles.T, body_params, tail_angles.T, tail_params, delta_tail_angles, delta_time)
        body_angles_out[i+1,:] = body_angles.copy()
        tail_angles_out[i+1,:] = tail_angles.copy()

    return body_angles_out, tail_angles_out


def kwon_F20(a0,a1):
    return np.square(a0)

def kwon_F21(a0,a1):
    return 2 * a0 * (a1-a0)

def kwon_F22(a0,a1):
    return np.square(a1-a0)

def kwon_F40(a0,a1):
    return np.power(a0,4)

def kwon_F41(a0,a1):
    return 4 * (a1-a0) * np.power(a0,3)

def kwon_F42(a0,a1):
    return 6 * np.square(a0) * np.square(a1-a0)

def kwon_F43(a0,a1):
    return 4 * a0 * np.power((a1-a0),3)

def kwon_F44(a0,a1):
    return np.power((a1-a0),4)

def kwon_G20(a0,a1):
    f22 = kwon_F22(a0,a1)
    f21 = kwon_F21(a0,a1)
    f20 = kwon_F20(a0,a1)
    return f22/3 + f21/2 + f20

def kwon_G21(a0,a1):
    f22 = kwon_F22(a0,a1)
    f21 = kwon_F21(a0,a1)
    f20 = kwon_F20(a0,a1)
    return f22/4 + f21/3 + f20/2

def kwon_G22(a0,a1):
    f22 = kwon_F22(a0,a1)
    f21 = kwon_F21(a0,a1)
    f20 = kwon_F20(a0,a1)
    return f22/5 + f21/4 + f20/3

def kwon_G40(a0,a1):
    f44 = kwon_F44(a0,a1)
    f43 = kwon_F43(a0,a1)
    f42 = kwon_F42(a0,a1)
    f41 = kwon_F41(a0,a1)
    f40 = kwon_F40(a0,a1)
    return f44/5 + f43/4 + f42/3 + f41/2 + f40

def kwon_m(a0,a1,rho,L):
    '''Get mass for a conical frustum, following Kwon 3D'''
    g20 = kwon_G20(a0,a1)
    return np.pi * rho * L * g20

def gz_com(a0,a1,L):
    '''Return the COM height for a conical frustum from the base (where r = a0). Following Kwon 3D.'''
    g21 = kwon_G21(a0,a1)
    g20 = kwon_G20(a0,a1)
    return g21 * L / g20


def Iz(a0,a1,rho,L):
    ''' Return Iz for a conical frustum, following Kwon 3D.'''
    g40 = kwon_G40(a0,a1)
    return np.pi * rho * L * g40 / 2

def Ix(a0,a1,rho,L):
    ''' Return Ix for a conical frustum swung at the base (where r = a0), following Kwon 3D.'''

    g40 = kwon_G40(a0,a1)
    g22 = kwon_G22(a0,a1)
    t1 = np.pi * rho * L * g40 / 4
    t2 = np.pi * rho * np.power(L,3) * g22
    return t1 + t2

def Ixx(a0,a1,rho,L):
    ''' Return principal Ixx for a conical frustum following Kwon 3D'''
    ix = Ix(a0,a1,rho,L)
    gz = gz_com(a0,a1,L)
    m = kwon_m(a0,a1,rho,L)
    return ix - m * np.square(gz)

def Iy(a0,a1,rho,L):
    ''' Return Iy for a conical frustum swung at the base (where r = a0), following Kwon 3D. Identical to Ix'''

    return Ix(a0,a1,rho,L)

def Iyy(a0,a1,rho,L):
    ''' Return principal Iyy for a conical frustum following Kwon 3D. Identical to Ixx'''

    return Ixx(a0,a1,rho,L)


def ifunc(v,ax):
    r0,r1,rho,L = v
    if ax in ['x','y']:
        return Ix(r0,r1,rho,L)
    if ax in ['xx','yy']:
        return Ixx(r0,r1,rho,L)
    if ax in ['z']:
        return Iz(r0,r1,rho,L)
    if ax in ['gz_com']:
        return gz_com(r0,r1,L)
    else:
        print('oops! second arg must be one of: x,y,xx,yy,z,gz_com')


def ellipsoid_MOI(m,a,b,c):
    ''' Solid ellipsoid MOI around COM.'''
    # from here: https://en.wikipedia.org/wiki/List_of_moments_of_inertia
    Ixx = 0.2 * m * (np.square(b) + np.square(c))
    Iyy = 0.2 * m * (np.square(a) + np.square(c))
    Izz = 0.2 * m * (np.square(a) + np.square(b))
    return Ixx, Iyy, Izz


