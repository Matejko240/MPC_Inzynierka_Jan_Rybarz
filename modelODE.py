import numpy as np
from scipy.integrate import ode
import matplotlib.pyplot as plt
from effectorTrajectoryGenerator3D import effectorTrajectoryGenerator3D

def model_ode(t, input_args, parameters):
    # get input values

    # manipulator joint velocities
    qr_d1 = input_args[0:3]
    q_d1 = qr_d1.reshape(-1, 1)
    # manipulator joint position
    qr = input_args[3:6]
    q = qr

    m1 = parameters['m1']
    m2 = parameters['m2']
    m3 = parameters['m3']
    mc = parameters['mc']

    l1 = parameters['l1']
    l2 = parameters['l2']
    l3 = parameters['l3']

    a = parameters['a']
    b = parameters['b']
    c = parameters['c']

    g = parameters['g']

    Tv = parameters['Tv']
    Ts = parameters['Ts']
    Tk = parameters['Tk']

    Kp = parameters['Kp']
    Kd = parameters['Kd']

    # Pozycja efektora końcowego
    xch = (
        a + l1 * np.cos(qr[0]) - l2 * (np.sin(qr[0]) * np.sin(qr[1]) - np.cos(qr[0]) * np.cos(qr[1])) - 
        l3 * np.cos(qr[2]) * (np.sin(qr[0]) * np.sin(qr[1]) - np.cos(qr[0]) * np.cos(qr[1]))
    )
    ych = (
        b + l1 * np.sin(qr[0]) + l2 * (np.cos(qr[0]) * np.sin(qr[1]) + np.cos(qr[1]) * np.sin(qr[0])) + 
        l3 * np.cos(qr[2]) * (np.cos(qr[0]) * np.sin(qr[1]) + np.cos(qr[1]) * np.sin(qr[0]))
    )
    zch = c + l3 * np.sin(qr[2])
    # Prędkość efektora końcowego
    xch_d1 = (
        - l3 * (
            np.cos(qr[2]) * (np.cos(qr[0]) * np.sin(qr[1]) * qr_d1[0] + np.cos(qr[1]) * np.sin(qr[0]) * qr_d1[0] + 
            np.cos(qr[0]) * np.sin(qr[1]) * qr_d1[1] + np.cos(qr[1]) * np.sin(qr[0]) * qr_d1[1]) - 
            np.sin(qr[2]) * (np.sin(qr[0]) * np.sin(qr[1]) - np.cos(qr[0]) * np.cos(qr[1])) * qr_d1[2]
        ) - l2 * (
            np.cos(qr[0]) * np.sin(qr[1]) * qr_d1[0] + np.cos(qr[1]) * np.sin(qr[0]) * qr_d1[0] + 
            np.cos(qr[0]) * np.sin(qr[1]) * qr_d1[1] + np.cos(qr[1]) * np.sin(qr[0]) * qr_d1[1]
        ) - l1 * np.sin(qr[0]) * qr_d1[0]
    )
    ych_d1 = (
        l3 * (
            np.cos(qr[2]) * (np.cos(qr[0]) * np.cos(qr[1]) * qr_d1[0] + np.cos(qr[0]) * np.cos(qr[1]) * qr_d1[1] - 
            np.sin(qr[0]) * np.sin(qr[1]) * qr_d1[0] - np.sin(qr[0]) * np.sin(qr[1]) * qr_d1[1]) - 
            np.sin(qr[2]) * (np.cos(qr[0]) * np.sin(qr[1]) + np.cos(qr[1]) * np.sin(qr[0])) * qr_d1[2]
        ) + l2 * (
            np.cos(qr[0]) * np.cos(qr[1]) * qr_d1[0] + np.cos(qr[0]) * np.cos(qr[1]) * qr_d1[1] - 
            np.sin(qr[0]) * np.sin(qr[1]) * qr_d1[0] - np.sin(qr[0]) * np.sin(qr[1]) * qr_d1[1]
        ) + l1 * np.cos(qr[0]) * qr_d1[0]
    )
    zch_d1 = l3 * np.cos(qr[2]) * qr_d1[2]

    # calculate dynamics
    # Macierz masy
    M = np.array([
        [(l1**2 * m1) / 3 + l1**2 * m2 + l1**2 * m3 + (l2**2 * m2) / 3 + l2**2 * m3 + (l3**2 * m3 * np.cos(qr[2])**2) / 3 + l1 * l2 * m2 * np.cos(qr[1]) + 2 * l1 * l2 * m3 * np.cos(qr[1]) + l2 * l3 * m3 * np.cos(qr[2]) + l1 * l3 * m3 * np.cos(qr[1]) * np.cos(qr[2]), (l2**2 * m2) / 3 + l2**2 * m3 + (l3**2 * m3 * np.cos(qr[2])**2) / 3 + (l1 * l2 * m2 * np.cos(qr[1])) / 2 + l1 * l2 * m3 * np.cos(qr[1]) + l2 * l3 * m3 * np.cos(qr[2]) + (l1 * l3 * m3 * np.cos(qr[1]) * np.cos(qr[2])) / 2, -(l1 * l3 * m3 * np.sin(qr[1]) * np.sin(qr[2])) / 2],
        [(l2**2 * m2) / 3 + l2**2 * m3 + (l3**2 * m3 * np.cos(qr[2])**2) / 3 + (l1 * l2 * m2 * np.cos(qr[1])) / 2 + l1 * l2 * m3 * np.cos(qr[1]) + l2 * l3 * m3 * np.cos(qr[2]) + (l1 * l3 * m3 * np.cos(qr[1]) * np.cos(qr[2])) / 2, (l2**2 * m2) / 3 + l2**2 * m3 + (l3**2 * m3 * np.cos(qr[2])**2) / 3 + l2 * l3 * m3 * np.cos(qr[2]), 0],
        [-(l1 * l3 * m3 * np.sin(qr[1]) * np.sin(qr[2])) / 2, 0, (l3**2 * m3) / 3]
    ])
    # Macierz Coriolisa
    C = np.array([
        [-qr_d1[1] * ((l1 * l2 * m2 * np.sin(qr[1])) / 2 + l1 * l2 * m3 * np.sin(qr[1]) + (l1 * l3 * m3 * np.cos(qr[2]) * np.sin(qr[1])) / 2) - qr_d1[2] * ((l3**2 * m3 * np.cos(qr[2]) * np.sin(qr[2])) / 3 + (l2 * l3 * m3 * np.sin(qr[2])) / 2 + (l1 * l3 * m3 * np.cos(qr[1]) * np.sin(qr[2])) / 2), -qr_d1[0] * ((l1 * l2 * m2 * np.sin(qr[1])) / 2 + l1 * l2 * m3 * np.sin(qr[1]) + (l1 * l3 * m3 * np.cos(qr[2]) * np.sin(qr[1])) / 2) - qr_d1[1] * ((l1 * l2 * m2 * np.sin(qr[1])) / 2 + l1 * l2 * m3 * np.sin(qr[1]) + (l1 * l3 * m3 * np.cos(qr[2]) * np.sin(qr[1])) / 2) - qr_d1[2] * ((l3**2 * m3 * np.cos(qr[2]) * np.sin(qr[2])) / 3 + (l2 * l3 * m3 * np.sin(qr[2])) / 2 + (l1 * l3 * m3 * np.cos(qr[1]) * np.sin(qr[2])) / 2), -qr_d1[0] * ((l3**2 * m3 * np.cos(qr[2]) * np.sin(qr[2])) / 3 + (l2 * l3 * m3 * np.sin(qr[2])) / 2 + (l1 * l3 * m3 * np.cos(qr[1]) * np.sin(qr[2])) / 2) - qr_d1[1] * ((l3**2 * m3 * np.cos(qr[2]) * np.sin(qr[2])) / 3 + (l2 * l3 * m3 * np.sin(qr[2])) / 2 + (l1 * l3 * m3 * np.cos(qr[1]) * np.sin(qr[2])) / 2) - (l1 * l3 * m3 * np.cos(qr[2]) * np.sin(qr[1]) * qr_d1[2]) / 2],
        [qr_d1[0] * ((l1 * l2 * m2 * np.sin(qr[1])) / 2 + l1 * l2 * m3 * np.sin(qr[1]) + (l1 * l3 * m3 * np.cos(qr[2]) * np.sin(qr[1])) / 2) - ((l3**2 * m3 * np.cos(qr[2]) * np.sin(qr[2])) / 3 + (l2 * l3 * m3 * np.sin(qr[2])) / 2) * qr_d1[2], -((l3**2 * m3 * np.cos(qr[2]) * np.sin(qr[2])) / 3 + (l2 * l3 * m3 * np.sin(qr[2])) / 2) * qr_d1[2], -((l3**2 * m3 * np.cos(qr[2]) * np.sin(qr[2])) / 3 + (l2 * l3 * m3 * np.sin(qr[2])) / 2) * qr_d1[0] - ((l3**2 * m3 * np.cos(qr[2]) * np.sin(qr[2])) / 3 + (l2 * l3 * m3 * np.sin(qr[2])) / 2) * qr_d1[1]],
        [((l3**2 * m3 * np.cos(qr[2]) * np.sin(qr[2])) / 3 + (l2 * l3 * m3 * np.sin(qr[2])) / 2) * qr_d1[1] + qr_d1[0] * ((l3**2 * m3 * np.cos(qr[2]) * np.sin(qr[2])) / 3 + (l2 * l3 * m3 * np.sin(qr[2])) / 2 + (l1 * l3 * m3 * np.cos(qr[1]) * np.sin(qr[2])) / 2), ((l3**2 * m3 * np.cos(qr[2]) * np.sin(qr[2])) / 3 + (l2 * l3 * m3 * np.sin(qr[2])) / 2) * qr_d1[0] + ((l3**2 * m3 * np.cos(qr[2]) * np.sin(qr[2])) / 3 + (l2 * l3 * m3 * np.sin(qr[2])) / 2) * qr_d1[1], 0]
    ])
    # sily grawitacyjne 
    D = np.array([
        [0],
        [0],
        [-(g * l3 * m3 * np.cos(qr[2])) / 2]
    ])
    # Macierz Jacobiego i jej pochodna
    J = np.array([
        [-l1 * np.sin(qr[0]) - l2 * np.sin(qr[0] + qr[1]) - l3 * np.cos(qr[2]) * np.sin(qr[0] + qr[1]), -l2 * np.sin(qr[0] + qr[1]) - l3 * np.cos(qr[2]) * np.sin(qr[0] + qr[1]), -l3 * np.sin(qr[2]) * np.cos(qr[0] + qr[1])],
        [l1 * np.cos(qr[0]) + l2 * np.cos(qr[0] + qr[1]) + l3 * np.cos(qr[2]) * np.cos(qr[0] + qr[1]), l2 * np.cos(qr[0] + qr[1]) + l3 * np.cos(qr[2]) * np.cos(qr[0] + qr[1]), -l3 * np.sin(qr[2]) * np.sin(qr[0] + qr[1])],
        [0, 0, l3 * np.cos(qr[2])]
    ])
    
    J_d1 = np.array([
        [-l3 * (np.cos(qr[2]) * np.cos(qr[0] + qr[1]) * (qr_d1[0] + qr_d1[1]) - np.sin(qr[2]) * np.sin(qr[0] + qr[1]) * qr_d1[2]) - l2 * np.cos(qr[0] + qr[1]) * (qr_d1[0] + qr_d1[1]) - l1 * np.cos(qr[0]) * qr_d1[0], -l3 * (np.cos(qr[2]) * np.cos(qr[0] + qr[1]) * (qr_d1[0] + qr_d1[1]) - np.sin(qr[2]) * np.sin(qr[0] + qr[1]) * qr_d1[2]) - l2 * np.cos(qr[0] + qr[1]) * (qr_d1[0] + qr_d1[1]), l3 * (np.sin(qr[2]) * np.sin(qr[0] + qr[1]) * (qr_d1[0] + qr_d1[1]) - np.cos(qr[2]) * np.cos(qr[0] + qr[1]) * qr_d1[2])],
        [-l3 * (np.sin(qr[2]) * np.cos(qr[0] + qr[1]) * qr_d1[2] + np.cos(qr[2]) * np.sin(qr[0] + qr[1]) * (qr_d1[0] + qr_d1[1])) - l2 * np.sin(qr[0] + qr[1]) * (qr_d1[0] + qr_d1[1]) - l1 * np.sin(qr[0]) * qr_d1[0], -l3 * (np.sin(qr[2]) * np.cos(qr[0] + qr[1]) * qr_d1[2] + np.cos(qr[2]) * np.sin(qr[0] + qr[1]) * (qr_d1[0] + qr_d1[1])) - l2 * np.sin(qr[0] + qr[1]) * (qr_d1[0] + qr_d1[1]), -l3 * np.cos(qr[2]) * np.sin(qr[0] + qr[1]) * qr_d1[2] - l3 * np.sin(qr[2]) * np.cos(qr[0] + qr[1]) * (qr_d1[0] + qr_d1[1])],
        [0, 0, -l3 * np.sin(qr[2]) * qr_d1[2]]
    ])
    # efekty Coriolisa i siły odśrodkowe
    P = np.array([
        [l3 * (np.sin(qr[2]) * np.sin(qr[0] + qr[1]) * (qr_d1[0] + qr_d1[1]) - np.cos(qr[2]) * np.cos(qr[0] + qr[1]) * qr_d1[2]) * qr_d1[2] - qr_d1[0] * (l3 * (np.cos(qr[2]) * np.cos(qr[0] + qr[1]) * (qr_d1[0] + qr_d1[1]) - np.sin(qr[2]) * np.sin(qr[0] + qr[1]) * qr_d1[2]) + l2 * np.cos(qr[0] + qr[1]) * (qr_d1[0] + qr_d1[1]) + l1 * np.cos(qr[0]) * qr_d1[0]) - qr_d1[1] * (l3 * (np.cos(qr[2]) * np.cos(qr[0] + qr[1]) * (qr_d1[0] + qr_d1[1]) - np.sin(qr[2]) * np.sin(qr[0] + qr[1]) * qr_d1[2]) + l2 * np.cos(qr[0] + qr[1]) * (qr_d1[0] + qr_d1[1]))],
        [-qr_d1[1] * (l3 * (np.sin(qr[2]) * np.cos(qr[0] + qr[1]) * qr_d1[2] + np.cos(qr[2]) * np.sin(qr[0] + qr[1]) * (qr_d1[0] + qr_d1[1])) + l2 * np.sin(qr[0] + qr[1]) * (qr_d1[0] + qr_d1[1])) - (l3 * np.cos(qr[2]) * np.sin(qr[0] + qr[1]) * qr_d1[2] + l3 * np.sin(qr[2]) * np.cos(qr[0] + qr[1]) * (qr_d1[0] + qr_d1[1])) * qr_d1[2] - qr_d1[0] * (l3 * (np.sin(qr[2]) * np.cos(qr[0] + qr[1]) * qr_d1[2] + np.cos(qr[2]) * np.sin(qr[0] + qr[1]) * (qr_d1[0] + qr_d1[1])) + l2 * np.sin(qr[0] + qr[1]) * (qr_d1[0] + qr_d1[1]) + l1 * np.sin(qr[0]) * qr_d1[0])],
        [-l3 * np.sin(qr[2]) * qr_d1[2]**2]
    ])
    # macierz sterowania
    B = np.identity(3)
    # wektor zakłóceń lub momentów zewnętrznych
    T = np.zeros((3, 1))
    
    # generate trajectory for end effector
    # remember to calculate 1st and 2nd derivative of desired end-effector trajectory
    qchd, qchd_d1, qchd_d2 = effectorTrajectoryGenerator3D(t, parameters)
    



    detJ = np.linalg.det(J)
    detM = np.linalg.det(M)
    if detM == 0 or detJ == 0:
        raise ValueError("Macierz M lub J jest osobliwa.")
    
    MInv = np.linalg.inv(M)
    
    
    # calculate F
    # F = P - J M^-1 C q' - J M^-1 D - J M^-1 T
    F = P - (J @ MInv @ C @ q_d1) - (J @ MInv @ D) - (J @ MInv @ T)
    
    # calculate G
    # G = J M^-1
    G = J @ MInv

    
    qch = np.array([xch, ych, zch])  # Aktualna pozycja efektora
    qch_d1 = np.array([xch_d1, ych_d1, zch_d1]) # Aktualna prędkość
   
    # Obliczanie błędu trajektorii:
    # calculate errors: e and e'
    e = qch - qchd
    e_d1 = qch_d1 - qchd_d1

    e = e.reshape(-1, 1)
    e_d1 = e_d1.reshape(-1, 1)
    # calculate new input to the system v
    #  v = qd'' - Kd e' - Kp e
    # v = qchd_d2 - Kd @ e_d1 - Kp @ e
    v = - Kd @ e_d1 - Kp @ e
    #energy_penalty = 0.01 * np.sum(v**2)
    #v -= energy_penalty
    

    detG = np.linalg.det(G)
    
    # calculate inverse of G as below
    Ginv = np.linalg.inv(G)
    
    # calculate control input u
    # u = G^-1 * (v - F)
    u = Ginv @ (v - F)
    #print("u=",u)
    # Ograniczenie maksymalnej prędkości

    max_velocity = 5000
    u = np.clip(u, -max_velocity, max_velocity)
    
    # y'' = P - J M^-1 C q' + J M^-1 u
    ych_d2 = P - J @ MInv @ C @ q_d1 + J @ MInv @ u
    
    # calculate state
    qr_d2 = MInv @ (B @ u - C @ q_d1 - D - T)
    dets = np.array([detG, detJ])
    
    output_args = np.zeros(6)
    output_args[0:3] = qr_d2.flatten()
    output_args[3:6] = qr_d1

    additional = {
        'qchd': qchd,
        'qchd_d1': qchd_d1,
        'qchd_d2': qchd_d2,
        'dets': dets,
        'k': qch,
        'end_effector_position': np.array([xch, ych, zch]),  # Store end-effector position
        'error': e, # Dodanie błędu do zwracanych wyników
        'error_d1':e_d1
    }

    return output_args, additional, parameters



