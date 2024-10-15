import numpy as np
import math

def effectorTrajectoryGenerator3D(t, parameters):
    
    # Wybór trajektorii dla efektora w zależności od potrzeb.
    #return moon(t, parameters)
    return ellipse(t, parameters)
    #return 0,0,0
    
    
def moon(t, parameters):
    w = 0.015 * 2 * math.pi  # Częstotliwość kątowa
    k = 0.5  # Stała skalująca trajektorię
    dx = 1.1  # Przesunięcie trajektorii w osi x
    dy = 0.8  # Przesunięcie trajektorii w osi y
    dz = 0.1  # Przesunięcie trajektorii w osi z
    qchd = np.array([k * np.cos(w * t) + dx,
                     k * np.sin(w * t) + dy,
                     0.1 * k * np.cos(w * t) + dz])

    qchd_d1 = np.array([-k * w * np.sin(w * t),
                     k * w * np.cos(w * t),
                     -0.1 * k * w * np.sin(w * t)])
    qchd_d2 = np.array([-k * w**2 * np.cos(w * t),
                       -k * w**2 * np.sin(w * t),
                       -0.1 * k * w**2 * np.cos(w * t)])
    return qchd, qchd_d1, qchd_d2
def ellipse(t, parameters):
    # Angular frequency
    omega = 0.015 * 2 * math.pi
    
    # Ellipse parameters: A is semi-major axis, B is semi-minor axis
    A = 0.5  # Semi-major axis in the x-direction
    B = 1  # Semi-minor axis in the y-direction
    C = 0.0  # Amplitude in the z-direction (optional for 3D)

    # Offsets to move the center of the ellipse
    dx = 1
    dy = 1.5
    dz = 0.1

    # Desired position (ellipse in xy-plane)
    qchd = np.array([
        A * np.cos(omega * t) + dx,
        B * np.sin(omega * t) + dy,
        C * np.cos(omega * t) + dz
    ])

    # First derivatives (velocity)
    qchd_d1 = np.array([
        -A * omega * np.sin(omega * t),
        B * omega * np.cos(omega * t),
        -C * omega * np.sin(omega * t)
    ])

    # Second derivatives (acceleration)
    qchd_d2 = np.array([
        -A * omega**2 * np.cos(omega * t),
        -B * omega**2 * np.sin(omega * t),
        -C * omega**2 * np.cos(omega * t)
    ])

    return qchd, qchd_d1, qchd_d2