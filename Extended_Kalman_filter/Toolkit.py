"""
@author: bheeshma
"""

import numpy as np
from math import sin, cos, sqrt

def cartesian_to_polar(x, y, vx, vy, THRESH = 0.0001):
  """   
  Converts 2d cartesian position and velocity coordinates to polar coordinates

  Args:
    x, y, vx, vy : floats - position and velocity components in cartesian respectively 
    THRESH : float - minimum value of rho to return non-zero values
  
  Returns: 
    rho, drho : floats - radius and velocity magnitude respectively
    phi : float - angle in radians
  """  

  rho = sqrt(x * x + y * y)
  phi = np.arctan2(y, x)
  if(x<0 and y<0):
      phi=-phi
  
  
  if rho < THRESH:
    print("WARNING: in cartesian_to_polar(): d_squared < THRESH")
    rho, phi, drho = 0, 0, 0
  else:
    drho = (x * vx + y * vy) / rho
    #drho = sqrt(vx * vx + vy * vy)
    
  return rho, phi, drho

def polar_to_cartesian(rho, phi, drho):
  """
  Converts 2D polar coordinates into cartesian coordinates

  Args:
    rho. drho : floats - radius magnitude and velocity magnitudes respectively
    phi : float - angle in radians

  Returns:
    x, y, vx, vy : floats - position and velocity components in cartesian respectively
  """
 
  x, y = rho * cos(phi), rho * sin(phi)
  vx, vy = drho * cos(phi) , drho * sin(phi)

  return x, y, vx, vy
 
  
def calculate_jacobian(px, py, vx, vy, THRESH = 0.0001, ZERO_REPLACEMENT = 0.0001):
  """
    Calculates the Jacobian given for four state variables

    Args:
      px, py, vx, vy : floats - four state variables in the system 
      THRESH - minimum value of squared distance to return a non-zero matrix
      ZERO_REPLACEMENT - value to replace zero to avoid division by zero error

    Returns:
      H : the jacobian matrix expressed as a 4 x 4 numpy matrix with float values
  """
    
  d_squared = px * px + py * py 
  d = sqrt(d_squared)
  d_cubed = d_squared * d
  
  if d_squared < THRESH:
 
    print("WARNING: in calculate_jacobian(): d_squared < THRESH")
    H = np.matrix(np.zeros([3, 4]))
 
  else:

    r11 = px / d
    r12 = py / d
    r21 = -py / d_squared
    r22 = px / d_squared
    r31 = py * (vx * py - vy * px) / d_cubed
    r32 = px * (vy * px - vx * py) / d_cubed
  
    H = np.matrix([[r11, r12, 0, 0], 
                  [r21, r22, 0, 0], 
                  [r31, r32, r11, r12]])

  return H