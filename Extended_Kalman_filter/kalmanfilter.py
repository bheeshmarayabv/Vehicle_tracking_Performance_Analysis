"""
@author: bheeshma
"""

import numpy as np

class KalmanFilter:
  """
    A class that predicts the next state of the system given sensor measurements 
    using the Kalman Filter algorithm 
  """

  def __init__(self, n):

    self.n = n
    self.I = np.matrix(np.eye(n))
    self.x = None
    self.P = None
    self.F = None
    self.Q = None
    self.i =0

  
  def start(self, x, P, F, Q): # ver

    self.x = x
    self.P = P
    self.F = F
    self.Q = Q
        
  def setQ(self, Q):
    self.Q = Q

  def updateF(self, dt):
    """[[1 0 t 0] ,[0 1 0 t],[0 0 1 0],[0 0 0 1]] """   
    self.F[0, 2], self.F[1, 3]  = dt, dt
 
  def getx(self):
    return self.x

  def predict(self):
        
    self.x = self.F * self.x
    
    self.P = self.F * self.P * self.F.T + self.Q

    
  def update(self, z, H, Hx, R):
    y = z - Hx
    PHt = self.P * H.T
    S = H * PHt + R    
    K = PHt * (S.I)
    self.i=self.i+1    
    self.x = self.x + K * y
    self.P = (self.I - K * H) * self.P
