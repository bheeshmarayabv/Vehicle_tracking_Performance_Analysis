# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 07:49:29 2021

@author: bhees
"""

from kalmanfilter import KalmanFilter
from DataSample import DataSample
from Extended_kf import Extended_KF
from Toolkit import polar_to_cartesian
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error ,accuracy_score
import csv


def parse_data(file_path):

  all_sensor_data = []
  all_ground_truths = []  
  count = 0
  
  
  i=38
  
  with open(file_path) as f:
      
    for line in f:
      data = line.split() 
      count =count+1
      if(count>=(i*101)+5 and count <=((i+1)*101)):
                  
          if data!=[]:
              
              sensor_data = DataSample({ 
              'name': 'radar',
              'rho': float(data[2]), 
              'phi': float(data[3]),
              'drho': float(data[4])
              })
           
              g = {
                 'name': 'state',
                 'x': float(data[12]),
                 'y': float(data[13]),
                 'vx': float(data[14]),
                 'vy': float(data[15])
                 }
    
              ground_truth = DataSample(g)  
    
              all_sensor_data.append(sensor_data)
              all_ground_truths.append(ground_truth)

  return all_sensor_data, all_ground_truths


def get_state_estimations(EKF, all_sensor_data):

  all_state_estimations = []
  x_predictions =[]
  y_predictions =[]
  #count =0 

  for data in all_sensor_data:
    
    EKF.process(data)
  
    x = EKF.get()
    px, py, vx, vy = x[0, 0], x[1, 0], x[2, 0], x[3, 0]

    g = {
         'name': 'state',
         'x': px,
         'y': py,
         'vx': vx,
         'vy': vy }

    state_estimation = DataSample(g)  
    all_state_estimations.append(state_estimation)
    
  x_predictions,y_predictions = EKF.get_predictions()
  return all_state_estimations,x_predictions,y_predictions


def positions_from_meas(sensor_data,all_ground_truths):
    
    x_meas=[]
    y_meas =[]
    vx_meas =[]
    vy_meas =[]
    x_ground_truth=[]
    y_ground_truth=[]
    vx_ground_truth =[]
    vy_ground_truth =[]
    
    
    for p,g in zip(sensor_data,all_ground_truths):
        r,phi,rr=p.get_raw()
        x_gt ,y_gt,vx_gt,vy_gt = g.get()
        x,y,vx,vy = polar_to_cartesian( r,phi,rr)

        x_ground_truth.append(x_gt)
        y_ground_truth.append(y_gt)
        vx_ground_truth.append(vx_gt)
        vy_ground_truth.append(vy_gt)

        x_meas.append(x)
        y_meas.append(y)
        vx_meas.append(vx)
        vy_meas.append(vy)
    np.array(x_ground_truth).reshape(-1,1)
    np.array(y_ground_truth).reshape(-1,1)
    np.array(x_meas).reshape(-1,1)
    np.array(y_meas).reshape(-1,1)
    
    meas = np.column_stack((x_meas,y_meas))
    ground_truth = np.column_stack((x_ground_truth,y_ground_truth))
    meas_rmse= sqrt(mean_squared_error(ground_truth, meas))
    print('meas_rmse: %.7f RMSE' % (meas_rmse))
    
    return x_meas, y_meas,vx_meas,vy_meas



def plot_data(all_ground_truths, all_state_estimations,sensor_data,x_predictions,y_predictions,filename):
    
    x_estimation=[] 
    y_estimation=[]
    vx_estimation =[]
    vy_estimation =[]


    x_ground_truth=[]
    y_ground_truth=[]
    vx_ground_truth =[]
    vy_ground_truth =[]
    
    for p,t in zip(all_state_estimations,all_ground_truths):
        
        x_est, y_est, vx_est, vy_est = p.get()
        x_gt ,y_gt,vx_gt,vy_gt = t.get()
        
        x_estimation.append(x_est)
        y_estimation.append(y_est)
        vx_estimation.append(vx_est)
        vy_estimation.append(vy_est)
            
        x_ground_truth.append(x_gt)
        y_ground_truth.append(y_gt)
        vx_ground_truth.append(vx_gt)
        vy_ground_truth.append(vy_gt)
        
    x_measurement, y_measurement ,vx_measuremnt ,vy_measuremnt = positions_from_meas(sensor_data,all_ground_truths) 
    plt.plot(x_ground_truth,y_ground_truth,'.')
    plt.plot(x_estimation,y_estimation,'.')
    plt.plot( x_measurement, y_measurement,'.')
    rmse = combined_RMSE(all_state_estimations, all_ground_truths)
    plt.legend(['Groundtruth','EKF output','measurment'])
    plt.xlabel("X")
    plt.ylabel("Y ")
   # plt.savefig("Graph" + str(i)+" " + str(rmse) +" "+ ".png", format="PNG")
    plt.show()
    
def combined_RMSE(all_state_estimations,all_ground_truths):
    x_estimation =[]
    y_estimation =[]
    x_ground_truth =[]
    y_ground_truth =[]
    
    for p,t in zip(all_state_estimations,all_ground_truths):
        x_est, y_est, vx_est, vy_est = p.get()
        x_gt ,y_gt,vx_gt,vy_gt = t.get()
        
        x_estimation.append(x_est)
        y_estimation.append(y_est)
        x_ground_truth.append(x_gt)
        y_ground_truth.append(y_gt)
    x_estimation = np.array(x_estimation)  
    y_estimation = np.array(y_estimation)
    x_ground_truth = np.array(x_ground_truth)
    y_ground_truth = np.array(y_ground_truth)
    
    
    x_estimation.reshape(-1,1)
    y_estimation.reshape(-1,1)
    x_ground_truth.reshape(-1,1)
    y_ground_truth.reshape(-1,1)

    
    estimation = np.column_stack((x_estimation,y_estimation))
    ground_truth = np.column_stack((x_ground_truth,y_ground_truth))
    
    testScore = sqrt(mean_squared_error(ground_truth, estimation))
    print('Test Score: %.7f RMSE' % (testScore))
    return testScore

def evaluate_prediction(combine_predictions,all_ground_truths):
    x_ground_truth =[]
    y_ground_truth =[]
    #np.savetxt('kalman_predictions.csv',combine_predictions,fmt ='%.7f',delimiter=',' )
    
    for g in all_ground_truths:
        x_gt ,y_gt,vx_gt,vy_gt = g.get()
        x_ground_truth.append(x_gt)
        y_ground_truth.append(y_gt)
        
    ground_truth = np.column_stack((x_ground_truth,y_ground_truth))
    predict_rmse= sqrt(mean_squared_error(ground_truth[:-1], combine_predictions))   
    print("Predict Score: %.7f RMSE" % (predict_rmse))
    
    

radar_R = np.matrix([[4,0,0], 
                     [0,  0.014, 0], 
                     [0, 0, 0.08]])


P = np.matrix([[1, 0, 0, 0],
               [0, 1, 0, 0],
               [0, 0, 1, 0], 
               [0, 0, 0, 1]])

Q = np.matrix(np.zeros([4, 4]))
F = np.matrix(np.eye(4))

d = {
  'number_of_states': 4, 
  'initial_process_matrix': P,
  'radar_covariance_matrix': radar_R,
  'inital_state_transition_matrix': F,
  'initial_noise_matrix': Q, 
  'acceleration_noise_x': 0.8,
  'acceleration_noise_y': 0.8
}       

EKF1 = Extended_KF(d)


filename = "test_n_ml_next1L.txt"


all_sensor_data, all_ground_truths = parse_data("data/"+filename)
all_state_estimations,x_predictions,y_predictions = get_state_estimations(EKF1, all_sensor_data)

combine_predictions = np.column_stack( (x_predictions,y_predictions))
evaluate_prediction(combine_predictions,all_ground_truths)

plot_data(all_ground_truths,all_state_estimations,all_sensor_data,x_predictions,y_predictions,filename)







