import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
import statistics
import pandas as pd
import sys
from sklearn.metrics import mean_squared_error ,accuracy_score

accel_estimate_variance_x =0.4

def prediction_State_Vector(x,vel_x,y,vel_y,t,accel):
    '''Prediction step:
    1) X(kp) = AX(k-1) + Bu(k) + w where w equals noise and we are assuming w to be 0

    Predicting the state vector
    '''
    A= np.reshape(np.array([[1,t,0,0],[0,1,0,0],[0,0,1,t],[0,0,0,1]]), (4,4))
    X_prev= np.reshape(np.array([x,vel_x,y,vel_y]),(4,1))
    X_curr= np.matmul(A,X_prev) 
    
    return (X_curr)
    
def prediction_process_covariance(cov,t):
    '''2) 
    P(kp) = AP(k-1)A(transpose) + Q where Q is process noise

    Predicting the process covariance
    '''
    A= np.reshape(np.array([[1,t,0,0],[0,1,0,0],[0,0,1,t],[0,0,0,1]]), (4,4))
    #B= np.reshape(([0.5*t**2,0],[t,0],[0,0.5*t**2],[0,t]), (4,2))
    B= np.reshape(np.array([0.5*t**2,t,0.5*t**2,t]), (4,1))
    P_curr_inter= np.matmul(A,cov)
    Q = np.dot(accel_estimate_variance_x,np.matmul(B,B.T))

    P_curr= np.matmul(P_curr_inter,A.T) + Q
    return (P_curr)

def Kalman_gain(p_curr,meas_cov_curr):
    '''
    3) KG= P(kp)H(transpose)/HP(kp)H(transpose)
    
    Calculating Kalman_gain
    '''
    H= np.reshape(np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]), (4,4))
    Num= np.matmul(p_curr,H.T)
    Den_intermediate=np.matmul(H,p_curr)
    Den= np.matmul(Den_intermediate,H.T) + meas_cov_curr
    KG= np.matmul(Num,np.linalg.inv(Den))   
    return (KG)

def update_State_vector(KG,X_curr,meas_curr,N):
    '''4) X= X(kp)+K[y-HX(kp)]
    
    Updating the state vector
    '''
    H= np.reshape(np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]), (4,4))
    '''
    Updating the state vector
    '''
    diff_meas_pred = np.reshape(meas_curr-np.matmul(H,X_curr),(4,1))
    balancing_v= np.matmul(diff_meas_pred.T,diff_meas_pred)
    X_updated= X_curr + np.matmul(KG,(meas_curr-np.matmul(H,X_curr)))
    return (X_updated)

def update_Process_Covar(KG, p_curr):
    '''
    P=(I-KH)P(kp)
    
    Updating the process Covariance
    '''
    H= np.reshape(np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]), (4,4))
    I=np.reshape(np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]), (4,4))
    ratio= I-np.matmul(KG,H)
    P_updated = np.matmul(p_curr,ratio)
    return(P_updated)

def kalman_estimation(iteration,Measurements,gt):
    i=0
    accel_var= 0.1
    Measurement_model_x=[]
    estimate_model_x=[]
    corrections_model_x=[]
    Measurement_model_y=[]
    acceleration_var=[]
    estimate_model_y=[]
    corrections_model_y=[]
    vel_estimate_x=[]
    vel_estimate_y=[]
    
    for number in range(0,96):
        # for the first iteration estimation is same as measurement
        # The state vector (x,dx,y,dy)
        if i==0:
            updated_state = np.reshape(np.array([Measurements[0,i],Measurements[1,i],Measurements[2,i],Measurements[3,i]]),(4,1))
            updated_vector = np.reshape(np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]), (4,4))
            i=i+1
            corrections_model_x.append(updated_state[0][0])
            corrections_model_y.append(updated_state[2][0])
            Measurements1=np.reshape(np.array([Measurements[0,i],Measurements[1,i],Measurements[2,i],Measurements[3,i]]),(4,1))
            Measurement_model_x.append(Measurements1[0][0])
            Measurement_model_y.append(Measurements1[2][0])
        else:
            estimate = prediction_State_Vector(updated_state[0][0],updated_state[1][0],updated_state[2][0],updated_state[3][0],1,0)
            process_covar = prediction_process_covariance(updated_vector,1)    
    
            Meas_cov= np.reshape(np.array([[2,0,0,0],[0,0.25,0,0],[0,0,1,0],[0,0,0,0.6]]), (4,4)) # R matrix 
    
            KG= Kalman_gain(process_covar,Meas_cov)
            identity=np.eye(4)
            KG=KG*identity
            Measurements1=np.reshape(np.array([Measurements[0,i],Measurements[1,i],Measurements[2,i],Measurements[3,i]]),(4,1))
    
            updated_state = update_State_vector(KG,estimate,Measurements1,i)
            updated_vector = update_Process_Covar(KG,process_covar)
            i=i+1
            Measurement_model_x.append(Measurements1[0][0])
            Measurement_model_y.append(Measurements1[2][0])
            estimate_model_x.append(estimate[0][0])
            estimate_model_y.append(estimate[2][0])
            corrections_model_x.append(updated_state[0][0])
            corrections_model_y.append(updated_state[2][0])
    print(len(corrections_model_x))
    plot_data(iteration,gt,Measurement_model_x,Measurement_model_y,corrections_model_x,corrections_model_y)
    
            
def calculate_rmse(gt,Measurement_model_x,Measurement_model_y,corrections_model_x,corrections_model_y):
    gt =gt.T
    meas = np.column_stack((Measurement_model_x,Measurement_model_y))
    prediction =  np.column_stack((corrections_model_x,corrections_model_y))
    meas_rmse = sqrt(mean_squared_error(meas, gt[0:96]))  
    prediction_rmse= sqrt(mean_squared_error(prediction, gt[0:96]))  
    print(f"meas_rsme:{meas_rmse} pred_rmse :{prediction_rmse}")
    return meas_rmse,prediction_rmse
    
def plot_data(iteration,gt,Measurement_model_x,Measurement_model_y,corrections_model_x,corrections_model_y):
    meas_rmse,prediction_rmse = calculate_rmse(gt,Measurement_model_x,Measurement_model_y,corrections_model_x,corrections_model_y)
    plt.plot(gt[0,:],gt[1,:],'.')
    plt.plot(corrections_model_x,corrections_model_y,'.')
    plt.plot(Measurement_model_x,Measurement_model_y,'.')
    plt.legend([ 'Groundtruth','kalman output','Measurement values'])
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

col_names =['sl_no','iteration','r_meas','phi_meas','rr_meas','x_m','y_m','dx_m','dy_m','r_gd','phi_gd','rr_gd','x_gd','y_gd','dx_gd','dy_gd']
df= pd.read_csv("data/test_data.csv",names=col_names,header=None)

iteration = 249

measurements_pd = df[['x_m','dx_m','y_m','dy_m']]
gt_pd = df[['x_gd','y_gd']]
gt_full= np.array(gt_pd).T
measurements=np.array(measurements_pd).T
    
Measurements = measurements[:,(iteration*101)+5 :((iteration+1)*101)]
gt = gt_full[:,(iteration*101)+5 :((iteration+1)*101)]
kalman_estimation(iteration,Measurements,gt)


