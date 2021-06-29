import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.metrics import mean_squared_error ,accuracy_score
class Sample:
   def  __init__(self,x,y):
       self.x =x
       self.y =y
       
"""" Class which calculate final estimation using AlphaBeta filter """
       
class AlphaBetaFilter:
    def __init__(self,x,y,alpha=1,beta=0.1,v_x=1,v_y=1):
        self.alpha = alpha
        self.beta = beta
        self.v_x_list =[v_x]
        self.v_y_list =[v_y]
        self.loc_x = [x]
        self.loc_y = [y]
        self.errors_x =[]
        self.errors_y =[]
        self.predictions_x=[]
        self.predictions_y=[]
        
    
    def last_velocity(self):
        return self.v_x_list[-1],self.v_y_list[-1]
    
    
    def last_sample(self):
        return self.loc_x[-1],self.loc_y[-1]
    
    """ Function where prediction , error calculation and the final estimated is calculated """ 
    def add_sample(self, s:Sample):
        expected_location_x ,expected_location_y = self.predictions()
        error_x = s.x-expected_location_x
        error_y =s.y-expected_location_y
        location_x = expected_location_x + self.alpha * error_x
        location_y = expected_location_y + self.alpha * error_y
        x_last_vel , y_last_vel = self.last_velocity() 
        x_vel = x_last_vel + (self.beta ) * error_x
        y_vel = y_last_vel + (self.beta ) * error_y
        
        #debug variables
        
        self.loc_x.append(location_x)
        self.loc_y.append(location_y)
        self.v_x_list.append(x_vel)
        self.v_y_list.append(y_vel)
        
        
        
        self.errors_x.append(error_x)
        self.errors_y.append(error_y)
        
        return location_x,location_y
    
    """ Function which predicts position and velocity based on previous estimations """    
    def predictions(self):
        last_sample_x ,last_sample_y = self.last_sample()
        x_last_vel , y_last_vel = self.last_velocity() 
        prediction_x = last_sample_x + x_last_vel
        prediction_y = last_sample_y + y_last_vel
        return prediction_x,prediction_y


def plot_data(gd,estimated_x,estimated_y,meas_x,meas_y):
    plt.plot(gd[:,0],gd[:,1],'.')
    plt.plot(estimated_x,estimated_y,'.')
    plt.plot(meas_x,meas_y,'.')
    plt.legend(['ground truth','alphabeta_op','measurement'] )
    rmse_m,rmse_p=calculate_rmse(gt,meas_x,meas_y,estimated_x,estimated_y)
    #plt.savefig("Graph" + str(iteration)+" " + str(rmse_p) +" "+ str(rmse_m) +".png", format="PNG")
    #plt.clf()

def calculate_rmse(gt,Measurement_model_x,Measurement_model_y,corrections_model_x,corrections_model_y):
    
    meas = np.column_stack((Measurement_model_x,Measurement_model_y))
    prediction =  np.column_stack((corrections_model_x,corrections_model_y))
    meas_rmse = sqrt(mean_squared_error(meas, gt[0:96]))  
    prediction_rmse= sqrt(mean_squared_error(prediction, gt[0:96]))  
    print(f"meas_rsme:{meas_rmse} pred_rmse :{prediction_rmse}")
    return meas_rmse,prediction_rmse
    
    
    
col_names =['sl_no','iteration','r_meas','phi_meas','rr_meas','x_m','y_m','dx_m','dy_m','r_gd','phi_gd','rr_gd','x_gd','y_gd','dx_gd','dy_gd']
df= pd.read_csv("data/test_data.csv",names=col_names,header=None)

iteration = 109

measurements_pd = df[['x_m','y_m']]
gt_pd = df[['x_gd','y_gd']]
gt_array= np.array(gt_pd)
measurements=np.array(measurements_pd)


Measurements = measurements[(iteration*101)+5 :((iteration+1)*101),:]
gt = gt_array[(iteration*101)+5 :((iteration+1)*101),:]

estimated_x_values =[Measurements[0,0]] # First estimation is same as the measurment 
estimated_y_values =[Measurements[0,1]] # First estimation is same as the measurment 


sample_x = Measurements[:,0]
sample_y = Measurements[:,1]
tracker = AlphaBetaFilter(sample_x[0],sample_y[0], alpha=0.7, beta=0.5, v_x=0.5, v_y =0.5)

for i in range(1,len(sample_x)):
    x_value ,y_value = tracker.add_sample(Sample(sample_x[i],sample_y[i]))
    estimated_x_values.append(x_value)
    estimated_y_values.append(y_value)

plot_data(gt,estimated_x_values,estimated_y_values,sample_x,sample_y)

    
        
       
  
       
        