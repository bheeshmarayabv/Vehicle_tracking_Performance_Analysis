from Toolkit import polar_to_cartesian

class DataSample:
  """
    A set of derived information from measurements of known sensors
    NOTE: Upon instantiation of a "radar" DataSample, state variables are computed from raw data 
  """
    
  def __init__(self, d):
    #self.timestamp = d['timestamp']
    self.name = d['name']
    self.all = d
    self.raw = []
    self.data = []

    
    if self.name == 'state':       
        self.data = [d['x'], d['y'], d['vx'], d['vy']]
        self.raw = self.data.copy()
      
    elif self.name == 'radar':     
        x, y, vx, vy = polar_to_cartesian(d['rho'], d['phi'], d['drho'])
        self.data = [x, y, vx, vy]
        self.raw = [d['rho'], d['phi'], d['drho']]

    
    self.all['processed_data'] = self.data
    self.all['raw'] = self.raw
    
  def get_dict(self):
    return self.all

  def get_raw(self):
    return self.raw

  def get(self):
    return self.data

  #def get_timestamp(self):
   # return self.timestamp

  def get_name(self):
    return self.name