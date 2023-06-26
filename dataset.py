from torch.utils.data import Dataset, DataLoader

import numpy as np
import torch
import matplotlib.pyplot as plt
import random

import netCDF4 as NCDF


def cart2pol(y,x):

    # переход к полярной системе координат
    '''
    Parameters:
    - x: float, x coord. of vector end
    - y: float, y coord. of vector end
    Returns:
    - r: float, vector amplitude
    - theta: float, vector angle
    '''

    z = x + y * 1j
    r,theta = np.abs(z), np.angle(z)

    return r,theta
    
#nc = NCDF.Dataset('20192021.nc')

def create_torch_DS(file_path: Path = '20192021.nc') - Dataset:
    nc = NCDF.Dataset(file_path)
    
    #Normalize 
    t = nc.variables['t2m'][:] # 2 meter temperature

    min_t2=t.min()
    max_t2=t.max()

    m_t2=t.mean()
    d_t2=(max_t2-min_t2)

    t = nc.variables['sp'][:] # mean sea level pressure
    min_mslp=t.min()
    max_mslp=t.max()

    m_mslp=t.mean()
    d_mslp=(max_mslp-min_mslp)

    t = nc.variables['u10'][:] # 10m u-component of winds
    min_u10=t.min()
    max_u10=t.max()

    m_u10=t.mean()
    d_u10=(max_u10-min_u10)


    t = nc.variables['v10'][:] # 10m v-component of winds
    min_v10=t.min()
    max_v10=t.max()

    m_v10=t.mean()
    d_v10=(max_v10-min_v10)

    r_temp,angle_temp=cart2pol(nc.variables['u10'][:],nc.variables['v10'][:])

    max_r=r_temp.max()
    min_r=r_temp.min()

    m_r=r_temp.mean()
    d_r=(max_r-min_r)

    max_a=angle_temp.max()
    min_a=angle_temp.min()

    m_a=angle_temp.mean()
    d_a=(max_a-min_a)
    
    train_ind,test_ind=split_set_to_train_and_val(nc,0.25)

    Train=NCDataset(nc,train_ind)
    Test=NCDataset(nc,test_ind)
    
    return Train, Test

#del t,r_temp,angle_temp


class NCDataset(Dataset):

    def __init__(self, dataset, ind,transform=None):

      self.transform=transform
      self.data=dataset
      self.lat = self.data.variables['latitude'][:]
      self.lon = self.data.variables['longitude'][:]
      self.time = self.data.variables['time'][ind]
      self.t2 = self.data.variables['t2m'][ind] # 2 meter temperature
      self.mslp = self.data.variables['sp'][ind] # mean sea level pressure
      self.u = self.data.variables['u10'][ind] # 10m u-component of winds
      self.v = self.data.variables['v10'][ind] # 10m v-component of winds

      self.r,self.angle=cart2pol(self.u,self.v)



    ####### СЕРЕГА ЗДЕСЬ ТОНКИЙ МОМЕНТ, ИНОГДА ТАМ 4 измерения а иногда 3, я сейачс оставил с 3. С 4 надо уточнить это только в послений год или нет и там есть параметр mask, если он TRUE то значений нет ...
    ######## В РИСОВАЛКАХ(plot) в это классе ЭТО ТОЖЕ ЕСТЬ !!!!!!!!!!!!!!!!
    ## ПОТОМ НАДО ПРОСТО ПРИВЕСТИ К 3-М везде где есть
    # def __getitem__(self, idx):

    #   if (np.unique(self.v[idx,1,:,:].mask)):
    #     mask=0
    #   else:
    #     mask=1

    #   return self.lat, self.lon, self.time[idx], self.t2[idx,mask,:,:], self.mslp[idx,mask,:,:] ,self.u[idx,mask,:,:], self.v[idx,mask,:,:]

    def __getitem__(self, idx):

       t=(self.t2[idx,:,:]-min_t2)/d_t2
       u=(self.u[idx,:,:]-min_u10)/d_u10
       pl=(self.mslp[idx,:,:]-min_mslp)/d_mslp
       v=(self.v[idx,:,:]-min_v10)/d_v10
       r=(self.r[idx,:,:]-min_r)/d_r
       a=(self.angle[idx,:,:]-min_a)/d_a

       out_main = torch.tensor(np.array([pl,t]).astype('float32'))


       return self.lat, self.lon, self.time[idx], out_main

    def __len__(self):
      return len(self.time)

    def plot(self, idx, **kwargs):

      map = Basemap(projection='merc',llcrnrlon=min(self.lon),llcrnrlat=min(self.lat),urcrnrlon=max(self.lon),urcrnrlat=max(self.lat),resolution='i') # projection, lat/lon extents and resolution of polygons to draw
      lons,lats= np.meshgrid(self.lon,self.lat) # for this dataset, longitude is 0 through 360, so you need to subtract 180 to properly display on map
      x,y = map(lons,lats)
      if("t2" in kwargs):

        fig,ax = plt.subplots()
       # temp_plot = map.contourf(x,y,self.t2[idx,0,:,:])
        temp_plot = map.contourf(x,y,self.t2[idx,:,:])


        cbar = fig.colorbar(temp_plot)
        plt.title('2m Temperature + mean sea pressure level')
        #cb.set_label('Temperature (K)')

        parallels = np.arange(min(self.lat),max(self.lat),5.)
        meridians = np.arange(min(self.lon),max(self.lon),5.)
        map.drawparallels(parallels,labels=[1,0,0,0],fontsize=10)
        map.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10)
        map.drawcoastlines()
        map.drawstates()
        map.drawcountries()
        map.drawlsmask(land_color='Linen', ocean_color='#CCFFFF') # can use HTML names or codes for colors
        map.drawcounties() # you can even add counties (and other shapefiles!)


        #ax.quiver(x.reshape(-1),y.reshape(-1),self.u[idx,:,:].reshape(-1),self.v[idx,:,:].reshape(-1), scale = 300)

        cs = map.contour(x,y,self.mslp[idx,:,:]/100, colors='blue',linewidths=1.)
        plt.clabel(cs, fontsize=9, inline=1) # contour labels
        plt.show()
        #plt.savefig('2m_temp.png')

      if("mslp" in kwargs):

        #cs = map.contour(x,y,self.mslp[idx,0,:,:]/100, colors='blue',linewidths=1.)
        cs = map.contour(x,y,self.mslp[idx,:,:]/100, colors='blue',linewidths=1.)
        #
        parallels = np.arange(min(self.lat),max(self.lat),5.)
        meridians = np.arange(min(self.lon),max(self.lon),5.)
        map.drawparallels(parallels,labels=[1,0,0,0],fontsize=10)
        map.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10)
        map.drawcoastlines()
        map.drawstates()
        map.drawcountries()
        map.drawlsmask(land_color='Linen', ocean_color='#CCFFFF') # can use HTML names or codes for colors
        map.drawcounties() # you can even add counties (and other shapefiles!)
        plt.clabel(cs, fontsize=9, inline=1) # contour labels
        plt.title('Mean Sea Level Pressure')
        plt.show()
        plt.savefig('2m_temp.png')

    def get_time(self,idx):

      nctime=self.data['time'][:]
      t_cal=self.data['time'].calendar
      t_unit = self.data.variables['time'].units

      return netcdftime.num2date(nctime[idx],units = t_unit,calendar = t_cal)
      
def split_set_to_train_and_val(Set,test_size):
  ind=np.arange(len(Set.variables['time'][:]-1))
  random.shuffle(ind)

  train_ind=ind[:int((1-test_size)*len(ind))]
  test_ind=ind[int((1-test_size)*len(ind))+1:]

  return train_ind,test_ind

