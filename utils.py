import pywt
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Cross Helicity and Residual Energy code from Sanchita
class Wavelet2Go:
    def __init__(self, f_n: int, f_min: float, f_max: float, dt: float) -> None:

        self.number_of_frequences: np.ndarray = np.array(int(f_n))
        self.frequency_range: np.ndarray = np.array((f_min, f_max))
        self.dt: np.ndarray = np.array(dt)

        self.s_spacing: np.ndarray = (1.0 / (self.number_of_frequences - 1)) * np.log2(
            self.frequency_range.max() / self.frequency_range.min()
        )

        self.scale: np.ndarray = np.power(
            2, np.arange(0, self.number_of_frequences) * self.s_spacing
        )

        self.frequency_axis: np.ndarray = self.frequency_range.min() * np.flip(
            self.scale
        )

        self.wave_scales: np.ndarray = 1.0 / (self.frequency_axis * self.dt)

        self.frequency_axis = (
            pywt.scale2frequency("cmor1.5-1.0", self.wave_scales) / self.dt
        )

        self.mother = pywt.ContinuousWavelet("cmor1.5-1.0")

        self.cone_of_influence: np.ndarray = np.ceil(
            np.sqrt(2) * self.wave_scales
        ).astype(np.int64)

    def get_frequency_axis(self) -> np.ndarray:
        return self.frequency_axis

    def get_time_axis(self, data: np.ndarray) -> np.ndarray:
        time_axis = np.linspace(0.0, data.shape[0] * self.dt, data.shape[0])
        return time_axis

    def perform_transform(self, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        complex_spectrum, frequency_axis = pywt.cwt(
            data, self.wave_scales, self.mother, self.dt
        )
        return complex_spectrum, frequency_axis

    def mask_invalid_data(
        self, complex_spectrum: np.ndarray, fill_value: float = 0
    ) -> np.ndarray:
        assert complex_spectrum.shape[0] == self.cone_of_influence.shape[0]

        for frequency_id in range(0, self.cone_of_influence.shape[0]):
            # Front side
            start_id: int = 0
            end_id: int = int(
                np.min(
                    (self.cone_of_influence[frequency_id], complex_spectrum.shape[1])
                )
            )
            complex_spectrum[frequency_id, start_id:end_id] = fill_value

            start_id = np.max(
                (
                    complex_spectrum.shape[1]
                    - self.cone_of_influence[frequency_id]
                    - 1,
                    0,
                )
            )
            end_id = complex_spectrum.shape[1]
            complex_spectrum[frequency_id, start_id:end_id] = fill_value

        return complex_spectrum

    def get_y_ticks(self, reduction_to: int) -> tuple[np.ndarray, np.ndarray]:
        output_ticks = np.arange(
            0,
            self.frequency_axis.shape[0],
            int(np.floor(self.frequency_axis.shape[0] / reduction_to)),
        )
        output_freq = self.frequency_axis[output_ticks]
        return output_ticks, output_freq

    def get_x_ticks(
        self, reduction_to: int, data: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        time_axis = self.get_time_axis(data)
        # output_ticks = np.arange(
        #     0, time_axis.shape[0], int(np.floor(time_axis.shape[0] / reduction_to))
        # )
        # output_time_axis = time_axis[output_ticks]
        output_ticks = np.arange(0, data.shape[0],reduction_to)
        output_ticks

        output_time_axis = time_axis[output_ticks]
        for i in range(len(output_time_axis)):
            output_time_axis[i]=str(i)
        output_time_axis
        return output_ticks, output_time_axis

def getProxyParameters(df, dt=60): # dt = time resolution in seconds
     
    df = df.interpolate()
    dt = 60
    mu_0 = 4*np.pi*1e-7
    m_proton = 1.673*1e-27#kg
    rho = 7*m_proton*1e6*df.Np/6 # kg m^{-3} see Good et al. 2022 MNRAS for more details
    vx_n = df.Vx*1e3 # m/s
    vy_n = df.Vy*1e3 # m/s
    vz_n = df.Vz*1e3 # m/s
    #tem = tem*1e3 # m s^{-1}
    bx_n = (1e-9*df.Bx)/np.sqrt(mu_0*rho) # m/s
    by_n = (1e-9*df.By)/np.sqrt(mu_0*rho) # m/s
    bz_n = (1e-9*df.Bz)/np.sqrt(mu_0*rho) # m/s
    z_plusx=vx_n+bx_n
    z_plusy=vy_n+by_n
    z_plusz=z=vz_n+bz_n

    z_minusx=vx_n-bx_n
    z_minusy=vy_n-by_n
    z_minusz=vz_n-bz_n
    #select the inertial range (1e-4-1e-2 hz), dt=1min=60sec
    my_wavelet2go = Wavelet2Go(f_n=11, f_min=1e-4, f_max=1e-2, dt=dt)
    y_ticks, y_ticks_freqs = my_wavelet2go.get_y_ticks(reduction_to=3)
    x_ticks, x_ticks_time = my_wavelet2go.get_x_ticks(reduction_to=24*60, data=df.Vx)
    ##############################################################################                                                  
    # <- This needs to be done only once
    cwtbx, freqs = my_wavelet2go.perform_transform(bx_n)
    cwtby, freqs = my_wavelet2go.perform_transform(by_n)
    cwtbz, freqs = my_wavelet2go.perform_transform(bz_n )
    eb=(abs(cwtbx)**2+abs(cwtby)**2+abs(cwtbz)**2)

    # Transformation -> Complex Spectrum
    cwtvx, freqs = my_wavelet2go.perform_transform(vx_n )
    cwtvy, freqs = my_wavelet2go.perform_transform(vy_n )
    cwtvz, freqs = my_wavelet2go.perform_transform(vz_n )
    ev=(abs(cwtvx)**2+abs(cwtvy)**2+abs(cwtvz)**2)
    #Cone of influence (COI) removing
    ev = my_wavelet2go.mask_invalid_data(ev, fill_value=0)
    eb = my_wavelet2go.mask_invalid_data(eb, fill_value=0)

    #residual energy calulation
    sigr=(ev-eb)/(ev+eb)
    #############################################################

    # <- This needs to be done only once
    cwtzpx, freqs = my_wavelet2go.perform_transform(z_plusx)
    cwtzpy, freqs = my_wavelet2go.perform_transform(z_plusy)
    cwtzpz, freqs = my_wavelet2go.perform_transform(z_plusz)
    ezp=(abs(cwtzpx)**2+abs(cwtzpy)**2+abs(cwtzpz)**2)

    # Transformation -> Complex Spectrum
    cwtzmx, freqs = my_wavelet2go.perform_transform(z_minusx)
    cwtzmy, freqs = my_wavelet2go.perform_transform(z_minusy)
    cwtzmz, freqs = my_wavelet2go.perform_transform(z_minusz)
    ezm=(abs(cwtzmx)**2+abs(cwtzmy)**2+abs(cwtzmz)**2)
    # Cone of influence (COI) removing
    ezp = my_wavelet2go.mask_invalid_data(ezp, fill_value=0)
    ezm = my_wavelet2go.mask_invalid_data(ezm, fill_value=0)

    #cross helicity calulation
    sigc=(ezp-ezm)/(ezp+ezm)

    #############################################################
    
    # PSD of B-field fluctuations
    #my_wavelet2go = Wavelet2Go(f_n=11, f_min=1e-3, f_max=1e-2, dt=60)
    #cwtbfx, freqs = my_wavelet2go.perform_transform(df.Bx)#-np.mean(df.Bx))
    #cwtbfy, freqs = my_wavelet2go.perform_transform(df.By)#-np.mean(df.By))
    #cwtbfz, freqs = my_wavelet2go.perform_transform(df.Bz)#-np.mean(df.Bz))
    #eb_fluc=(abs(cwtbfx)**2+abs(cwtbfy)**2+abs(cwtbfz)**2)
    
    # convert spectrogram to time series, i.e. average over frequency ranges
    ch = np.mean(sigc, axis=0)
    re = np.mean(sigr, axis=0)
    #bf = np.mean(eb_fluc, axis=0)

    #return ch, re
    return np.abs(ch), np.abs(re)
    
# Helper functions
def totalPressure( data ):
    
    #kBz = np.float64(1.3806503e-23)        # Boltzmann constant (kg*m^2)/(K*s^2)
    #mu0 = np.float64(4.0 * np.pi * 1e-7)   # Permeability (H/m) = (N/A^2) = (kg*m/s^2)/A^2
    
    z = data['Bx']*data['Bx'] + data['By']*data['By'] + data['Bz']*data['Bz']
    z = z.astype(float)
    b = np.sqrt( z )
    
    #tp = ( (b * 1e-9)**2 / (2*mu0) ) + ( 1e6*data['Np']*kBz*data['Tp'] )
    
    pt = (data['Tp']*1.4e-16)*data['Np'] + (b*1e-5)**2/(8*np.pi) # pressure in cgs metric

    pt = pt*0.1/1e-12 # pressure in pico pascal

    return pt
    
def alfvenVelocity( df ):
        
    z = df['Bx']*df['Bx'] + df['By']*df['By'] + df['Bz']*df['Bz']
    z = z.astype(float)
    b = np.sqrt( z )
    
    va = (21.81 * b) / np.sqrt(df['Np'])
    
    df['Va'] = va # in km/s
    
    return df
    
def checkNegative( df, array ):
    
    ix = np.where( df[array] <= 0)
    n = len(ix[0])
    
    return n

def processOMNI(data, file):
    
    # Thermal Speed = sqrt(2kT/M)
    # 2kT/M = (speed)**2
    # 2kT = M * (speed)**2
    # T = ( M * speed**2 ) / 2k
    
    kBz = np.float64(1.3806503e-23)                # Boltzmann constant (kg*m^2)/(K*s^2)
    Mp  = np.float64(1.672621777e-27)              # Proton mass (kg)
    Tp = (Mp * (1e3 * data['Vth'])**2) / (2.*kBz)  # Temperature (K) (need Vth in m/s not km/s)
    
    # need temp in units of eV, 1 Kelvin = 8.61732814974493E-05 Electron-Volt
    data['Tp'] = Tp * 8.61732814974493E-05 # temperature in eV
    
    z = data['Bx']*data['Bx'] + data['By']*data['By'] + data['Bz']*data['Bz']
    z = z.astype(float)
    b = np.sqrt( z )

    # magnetic field fluctuations
    data['date_time'] = pd.to_datetime(data['date_time'], format='%Y-%m-%d %H:%M:%S')
    data = data.set_index(['date_time'])
    tmp = data.resample('10Min', label='right').mean()
    previous = 0

    z = data['Bx']*data['Bx'] + data['By']*data['By'] + data['Bz']*data['Bz']
    z = z.astype(float)
    b = np.sqrt( z )
    
    rmsB = []
    for indx, row in tmp.iterrows():
    
        if ( previous == 0 ):
            ix = np.where( data.index < indx )
        else:
            ix = np.where( (data.index < indx) & (data.index > previousIndex) )
        previousIndex = indx
        previous += 1
    
        rowB = np.sqrt( row.Bx*row.Bx + row.By*row.By + row.Bz*row.Bz )
        rms = ( (b[ix[0]] - rowB)**2 ) / len(ix[0]) 
        rms = np.sqrt( np.sum(rms) )
        rmsB.append( rms ) 

    # total pressure
    tp = totalPressure(data)
    
    Texp = (data['V']/258.)**3.113 # in eV per formula from classification publication
    Tratio = Texp / data['Tp'] # eV / eV
    
    va = alfvenVelocity( data ) # in km/s
    
    entropy = data['Tp'] / ( data['Np']**(2/3) ) # in eV cm**2
    
    crossHelicity, residualEnergy = getProxyParameters(data, dt=60)
    
    data['Tratio'] = Tratio
    data['entropy'] = entropy
    data['crossHelicity'] = crossHelicity
    data['residualEnergy'] = residualEnergy
    data['TotalPressure'] = tp
    
    # average to 10-minute
    data = data.resample('10Min').mean()
    data['bFluctuations'] = rmsB
    
    data = data.drop( columns=['ddoy', 'Bx', 'By', 'Bz', 'V', 'Vx', 'Vy', 'Vz', \
                               'Vth', 'Np', 'Tp', 'IMF', 'PLS'] )
    data = data.rename(columns={"Beta": "Beta_p"})
    
    # check for negative values
    n1 = checkNegative( data, 'Va' )
    n2 = checkNegative( data, 'Tratio' )
    n3 = checkNegative( data, 'entropy' )
    n4 = checkNegative( data, 'Beta_p' )
    
    if ( (n1 > 0) or (n2 > 0) or (n3 > 0) or (n4 > 0) ): 
        print('Negative values found in:', file)
    
    data = data[ data['Va'] > 0 ]
    data = data[ data['Tratio'] > 0 ]
    data = data[ data['entropy'] > 0 ]
    data = data[ data['Beta_p'] > 0 ]
    
    n = data.shape[0]
    parts = file.split('/')
    data['files'] = [parts[-1]] * n
    
    data = data.dropna()
    
    return data

def readSpc(file):
    
    data = pd.read_csv(file)
    
    # convert RTN to GSE
    if 'Br' in data.columns: 
        data['Bx'] = -1. * data['Br']
        data = data.drop( columns=['Br'] )

    if 'Bt' in data.columns: 
        data['By'] = -1. * data['Bt']
        data = data.drop( columns=['Bt'] )
        
    if 'Bn' in data.columns: 
        data['Bz'] = data['Bn']
        data = data.drop( columns=['Bn'] )
     
    z = data['Bx']*data['Bx'] + data['By']*data['By'] + data['Bz']*data['Bz']
    z = z.astype(float)
    b = np.sqrt( z )
    
    # magnetic field fluctuations
    data['date_time'] = pd.to_datetime(data['date_time'], format='%Y-%m-%d %H:%M:%S')
    data = data.set_index(['date_time'])
    tmp = data.resample('10Min', label='right').mean()
    previous = 0

    z = data['Bx']*data['Bx'] + data['By']*data['By'] + data['Bz']*data['Bz']
    z = z.astype(float)
    b = np.sqrt( z )

    rmsB = []
    for indx, row in tmp.iterrows():
    
        if ( previous == 0 ):
            ix = np.where( data.index < indx )
        else:
            ix = np.where( (data.index < indx) & (data.index > previousIndex) )
        previousIndex = indx
        previous += 1
    
        rowB = np.sqrt( row.Bx*row.Bx + row.By*row.By + row.Bz*row.Bz )
        rms = ( (b[ix[0]] - rowB)**2 ) / len(ix[0]) 
        rms = np.sqrt( np.sum(rms) )
        rmsB.append( rms ) 

    # total pressure
    tp = totalPressure(data)
    
    v = data['Vx']*data['Vx'] + data['Vy']*data['Vy'] + data['Vz']*data['Vz']
    v = v.astype(float)
    v = np.sqrt( v )
    
    # need temp in units of eV
    # 1 Kelvin = 8.61732814974493E-05 Electron-Volt
    data['Tp'] = data['Tp'] * 8.61732814974493E-05 # Temp in eV
    
    Texp = (v/258.)**3.113 # in eV
    Tratio = Texp / data['Tp'] # ratio is eV / eV
    
    va = alfvenVelocity( data )
    
    entropy = data['Tp'] / ( data['Np']**(2/3) ) # in eV cm**2
    
    crossHelicity, residualEnergy = getProxyParameters(data, dt=60)
    
    data['Tratio'] = Tratio
    data['entropy'] = entropy
    data['crossHelicity'] = crossHelicity
    data['residualEnergy'] = residualEnergy
    data['TotalPressure'] = tp
    
    # average to 10-minute
    data = data.resample('10Min').mean()
    data['bFluctuations'] = rmsB
        
    data = data.drop( columns=['ddoy', 'Bx', 'By', 'Bz', 'Vx', 'Vy', 'Vz', 'Np', 'Vsw', 'Tp'] )
        
    # check for negative values
    n1 = checkNegative( data, 'Va' )
    n2 = checkNegative( data, 'Tratio' )
    n3 = checkNegative( data, 'entropy' )
    n4 = checkNegative( data, 'Beta_p' )
    
    if ( (n1 > 0) or (n2 > 0) or (n3 > 0) or (n4 > 0) ): 
        print('Negative values found in:', file)
    
    data = data[ data['Va'] > 0 ]
    data = data[ data['Tratio'] > 0 ]
    data = data[ data['entropy'] > 0 ]
    data = data[ data['Beta_p'] > 0 ]
        
    data = data.dropna()
    
    return data

def getNormalizingValues( df ):
    
    return df.min(), df.max() 

def normalize( df, mn=-1, mx=-1 ):
    
    # decimal point normalization
    #m = df.max()
    #q = len( str(abs(int(m))) )
    #return df / 10**q

    #return (df-df.mean()) / df.std() # z-score normalization
    
    # min-max normalization
    if ( (mn != -1) and (mx != -1) ):
        return (df-df.min()) / (df.max()-df.min())
    else: 
        return (df-mn) / (mx-mn)