from sfe.BispectrogramFeatures import BispectrogramFeatures
from sfe.SpectrogramFeatures import SpectrogramFeatures
from sfe.feature_extractor_1D import ODFeatureExtractor
from sfe.sfe_aux_tools import getfgrid, spectrum_filter
from sfe.signal_transform import SignalTransform
from SleepStageDetection.utils import merge_dicts
import numpy as np
import pandas as pd

class Features:
    def __init__(self, signal:pd.DataFrame, fs, label, window_len=100) -> None:
        self.signal = signal.to_numpy()
        self.fs = fs
        self.wl = int(window_len/2) # int(/2)
        self.label = label
        self.signal_transformed = SignalTransform(self.signal, Fs=self.fs)   
    
    def __bispectogram_features(self):
        #não funcionando devido a função bispectrum_filter não estar implementada
        #spectogram generation
        bg = self.signal_transformed.get_bispectrogram()
        bg_signal = BispectrogramFeatures(bg, self.fs, len(bg))
        features = bg_signal.extract_features()
        return features
    
    def __spectogram_features(self, delta_f = True, theta_f = True, alpha_f = True, beta_f = True, gamma_f = True, entire_f = True):
        #funciona com window_len = 100
        """
        - Caracteristicas extraidas: sg_average, sg_std, sg_min, sg_peak, sg_kurtosis, sg_skewness, sg_centroid, sg_sbe, sg_sbw, sg_rms, sg_crest_factor, 
        sg_var_coef, sg_q1, sg_q2, sg_q3, sg_interquatile_range

        - Caracteristicas não extraidas: line_length, nonlinear_energy, Hurst_exponent, Shanon_entropy, Renyi_entropy

        """
        sg = self.signal_transformed.get_spectrogram(window_len=self.wl)
        sg_signal = SpectrogramFeatures(sg, self.fs, len(sg), label=self.label)
        features = sg_signal.extract_features(delta_f = True, theta_f = True, alpha_f = True, beta_f = True, gamma_f = True, entire_f = True)
        return features
    
    def __power_spectrum_features(self):
        #sample_entropy = nan ou inf
        __delta_F = [1, 4]
        __theta_F = [4, 8]
        __alpha_F = [8, 12]
        __beta_F = [12, 30]
        __gamma_F = [30, 60]
        __entire_F = [1, 60]

        ps = self.signal_transformed.get_power_spectrum()

        # --- DELTA --- #funciona com window_len = 100
        # power spectrum of Delta Waves
        ps_delta = spectrum_filter(ps, self.fs, __delta_F)
        f_gridDelta = getfgrid(self.fs, len(ps), fpassMin=__delta_F[0], fpassMax=__delta_F[-1])
        # power spectrum feature extraction of Delta Waves
        psDeltaO = ODFeatureExtractor(ps_delta, freq_grid=f_gridDelta[:-1], label='PS_Delta_' + self.label)
        psDeltaO.extract_all_features()
        psDeltaF = psDeltaO.get_extracted_features()
        
        # --- THETA --- #funciona com window_len = 100
        # power spectrum of Theta Waves
        ps_theta = spectrum_filter(ps, self.fs, __theta_F)
        f_gridTheta = getfgrid(self.fs, len(ps), fpassMin=__theta_F[0], fpassMax=__theta_F[-1])
        # power spectrum feature extraction of Theta Waves
        psThetaO = ODFeatureExtractor(ps_theta, freq_grid=f_gridTheta[:-1], label='PS_Theta_' + self.label)
        psThetaO.extract_all_features()
        psThetaF = psThetaO.get_extracted_features()

        # --- ALPHA --- 
        # power spectrum of Alpha Waves
        ps_alpha = spectrum_filter(ps, self.fs, __alpha_F)
        f_gridAlpha = getfgrid(self.fs, len(ps), fpassMin=__alpha_F[0], fpassMax=__alpha_F[-1])
        # power spectrum feature extraction of Alpha Waves
        psAlphaO = ODFeatureExtractor(ps_alpha, freq_grid=f_gridAlpha[:-1], label='PS_Alpha_' + self.label)
        psAlphaO.extract_all_features()
        psAlphaF = psAlphaO.get_extracted_features()
        
        # --- BETA --- 
        # power spectrum of Beta Waves
        ps_beta = spectrum_filter(ps, self.fs, __beta_F)
        f_gridBeta = getfgrid(self.fs, len(ps), fpassMin=__beta_F[0], fpassMax=__beta_F[-1])
        # power spectrum feature extraction of Beta Waves
        psBetaO = ODFeatureExtractor(ps_beta, freq_grid=f_gridBeta[:-1], label='PS_Beta_' + self.label)
        psBetaO.extract_all_features()
        psBetaF = psAlphaO.get_extracted_features()
        
        # --- GAMA --- 
        # power spectrum of Gama Waves
        ps_gama = spectrum_filter(ps, self.fs, __gamma_F)
        f_gridGama = getfgrid(self.fs, len(ps), fpassMin=__gamma_F[0], fpassMax=__gamma_F[-1])
        # power spectrum feature extraction of Gama Waves
        psGamaO = ODFeatureExtractor(ps_gama, freq_grid=f_gridGama[:-1], label='PS_Gama_' + self.label)
        psGamaO.extract_all_features()
        psGamaF = psGamaO.get_extracted_features()

        # --- Entire --- 
        ps_entire = spectrum_filter(ps, self.fs, __entire_F)
        f_gridEntire = getfgrid(self.fs, len(ps), fpassMin=__entire_F[0], fpassMax=__entire_F[-1])
        psEntireO = ODFeatureExtractor(ps_entire, freq_grid=f_gridEntire[:-1], label='PS_Entire_' + self.label)
        psEntireO.extract_all_features()
        psEntireF = psEntireO.get_extracted_features()

        return merge_dicts(psDeltaF, psThetaF, psAlphaF, psBetaF, psGamaF, psEntireF)

    def __timeseries_features(self):
        signal = np.asarray(self.signal)
        f_grid = getfgrid(self.fs, len(signal))
        tsO = ODFeatureExtractor(signal, freq_grid=f_grid, label='TS_Entire_' + self.label)
        tsO.extract_all_features()
        tsFeatures = tsO.get_extracted_features()
        return tsFeatures
    
    def extract_features(self):
        return merge_dicts(
            #self.__bispectogram_features(),
            self.__spectogram_features(),
            self.__power_spectrum_features(),
            self.__timeseries_features()
        )