
from SleepStageDetection.utils import preparing_for_binary_classification, balanceamento_por_classe_undersampling

from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import CategoricalNB, ComplementNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import MinMaxScaler

import pandas as pd
import numpy as np
import time

pd.options.mode.use_inf_as_na = True

features_Fz_Cz = [
    #Fz - Cz Channel
    "SG_average_delta_EEG Fpz-Cz", "SG_std_delta_EEG Fpz-Cz", "SG_min_delta_EEG Fpz-Cz", "SG_peak_delta_EEG Fpz-Cz", 
    "SG_kurtosis_delta_EEG Fpz-Cz", "SG_skewness_delta_EEG Fpz-Cz", "SG_centroid_delta_EEG Fpz-Cz", "SG_sbe_delta_EEG Fpz-Cz", "SG_sbw_delta_EEG Fpz-Cz", 
    "SG_rms_delta_EEG Fpz-Cz", "SG_crest_factor_delta_EEG Fpz-Cz", "SG_var_coef_delta_EEG Fpz-Cz", "SG_Q1_delta_EEG Fpz-Cz", "SG_Q2_delta_EEG Fpz-Cz", 
    "SG_Q3_delta_EEG Fpz-Cz", "SG_interquatile_range_delta_EEG Fpz-Cz", "SG_average_theta_EEG Fpz-Cz", "SG_std_theta_EEG Fpz-Cz", "SG_min_theta_EEG Fpz-Cz", 
    "SG_peak_theta_EEG Fpz-Cz", "SG_kurtosis_theta_EEG Fpz-Cz", "SG_skewness_theta_EEG Fpz-Cz", "SG_centroid_theta_EEG Fpz-Cz", "SG_sbe_theta_EEG Fpz-Cz", 
    "SG_sbw_theta_EEG Fpz-Cz", "SG_rms_theta_EEG Fpz-Cz", "SG_crest_factor_theta_EEG Fpz-Cz", "SG_var_coef_theta_EEG Fpz-Cz", "SG_Q1_theta_EEG Fpz-Cz", 
    "SG_Q2_theta_EEG Fpz-Cz", "SG_Q3_theta_EEG Fpz-Cz", "SG_interquatile_range_theta_EEG Fpz-Cz", "SG_average_alpha_EEG Fpz-Cz", "SG_std_alpha_EEG Fpz-Cz", 
    "SG_min_alpha_EEG Fpz-Cz", "SG_peak_alpha_EEG Fpz-Cz", "SG_kurtosis_alpha_EEG Fpz-Cz", "SG_skewness_alpha_EEG Fpz-Cz", "SG_centroid_alpha_EEG Fpz-Cz", 
    "SG_sbe_alpha_EEG Fpz-Cz", "SG_sbw_alpha_EEG Fpz-Cz", "SG_rms_alpha_EEG Fpz-Cz", "SG_crest_factor_alpha_EEG Fpz-Cz", "SG_var_coef_alpha_EEG Fpz-Cz", 
    "SG_Q1_alpha_EEG Fpz-Cz", "SG_Q2_alpha_EEG Fpz-Cz", "SG_Q3_alpha_EEG Fpz-Cz", "SG_interquatile_range_alpha_EEG Fpz-Cz", "SG_average_beta_EEG Fpz-Cz", 
    "SG_std_beta_EEG Fpz-Cz", "SG_min_beta_EEG Fpz-Cz", "SG_peak_beta_EEG Fpz-Cz", "SG_kurtosis_beta_EEG Fpz-Cz", "SG_skewness_beta_EEG Fpz-Cz", 
    "SG_centroid_beta_EEG Fpz-Cz", "SG_sbe_beta_EEG Fpz-Cz", "SG_sbw_beta_EEG Fpz-Cz", "SG_rms_beta_EEG Fpz-Cz", "SG_crest_factor_beta_EEG Fpz-Cz", 
    "SG_var_coef_beta_EEG Fpz-Cz", "SG_Q1_beta_EEG Fpz-Cz", "SG_Q2_beta_EEG Fpz-Cz", "SG_Q3_beta_EEG Fpz-Cz", "SG_interquatile_range_beta_EEG Fpz-Cz", 
    "SG_average_gamma_EEG Fpz-Cz", "SG_std_gamma_EEG Fpz-Cz", "SG_min_gamma_EEG Fpz-Cz", "SG_peak_gamma_EEG Fpz-Cz", "SG_kurtosis_gamma_EEG Fpz-Cz", 
    "SG_skewness_gamma_EEG Fpz-Cz", "SG_centroid_gamma_EEG Fpz-Cz", "SG_sbe_gamma_EEG Fpz-Cz", "SG_sbw_gamma_EEG Fpz-Cz", "SG_rms_gamma_EEG Fpz-Cz", 
    "SG_crest_factor_gamma_EEG Fpz-Cz", "SG_var_coef_gamma_EEG Fpz-Cz", "SG_Q1_gamma_EEG Fpz-Cz", "SG_Q2_gamma_EEG Fpz-Cz", "SG_Q3_gamma_EEG Fpz-Cz", 
    "SG_interquatile_range_gamma_EEG Fpz-Cz", "SG_average_entire_EEG Fpz-Cz", "SG_std_entire_EEG Fpz-Cz", "SG_min_entire_EEG Fpz-Cz", "SG_peak_entire_EEG Fpz-Cz", 
    "SG_kurtosis_entire_EEG Fpz-Cz", "SG_skewness_entire_EEG Fpz-Cz", "SG_centroid_entire_EEG Fpz-Cz", "SG_sbe_entire_EEG Fpz-Cz", "SG_sbw_entire_EEG Fpz-Cz", 
    "SG_rms_entire_EEG Fpz-Cz", "SG_crest_factor_entire_EEG Fpz-Cz", "SG_var_coef_entire_EEG Fpz-Cz", "SG_Q1_entire_EEG Fpz-Cz", "SG_Q2_entire_EEG Fpz-Cz", 
    "SG_Q3_entire_EEG Fpz-Cz", "SG_interquatile_range_entire_EEG Fpz-Cz", "mean_PS_Delta_EEG Fpz-Cz", "std_PS_Delta_EEG Fpz-Cz", "peak_PS_Delta_EEG Fpz-Cz", 
    "min_PS_Delta_EEG Fpz-Cz", "peak_frequency_PS_Delta_EEG Fpz-Cz", "Q1_PS_Delta_EEG Fpz-Cz", "Q2_PS_Delta_EEG Fpz-Cz", "Q3_PS_Delta_EEG Fpz-Cz", 
    "Q_range_PS_Delta_EEG Fpz-Cz", "amplitude_PS_Delta_EEG Fpz-Cz", "spec_centroid_PS_Delta_EEG Fpz-Cz", "kurtosis_PS_Delta_EEG Fpz-Cz", 
    "skewness_PS_Delta_EEG Fpz-Cz", "coef_var_PS_Delta_EEG Fpz-Cz", "flatness_PS_Delta_EEG Fpz-Cz", "1st_moment_PS_Delta_EEG Fpz-Cz", 
    "2nd_moment_PS_Delta_EEG Fpz-Cz", "rms_PS_Delta_EEG Fpz-Cz", "crest_factor_PS_Delta_EEG Fpz-Cz", "line_length_PS_Delta_EEG Fpz-Cz", 
    "nonlinear_energy_PS_Delta_EEG Fpz-Cz", "Hurst_exponent_PS_Delta_EEG Fpz-Cz", "Hjorth_activity_PS_Delta_EEG Fpz-Cz", "Hjorth_mobility_PS_Delta_EEG Fpz-Cz", 
    "Hjorth_complexity_PS_Delta_EEG Fpz-Cz", "Shannon_entropy_PS_Delta_EEG Fpz-Cz", "Renyi_entropy_PS_Delta_EEG Fpz-Cz", "approx_entropy_PS_Delta_EEG Fpz-Cz", 
    "mean_PS_Theta_EEG Fpz-Cz", "std_PS_Theta_EEG Fpz-Cz", "peak_PS_Theta_EEG Fpz-Cz", "min_PS_Theta_EEG Fpz-Cz", "peak_frequency_PS_Theta_EEG Fpz-Cz", 
    "Q1_PS_Theta_EEG Fpz-Cz", "Q2_PS_Theta_EEG Fpz-Cz", "Q3_PS_Theta_EEG Fpz-Cz", "Q_range_PS_Theta_EEG Fpz-Cz", "amplitude_PS_Theta_EEG Fpz-Cz", 
    "spec_centroid_PS_Theta_EEG Fpz-Cz", "kurtosis_PS_Theta_EEG Fpz-Cz", "skewness_PS_Theta_EEG Fpz-Cz", "coef_var_PS_Theta_EEG Fpz-Cz", 
    "flatness_PS_Theta_EEG Fpz-Cz", "1st_moment_PS_Theta_EEG Fpz-Cz", "2nd_moment_PS_Theta_EEG Fpz-Cz", "rms_PS_Theta_EEG Fpz-Cz", 
    "crest_factor_PS_Theta_EEG Fpz-Cz", "line_length_PS_Theta_EEG Fpz-Cz", "nonlinear_energy_PS_Theta_EEG Fpz-Cz", "Hurst_exponent_PS_Theta_EEG Fpz-Cz", 
    "Hjorth_activity_PS_Theta_EEG Fpz-Cz", "Hjorth_mobility_PS_Theta_EEG Fpz-Cz", "Hjorth_complexity_PS_Theta_EEG Fpz-Cz", "Shannon_entropy_PS_Theta_EEG Fpz-Cz", 
    "Renyi_entropy_PS_Theta_EEG Fpz-Cz", "approx_entropy_PS_Theta_EEG Fpz-Cz", "mean_PS_Alpha_EEG Fpz-Cz", "std_PS_Alpha_EEG Fpz-Cz", "peak_PS_Alpha_EEG Fpz-Cz", 
    "min_PS_Alpha_EEG Fpz-Cz", "peak_frequency_PS_Alpha_EEG Fpz-Cz", "Q1_PS_Alpha_EEG Fpz-Cz", "Q2_PS_Alpha_EEG Fpz-Cz", "Q3_PS_Alpha_EEG Fpz-Cz", 
    "Q_range_PS_Alpha_EEG Fpz-Cz", "amplitude_PS_Alpha_EEG Fpz-Cz", "spec_centroid_PS_Alpha_EEG Fpz-Cz", "kurtosis_PS_Alpha_EEG Fpz-Cz", 
    "skewness_PS_Alpha_EEG Fpz-Cz", "coef_var_PS_Alpha_EEG Fpz-Cz", "flatness_PS_Alpha_EEG Fpz-Cz", "1st_moment_PS_Alpha_EEG Fpz-Cz", 
    "2nd_moment_PS_Alpha_EEG Fpz-Cz", "rms_PS_Alpha_EEG Fpz-Cz", "crest_factor_PS_Alpha_EEG Fpz-Cz", "line_length_PS_Alpha_EEG Fpz-Cz", 
    "nonlinear_energy_PS_Alpha_EEG Fpz-Cz", "Hurst_exponent_PS_Alpha_EEG Fpz-Cz", "Hjorth_activity_PS_Alpha_EEG Fpz-Cz", "Hjorth_mobility_PS_Alpha_EEG Fpz-Cz", 
    "Hjorth_complexity_PS_Alpha_EEG Fpz-Cz", "Shannon_entropy_PS_Alpha_EEG Fpz-Cz", "Renyi_entropy_PS_Alpha_EEG Fpz-Cz", "approx_entropy_PS_Alpha_EEG Fpz-Cz", 
    "mean_PS_Beta_EEG Fpz-Cz", "std_PS_Beta_EEG Fpz-Cz", "peak_PS_Beta_EEG Fpz-Cz", "min_PS_Beta_EEG Fpz-Cz", "peak_frequency_PS_Beta_EEG Fpz-Cz", 
    "Q1_PS_Beta_EEG Fpz-Cz", "Q2_PS_Beta_EEG Fpz-Cz", "Q3_PS_Beta_EEG Fpz-Cz", "Q_range_PS_Beta_EEG Fpz-Cz", "amplitude_PS_Beta_EEG Fpz-Cz", 
    "spec_centroid_PS_Beta_EEG Fpz-Cz", "kurtosis_PS_Beta_EEG Fpz-Cz", "skewness_PS_Beta_EEG Fpz-Cz", "coef_var_PS_Beta_EEG Fpz-Cz", "flatness_PS_Beta_EEG Fpz-Cz", 
    "1st_moment_PS_Beta_EEG Fpz-Cz", "2nd_moment_PS_Beta_EEG Fpz-Cz", "rms_PS_Beta_EEG Fpz-Cz", "crest_factor_PS_Beta_EEG Fpz-Cz", "line_length_PS_Beta_EEG Fpz-Cz", 
    "nonlinear_energy_PS_Beta_EEG Fpz-Cz", "Hurst_exponent_PS_Beta_EEG Fpz-Cz", "Hjorth_activity_PS_Beta_EEG Fpz-Cz", "Hjorth_mobility_PS_Beta_EEG Fpz-Cz", 
    "Hjorth_complexity_PS_Beta_EEG Fpz-Cz", "Shannon_entropy_PS_Beta_EEG Fpz-Cz", "Renyi_entropy_PS_Beta_EEG Fpz-Cz", "approx_entropy_PS_Beta_EEG Fpz-Cz", 
    "mean_PS_Gama_EEG Fpz-Cz", "std_PS_Gama_EEG Fpz-Cz", "peak_PS_Gama_EEG Fpz-Cz", "min_PS_Gama_EEG Fpz-Cz", "peak_frequency_PS_Gama_EEG Fpz-Cz", 
    "Q1_PS_Gama_EEG Fpz-Cz", "Q2_PS_Gama_EEG Fpz-Cz", "Q3_PS_Gama_EEG Fpz-Cz", "Q_range_PS_Gama_EEG Fpz-Cz", "amplitude_PS_Gama_EEG Fpz-Cz", 
    "spec_centroid_PS_Gama_EEG Fpz-Cz", "kurtosis_PS_Gama_EEG Fpz-Cz", "skewness_PS_Gama_EEG Fpz-Cz", "coef_var_PS_Gama_EEG Fpz-Cz", "flatness_PS_Gama_EEG Fpz-Cz", 
    "1st_moment_PS_Gama_EEG Fpz-Cz", "2nd_moment_PS_Gama_EEG Fpz-Cz", "rms_PS_Gama_EEG Fpz-Cz", "crest_factor_PS_Gama_EEG Fpz-Cz", "line_length_PS_Gama_EEG Fpz-Cz", 
    "nonlinear_energy_PS_Gama_EEG Fpz-Cz", "Hurst_exponent_PS_Gama_EEG Fpz-Cz", "Hjorth_activity_PS_Gama_EEG Fpz-Cz", "Hjorth_mobility_PS_Gama_EEG Fpz-Cz", 
    "Hjorth_complexity_PS_Gama_EEG Fpz-Cz", "Shannon_entropy_PS_Gama_EEG Fpz-Cz", "Renyi_entropy_PS_Gama_EEG Fpz-Cz", "approx_entropy_PS_Gama_EEG Fpz-Cz", 
    #"sample_entropy_PS_Gama_EEG Fpz-Cz",
    "mean_PS_Entire_EEG Fpz-Cz", "std_PS_Entire_EEG Fpz-Cz", "peak_PS_Entire_EEG Fpz-Cz", "min_PS_Entire_EEG Fpz-Cz", 
    "peak_frequency_PS_Entire_EEG Fpz-Cz", "Q1_PS_Entire_EEG Fpz-Cz", "Q2_PS_Entire_EEG Fpz-Cz", "Q3_PS_Entire_EEG Fpz-Cz", "Q_range_PS_Entire_EEG Fpz-Cz", 
    "amplitude_PS_Entire_EEG Fpz-Cz", "spec_centroid_PS_Entire_EEG Fpz-Cz", "kurtosis_PS_Entire_EEG Fpz-Cz", "skewness_PS_Entire_EEG Fpz-Cz", 
    "coef_var_PS_Entire_EEG Fpz-Cz", "flatness_PS_Entire_EEG Fpz-Cz", "1st_moment_PS_Entire_EEG Fpz-Cz", "2nd_moment_PS_Entire_EEG Fpz-Cz", 
    "rms_PS_Entire_EEG Fpz-Cz", "crest_factor_PS_Entire_EEG Fpz-Cz", "line_length_PS_Entire_EEG Fpz-Cz", "nonlinear_energy_PS_Entire_EEG Fpz-Cz", 
    "Hurst_exponent_PS_Entire_EEG Fpz-Cz", "Hjorth_activity_PS_Entire_EEG Fpz-Cz", "Hjorth_mobility_PS_Entire_EEG Fpz-Cz", "Hjorth_complexity_PS_Entire_EEG Fpz-Cz", 
    "Shannon_entropy_PS_Entire_EEG Fpz-Cz", "Renyi_entropy_PS_Entire_EEG Fpz-Cz", "approx_entropy_PS_Entire_EEG Fpz-Cz", 
    #"sample_entropy_PS_Entire_EEG Fpz-Cz", 
    "mean_TS_Entire_EEG Fpz-Cz", "std_TS_Entire_EEG Fpz-Cz", "peak_TS_Entire_EEG Fpz-Cz", "min_TS_Entire_EEG Fpz-Cz", "peak_frequency_TS_Entire_EEG Fpz-Cz", 
    "Q1_TS_Entire_EEG Fpz-Cz", "Q2_TS_Entire_EEG Fpz-Cz", "Q3_TS_Entire_EEG Fpz-Cz", "Q_range_TS_Entire_EEG Fpz-Cz", "amplitude_TS_Entire_EEG Fpz-Cz", 
    "spec_centroid_TS_Entire_EEG Fpz-Cz", "kurtosis_TS_Entire_EEG Fpz-Cz", "skewness_TS_Entire_EEG Fpz-Cz", "coef_var_TS_Entire_EEG Fpz-Cz", 
    "flatness_TS_Entire_EEG Fpz-Cz", "1st_moment_TS_Entire_EEG Fpz-Cz", "2nd_moment_TS_Entire_EEG Fpz-Cz", "rms_TS_Entire_EEG Fpz-Cz", 
    "crest_factor_TS_Entire_EEG Fpz-Cz", "line_length_TS_Entire_EEG Fpz-Cz", "nonlinear_energy_TS_Entire_EEG Fpz-Cz", "Hurst_exponent_TS_Entire_EEG Fpz-Cz", 
    "Hjorth_activity_TS_Entire_EEG Fpz-Cz", "Hjorth_mobility_TS_Entire_EEG Fpz-Cz", "Hjorth_complexity_TS_Entire_EEG Fpz-Cz", "Shannon_entropy_TS_Entire_EEG Fpz-Cz", 
    "Renyi_entropy_TS_Entire_EEG Fpz-Cz", "approx_entropy_TS_Entire_EEG Fpz-Cz", 
    #"sample_entropy_TS_Entire_EEG Fpz-Cz", 
]

features_Pz_Oz = [
    #Pz - Oz Channel
    "SG_average_delta_EEG Pz-Oz", "SG_std_delta_EEG Pz-Oz", "SG_min_delta_EEG Pz-Oz", "SG_peak_delta_EEG Pz-Oz", "SG_kurtosis_delta_EEG Pz-Oz", 
    "SG_skewness_delta_EEG Pz-Oz", "SG_centroid_delta_EEG Pz-Oz", "SG_sbe_delta_EEG Pz-Oz", "SG_sbw_delta_EEG Pz-Oz", "SG_rms_delta_EEG Pz-Oz", 
    "SG_crest_factor_delta_EEG Pz-Oz", "SG_var_coef_delta_EEG Pz-Oz", "SG_Q1_delta_EEG Pz-Oz", "SG_Q2_delta_EEG Pz-Oz", "SG_Q3_delta_EEG Pz-Oz", 
    "SG_interquatile_range_delta_EEG Pz-Oz", "SG_average_theta_EEG Pz-Oz", "SG_std_theta_EEG Pz-Oz", "SG_min_theta_EEG Pz-Oz", "SG_peak_theta_EEG Pz-Oz", 
    "SG_kurtosis_theta_EEG Pz-Oz", "SG_skewness_theta_EEG Pz-Oz", "SG_centroid_theta_EEG Pz-Oz", "SG_sbe_theta_EEG Pz-Oz", "SG_sbw_theta_EEG Pz-Oz", 
    "SG_rms_theta_EEG Pz-Oz", "SG_crest_factor_theta_EEG Pz-Oz", "SG_var_coef_theta_EEG Pz-Oz", "SG_Q1_theta_EEG Pz-Oz", "SG_Q2_theta_EEG Pz-Oz", 
    "SG_Q3_theta_EEG Pz-Oz", "SG_interquatile_range_theta_EEG Pz-Oz", "SG_average_alpha_EEG Pz-Oz", "SG_std_alpha_EEG Pz-Oz", "SG_min_alpha_EEG Pz-Oz", 
    "SG_peak_alpha_EEG Pz-Oz", "SG_kurtosis_alpha_EEG Pz-Oz", "SG_skewness_alpha_EEG Pz-Oz", "SG_centroid_alpha_EEG Pz-Oz", "SG_sbe_alpha_EEG Pz-Oz", 
    "SG_sbw_alpha_EEG Pz-Oz", "SG_rms_alpha_EEG Pz-Oz", "SG_crest_factor_alpha_EEG Pz-Oz", "SG_var_coef_alpha_EEG Pz-Oz", "SG_Q1_alpha_EEG Pz-Oz", 
    "SG_Q2_alpha_EEG Pz-Oz", "SG_Q3_alpha_EEG Pz-Oz", "SG_interquatile_range_alpha_EEG Pz-Oz", "SG_average_beta_EEG Pz-Oz", "SG_std_beta_EEG Pz-Oz", 
    "SG_min_beta_EEG Pz-Oz", "SG_peak_beta_EEG Pz-Oz", "SG_kurtosis_beta_EEG Pz-Oz", "SG_skewness_beta_EEG Pz-Oz", "SG_centroid_beta_EEG Pz-Oz", 
    "SG_sbe_beta_EEG Pz-Oz", "SG_sbw_beta_EEG Pz-Oz", "SG_rms_beta_EEG Pz-Oz", "SG_crest_factor_beta_EEG Pz-Oz", "SG_var_coef_beta_EEG Pz-Oz", 
    "SG_Q1_beta_EEG Pz-Oz", "SG_Q2_beta_EEG Pz-Oz", "SG_Q3_beta_EEG Pz-Oz", "SG_interquatile_range_beta_EEG Pz-Oz", "SG_average_gamma_EEG Pz-Oz", 
    "SG_std_gamma_EEG Pz-Oz", "SG_min_gamma_EEG Pz-Oz", "SG_peak_gamma_EEG Pz-Oz", "SG_kurtosis_gamma_EEG Pz-Oz", "SG_skewness_gamma_EEG Pz-Oz", 
    "SG_centroid_gamma_EEG Pz-Oz", "SG_sbe_gamma_EEG Pz-Oz", "SG_sbw_gamma_EEG Pz-Oz", "SG_rms_gamma_EEG Pz-Oz", "SG_crest_factor_gamma_EEG Pz-Oz", 
    "SG_var_coef_gamma_EEG Pz-Oz", "SG_Q1_gamma_EEG Pz-Oz", "SG_Q2_gamma_EEG Pz-Oz", "SG_Q3_gamma_EEG Pz-Oz", "SG_interquatile_range_gamma_EEG Pz-Oz", 
    "SG_average_entire_EEG Pz-Oz", "SG_std_entire_EEG Pz-Oz", "SG_min_entire_EEG Pz-Oz", "SG_peak_entire_EEG Pz-Oz", "SG_kurtosis_entire_EEG Pz-Oz", 
    "SG_skewness_entire_EEG Pz-Oz", "SG_centroid_entire_EEG Pz-Oz", "SG_sbe_entire_EEG Pz-Oz", "SG_sbw_entire_EEG Pz-Oz", "SG_rms_entire_EEG Pz-Oz", 
    "SG_crest_factor_entire_EEG Pz-Oz", "SG_var_coef_entire_EEG Pz-Oz", "SG_Q1_entire_EEG Pz-Oz", "SG_Q2_entire_EEG Pz-Oz", "SG_Q3_entire_EEG Pz-Oz", 
    "SG_interquatile_range_entire_EEG Pz-Oz", "mean_PS_Delta_EEG Pz-Oz", "std_PS_Delta_EEG Pz-Oz", "peak_PS_Delta_EEG Pz-Oz", "min_PS_Delta_EEG Pz-Oz", 
    "peak_frequency_PS_Delta_EEG Pz-Oz", "Q1_PS_Delta_EEG Pz-Oz", "Q2_PS_Delta_EEG Pz-Oz", "Q3_PS_Delta_EEG Pz-Oz", "Q_range_PS_Delta_EEG Pz-Oz", 
    "amplitude_PS_Delta_EEG Pz-Oz", "spec_centroid_PS_Delta_EEG Pz-Oz", "kurtosis_PS_Delta_EEG Pz-Oz", "skewness_PS_Delta_EEG Pz-Oz", 
    "coef_var_PS_Delta_EEG Pz-Oz", "flatness_PS_Delta_EEG Pz-Oz", "1st_moment_PS_Delta_EEG Pz-Oz", "2nd_moment_PS_Delta_EEG Pz-Oz", 
    "rms_PS_Delta_EEG Pz-Oz", "crest_factor_PS_Delta_EEG Pz-Oz", "line_length_PS_Delta_EEG Pz-Oz", "nonlinear_energy_PS_Delta_EEG Pz-Oz", 
    "Hurst_exponent_PS_Delta_EEG Pz-Oz", "Hjorth_activity_PS_Delta_EEG Pz-Oz", "Hjorth_mobility_PS_Delta_EEG Pz-Oz", "Hjorth_complexity_PS_Delta_EEG Pz-Oz", 
    "Shannon_entropy_PS_Delta_EEG Pz-Oz", "Renyi_entropy_PS_Delta_EEG Pz-Oz", "approx_entropy_PS_Delta_EEG Pz-Oz", "mean_PS_Theta_EEG Pz-Oz", 
    "std_PS_Theta_EEG Pz-Oz", "peak_PS_Theta_EEG Pz-Oz", "min_PS_Theta_EEG Pz-Oz", "peak_frequency_PS_Theta_EEG Pz-Oz", "Q1_PS_Theta_EEG Pz-Oz", 
    "Q2_PS_Theta_EEG Pz-Oz", "Q3_PS_Theta_EEG Pz-Oz", "Q_range_PS_Theta_EEG Pz-Oz", "amplitude_PS_Theta_EEG Pz-Oz", "spec_centroid_PS_Theta_EEG Pz-Oz", 
    "kurtosis_PS_Theta_EEG Pz-Oz", "skewness_PS_Theta_EEG Pz-Oz", "coef_var_PS_Theta_EEG Pz-Oz", "flatness_PS_Theta_EEG Pz-Oz", "1st_moment_PS_Theta_EEG Pz-Oz", 
    "2nd_moment_PS_Theta_EEG Pz-Oz", "rms_PS_Theta_EEG Pz-Oz", "crest_factor_PS_Theta_EEG Pz-Oz", "line_length_PS_Theta_EEG Pz-Oz", 
    "nonlinear_energy_PS_Theta_EEG Pz-Oz", "Hurst_exponent_PS_Theta_EEG Pz-Oz", "Hjorth_activity_PS_Theta_EEG Pz-Oz", "Hjorth_mobility_PS_Theta_EEG Pz-Oz", 
    "Hjorth_complexity_PS_Theta_EEG Pz-Oz", "Shannon_entropy_PS_Theta_EEG Pz-Oz", "Renyi_entropy_PS_Theta_EEG Pz-Oz", "approx_entropy_PS_Theta_EEG Pz-Oz", 
    "mean_PS_Alpha_EEG Pz-Oz", "std_PS_Alpha_EEG Pz-Oz", "peak_PS_Alpha_EEG Pz-Oz", "min_PS_Alpha_EEG Pz-Oz", "peak_frequency_PS_Alpha_EEG Pz-Oz", 
    "Q1_PS_Alpha_EEG Pz-Oz", "Q2_PS_Alpha_EEG Pz-Oz", "Q3_PS_Alpha_EEG Pz-Oz", "Q_range_PS_Alpha_EEG Pz-Oz", "amplitude_PS_Alpha_EEG Pz-Oz", 
    "spec_centroid_PS_Alpha_EEG Pz-Oz", "kurtosis_PS_Alpha_EEG Pz-Oz", "skewness_PS_Alpha_EEG Pz-Oz", "coef_var_PS_Alpha_EEG Pz-Oz", 
    "flatness_PS_Alpha_EEG Pz-Oz", "1st_moment_PS_Alpha_EEG Pz-Oz", "2nd_moment_PS_Alpha_EEG Pz-Oz", "rms_PS_Alpha_EEG Pz-Oz", "crest_factor_PS_Alpha_EEG Pz-Oz", 
    "line_length_PS_Alpha_EEG Pz-Oz", "nonlinear_energy_PS_Alpha_EEG Pz-Oz", "Hurst_exponent_PS_Alpha_EEG Pz-Oz", "Hjorth_activity_PS_Alpha_EEG Pz-Oz", 
    "Hjorth_mobility_PS_Alpha_EEG Pz-Oz", "Hjorth_complexity_PS_Alpha_EEG Pz-Oz", "Shannon_entropy_PS_Alpha_EEG Pz-Oz", "Renyi_entropy_PS_Alpha_EEG Pz-Oz", 
    "approx_entropy_PS_Alpha_EEG Pz-Oz", "mean_PS_Beta_EEG Pz-Oz", "std_PS_Beta_EEG Pz-Oz", "peak_PS_Beta_EEG Pz-Oz", "min_PS_Beta_EEG Pz-Oz", 
    "peak_frequency_PS_Beta_EEG Pz-Oz", "Q1_PS_Beta_EEG Pz-Oz", "Q2_PS_Beta_EEG Pz-Oz", "Q3_PS_Beta_EEG Pz-Oz", "Q_range_PS_Beta_EEG Pz-Oz", 
    "amplitude_PS_Beta_EEG Pz-Oz", "spec_centroid_PS_Beta_EEG Pz-Oz", "kurtosis_PS_Beta_EEG Pz-Oz", "skewness_PS_Beta_EEG Pz-Oz", "coef_var_PS_Beta_EEG Pz-Oz", 
    "flatness_PS_Beta_EEG Pz-Oz", "1st_moment_PS_Beta_EEG Pz-Oz", "2nd_moment_PS_Beta_EEG Pz-Oz", "rms_PS_Beta_EEG Pz-Oz", "crest_factor_PS_Beta_EEG Pz-Oz", 
    "line_length_PS_Beta_EEG Pz-Oz", "nonlinear_energy_PS_Beta_EEG Pz-Oz", "Hurst_exponent_PS_Beta_EEG Pz-Oz", "Hjorth_activity_PS_Beta_EEG Pz-Oz", 
    "Hjorth_mobility_PS_Beta_EEG Pz-Oz", "Hjorth_complexity_PS_Beta_EEG Pz-Oz", "Shannon_entropy_PS_Beta_EEG Pz-Oz", "Renyi_entropy_PS_Beta_EEG Pz-Oz", 
    "approx_entropy_PS_Beta_EEG Pz-Oz", "mean_PS_Gama_EEG Pz-Oz", "std_PS_Gama_EEG Pz-Oz", "peak_PS_Gama_EEG Pz-Oz", "min_PS_Gama_EEG Pz-Oz", 
    "peak_frequency_PS_Gama_EEG Pz-Oz", "Q1_PS_Gama_EEG Pz-Oz", "Q2_PS_Gama_EEG Pz-Oz", "Q3_PS_Gama_EEG Pz-Oz", "Q_range_PS_Gama_EEG Pz-Oz", 
    "amplitude_PS_Gama_EEG Pz-Oz", "spec_centroid_PS_Gama_EEG Pz-Oz", "kurtosis_PS_Gama_EEG Pz-Oz", "skewness_PS_Gama_EEG Pz-Oz", "coef_var_PS_Gama_EEG Pz-Oz", 
    "flatness_PS_Gama_EEG Pz-Oz", "1st_moment_PS_Gama_EEG Pz-Oz", "2nd_moment_PS_Gama_EEG Pz-Oz", "rms_PS_Gama_EEG Pz-Oz", "crest_factor_PS_Gama_EEG Pz-Oz", 
    "line_length_PS_Gama_EEG Pz-Oz", "nonlinear_energy_PS_Gama_EEG Pz-Oz", "Hurst_exponent_PS_Gama_EEG Pz-Oz", "Hjorth_activity_PS_Gama_EEG Pz-Oz", 
    "Hjorth_mobility_PS_Gama_EEG Pz-Oz", "Hjorth_complexity_PS_Gama_EEG Pz-Oz", "Shannon_entropy_PS_Gama_EEG Pz-Oz", "Renyi_entropy_PS_Gama_EEG Pz-Oz", 
    "approx_entropy_PS_Gama_EEG Pz-Oz", 
    #"sample_entropy_PS_Gama_EEG Pz-Oz",
    "mean_PS_Entire_EEG Pz-Oz", "std_PS_Entire_EEG Pz-Oz", "peak_PS_Entire_EEG Pz-Oz", 
    "min_PS_Entire_EEG Pz-Oz", "peak_frequency_PS_Entire_EEG Pz-Oz", "Q1_PS_Entire_EEG Pz-Oz", "Q2_PS_Entire_EEG Pz-Oz", "Q3_PS_Entire_EEG Pz-Oz", 
    "Q_range_PS_Entire_EEG Pz-Oz", "amplitude_PS_Entire_EEG Pz-Oz", "spec_centroid_PS_Entire_EEG Pz-Oz", "kurtosis_PS_Entire_EEG Pz-Oz", 
    "skewness_PS_Entire_EEG Pz-Oz", "coef_var_PS_Entire_EEG Pz-Oz", "flatness_PS_Entire_EEG Pz-Oz", "1st_moment_PS_Entire_EEG Pz-Oz", 
    "2nd_moment_PS_Entire_EEG Pz-Oz", "rms_PS_Entire_EEG Pz-Oz", "crest_factor_PS_Entire_EEG Pz-Oz", "line_length_PS_Entire_EEG Pz-Oz", 
    "nonlinear_energy_PS_Entire_EEG Pz-Oz", "Hurst_exponent_PS_Entire_EEG Pz-Oz", "Hjorth_activity_PS_Entire_EEG Pz-Oz", "Hjorth_mobility_PS_Entire_EEG Pz-Oz", 
    "Hjorth_complexity_PS_Entire_EEG Pz-Oz", "Shannon_entropy_PS_Entire_EEG Pz-Oz", "Renyi_entropy_PS_Entire_EEG Pz-Oz", "approx_entropy_PS_Entire_EEG Pz-Oz", 
    #"sample_entropy_PS_Entire_EEG Pz-Oz", 
    "mean_TS_Entire_EEG Pz-Oz", "std_TS_Entire_EEG Pz-Oz", "peak_TS_Entire_EEG Pz-Oz", "min_TS_Entire_EEG Pz-Oz", 
    "peak_frequency_TS_Entire_EEG Pz-Oz", "Q1_TS_Entire_EEG Pz-Oz", "Q2_TS_Entire_EEG Pz-Oz", "Q3_TS_Entire_EEG Pz-Oz", "Q_range_TS_Entire_EEG Pz-Oz", 
    "amplitude_TS_Entire_EEG Pz-Oz", "spec_centroid_TS_Entire_EEG Pz-Oz", "kurtosis_TS_Entire_EEG Pz-Oz", "skewness_TS_Entire_EEG Pz-Oz", 
    "coef_var_TS_Entire_EEG Pz-Oz", "flatness_TS_Entire_EEG Pz-Oz", "1st_moment_TS_Entire_EEG Pz-Oz", "2nd_moment_TS_Entire_EEG Pz-Oz", "rms_TS_Entire_EEG Pz-Oz", 
    "crest_factor_TS_Entire_EEG Pz-Oz", "line_length_TS_Entire_EEG Pz-Oz", "nonlinear_energy_TS_Entire_EEG Pz-Oz", "Hurst_exponent_TS_Entire_EEG Pz-Oz", 
    "Hjorth_activity_TS_Entire_EEG Pz-Oz", "Hjorth_mobility_TS_Entire_EEG Pz-Oz", "Hjorth_complexity_TS_Entire_EEG Pz-Oz", "Shannon_entropy_TS_Entire_EEG Pz-Oz", 
    "Renyi_entropy_TS_Entire_EEG Pz-Oz", "approx_entropy_TS_Entire_EEG Pz-Oz", "sample_entropy_TS_Entire_EEG Pz-Oz"
]

class BinaryClassification:
    def __init__(self, train_features_path:str, test_features_path:str, class_column:str, fz_cz:bool=True, pz_oz:bool=True) -> None:
        self.test_data = pd.read_csv(test_features_path)

        self.test_awake = preparing_for_binary_classification(self.test_data, class_column, 0)
        self.test_awake.dropna(inplace=True)
        self.test_awake = self.test_awake.sample(frac=1).reset_index(drop=True) #shuffle

        self.test_stage1 = preparing_for_binary_classification(self.test_data, class_column, 1)
        self.test_stage1.dropna(inplace=True)
        self.test_stage1 = self.test_stage1.sample(frac=1).reset_index(drop=True) #shuffle

        self.test_stage2 = preparing_for_binary_classification(self.test_data, class_column, 2)
        self.test_stage2.dropna(inplace=True)
        self.test_stage2 = self.test_stage2.sample(frac=1).reset_index(drop=True) #shuffle

        self.test_stage3 = preparing_for_binary_classification(self.test_data, class_column, 3)
        self.test_stage3.dropna(inplace=True)
        self.test_stage3 = self.test_stage3.sample(frac=1).reset_index(drop=True) #shuffle

        self.test_stage4 = preparing_for_binary_classification(self.test_data, class_column, 4)
        self.test_stage4.dropna(inplace=True)
        self.test_stage4 = self.test_stage4.sample(frac=1).reset_index(drop=True) #shuffle

        self.test_rem = preparing_for_binary_classification(self.test_data, class_column, 5)
        self.test_rem.dropna(inplace=True)
        self.test_rem = self.test_rem.sample(frac=1).reset_index(drop=True) #shuffle

        self.train_data = pd.read_csv(train_features_path)

        self.train_awake = preparing_for_binary_classification(self.train_data, class_column, 0)
        self.train_awake.dropna(inplace=True)
        self.train_awake = self.train_awake.sample(frac=1).reset_index(drop=True) #shuffle
        
        self.train_stage1 = preparing_for_binary_classification(self.train_data, class_column, 1)
        self.train_stage1.dropna(inplace=True)
        self.train_stage1 = self.train_stage1.sample(frac=1).reset_index(drop=True) #shuffle

        self.train_stage2 = preparing_for_binary_classification(self.train_data, class_column, 2)
        self.train_stage2.dropna(inplace=True)
        self.train_stage2 = self.train_stage2.sample(frac=1).reset_index(drop=True) #shuffle

        self.train_stage3 = preparing_for_binary_classification(self.train_data, class_column, 3)
        self.train_stage3.dropna(inplace=True)
        self.train_stage3 = self.train_stage3.sample(frac=1).reset_index(drop=True) #shuffle

        self.train_stage4 = preparing_for_binary_classification(self.train_data, class_column, 4)
        self.train_stage4.dropna(inplace=True)
        self.train_stage4 = self.train_stage4.sample(frac=1).reset_index(drop=True) #shuffle

        self.train_rem = preparing_for_binary_classification(self.train_data, class_column, 5)
        self.train_rem.dropna(inplace=True)
        self.train_rem = self.train_rem.sample(frac=1).reset_index(drop=True) #shuffle

        self.features = []
        
        if(fz_cz):
            self.features += features_Fz_Cz
        if(pz_oz):
            self.features += features_Pz_Oz

    def __balance(self, df:pd.DataFrame, class_column:str):
        X, y = balanceamento_por_classe_undersampling(df, self.features, class_column, False)
        return X, y
    
    def __decision_tree_classifier(self, stage:str, njobs:int=4):#implementado (220 iterações)
        cont = 0
        
        # https://towardsdatascience.com/how-to-tune-a-decision-tree-f03721801680
        #criterion = ["gini", "entropy"]
        criterion = ["gini"]
        min_samples_split = [2, 6, 10, 14, 18, 22, 26, 32, 36, 40]
        min_samples_leaf = [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]

        dt_results = []

        dataframes_train = {
            "awake": self.train_awake, 
            "stage1": self.train_stage1, 
            "stage2": self.train_stage2, 
            "stage3": self.train_stage3, 
            "stage4": self.train_stage4, 
            "stage5": self.train_rem
        }

        dataframes_test = {
            "awake": self.test_awake, 
            "stage1": self.test_stage1, 
            "stage2": self.test_stage2, 
            "stage3": self.test_stage3, 
            "stage4": self.test_stage4, 
            "stage5": self.test_rem
        }

        X_train, y_train = self.__balance(dataframes_train[stage], "stage")
        X_test, y_test = self.__balance(dataframes_test[stage], "stage")

        for crit in criterion:
            for mss in min_samples_split:
                for msl in min_samples_leaf:
                    
                    print(f"Iteração: {cont}")
                    cont += 1
                    
                    try:
                        inicio = time.time()

                        classifier = DecisionTreeClassifier(
                            criterion = crit,
                            min_samples_split = mss,
                            min_samples_leaf = msl 
                        )
                        classifier.fit(X_train, y_train)

                        scores = cross_val_score(classifier, X_test, y_test, cv = 10, n_jobs = njobs)
                        fim = time.time()

                        result = {
                            "stage": stage,
                            "criterion": crit,
                            "min_samples_split": mss,
                            "min_samples_leaf": msl,
                            "accuracy(%)": (scores.mean() * 100),
                            "score_standard_deviation(%)": (scores.std() * 100),
                            "tempo(s)": fim - inicio
                        }

                        print(result)

                        dt_results.append(result)
                    except Exception as e:
                        print(e)

        try:
            dt_tests = pd.DataFrame(dt_results)
            dt_tests.sort_values(by="accuracy(%)", ascending=False, inplace=True)
            dt_tests.to_csv(f"./Scores/decison_tree_{stage}_scores.csv", index=False)
        except Exception as e:
            print(e)

    def __naive_bayes_classifier(self, stage:str, njobs:int=4):#implementado (2 iterações)
        nb_results = []
        
        scaler = MinMaxScaler()

        dataframes_train = {
            "awake": self.train_awake, 
            "stage1": self.train_stage1, 
            "stage2": self.train_stage2, 
            "stage3": self.train_stage3, 
            "stage4": self.train_stage4, 
            "stage5": self.train_rem
        }

        dataframes_test = {
            "awake": self.test_awake, 
            "stage1": self.test_stage1, 
            "stage2": self.test_stage2, 
            "stage3": self.test_stage3, 
            "stage4": self.test_stage4, 
            "stage5": self.test_rem
        }
        
        X_train, y_train = self.__balance(dataframes_train[stage], "stage")
        X_test, y_test = self.__balance(dataframes_test[stage], "stage")

        #ComplementNB
        try:
            for fit_prior in [False, True]:
                inicio = time.time()    

                X_train = scaler.fit_transform(X_train)
                X_test = scaler.fit_transform(X_test)

                classifier = ComplementNB(fit_prior = fit_prior)
                classifier.fit(X_train, y_train)
                scores = cross_val_score(classifier, X_test, y_test, cv = 10, n_jobs = njobs)
                fim = time.time()
                result = {
                    "stage": stage,
                    "type": "ComplementNB",
                    "fit_prior": fit_prior,
                    "accuracy(%)": (scores.mean() * 100),
                    "score_standard_deviation(%)": (scores.std() * 100),
                    "tempo(s)": fim - inicio
                }
                print(result)
                nb_results.append(result)
        except Exception as e:
            print(e)

        try:
            nb_tests = pd.DataFrame(nb_results)
            nb_tests.sort_values(by="accuracy(%)", ascending=False, inplace=True)
            nb_tests.to_csv(f"./Scores/naive_bayes_{stage}_scores.csv", index=False)
        except Exception as e:
            print(e)

    def __random_forest_classifier(self, stage:str, njobs:int=4):#implementado (360 iterações)
        cont = 0
        
        criterion_ = ["gini", "entropy"]
        max_features_ = ['auto', 'sqrt']
        bootstrap_ = [True, False]
        min_samples_split_ = [2, 5, 10]
        min_samples_leaf_ = [1, 2, 4]
        max_depth_ =  [10, 30, 50, 70, "None"]

        dataframes_train = {
            "awake": self.train_awake, 
            "stage1": self.train_stage1, 
            "stage2": self.train_stage2, 
            "stage3": self.train_stage3, 
            "stage4": self.train_stage4, 
            "stage5": self.train_rem
        }

        dataframes_test = {
            "awake": self.test_awake, 
            "stage1": self.test_stage1, 
            "stage2": self.test_stage2, 
            "stage3": self.test_stage3, 
            "stage4": self.test_stage4, 
            "stage5": self.test_rem
        }

        rf_results = []

        X_train, y_train = self.__balance(dataframes_train[stage], "stage")
        X_test, y_test = self.__balance(dataframes_test[stage], "stage")
        
        for criterion in criterion_:
            for bootstrap in bootstrap_:
                for max_features in max_features_:
                    for min_samples_leaf in min_samples_leaf_:
                        for min_samples_split in min_samples_split_:
                            for max_depth in max_depth_:
                                print(f"Iteração: {cont}")
                                cont += 1

                                if max_depth == "None":
                                    md = None
                                else:
                                    md = max_depth

                                try:
                                    inicio = time.time()
                                    
                                    classifier = RandomForestClassifier(
                                        n_estimators = 100, #usando constante (quanto maior maior a acuracia e o tempo)
                                        criterion = criterion,
                                        max_depth = md,
                                        min_samples_split = min_samples_split,
                                        min_samples_leaf = min_samples_leaf,
                                        bootstrap = bootstrap,
                                        max_features = max_features,
                                        n_jobs = njobs
                                    )
                                    classifier.fit(X_train, y_train)

                                    scores = cross_val_score(classifier, X_test, y_test, cv = 10, n_jobs = njobs)

                                    fim = time.time()

                                    result = {
                                        "stage": stage,
                                        "criterion": criterion,
                                        "bootstrap": bootstrap,
                                        "max_features": max_features,
                                        "min_samples_leaf": min_samples_leaf,
                                        "min_samples_split": min_samples_split,
                                        "n_estimators": 100,
                                        "max_depth": max_depth,
                                        "accuracy(%)": (scores.mean() * 100),
                                        "score_standard_deviation(%)": (scores.std() * 100),
                                        "tempo(s)": fim - inicio
                                    }
                                    print(result)
                                    rf_results.append(result)
                                except Exception as e:
                                    print(e)
        
        try:
            rf_tests = pd.DataFrame(rf_results)
            rf_tests.sort_values(by="accuracy(%)", ascending=False, inplace=True)
            rf_tests.to_csv(f"./Scores/random_forest_{stage}_scores.csv", index=False)
        except Exception as e:
            print(e)

    def __ridge_classifier(self, stage:str, njobs:int=4): #implementado (80 iterações)
        #alpha_ = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        #solver_ = ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga", "lbfgs"]

        alpha_ = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        solver_ = ["auto", "svd"]

        cont = 0

        dataframes_train = {
            "awake": self.train_awake, 
            "stage1": self.train_stage1, 
            "stage2": self.train_stage2, 
            "stage3": self.train_stage3, 
            "stage4": self.train_stage4, 
            "stage5": self.train_rem
        }

        dataframes_test = {
            "awake": self.test_awake, 
            "stage1": self.test_stage1, 
            "stage2": self.test_stage2, 
            "stage3": self.test_stage3, 
            "stage4": self.test_stage4, 
            "stage5": self.test_rem
        }

        rc_results = []

        X_train, y_train = self.__balance(dataframes_train[stage], "stage")
        X_test, y_test = self.__balance(dataframes_test[stage], "stage")

        for alpha in alpha_:
            for solver in solver_:
                print(f"Iteração: {cont}")
                cont += 1

                try:
                    inicio = time.time()
                    
                    classifier = RidgeClassifier(
                        alpha=alpha,
                        solver=solver,
                        #normalize=True # --> depreciado
                    )
                    classifier.fit(X_train, y_train)
                    scores = cross_val_score(classifier, X_test, y_test, cv = 10, n_jobs = njobs)

                    fim = time.time()

                    result = {
                        "stage": stage,
                        "alpha": alpha,
                        "solver": solver,        
                        "accuracy(%)": (scores.mean() * 100),
                        "score_standard_deviation(%)": (scores.std() * 100),
                        "tempo(s)": fim - inicio
                    }
                    print(result)
                    rc_results.append(result)
                except Exception as e:
                    print(e)

        try:
            rc_tests = pd.DataFrame(rc_results)
            rc_tests.sort_values(by="accuracy(%)", ascending=False, inplace=True)
            rc_tests.to_csv(f"./Scores/ridge_{stage}_scores.csv", index=False)
        except Exception as e:
            print(e)

    def __bagged_decision_tree_classifier(self, stage:str, njobs:int=4):
        cont = 0
        bdt_results = []

        dataframes_train = {
            "awake": self.train_awake, 
            "stage1": self.train_stage1, 
            "stage2": self.train_stage2, 
            "stage3": self.train_stage3, 
            "stage4": self.train_stage4, 
            "stage5": self.train_rem
        }

        dataframes_test = {
            "awake": self.test_awake, 
            "stage1": self.test_stage1, 
            "stage2": self.test_stage2, 
            "stage3": self.test_stage3, 
            "stage4": self.test_stage4, 
            "stage5": self.test_rem
        }

        X_train, y_train = self.__balance(dataframes_train[stage], "stage")
        X_test, y_test = self.__balance(dataframes_test[stage], "stage")
        
        pass

    def __gradient_boosting_classifier(self, stage:str, njobs:int=4):
        pass
    
    def __logistic_regression_classifier(self, stage:str, njobs:int=4):
        pass

    def __knn_classifier(self, stage:str, njobs:int=4):
        pass

    def __svm_classifier(self, stage:str, njobs:int=4):
        pass

    def __deep_learning_classification(self, stage:str, njobs:int=4): #não sei como implementar
        pass

    def test_parameters(self):
        stages = [
            "awake", 
            "stage1", 
            "stage2", 
            "stage3", 
            "stage4", 
            "stage5"
        ]

        #print("--------- Naive Bayes ---------")
        #for stage in stages:
        #    print(f"Stage: {stage}")
        #    self.__naive_bayes_classifier(stage)

        #print("--------- Ridge ---------")
        #for stage in stages:
        #    print(f"Stage: {stage}")
        #    self.__ridge_classifier(stage)

        print("--------- Decision Tree ---------")
        for stage in stages:
            print(f"Stage: {stage}")
            self.__decision_tree_classifier(stage)

        print("--------- Random Forest ---------")
        for stage in stages:
            print(f"Stage: {stage}")
            self.__random_forest_classifier(stage)
    



if __name__ == "__main__":
    teste = BinaryClassification(
        "./Features/0N1_100_100_noNaN_3000.csv", #train
        "./Features/0N2_100_100_noNaN_3000.csv", #test
        "stage"
    )
    
    teste.test_parameters()