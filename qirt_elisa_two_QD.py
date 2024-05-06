#Setup required packages------------
import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
#Imports------------
from streamlit_option_menu import option_menu
from pandas import Series, DataFrame
from scipy.signal import find_peaks as fp
from scipy.signal import savgol_filter
from scipy.signal import peak_widths
from peakutils import indexes
from peakutils import baseline
from scipy import integrate
#------------
#Title of the app
st.set_page_config(page_icon=None, layout="wide", initial_sidebar_state='auto')
st.title("Two Biomarker QIRT-ELISA Peak Data Analysis")
#------------
#Setup tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Data Loading", "Signal Smoothing", "Baseline Correction", "Peak Analysis", "Results"])
#------------
#Load data
with tab1:
    st.header('Load Your Experimental Data')
    def load_data(file):
     data = pd.read_csv(file, usecols=['#Digilent WaveForms Oscilloscope Acquisition','Unnamed: 1','Unnamed: 2']).rename(columns = {'Unnamed: 1': 'Signal605', 'Unnamed: 2': 'Signal655', '#Digilent WaveForms Oscilloscope Acquisition': 'time'})
     return data
    uploaded_file = st.file_uploader ("Upload your CSV data file", type={"csv", "txt"})
    if uploaded_file is None:
        st.info(" Please upload a file to continue.", icon="ℹ️")
        st.stop()
    df = load_data(uploaded_file)
 #------------
 #data clean-up to remove the headers from the csv files
    df.drop(df.index[pd.Series(range(0,13))],inplace=True)
    df=df.astype(float)
    with st.expander("Data Preview"):
            st.dataframe(df)
 #------------
    col1, col2 = st.columns(2)
    with col1:
        multiplier1 = st.number_input('Enter the sensitivity of QDot605 on the preamplifier in pA:', min_value=1)
        df['Signal605'] = df['Signal605'] * multiplier1 
        #if item == 'QDot655':
    with col2:
        multiplier2 = st.number_input('Enter the sensitivity of QDot655 on the preamplifier in pA:', min_value=1)
        df['Signal655'] = df['Signal655'] * multiplier2
    if multiplier1 is 1:
        st.info(" Please enter the sensitivities values", icon="ℹ️")
        st.stop()
    if multiplier2 is 1:
        st.info(" Please enter the sensitivities values", icon="ℹ️")
        st.stop() 
    st.sidebar.info('Move to the *Signal Smoothing* tab once you have entered your preamplifier sensitivities.', icon="1️⃣")
    with st.expander("Data Preview"):
            st.dataframe(df)
with tab2:
    st.header('Signal Smoothing')
    st.write('By default, there is no signal smoothing. Please check the boxes below to apply a Savitzky-Golay smoothing filter if necessary.',icon="1️⃣")    
    col1, col2 = st.columns(2) 
    with col1:
        st.header("QDot 605")
        filter_window1 = 1
        order_value1 = 0
        Signal605_smoothed = savgol_filter(df.Signal605, filter_window1, order_value1, deriv=0, delta=1.0, axis=-1, mode='interp', cval=0.0)      
        fig_smooth_605 = plt.figure(figsize=(9, 7))
        sns.lineplot(y =Signal605_smoothed, x = df.time, label = "QDot 605 Signal")
        plt.xlabel("Time [sec]")
        plt.ylabel("Fluorscent Signal [pA]")
        plt.title("Signals")
        st.pyplot(fig_smooth_605)
        change_smooth_605 = st.checkbox("Apply a signal smoothing filter for QDot 605")
        if change_smooth_605:
            filter_window1 = st.number_input("Length of the filter window", min_value=1  , max_value=None, value=None, placeholder="Type a number...", key='fw1', help = 'Please enter a value less the length of your data frame. For no signal smoothing, enter 1')
            if filter_window1 is None:
                st.info(" Please choose a filter window", icon="ℹ️")
                st.stop() 
            order_value1 = st.number_input("Order of the polynomial", min_value=0, max_value=filter_window1, value=None, placeholder="Type a number...", key='ov1', help = 'Please enter a value less the filter window. For no signal smoothing, enter 0')
            if order_value1 is None:
                st.info(" Please choose the order of the filter", icon="ℹ️")
                st.stop() 
            Signal605_smoothed = savgol_filter(df.Signal605, filter_window1, order_value1, deriv=0, delta=1.0, axis=-1, mode='interp', cval=0.0)
            fig_smooth_605 = plt.figure(figsize=(9, 7))
            sns.lineplot(y =df.Signal605, x = df.time, label = "605 Raw Signal") 
            sns.lineplot(y =Signal605_smoothed, x = df.time, label = "QDot 605 Signal")
            plt.xlabel("Time [sec]")
            plt.ylabel("Fluorscent Signal [pA]")
            plt.title("Signals")
            st.pyplot(fig_smooth_605)
    with col2:
        st.header("QDot 655")
        filter_window2 = 1
        order_value2 = 0
        Signal655_smoothed = savgol_filter(df.Signal655, filter_window2, order_value2, deriv=0, delta=1.0, axis=-1, mode='interp', cval=0.0)
        fig_smooth_655 = plt.figure(figsize=(9, 7))   
        sns.lineplot(y =Signal655_smoothed, x = df.time, label = "QDot 655 Signal")
        plt.title("Signals")
        plt.xlabel("Time [sec]")
        plt.ylabel("Fluorscent Signal [pA]")
        st.pyplot(fig_smooth_655)
        st.sidebar.info('Move to the *Baseline correction* tab if you are satisfied with your filter selection.', icon="2️⃣")
        change_smooth_655 = st.checkbox("Apply a signal smoothing filter for QDot 655")
        if change_smooth_655:
            filter_window2 = st.number_input("Length of the filter window", min_value=1  , max_value=None, value=None, placeholder="Type a number...", key='fw2', help = 'Please enter a value less the length of your data frame. For no signal smoothing, enter 1')
            if filter_window2 is None:
                st.info(" Please choose a filter window", icon="ℹ️")
                st.stop() 
            order_value2 = st.number_input("Order of the polynomial", min_value=0, max_value=filter_window2, value=None, placeholder="Type a number...", key='ov2', help = 'Please enter a value less the filter window. For no signal smoothing, enter 0')
            if order_value2 is None:
                st.info(" Please choose the order of the filter", icon="ℹ️")
                st.stop()  
            Signal655_smoothed = savgol_filter(df.Signal655, filter_window2, order_value2, deriv=0, delta=1.0, axis=-1, mode='interp', cval=0.0)
            fig_smooth_655 = plt.figure(figsize=(9, 7))   
            sns.lineplot(y =df.Signal655, x = df.time, label = "655 Raw Signal")
            sns.lineplot(y =Signal655_smoothed, x = df.time, label = "QDot 655 Signal")
            plt.title("Signals")
            plt.xlabel("Time [sec]")
            plt.ylabel("Fluorscent Signal [pA]")
            st.pyplot(fig_smooth_655) 
with tab3:
    st.header('Baseline Correction')
    st.write('Remove the background noise.')
    col1, col2 = st.columns(2)
    with col1:
     base_deg1 = 100
     base_line1 = baseline(Signal605_smoothed, deg = base_deg1) 
     Signal605_smoothed_adjusted = Signal605_smoothed - base_line1 
     fig_baseline_605 = plt.figure(figsize=(9, 7))
     sns.lineplot(y = base_line1, x = df.time, label = "Generated Baseline for QDot 605")
     sns.lineplot(y = Signal605_smoothed, x = df.time, label = "Adjusted QDot 605 Signal")
     plt.title( "Signals")
     plt.xlabel("Time [sec]")
     plt.ylabel("Fluorscent Signal [pA]")
     st.pyplot(fig_baseline_605)
    with col2:
     base_deg2 = 100
     base_line2 = baseline(Signal655_smoothed, deg = base_deg2) 
     Signal655_smoothed_adjusted = Signal655_smoothed - base_line2 

     fig_baseline_655 = plt.figure(figsize=(9, 7))
     sns.lineplot(y = base_line2, x = df.time, label = "Generated Baseline for QDot 655")
     sns.lineplot(y = Signal655_smoothed, x = df.time, label = "Adjusted QDot 655 Signal")
     plt.title( "Signals")
     plt.xlabel("Time [sec]")
     plt.ylabel("Fluorscent Signal [pA]")
     st.pyplot(fig_baseline_655)
     st.sidebar.info('Move to the *Peak Analysis* tab after the signals are adjusted.', icon="3️⃣")
with tab4:
    st.header('Peak Analysis')
    df21 = pd.DataFrame({'time': df.time,'Signal':Signal605_smoothed_adjusted}).set_index('time')
    df22 = pd.DataFrame({'time': df.time,'Signal':Signal655_smoothed_adjusted}).set_index('time') 
    st.write('Find the peaks in the adjusted signals and integrate the found peaks.')
    col1, col2 = st.columns(2)
    with col1:
         st.header("QDot 605")
         maxval_605 = df21.Signal.max()
         h1 = 0.1*maxval_605
         prom1 = h1/2
         def set_inputs_zero():
          st.session_state.h2 = 0
          st.session_state.prom2 = 0  
         peaks21, property1 = fp(df21.Signal,
            height = h1,
            prominence = prom1,
            distance = 10, wlen = 350)
         peak_found_605 = property1
         change605 = st.checkbox("Change the pre-set prominence for QDot 605")
         if change605:
            new_prom1 = st.slider(label = "Choose the new prominence", min_value = 1 , max_value = None, value = None, step = 1, key = "changeprom1")
            prom1 = new_prom1
            peaks21, property1 = fp(df21.Signal,
               height = h1,
               prominence = prom1,
               distance = 1, wlen = 350)
            peak_found_605 = property1
         df21.plot()
         sns.scatterplot(data = df21.iloc[peaks21], x = 'time', y = 'Signal', 
                color = 'red', alpha = 0.5)
         st.pyplot(plt.gcf( ))
    with col2:
         st.header("QDot 655")
         maxval_655 = df22.Signal.max()
         h2 = 0.1*maxval_655
         prom2 = h2/2
         peaks22, property2 = fp(df22.Signal,
            height = h2,
            prominence = prom2,
            distance = 10, wlen = 350)
         peak_found_655 = property2
         change655 = st.checkbox("Change the preset prominence for QDot 655")
         if change655:
            new_prom2 = st.slider(label = "Choose the new prominence", min_value = 1 , max_value = None, value = None, step = 1, key = "changeprom2")
            prom2 = new_prom2
            peaks22, property2 = fp(df22.Signal,
               height = h2,
               prominence = prom2,
               distance = 10, wlen = 350)
            peak_found_655 = property2
         df22.plot()
         sns.scatterplot(data = df22.iloc[peaks22], x = 'time', y = 'Signal', 
                color = 'red', alpha = 0.5)
         st.pyplot(plt.gcf( ))
         st.sidebar.info('Move to the *Results* tab if you are satisfied with the identified peaks.', icon="4️⃣")
with tab5:
    st.header('Results')
    col1, col2 = st.columns(2)
    with col1:
        df21_2 = pd.DataFrame({'time': df.time,'Signal':Signal605_smoothed_adjusted}).set_index('time')
        df21_3 = pd.DataFrame({'time': df.time,'Signal':Signal605_smoothed_adjusted})
        time_array_605 = df21_3[["time"]].to_numpy( )
        peak_number_605 = len(peak_found_605['peak_heights'])
        n_605 = 0
        area_idv_605 = 0
        area_sum_605 = 0
        area_peaks_605 = 0 
        array_605_peaks = np.array([])
        for n_605 in range(peak_number_605):
         left_base_index_605 = peak_found_605['left_bases'][n_605]
         right_base_index_605 = peak_found_605['right_bases'][n_605]
         left_base_605 = time_array_605[left_base_index_605][0]
         right_base_605 = time_array_605[right_base_index_605][0]
         df5 = df21_2.Signal[left_base_605:right_base_605]
         timestamp2_605 = df5.index
         area_idv_605 = np.trapz(df21_2.Signal[left_base_605:right_base_605], x = timestamp2_605)
         array_605_peaks = np.append(array_605_peaks, area_idv_605)
         area_sum_605 = area_sum_605 + area_idv_605 
         n_605 = n_605 + 1
         area_peaks_605 = area_sum_605/peak_number_605
        y1 = [1.14493664, -0.5470587, -0.597878, 10.2377709, 11.0072546, 7.76470096, 22.4991946, 24.8191155, 24.1932251, 40.8335942, 41.9158234, 38.4410576, 53.0854621, 61.1917111, 70.4062313]
        x1 = [0, 0, 0, 10, 10, 10, 40, 40, 40, 80, 80, 80, 100, 100, 100]
        x_min_1 = min(x1)
        x_max_1 = max(x1)
        step1 = 0.01
        x_new_1 = np.arange(x_min_1, x_max_1, step1)
        B1 = 1.361
        m = 0.5594
        def log4pl_conc_gluc(y_exp_gluc):
            return (y_exp_gluc-B1)/m 
        yfit1_new = (m*x_new_1)+B1
        f_605 = area_peaks_605
        f_0_gluc = 0.812388566
        normalized_f_605 = (f_605 - f_0_gluc)/f_0_gluc
        y_exp_gluc = normalized_f_605
        gluc_concent = log4pl_conc_gluc(y_exp_gluc)
        fig_calib_glucagon = plt.figure(figsize=(9, 7))
        plt.plot(x1, y1, 'r+', label="Experimental replicates")
        plt.plot(x_new_1, yfit1_new, label="Linear fit")
        plt.plot(gluc_concent,y_exp_gluc,'bd')
        plt.xlabel('Glucagon Concentration [pM]')
        plt.ylabel('Fluorscent Signal (peak AUC)')
        plt.legend(loc='best', fancybox=True, shadow=True)
        st.pyplot(fig_calib_glucagon)
        st.write('The average peak AUC is:', area_peaks_605)
        st.write('The normalized average peak AUC is:', normalized_f_605)
        st.write('The corresponding glucagon concentrations in [pM] is:', gluc_concent)
    with col2:    
        df22_2 = pd.DataFrame({'time': df.time,'Signal':Signal655_smoothed_adjusted}).set_index('time')
        df22_3 = pd.DataFrame({'time': df.time,'Signal':Signal655_smoothed_adjusted})
        time_array_655 = df22_3[["time"]].to_numpy( )
        peak_number_655 = len(peak_found_655['peak_heights'])
        n_655 = 0
        area_idv_655 = 0
        area_sum_655 = 0
        area_peaks_655 = 0
        array_655_peaks = np.array([])
        for n_655 in range(peak_number_655):
           left_base_index_655 = peak_found_655['left_bases'][n_655]
           right_base_index_655 = peak_found_655['right_bases'][n_655]
           left_base_655 = time_array_655[left_base_index_655][0]
           right_base_655 = time_array_655[right_base_index_655][0]
           df6 = df22_2.Signal[left_base_655:right_base_655]
           timestamp2_655 = df6.index
           area_idv_655 = np.trapz(df22_2.Signal[left_base_655:right_base_655], x = timestamp2_655)
           array_655_peaks = np.append(array_655_peaks, area_idv_655)
           area_sum_655 = area_sum_655 + area_idv_655 
           n_655 = n_655 + 1
           area_peaks_655 = area_sum_655/peak_number_655
        y2 = [0.58123804, -0.281096, -0.3001421, 8.06791665,	9.20469324,	7.49843809, 17.1460606, 14.6611185, 15.8027406, 31.3812569, 38.9815661, 35.3228082, 57.5532343, 63.0998399, 62.60464]
        x2 = [0, 0, 0, 100, 100, 100, 200, 200, 200, 400, 400, 400, 1000, 1000, 1000]
        x_min_2 = min(x2)
        x_max_2 = max(x2)
        step2 = 0.01
        x_new_2 = np.arange(x_min_2, x_max_2, step2)
        A2 = 0.3460
        B2 = 1.495
        C2 = 500.5
        D2 = 82.79
        def log4pl_conc_ins(y_exp_insulin):
            return C2*((A2-D2)/(y_exp_insulin-D2)-1)**(1/B2)
        yfit2_new = ((A2-D2)/(1.0+((x_new_2/C2)**B2))) + D2
        f_655 = area_peaks_655
        f_0_insulin = 1.426343498
        normalized_f_655 = (f_655 - f_0_insulin)/f_0_insulin
        y_exp_insulin = normalized_f_655
        insulin_concent = log4pl_conc_ins(y_exp_insulin)
        fig_calib_insulin = plt.figure(figsize=(9, 7))
        plt.plot(x2, y2, 'r+', label="Experimental replicates")
        plt.plot(x_new_2, yfit2_new, label="Sigmoidal, 4PL non-linear fit")
        plt.plot(insulin_concent,y_exp_insulin,'bd')
        plt.xlabel('Insulin Concentration [pM]')
        plt.ylabel('Fluorscent Signal (peak AUC)')
        plt.legend(loc='best', fancybox=True, shadow=True)
        st.pyplot(fig_calib_insulin)
        st.write('The average peak AUC is:', area_peaks_655)
        st.write('The normalized average peak AUC is:', normalized_f_655)
        st.write('The corresponding insulin concentrations in [pM] is:', insulin_concent)