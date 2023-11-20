# bme 599 advanced mri
# codes for hw 1
# author: jiayao


import os
import numpy as np
import matplotlib.pyplot as plt

import Phy


def problem_1():
    gamma = 42.58*1e6  #(Hz/T)

    print('problem 1 ================')
    print('----------question a ')
    slewrate = 180  # (T/m/s)
    G_max = 25e-3  #(T/m)
    dur = 1e-3  #(s)
    tbw = 2  # time-bandwidth product

    bw = tbw/dur  #(Hz)
    print('TBW = {}'.format(tbw))
    print('BW = {} Hz'.format(bw))
    
    slicethickness = 3e-3  #(m)
    G_ss = bw/(gamma*slicethickness)  #Hz/ (Hz/T) / m = (T/m)
    print('Gss = {} mT/m'.format(G_ss*1000))
    if G_ss > G_max:
        G_ss = G_max
    print('Gss = {} mT/m'.format(G_ss*1000))
    print('slice thickness can acieve: {} mm'.format(bw/(G_ss*gamma)*1000))

    t_rise = G_ss / slewrate  # (s) = (T/m) / (T/m/s)
    print('t_rise = {} ms'.format(t_rise*1000))
    t_total_Gslice = 2*t_rise + dur
    print('total gradient time: {} ms'.format(t_total_Gslice*1000))

    # rephasing gradient time:
    area_flat = (t_rise*G_ss + G_ss*dur)*0.5 - G_max*G_max/slewrate
    t_reph_flat = area_flat/G_max
    print('rephasing flat time(ms):',t_reph_flat*1000)
    t_reph_total = G_max/slewrate*2 + t_reph_flat
    print('total rephasing time(ms):',t_reph_total*1000)



    print('-----------question b ')
    FOV_y = 256*1.2e-3  #(m)
    delta_ky = 1/FOV_y # (1/m)
    delta_kymax = delta_ky*256/2
    area_pemax = delta_kymax/gamma  #1/m / Hz*T = T/(Hz*m)
    print('delta ky = {} 1/m'.format(delta_ky))
    print('delta kymax = {} 1/m'.format(delta_kymax),1/(1.2e-3)/2)
    print('Area_pemax = {} T/Hz/m'.format(area_pemax))

    # try no flat gradient
    area_maxnoflat = G_max/slewrate*G_max  #(T/m*s)
    print('no flat gradient -> {} T/m*s'.format(area_maxnoflat))
    t_flat = (area_pemax - area_maxnoflat)/G_max  #(T/m*s) / (T/m) = s 
    print('time flat-top: {} ms'.format(t_flat*1000))
    t_total_phaseencoding = G_max/slewrate*2 + t_flat #(s)
    print('t_total(ms): {}'.format(t_total_phaseencoding*1000))

    
    print('-------------question c')
    # frequency encoding gradient
    FOV_x = 256*1.2e-3  #(m)
    delta_kx = 1/FOV_x
    width_kx = delta_kx*256
    receiver_bw = 750  #(Hz)
    delta_t = 1/(receiver_bw*256)
    adc_time = delta_t*256
    print('delta kx = {} 1/m'.format(delta_kx))
    print('kx-width: {} 1/m'.format(width_kx))
    print('delta time = {} ms'.format(delta_t*1000))
    print('total adc time: {} ms'.format(adc_time*1000), '|', 1/receiver_bw)
    Gf = width_kx/gamma/adc_time  # (1/m) / (Hz/T) / s = T/m
    print('frequency encoding gradient is: {} mT/m'.format(Gf*1000))
    t_rise = Gf/slewrate
    print('gradient rise time is {} ms'.format(t_rise*1000))
    t_total_adc = 2*t_rise + adc_time
    print('total frequency encoding gradient time: {} ms'.format(t_total_adc*1000))

    # prephasing gradient
    t_pre = ((Gf*t_rise + adc_time*Gf)*0.5 - G_max/slewrate*G_max) / (G_max)
    print('prephasing gradient flat time: {} ms'.format(t_pre*1000))
    t_total_preph = t_pre + 2*G_max/slewrate
    print('total time for prephaing gradient(ms):',t_total_preph*1000)


    # print('-------------question d') # diagram
    TR = t_total_Gslice + t_total_adc + t_total_preph*2
    print(TR)
    print(1.174 + 0.554 + 1.486 + 0.554)
    print(TR/2)

    return




def problem_2():
    print('\nProblem 2 =======================')
    print('------------ 2.a')

    def plot_response(f_range,x,TR,TE,picname=None):
        '''x: is the complex response'''
        fig,axs = plt.subplots(nrows=1,ncols=2,figsize=(8,4))

        axs[0].plot(f_range,np.abs(x))
        axs[0].set_title('magnitude')
        axs[0].set_xlabel('frquency (Hz)')
        axs[1].plot(f_range,np.angle(x))
        axs[1].set_title('phase')
        axs[1].set_xlabel('frquency (Hz)')
        fig.suptitle('TR = {} ms, TE = {} ms'.format(TR,TE))
        # plt.show()
        plt.savefig(picname)
        plt.close(fig)

        return
    

    offres_range = np.linspace(-200,200,200)

    # (1) 
    T1 = 1000 #(ms)  
    T2 = 100 #(ms)
    flip = 60/180*np.pi
    TR = 5 #(ms)
    TE = 2.5 #(ms)
    # a test of how many TR neede to get the steady state
    Phy.bSSFP_steadystate_signal_revolution(flip,TR,T1,T2,df=0,nTR=200,
                                            alternating_rf_phase=True,
                                            spoiling_gradient_case='none',
                                            savepicname='hw1/bssfp-steadystate.png')

    M_resp_1 = Phy.bSSFP_frequency_response(flip,TR,TE,T1,T2,offres_range,nTR=500)
    Mxy1 = M_resp_1[:,0] + 1j*M_resp_1[:,1]
    # plt.figure()
    # Mxy = M_resp[:,0] + 1j*M_resp[:,1]
    # plt.plot(offres_range,np.abs(Mxy))
    # plt.show()
    plot_response(offres_range,Mxy1,TR,TE,picname='hw1/2a-1.png')

    # (2)
    T1 = 1000 #(ms)  
    T2 = 100 #(ms)
    flip = 60/180*np.pi
    TR = 10 #(ms)
    TE = 5 #(ms)
    # a test of how many TR neede to get the steady state
    # Phy.bSSFP_steadystate_signal_revolution(flip,TR,T1,T2,df=0,nTR=500,
    #                                         alternating_rf_phase=True,
    #                                         spoiling_gradient_case='none')

    M_resp_2 = Phy.bSSFP_frequency_response(flip,TR,TE,T1,T2,offres_range,nTR=500)
    Mxy2 = M_resp_2[:,0] + 1j*M_resp_2[:,1]
    plot_response(offres_range,Mxy2,TR,TE,picname='hw1/2a-2.png')

    # (3)
    T1 = 1000 #(ms)  
    T2 = 100 #(ms)
    flip = 60/180*np.pi
    TR = 20 #(ms)
    TE = 10 #(ms)
    # a test of how many TR neede to get the steady state
    # Phy.bSSFP_steadystate_signal_revolution(flip,TR,T1,T2,df=0,nTR=500,
    #                                         alternating_rf_phase=True,
    #                                         spoiling_gradient_case='none')

    M_resp_3 = Phy.bSSFP_frequency_response(flip,TR,TE,T1,T2,offres_range,nTR=500)
    Mxy3 = M_resp_3[:,0] + 1j*M_resp_3[:,1]
    plot_response(offres_range,Mxy3,TR,TE,picname='hw1/2a-3.png')



    print('------------- 2.b')
    # FLASH sequence 
    T1 = 1000 #(ms)
    T2 = 100 #(ms)
    TR = 10  #(ms)
    TE = 5 #(ms)
    flip = 10/180*np.pi

    # (i) perfect spoiler gradient, do this mannual
    # check the signal
    # Phy.bSSFP_steadystate_signal_revolution(flip,TR,T1,T2,df=0,nTR=500,
    #                                         spoiling_gradient_case='perfect-gradient-spoiling')
    # Phy.bSSFP_steadystate_signal_revolution(flip,TR,T1,T2,df=0,nTR=100,
    #                                     spoiling_gradient_case='perfect-gradient-spoiling')
    # calculate the steady-state signal
    M = Phy.SSFP_complex_steadystate_mag(flip,TR,TE,T1,T2,df=0,nTR=500,
                                                   alternating_rf_phase=True,
                                                   spoiling_gradient_case='perfect-gradient-spoiling')
    print(M)
    M = Phy.SSFP_complex_steadystate_mag(flip,TR,TE,T1,T2,df=0,nTR=501,
                                                   alternating_rf_phase=True,
                                                   spoiling_gradient_case='perfect-gradient-spoiling')
    print(M)
    M = Phy.SSFP_complex_steadystate_mag(flip,TR,TE,T1,T2,df=0,nTR=502,
                                                   alternating_rf_phase=True,
                                                   spoiling_gradient_case='perfect-gradient-spoiling')
    print(M)

    M_perfect_no_transverse = np.sqrt(M[0]**2+M[1]**2)
    print(M_perfect_no_transverse)


    # (ii) with different gradient spoiler
    print('>> gradient spoiling')
    def bSSFP_gradientspoiling_voxel_signal(flip,TR,TE,T1,T2,df,nTR,alternating_rf,spoiler_phase_range):
        Nsp = len(spoiler_phase_range)
        M_total = 0
        for spoiler_phase in spoiler_phase_range:
            M = Phy.SSFP_complex_steadystate_mag(flip,TR,TE,T1,T2,df,nTR,
                                                           alternating_rf_phase=alternating_rf,
                                                           spoiling_gradient_case='gradient-spoiling',
                                                           spoiling_gradient=spoiler_phase)
            M_total = M_total + M
        M_normalized = M_total/Nsp
        return M_normalized
            
    # check the signal
    # Phy.bSSFP_steadystate_signal_revolution(flip,TR,T1,T2,df=0,nTR=500,
    #                                         spoiling_gradient_case='gradient-spoiling',
    #                                         spoiling_gradient=np.pi)
    
    test_spoiler_phase = np.array([0*np.pi, 1*np.pi, 2*np.pi, 4*np.pi, 8*np.pi, 16*np.pi])

    # calculate the steady-state signal, a test part
    # see how many samples i need
    # for pp in test_spoiler_phase:
    #     print('--- test spoiler: {} pi'.format(pp/np.pi))
    #     for nsamples in [10,50,100,200,300,400]:
    #         print('nsamples:',nsamples)
    #         spoiler_phase_range = np.linspace(0,pp,nsamples)
    #         M = bSSFP_gradientspoiling_voxel_signal(flip,TR,TE,T1,T2,df=0,nTR=500,
    #                                                 spoiler_phase_range=spoiler_phase_range)
    #         print(M)

    Mx = np.zeros_like(test_spoiler_phase)
    My = np.zeros_like(test_spoiler_phase)
    Mz = np.zeros_like(test_spoiler_phase)
    k = 0
    for pp in test_spoiler_phase:
        print('--- test spoiler: {} pi'.format(pp/np.pi))
        nsamples = 500
        spoiler_phase_range = np.linspace(0,pp,nsamples)
        M = bSSFP_gradientspoiling_voxel_signal(flip,TR,TE,T1,T2,df=0,nTR=500,
                                                alternating_rf=True,
                                                spoiler_phase_range=spoiler_phase_range)
        print(M)
        Mx[k] = M[0]
        My[k] = M[1]
        Mz[k] = M[2]
        k = k+1

    # plot of the signal
    fig,ax = plt.subplots()
    xdata = test_spoiler_phase/np.pi
    plt.plot(xdata,Mx,label='Mx',marker='o',linestyle='--')
    plt.plot(xdata,My,label='My',marker='o',linestyle='--')
    plt.plot(xdata,Mz,label='Mz',marker='o',linestyle='--')
    plt.plot(xdata,np.sqrt(Mx**2+My**2),label='|Mxy|',marker='o',linestyle='--')
    plt.legend()
    plt.xlabel(r'dephasing ($\pi$)')
    plt.savefig('hw1/2b-ii.png')
    plt.close(fig)



    # (iii) gradient spoiler and RF spoiling
    print('>> rf spoiling')
    T1 = 1000 #(ms)
    T2 = 100 #(ms)
    TR = 10  #(ms)
    TE = 5 #(ms)
    flip = 10/180*np.pi

    Phy.bSSFP_steadystate_signal_revolution(flip,TR,T1,T2,df=0,nTR=100,
                                            alternating_rf_phase=False,
                                            spoiling_gradient_case='gradient-spoiling',
                                            spoiling_gradient=0.3*np.pi,
                                            spoiling_rf=True,
                                            spoiling_rf_phase=0.1*np.pi)
    M = Phy.SSFP_complex_steadystate_mag(flip,TR,TE,T1,T2,df=0,nTR=100,
                                        alternating_rf_phase=False,
                                        spoiling_gradient_case='gradient-spoiling',
                                        spoiling_gradient=0.3*np.pi,
                                        spoiling_rf=True,
                                        spoiling_rf_phase=0.1*np.pi)
    print(M)


    # def rf_spoiled_voxel_signal_revolution(flip,TR,TE,T1,T2,df,nTR,spoiling_rf_phase):
    #     Nspins = 200
    #     spoiling_gradient_phase = np.linspace(0,2*np.pi,Nspins)
    #     Mx_total = 0
    #     My_total = 0
    #     Mz_total = 0
    #     for spoiler_phase in spoiling_gradient_phase:
    #         Mx,My,Mz = Phy.bSSFP_steadystate_signal_revolution(flip,TR,TE,T1,T2,df,nTR,
    #                                             alternating_rf_phase=False,
    #                                             spoiling_gradient_case='gradient-spoiling',
    #                                             spoiling_gradient=spoiler_phase,
    #                                             spoiling_rf=True,
    #                                             spoiling_rf_phase=spoiling_rf_phase,
    #                                             savepicname='none')
    #         Mx_total = Mx_total + Mx
    #         My_total = My_total + My
    #         Mz_total = Mz_total + Mz
    #     Mx_normalized = Mx_total/Nspins
    #     My_normalized = My_total/Nspins
    #     Mz_normalized = Mz_total/Nspins
    #     return Mx_normalized,My_normalized,Mz_normalized
    # Mx,My,Mz = rf_spoiled_voxel_signal_revolution(flip,TR,TE,T1,T2,df=0,nTR=100,
    #                                               spoiling_rf_phase=0.5*np.pi)
    # plt.figure()
    # plt.plot(Mx,label='x')
    # plt.plot(My,label='y')
    # plt.plot(Mz,label='z')
    # plt.legend()
    # plt.savefig('hw1/2b-iii-rfspoiled-signal.png')

    def rf_spoiled_voxel_signal(flip,TR,TE,T1,T2,df,nTR,spoiling_rf_method,spoiling_rf_phase):
        Nspins = 200
        spoiling_gradient_phase = np.linspace(0,4*np.pi,Nspins)
        M_total = 0
        for spoiler_phase in spoiling_gradient_phase:
            M = Phy.SSFP_complex_steadystate_mag(flip,TR,TE,T1,T2,df,nTR,
                                                alternating_rf_phase=False,
                                                spoiling_gradient_case='gradient-spoiling',
                                                spoiling_gradient=spoiler_phase,
                                                spoiling_rf=spoiling_rf_method,
                                                spoiling_rf_phase=spoiling_rf_phase)
            M_total = M_total + M
        M_normalized = M_total/Nspins
        return M_normalized
    

    spoiling_rf_phase_range = np.linspace(0,np.pi,180)
    Mx = np.zeros_like(spoiling_rf_phase_range)
    My = np.zeros_like(spoiling_rf_phase_range)
    Mz = np.zeros_like(spoiling_rf_phase_range)
    k = 0
    for spoiling_rf_phase in spoiling_rf_phase_range:
        _ = print('test rf phase # =',k) if ((k%5)==0) else None
        M = rf_spoiled_voxel_signal(flip,TR,TE,T1,T2,df=0,nTR=600,spoiling_rf_method='linear',
                                    spoiling_rf_phase=spoiling_rf_phase)
        Mx[k] = M[0]
        My[k] = M[1]
        Mz[k] = M[2]
        k = k+1
    # plot
    best_line = np.ones_like(spoiling_rf_phase_range)*M_perfect_no_transverse
    plt.figure(figsize=(8,4))
    plt.plot(spoiling_rf_phase_range/np.pi*180,np.sqrt(Mx**2+My**2),label='|Mxy|')
    plt.plot(spoiling_rf_phase_range/np.pi*180,best_line,color='red',ls='--')
    plt.legend()
    plt.xlabel('RF phase')
    plt.savefig('hw1/2b-iii-linear.png')


    spoiling_rf_phase_range = np.linspace(0,np.pi,180)
    Mx = np.zeros_like(spoiling_rf_phase_range)
    My = np.zeros_like(spoiling_rf_phase_range)
    Mz = np.zeros_like(spoiling_rf_phase_range)
    k = 0
    for spoiling_rf_phase in spoiling_rf_phase_range:
        _ = print('test rf phase # =',k) if ((k%5)==0) else None
        M = rf_spoiled_voxel_signal(flip,TR,TE,T1,T2,df=0,nTR=600,spoiling_rf_method='quadratic',
                                    spoiling_rf_phase=spoiling_rf_phase)
        Mx[k] = M[0]
        My[k] = M[1]
        Mz[k] = M[2]
        k = k+1
    # plot
    best_line = np.ones_like(spoiling_rf_phase_range)*M_perfect_no_transverse
    plt.figure(figsize=(8,4))
    plt.plot(spoiling_rf_phase_range/np.pi*180,np.sqrt(Mx**2+My**2),label='|Mxy|')
    plt.plot(spoiling_rf_phase_range/np.pi*180,best_line,color='red',ls='--')
    plt.legend()
    plt.xlabel('RF phase')
    plt.savefig('hw1/2b-iii-quadratic.png')


    return

def problem_3():
    print('\nProblem 3 =======================')
    print('------------question 1')
    gamma = 42.58  #(MHz/T)
    gmax = 25 #(mT/m)
    slewrate = 180 #(mT/m/ms)
    
    slicethickness = 5e-3 #(m)
    tbw = 8
    dur = 2  #(ms)
    bw = tbw/(dur/1000)  #(Hz)
    print('bandwidth:',bw)
    Gs = bw/(gamma*slicethickness)/1000  #(mT/m)
    print('slice-selective gradient: {} mT/m'.format(Gs))
    t_rise = Gs/slewrate  #(ms)
    print('gradient rise time: {} ms'.format(t_rise))
    print('total time: {} ms'.format(t_rise*2+dur))

    # calculate for the rephasing gradient
    t_rep_flat = (0.5*(t_rise*Gs + Gs*dur) - gmax/slewrate*gmax)/gmax
    print('rephasing gradient flat time (ms):',t_rep_flat)
    print('rephasing gradient rise time (ms):',gmax/slewrate)

    
    
    
    print('--------------question 2')
    Nt = 200
    dt = 2/Nt #(ms)
    t = np.arange(Nt)*dt  #(ms)
    rf = np.sinc((t-1)/(0.25*1))

    T1 = 1000  #(ms)
    T2 = 100 #(ms)
    
    # calculate the magnitude of RF which gives 90 degree flip
    flip = np.pi/2
    rf_scale = flip / (np.sum(rf*dt)*gamma*2*np.pi)
    rf_90 = rf*rf_scale  # (mT)
    print('rf-90',rf_90.shape)
    print('rf-90: max (mT)',np.max(rf_90))
    print(np.sum(rf_90)*dt*gamma*np.pi*2)

    # the magnetization change at 0Hz
    df = 0
    M_end, M_hist = Phy.simulate_rf(rf_90,dt=dt,df=df,T1=T1,T2=T2,M_init=np.array([0,0,1]),
                                    rephasing_gradient=False)
    tlist = np.arange(M_hist.shape[1])*dt
    fig,ax = plt.subplots()
    plt.plot(tlist,M_hist[0,:],label='Mx')
    plt.plot(tlist,M_hist[1,:],label='My')
    plt.plot(tlist,M_hist[2,:],label='Mz')
    plt.plot(tlist,np.sqrt(M_hist[0,:]**2+M_hist[1,:]**2),label='|Mxy|')
    plt.title('df = {} Hz'.format(df))
    plt.xlabel('time (ms)')
    plt.legend()
    plt.savefig('hw1/rf-90-0hz.png')
    print(df,M_end)

    # the magnetization change at 200Hz
    df = 200
    M_end, M_hist = Phy.simulate_rf(rf_90,dt=dt,df=df,T1=T1,T2=T2,M_init=np.array([0,0,1]),
                                    rephasing_gradient=False)
    tlist = np.arange(M_hist.shape[1])*dt
    fig,ax = plt.subplots()
    plt.plot(tlist,M_hist[0,:],label='Mx')
    plt.plot(tlist,M_hist[1,:],label='My')
    plt.plot(tlist,M_hist[2,:],label='Mz')
    plt.plot(tlist,np.sqrt(M_hist[0,:]**2+M_hist[1,:]**2),label='|Mxy|')
    plt.title('df = {} Hz'.format(df))
    plt.xlabel('time (ms)')
    plt.legend()
    plt.savefig('hw1/rf-90-200hz.png')
    print(df,M_end)

    # the magnetization change at 0Hz
    df = 0
    M_end, M_hist = Phy.simulate_rf(rf_90,dt=dt,df=df,T1=T1,T2=T2,M_init=np.array([0,0,1]),
                                    rephasing_gradient=True)
    tlist = np.arange(M_hist.shape[1])*dt
    fig,ax = plt.subplots()
    plt.plot(tlist,M_hist[0,:],label='Mx')
    plt.plot(tlist,M_hist[1,:],label='My')
    plt.plot(tlist,M_hist[2,:],label='Mz')
    plt.plot(tlist,np.sqrt(M_hist[0,:]**2+M_hist[1,:]**2),label='|Mxy|')
    plt.title('df = {} Hz'.format(df))
    plt.xlabel('time (ms)')
    plt.legend()
    plt.savefig('hw1/rf-90-0hz-r.png')
    print(df,M_end)

    # the magnetization change at 200Hz
    df = 200
    M_end, M_hist = Phy.simulate_rf(rf_90,dt=dt,df=df,T1=T1,T2=T2,M_init=np.array([0,0,1]),
                                    rephasing_gradient=True)
    tlist = np.arange(M_hist.shape[1])*dt
    fig,ax = plt.subplots()
    plt.plot(tlist,M_hist[0,:],label='Mx')
    plt.plot(tlist,M_hist[1,:],label='My')
    plt.plot(tlist,M_hist[2,:],label='Mz')
    plt.plot(tlist,np.sqrt(M_hist[0,:]**2+M_hist[1,:]**2),label='|Mxy|')
    plt.title('df = {} Hz'.format(df))
    plt.xlabel('time (ms)')
    plt.legend()
    plt.savefig('hw1/rf-90-200hz-r.png')
    print(df,M_end)


    # plot of the shape of RF pulse
    if True:
        fig,ax = plt.subplots(figsize=(3,3))
        plt.plot(t,rf_90*1000)
        # plt.xlabel('ms')
        # ax.set_xticks([])
        # ax.set_yticks([])
        # plt.axis('off')
        ax.set_ylabel(r'${\mu}T$')
        ax.set_xlabel('ms')
        fig.patch.set_alpha(0.0)
        fig.tight_layout()
        # fig.set_facecolor('none')
        plt.savefig('hw1/prob3-q1-rf.png',transparent=True)
        plt.close(fig)



    # function simulate the slice profile of the rf pulse
    def simu_rf_profile(rf,dt,df_range,T1,T2,rephasing_gradient=False):
        '''
        input:
            rf: (mT)
            dt: (ms)
            df_range: (Hz)
        '''
        Nf = len(df_range)
        print('{} frequency points'.format(Nf))
        M = np.zeros((3,Nf))
        for k in range(Nf):
            df = df_range[k]
            # print('df:{} Hz'.format(df))
            Mexc,_ = Phy.simulate_rf(rf,dt,df,T1,T2,rephasing_gradient=rephasing_gradient)
            # print(Mexc)
            M[:,k] = Mexc
        return M


    print('--------------question 3')
    # generate rf pulse
    rf_30 = rf_90/3
    rf_10 = rf_90/9

    print(Gs,gamma)
    

    # simulate the profile -------- 90 degree
    df_range = np.linspace(-4000,4000,200)  #(Hz)
    # also translate the off-resonance into position
    dz_range = df_range/gamma/Gs # (mm)
    M = simu_rf_profile(rf_90,dt,df_range,T1,T2)
    # plot the profile
    plt.figure()
    plt.plot(dz_range,M[0,:],label='Mx')
    plt.plot(dz_range,M[1,:],label='My')
    plt.plot(dz_range,M[2,:],label='Mz')
    plt.plot(dz_range,np.sqrt(M[0,:]**2 + M[1,:]**2),label='|Mxy|',ls='--')
    plt.legend()
    plt.xlabel('mm')
    # plt.show()
    plt.savefig('hw1/rf-90-profile.png')


    df_range = np.linspace(-4000,4000,200)  #(Hz)
    # also translate the off-resonance into position
    dz_range = df_range/gamma/Gs
    M = simu_rf_profile(rf_90,dt,df_range,T1,T2,rephasing_gradient=True)
    # plot the profile
    plt.figure()
    plt.plot(dz_range,M[0,:],label='Mx')
    plt.plot(dz_range,M[1,:],label='My')
    plt.plot(dz_range,M[2,:],label='Mz')
    plt.plot(dz_range,np.sqrt(M[0,:]**2 + M[1,:]**2),label='|Mxy|',ls='--')
    plt.legend()
    plt.xlabel('mm')
    # plt.show()
    plt.savefig('hw1/rf-90-profile-r.png')

    M_90 = M
    M_90_ideal = np.ones_like(df_range)
    M_90_ideal[df_range>2000] = 0
    M_90_ideal[df_range<-2000] = 0


    # simulate the profile -------- 30 degree
    df_range = np.linspace(-4000,4000,200)  #(Hz)
    # also translate the off-resonance into position
    dz_range = df_range/gamma/Gs
    M = simu_rf_profile(rf_30,dt,df_range,T1,T2)
    # plot the profile
    plt.figure()
    plt.plot(dz_range,M[0,:],label='Mx')
    plt.plot(dz_range,M[1,:],label='My')
    plt.plot(dz_range,M[2,:],label='Mz')
    plt.plot(dz_range,np.sqrt(M[0,:]**2 + M[1,:]**2),label='|Mxy|',ls='--')
    plt.legend()
    plt.xlabel('mm')
    # plt.show()
    plt.savefig('hw1/rf-30-profile.png')


    df_range = np.linspace(-4000,4000,200)  #(Hz)
    # also translate the off-resonance into position
    dz_range = df_range/gamma/Gs
    M = simu_rf_profile(rf_30,dt,df_range,T1,T2,rephasing_gradient=True)
    # plot the profile
    plt.figure()
    plt.plot(dz_range,M[0,:],label='Mx')
    plt.plot(dz_range,M[1,:],label='My')
    plt.plot(dz_range,M[2,:],label='Mz')
    plt.plot(dz_range,np.sqrt(M[0,:]**2 + M[1,:]**2),label='|Mxy|',ls='--')
    plt.legend()
    plt.xlabel('mm')
    # plt.show()
    plt.savefig('hw1/rf-30-profile-r.png')


    # simulate the profile -------- 10 degree
    df_range = np.linspace(-4000,4000,200)  #(Hz)
    # also translate the off-resonance into position
    dz_range = df_range/gamma/Gs
    M = simu_rf_profile(rf_10,dt,df_range,T1,T2)
    # plot the profile
    plt.figure()
    plt.plot(dz_range,M[0,:],label='Mx')
    plt.plot(dz_range,M[1,:],label='My')
    plt.plot(dz_range,M[2,:],label='Mz')
    plt.plot(dz_range,np.sqrt(M[0,:]**2 + M[1,:]**2),label='|Mxy|',ls='--')
    plt.legend()
    plt.xlabel('mm')
    # plt.show()
    plt.savefig('hw1/rf-10-profile.png')


    df_range = np.linspace(-4000,4000,200)  #(Hz)
    # also translate the off-resonance into position
    dz_range = df_range/gamma/Gs
    M = simu_rf_profile(rf_10,dt,df_range,T1,T2,rephasing_gradient=True)
    # plot the profile
    plt.figure()
    plt.plot(dz_range,M[0,:],label='Mx')
    plt.plot(dz_range,M[1,:],label='My')
    plt.plot(dz_range,M[2,:],label='Mz')
    plt.plot(dz_range,np.sqrt(M[0,:]**2 + M[1,:]**2),label='|Mxy|',ls='--')
    plt.legend()
    plt.xlabel('mm')
    # plt.show()
    plt.savefig('hw1/rf-10-profile-r.png')

    M_10 = M
    M_10_ideal = np.ones_like(df_range)*np.sin(10/180*np.pi)
    M_10_ideal[df_range>2000] = 0
    M_10_ideal[df_range<-2000] = 0

    # plot of 90 and 10 deg pulse together, and show idea profile
    fig,axs = plt.subplots()
    axs.plot(dz_range,np.sqrt(M_90[0,:]**2+M_90[1,:]**2),label='90-simulation')
    axs.plot(dz_range,np.sqrt(M_10[0,:]**2+M_10[1,:]**2),label='10-simulation')
    axs.plot(dz_range,M_90_ideal,label='90-ideal')
    axs.plot(dz_range,M_10_ideal,label='10-ideal')
    plt.xlabel('mm')
    plt.legend()
    plt.savefig('hw1/rf-compare-profile-10-90.png')




    # what if the T2=2(ms)
    # simulate the profile -------- 90 degree
    df_range = np.linspace(-4000,4000,200)  #(Hz)
    # also translate the off-resonance into position
    dz_range = df_range/gamma/Gs # (mm)
    M = simu_rf_profile(rf_90,dt,df_range,T1,T2=2)
    # plot the profile
    plt.figure()
    plt.plot(dz_range,M[0,:],label='Mx')
    plt.plot(dz_range,M[1,:],label='My')
    plt.plot(dz_range,M[2,:],label='Mz')
    plt.plot(dz_range,np.sqrt(M[0,:]**2 + M[1,:]**2),label='|Mxy|',ls='--')
    plt.legend()
    plt.xlabel('mm')
    # plt.show()
    plt.savefig('hw1/rf-smallT2-90-profile.png')

    df_range = np.linspace(-4000,4000,200)  #(Hz)
    # also translate the off-resonance into position
    dz_range = df_range/gamma/Gs
    M = simu_rf_profile(rf_90,dt,df_range,T1,T2=2,rephasing_gradient=True)
    # plot the profile
    plt.figure()
    plt.plot(dz_range,M[0,:],label='Mx')
    plt.plot(dz_range,M[1,:],label='My')
    plt.plot(dz_range,M[2,:],label='Mz')
    plt.plot(dz_range,np.sqrt(M[0,:]**2 + M[1,:]**2),label='|Mxy|',ls='--')
    plt.legend()
    plt.xlabel('mm')
    # plt.show()
    plt.savefig('hw1/rf-smallT2-90-profile-r.png')


    print('--------------question 4')
    # simulate Mx,My,Mz at the point before slice refocusing gradient
    # use one example above


    print('--------------question 5')
    # design rf pulse excite 5 slices
    # modulate the 90-deg pulse to 5 slices
    gap = 20e-3 #(m)
    # comput the modulation
    freq_shift = gap*gamma*Gs*1e3
    print('frequency shift (Hz):', freq_shift)

    # design sms pulse
    rf_modulater = 2*np.cos(2*np.pi*freq_shift*t*1e-3) + 2*np.cos(2*np.pi*freq_shift*2*t*1e-3) + 1
    rf_sms = rf_90*rf_modulater

    # simulate the slice profile
    df_range = np.linspace(-50e3,50e3,1000)  #(Hz)
    # also translate the off-resonance into position
    dz_range = df_range/gamma/Gs
    M = simu_rf_profile(rf_sms,dt,df_range,T1,T2,rephasing_gradient=False)
    # plot the profile
    plt.figure()
    plt.plot(dz_range,M[0,:],label='Mx')
    plt.plot(dz_range,M[1,:],label='My')
    plt.plot(dz_range,M[2,:],label='Mz')
    plt.plot(dz_range,np.sqrt(M[0,:]**2 + M[1,:]**2),label='|Mxy|',ls='--')
    plt.legend()
    plt.xlabel('mm')
    # plt.show()
    plt.savefig('hw1/rf-sms.png')

    # simulate the slice profile
    df_range = np.linspace(-50e3,50e3,1000)  #(Hz)
    # also translate the off-resonance into position
    dz_range = df_range/gamma/Gs
    M = simu_rf_profile(rf_sms,dt,df_range,T1,T2,rephasing_gradient=True)
    # plot the profile
    plt.figure()
    plt.plot(dz_range,M[0,:],label='Mx')
    plt.plot(dz_range,M[1,:],label='My')
    plt.plot(dz_range,M[2,:],label='Mz')
    plt.plot(dz_range,np.sqrt(M[0,:]**2 + M[1,:]**2),label='|Mxy|',ls='--')
    plt.legend()
    plt.xlabel('mm')
    # plt.show()
    plt.savefig('hw1/rf-sms-r.png')


    return


if __name__ == '__main__':
    pass
    # priblem 1
    problem_1()

    # Problem 2
    problem_2()


    # Problem 3
    problem_3()




