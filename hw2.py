# umich bme hw 2
# jiayao


import numpy as np
import matplotlib.pyplot as plt
import scipy.io as spio
import mat73
from time import time

import simu



def p1_q1(T1,T2,alpha,lim=None,epgpic='xxx.png',echopic='xxx.png'):
    '''for problem 1 question 1'''
    epg = simu.ExtendedPhaseGraph(dt=0.25,t_total=400,T1=np.array([T1]),T2=np.array([T2]),relaxation=True)
    # add rf pulses
    epg.add_rf(flip=90,phase=0,t=0)
    flip1 = alpha + 90 - alpha/2
    t = 2.5
    for _ in range(64):
        epg.add_rf(flip=flip1,phase=90,t=t)
        t = t+5

    epg.show_info()
    # epg.simulate()
    epg.plot_epg_graph(state_lim=lim,dur=100,picname=epgpic,savefig=True)
    epg.plot_echo_states(picname=echopic,savefig=True)
    return

def p1_q2(T1list,T2list,alpha):
    '''for problem 1 question 1'''
    nt1 = len(T1list)
    nt2 = len(T2list)
    print('T1,T2 # = {},{}'.format(nt1,nt2))
    echo_mag = np.zeros((4,nt1,nt2))

    T1listlong = np.ones((nt1,nt2))*T1list.reshape(-1,1)
    T2listlong = np.ones((nt1,nt2))*T2list.reshape(1,-1)
    T1listlong = T1listlong.reshape(-1)
    T2listlong = T2listlong.reshape(-1)

    epg = simu.ExtendedPhaseGraph(dt=0.25,t_total=400,T1=T1listlong,T2=T2listlong,relaxation=True)
    # add rf pulses
    epg.add_rf(flip=90,phase=0,t=0)
    flip1 = alpha + 90 - alpha/2
    t = 2.5
    for _ in range(64):
        epg.add_rf(flip=flip1,phase=90,t=t)
        t = t+5

    epg.show_info()
    epg.simulate()
    # echo_signal = epg.echo_states
    # get the 6th, 16th, 32th, 48th echoes
    # echo_mag[0] = echo_signal[:,120].reshape(nt1,nt2)
    # echo_mag[0] = echo_signal[:,320].reshape(nt1,nt2)
    # echo_mag[0] = echo_signal[:,640].reshape(nt1,nt2)
    # echo_mag[0] = echo_signal[:,960].reshape(nt1,nt2)
    echo_mag[0] = np.abs(epg.detect_echo(5*6)).reshape(nt1,nt2)
    echo_mag[1] = np.abs(epg.detect_echo(5*16)).reshape(nt1,nt2)
    echo_mag[2] = np.abs(epg.detect_echo(5*32)).reshape(nt1,nt2)
    echo_mag[3] = np.abs(epg.detect_echo(5*48)).reshape(nt1,nt2)
    print()
    return echo_mag

def p2_q1(M0map,T1map,T2map,TR,TE):
    '''for problem 2 question a'''
    print('>> problem 2 question a')
    nTR = 5
    nx,ny = M0map.shape
    mask = np.nonzero(M0map>0)

    T1list = T1map[mask].reshape(-1)
    T2list = T2map[mask].reshape(-1)
    # T1list = np.array([1000])
    # T2list = np.array([200])

    t_total = (nTR+0)*TR
    dt = TE/20
    epg = simu.ExtendedPhaseGraph(
        dt=dt,t_total=t_total,T1=T1list,T2=T2list,
        relaxation=True,unit_gradient_shift_time=dt)
    for TRitr in range(nTR):
        print('add rf for TR #{}'.format(TRitr))
        epg.add_rf(flip=90,phase=0,t=TR*TRitr)
        epg.add_rf(flip=180,phase=90,t=TR*TRitr+TE/2)
    epg.add_key_times([4*TR+TE])
    epg.show_info()
    epg.simulate(efficient=True)
    
    # epg.plot_echo_states(efficient=True,savefig=True)
    # return

    echo = epg.detect_echo(4*TR+TE)

    img = np.zeros((nx,ny))*1j
    img[mask] = echo*M0map[mask]

    # print(echo)
    # print(epg.echo_states[int(TE/epg.dt)-3:int(TE/epg.dt)+1])
    # plt.figure()
    # plt.plot(np.abs(epg.echo_states))
    # plt.savefig('hw2/tmp.png')
    return img

def p2_q2_i(T1,T2,TR=3000):
    ETL = 32
    ESP = 5 #(ms)
    dt = ESP/10
    t_total = TR*5
    epg = simu.ExtendedPhaseGraph(
        dt=dt,t_total=t_total,T1=np.array([T1]),T2=np.array([T2]),
        relaxation=True,unit_gradient_shift_time=dt,
    )
    # simulate first 4 TR
    for TRitr in range(5):
        print('add rf for TR #{}'.format(TRitr+1))
        epg.add_rf(flip=90,phase=0,t=TR*TRitr)
        for k in range(ETL):
            epg.add_rf(flip=180,phase=90,t=TR*TRitr+ESP/2+ESP*k)
    # 
    epg.plot_epg_graph(state_lim=0.0005,starttime=TR*4,dur=ESP*ETL+ESP,
                       picname='hw2/p2-b-i-epg-T1{}-T2{}.png'.format(T1,T2),savefig=True)
    
    epg.plot_echo_states(starttime=4*TR,dur=ESP*ETL+ESP,efficient=False,
                        picname='hw2/p2-b-i-echo-T1{}-T2{}.png'.format(T1,T2),savefig=True)
    
    # epg.add_key_times([4*TR])
    # # print(epg.rf_events_time)
    # # return
    # # epg.plot_echo_states(picname='hw2/test.png',savefig=True)
    # # return
    # epg.simulate(efficient=True)
    # state_p,state_n,state_z = epg.get_all_states()
    

    # # plot the 5th TR
    # epg.clean_all_events()
    # epg.add_rf(flip=90,phase=0,t=0)
    # for k in range(ETL):
    #     epg.add_rf(flip=180,phase=90,t=ESP/2+ESP*k)

    # epg.set_all_states(state_p,state_n,state_z)
    # epg.plot_epg_graph(reset=False,state_lim=0.001,starttime=0,dur=ESP*ETL+ESP,
    #                    picname='hw2/p2-b-i-epg-T1{}-T2{}.png'.format(T1,T2),savefig=True)

    # epg.set_all_states(state_p,state_n,state_z)
    # epg.plot_echo_states(reset=False,dur=ESP*ETL+ESP,efficient=False,
    #                     picname='hw2/p2-b-i-echo-T1{}-T2{}.png'.format(T1,T2),savefig=True)

    return

def q2_p2_echo_images(M0map,T1map,T2map,TR,ESP,ETL):
    '''acquire images from different echoes'''
    nTR = 3
    nx,ny = M0map.shape
    mask = np.nonzero(M0map>0)

    T1list = T1map[mask].reshape(-1)
    T2list = T2map[mask].reshape(-1)
    # T1list = np.array([1000])
    # T2list = np.array([200])

    ttt = time()

    t_total = (nTR+0)*TR
    dt = ESP/4


    # pixel by pixel
    if False:
        echo_image_flat = np.zeros((ETL,len(T1list)))
        for pixitr in range(len(T1list)):
            print('{}th pixel'.format(pixitr))
            T1 = T1list[pixitr]
            T2 = T2list[pixitr]

            epg = simu.ExtendedPhaseGraph(dt=dt,t_total=t_total,T1=np.array([T1]),T2=np.array([T2]),
                relaxation=True,unit_gradient_shift_time=dt)
            # add rf pulse
            for TRitr in range(nTR):
                print('add rf for TR #{}'.format(TRitr))
                epg.add_rf(flip=90,phase=0,t=TR*TRitr)
                for ee in range(ETL):
                    epg.add_rf(flip=180,phase=90,t=TR*TRitr + ESP/2 + ee*ESP)
            # add echo times for detection
            for ee in range(ETL):
                epg.add_key_times([2*TR + ESP + ESP*ee])
        
            # epg.show_info()
            epg.simulate(efficient=True)

            # detect echoes
            for ee in range(ETL):
                echo = epg.detect_echo(2*TR+ESP+ESP*ee)
                print(echo.shape)
                echo_image_flat[ee,pixitr]

        print('use time:',time()-ttt)

        echo_images = np.zeros((ETL,nx,ny))*1j
        for k in range(ETL):

            img = np.zeros((nx,ny))*1j
            img[mask] = echo_image_flat[k,:]*M0map[mask]

            echo_images[k] = img
    
    # as a list
    if True:
        epg = simu.ExtendedPhaseGraph(dt=dt,t_total=t_total,T1=T1list,T2=T2list,
            relaxation=True,unit_gradient_shift_time=dt)
        # add rf pulse
        for TRitr in range(nTR):
            epg.add_rf(flip=90,phase=0,t=TR*TRitr)
            if TRitr>1:
                print('add rf for TR #{}'.format(TRitr))
                for ee in range(ETL):
                    epg.add_rf(flip=180,phase=90,t=TR*TRitr + ESP/2 + ee*ESP)
        # add echo times for detection
        for ee in range(ETL):
            epg.add_key_times([2*TR + ESP + ESP*ee])

        epg.show_info()
        epg.simulate(efficient=True)


        echo_images = np.zeros((ETL,nx,ny))*1j

        # detect echoes
        for ee in range(ETL):

            echo = epg.detect_echo(2*TR+ESP+ESP*ee)

            img = np.zeros((nx,ny))*1j
            img[mask] = echo*M0map[mask]

            echo_images[ee] = img

    return echo_images

def q2_p2_sample_kspace(echo_images,pe_order,picname='tmp.png',title='',savefig=True):
    '''sample the kspace according the pe_order'''
    n_echo,nx,ny = echo_images.shape
    n_TR = pe_order.shape[0]

    # get kspace of different echo images
    kspaces = np.zeros_like(echo_images)*1j
    for echo_itr in range(n_echo):
        ksp = np.fft.fftshift(np.fft.fft2(echo_images[echo_itr]))
        kspaces[echo_itr] = ksp

    # acquire k-space according to pe_order
    kspace_acq = np.zeros((nx,ny))*1.0j
    for acqitr in range(n_TR):
        print('TR: #{}'.format(acqitr))
        echo_lines = pe_order[acqitr]
        echo_cnt = 0
        for k in echo_lines:
            kspace_acq[:,k] = kspaces[echo_cnt,:,k]
            echo_cnt = echo_cnt + 1

    # recon image
    img = np.fft.ifft2(np.fft.fftshift(kspace_acq))

    fig = plt.figure()
    plt.imshow(np.abs(img),cmap='gray')
    plt.title(title)
    if savefig:
        plt.savefig(picname)
        plt.close(fig)
    else:
        plt.show()
    return img











if __name__ == '__main__':
    print('hw2'.center(50,'-'))
    print('problem 1:')


    # question 1
    # T1,T2 = 200,50 #ms
    # ---------------------------
    if False:
        p1_q1(T1=200,T2=50,alpha=180,lim=0.025,epgpic='hw2/180-1-epg.png',echopic='hw2/180-1-echo.png')
        p1_q1(T1=500,T2=50,alpha=180,lim=0.025,epgpic='hw2/180-2-epg.png',echopic='hw2/180-2-echo.png')
        p1_q1(T1=1000,T2=50,alpha=180,lim=0.025,epgpic='hw2/180-3-epg.png',echopic='hw2/180-3-echo.png')
        p1_q1(T1=1000,T2=110,alpha=180,lim=0.025,epgpic='hw2/180-4-epg.png',echopic='hw2/180-4-echo.png')
        p1_q1(T1=1000,T2=200,alpha=180,lim=0.025,epgpic='hw2/180-5-epg.png',echopic='hw2/180-5-echo.png')


        p1_q1(T1=200,T2=50,alpha=120,lim=0.05,epgpic='hw2/120-1-epg.png',echopic='hw2/120-1-echo.png')
        p1_q1(T1=500,T2=50,alpha=120,lim=0.05,epgpic='hw2/120-2-epg.png',echopic='hw2/120-2-echo.png')
        p1_q1(T1=1000,T2=50,alpha=120,lim=0.05,epgpic='hw2/120-3-epg.png',echopic='hw2/120-3-echo.png')
        p1_q1(T1=1000,T2=110,alpha=120,lim=0.1,epgpic='hw2/120-4-epg.png',echopic='hw2/120-4-echo.png')
        p1_q1(T1=1000,T2=200,alpha=120,lim=0.1,epgpic='hw2/120-5-epg.png',echopic='hw2/120-5-echo.png')


        p1_q1(T1=200,T2=50,alpha=60,lim=0.1,epgpic='hw2/60-1-epg.png',echopic='hw2/60-1-echo.png')
        p1_q1(T1=500,T2=50,alpha=60,lim=0.1,epgpic='hw2/60-2-epg.png',echopic='hw2/60-2-echo.png')
        p1_q1(T1=1000,T2=50,alpha=60,lim=0.1,epgpic='hw2/60-3-epg.png',echopic='hw2/60-3-echo.png')
        p1_q1(T1=1000,T2=110,alpha=60,lim=0.1,epgpic='hw2/60-4-epg.png',echopic='hw2/60-4-echo.png')
        p1_q1(T1=1000,T2=200,alpha=60,lim=0.1,epgpic='hw2/60-5-epg.png',echopic='hw2/60-5-echo.png')

    # question 2
    if False:
        T1list = np.linspace(200,1500,30)
        T2list = np.linspace(50,300,30)
        alpha_list = [180,120,60]
        nt1 = len(T1list)
        nt2 = len(T2list)
        
        xx,yy = np.meshgrid(T1list,T2list)

        for alpha in alpha_list:
            echo_mag = p1_q2(T1list=T1list,T2list=T2list,alpha=alpha)

            fig = plt.figure(figsize=(10,7))
            ax = plt.subplot(2,2,1)
            # im = ax.imshow(echo_mag[0],vmin=0,vmax=1)
            im = ax.contourf(xx,yy,echo_mag[0])
            fig.colorbar(im,ax=ax)
            ax.set_title('6th echo')
            ax.set_xlabel('T1 (ms)')
            ax.set_ylabel('T2 (ms)')
            
            ax = plt.subplot(2,2,2)
            # im = ax.imshow(echo_mag[1],vmin=0,vmax=1)
            im = ax.contourf(xx,yy,echo_mag[1])
            fig.colorbar(im,ax=ax)
            ax.set_title('16th echo')
            ax.set_xlabel('T1 (ms)')
            ax.set_ylabel('T2 (ms)')

            ax = plt.subplot(2,2,3)
            # im = ax.imshow(echo_mag[2],vmin=0,vmax=1)
            im = ax.contourf(xx,yy,echo_mag[2])
            fig.colorbar(im,ax=ax)
            ax.set_title('32th echo')
            ax.set_xlabel('T1 (ms)')
            ax.set_ylabel('T2 (ms)')

            ax = plt.subplot(2,2,4)
            # im = ax.imshow(echo_mag[3],vmin=0,vmax=1)
            im = ax.contourf(xx,yy,echo_mag[3])
            fig.colorbar(im,ax=ax)
            ax.set_title('48th echo')
            ax.set_xlabel('T1 (ms)')
            ax.set_ylabel('T2 (ms)')

            plt.tight_layout()
            plt.suptitle(r'$\alpha$={}$^\circ$'.format(alpha))
            plt.savefig('hw2/p1-b-{}.png'.format(alpha))


        

    print('problem 2:')
    data = mat73.loadmat('hw2/brain_maps.mat')
    print(data.keys())
    brain_proton = data['M0map']
    brain_T1 = data['T1map']+1e-5
    brain_T2 = data['T2map']+1e-5
    print(brain_proton.shape)
    # 
    for var in [brain_proton,brain_T1,brain_T2]:
        print('nan:',np.sum(np.isnan(var)))
        print('sum:',np.sum(var==0))
        print('>0:',np.sum(var>0))
        print('max;min:',np.max(var),np.min(var))
        print()

    if False:
        print('test')
        fig = plt.figure()
        ax = plt.subplot(1,3,1)
        ax.imshow(brain_proton,cmap='gray')
        ax = plt.subplot(1,3,2)
        ax.imshow(brain_T1,cmap='gray')
        ax = plt.subplot(1,3,3)
        ax.imshow(brain_T2,cmap='gray')
        plt.savefig('hw2/p2-test.png')
        plt.close(fig)

    # question a:
    # ------------------------
    # use epg to create different weighted images
    if False:
        # T1-weighted
        TR,TE = 400,10 #ms
        img = p2_q1(M0map=brain_proton,T1map=brain_T1,T2map=brain_T2,TR=TR,TE=TE)
        fig,ax = plt.subplots()
        ax.imshow(np.abs(img),cmap='gray')
        plt.title('TE={}ms,TR={}ms'.format(TE,TR))
        plt.savefig('hw2/p2-a-T1.png')
        plt.close(fig)

        # T2-weighted
        TR,TE = 3000,80 #ms
        img = p2_q1(M0map=brain_proton,T1map=brain_T1,T2map=brain_T2,TR=TR,TE=TE)
        fig,ax = plt.subplots()
        ax.imshow(np.abs(img),cmap='gray')
        plt.title('TE={}ms,TR={}ms'.format(TE,TR))
        plt.savefig('hw2/p2-a-T2.png')
        plt.close(fig)

        # proton density weighted
        TR,TE = 3000,10 #ms
        img = p2_q1(M0map=brain_proton,T1map=brain_T1,T2map=brain_T2,TR=TR,TE=TE)
        fig,ax = plt.subplots()
        ax.imshow(np.abs(img),cmap='gray')
        plt.title('TE={}ms,TR={}ms'.format(TE,TR))
        plt.savefig('hw2/p2-a-proton.png')
        plt.close(fig)


    # question b:
    # ------------------------
    # different T1,T2 combination
    if False:
        T1,T2 = 1000,50
        p2_q2_i(T1=T1,T2=T2)
        
        T1,T2 = 1000,100
        p2_q2_i(T1=T1,T2=T2)
        
        T1,T2 = 2000,50
        p2_q2_i(T1=T1,T2=T2)

        T1,T2 = 2000,100
        p2_q2_i(T1=T1,T2=T2)
    


    # design FSE k-space filling
    nx,ny = brain_proton.shape

    # test
    if False:
        echo_images_128 = np.zeros((128,nx,ny))*1j
        print(echo_images_128.shape)

    TR = 3000
    ETL = 128
    ESP = 5
    if False:
        echo_images_128 = q2_p2_echo_images(brain_proton,brain_T1,brain_T2,TR=TR,ESP=ESP,ETL=ETL)
        # echo_images_32 = np.zeros()
        spio.savemat('hw2/echoimages_128.mat',{'echo_images_128':echo_images_128})
    else:
        data = spio.loadmat('hw2/echoimages_128.mat')
        echo_images_128 = data['echo_images_128']
    
    # test
    fig,axes = plt.subplots(5,5)
    n = 0
    for ax in axes.flat:
        ax.imshow(np.abs(echo_images_128[n]),cmap='gray')
        n = n+1
    plt.tight_layout()
    plt.savefig('hw2/test.png')
    plt.close(fig)
    # exit()

    # ii. 
    # TEeff = 80 ms, ETL = 32
    pe_order = np.arange(256)
    pe_order = np.transpose(pe_order.reshape(32,8))
    pe_order = np.roll(pe_order,-1,axis=1)
    
    def print_matr(ma):
        x,y = ma.shape
        for xi in range(x):
            for yi in range(y):
                if yi==y-1:
                    print(ma[xi,yi],'\\\\')
                else:
                    print(ma[xi,yi],',',end=' ')
        print()
        return

    print_matr(pe_order)
    q2_p2_sample_kspace(
        echo_images=echo_images_128, pe_order=pe_order, picname='hw2/TEeff80-ETL32.png',
        title='TEeff=80ms,ETL=32',savefig=True
    )


    # iii. 
    # TEeff = 40 ms, ETL = 32
    pe_order = np.arange(256)
    pe_order = np.transpose(pe_order.reshape(32,8))
    pe_order = np.roll(pe_order,-9,axis=1)
    print_matr(pe_order)
    q2_p2_sample_kspace(
        echo_images=echo_images_128, pe_order=pe_order, picname='hw2/TEeff40-ETL32.png',
        title='TEeff=40ms,ETL=32',savefig=True
    )

    # TEeff = 120 ms, ETL = 32
    pe_order = np.arange(256)
    pe_order = np.transpose(pe_order.reshape(32,8))
    pe_order = np.roll(pe_order,7,axis=1)
    print_matr(pe_order)
    q2_p2_sample_kspace(
        echo_images=echo_images_128, pe_order=pe_order, picname='hw2/TEeff120-ETL32.png',
        title='TEeff=120ms,ETL=32',savefig=True
    )




    print('question iv:')
    # iv. TEeff = 80 ms, ETL = 16
    pe_order = np.arange(256)
    pe_order = np.transpose(pe_order.reshape(16,16))
    pe_order = np.roll(pe_order,7,axis=1)
    print_matr(pe_order)
    q2_p2_sample_kspace(
        echo_images=echo_images_128, pe_order=pe_order, picname='hw2/TEeff80-ETL16.png',
        title='TEeff=80ms,ETL=16',savefig=True
    )


    # iv. TEeff = 80 ms, ETL = 32
    pe_order = np.arange(256)
    pe_order = np.transpose(pe_order.reshape(32,8))
    pe_order = np.roll(pe_order,-1,axis=1)
    print_matr(pe_order)
    q2_p2_sample_kspace(
        echo_images=echo_images_128, pe_order=pe_order, picname='hw2/TEeff80-ETL32.png',
        title='TEeff=80ms,ETL=32',savefig=True
    )


    # iv. TEeff = 80 ms, ETL = 64
    pe_order = np.arange(256)
    pe_order = np.transpose(pe_order.reshape(64,4))
    pe_order = np.roll(pe_order,-17,axis=1)
    print_matr(pe_order)
    q2_p2_sample_kspace(
        echo_images=echo_images_128, pe_order=pe_order, picname='hw2/TEeff80-ETL64.png',
        title='TEeff=80ms,ETL=64',savefig=True
    )



    # iv. TEeff = 80 ms, ETL = 128
    pe_order = np.arange(256)
    pe_order = np.transpose(pe_order.reshape(128,2))
    pe_order = np.roll(pe_order,-49,axis=1)
    print_matr(pe_order)
    q2_p2_sample_kspace(
        echo_images=echo_images_128, pe_order=pe_order, picname='hw2/TEeff80-ETL128.png',
        title='TEeff=80ms,ETL=128',savefig=True
    )
