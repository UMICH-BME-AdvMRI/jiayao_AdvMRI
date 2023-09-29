# simulation functions
# mainly in physics and for MRI
# author: jiayao

import numpy as np

import matplotlib.pyplot as plt


GAMMA = 42.58  # (MHz/T)

# compute rotation matrix
# https://en.wikipedia.org/wiki/Rotation_matrix

def rotation_matrix_z2x(ang):
	'''rotaion matrix from z-axis to x-axis (in numpy)
	
	right-handed, around y-axis

	input:
		ang: rotation angle, e.g., 0.5*np.pi
	output:
		R: (3*3) rotation matrix	
	'''
	R = np.array([[np.cos(ang), 0, np.sin(ang)],
				[0, 1, 0],
				[-np.sin(ang), 0, np.cos(ang)]])
	return R
def rotation_matrix_x2y(ang):
	'''rotation matrix from x-axis to y-axis (in numpy)

	rotate around z-axis

	input:
		ang: rotation angle, e.g., 1/4*np.pi
	output:
		R: (3*3) rotaion matrix
	'''
	R = np.array([[np.cos(ang), -np.sin(ang), 0],
				[np.sin(ang), np.cos(ang), 0],
				[0, 0, 1]])
	return R

# def rotation_matrix_np(rotaxis,rotang):
#     '''return rotation matrix, right handed (in numpy)
    
#     input:
#         rotaxis: (1*3) unit vector
#         rotang: (e.g., 0.5*pi)
#     output:
#         R: (3*3) rotation matrix
#     '''
#     if np.linalg.norm(rotaxis) == 0:
#         R = np.eye(3)
#     else:
#         rotaxis = rotaxis/np.linalg.norm(rotaxis)

#     return


# ===========================================================
# free precession
def free_precession(M_init,dur,T1,T2,df):
	'''free precession of spin in mri (in numpy)

	input:
		M_init: (3*1) assume the vector length 1
		dur: (ms) time for free precession
		T1: (ms)
		T2: (ms)
		df: (Hz) off-resonance effect
	output:
		M_end
	'''
	# consider the relaxation effects
	Mx = M_init[0]*np.exp(-dur/T2)
	My = M_init[1]*np.exp(-dur/T2)
	Mz = M_init[2]*np.exp(-dur/T1) + 1*(1 - np.exp(-dur/T1))
	M_end = np.array([Mx,My,Mz])
	# consider the off-resonance effect
	# which is a rotation in transverse plane, from x -> y
	phi = 2*np.pi*df*dur/1000
	R = rotation_matrix_x2y(phi)
	M_end = np.matmul(R,M_end)
	return M_end


# ===========================================================
# for gradient sequence, steady-state simulations

# def steady_state_revolution(flip,TR,T1,T2,df,nTR=10):
#     '''how signal change until steady state

#     input:
#         flip: e.g., 1/*np.pi
#         TR: (ms)
#         T1: (ms)
#         T2: (ms)
#         df: (Hz) off-resonance
#     output:
#         None
#     '''
#     # T1 = 600 # (ms)
#     # T2 = 100 # (ms)
#     # df = 0 # (Hz)
#     # flip = 60/180*np.pi
#     # TR = 500 #(ms)
#     # 
#     Mag = np.array([0,0,1])
#     Mx = []
#     My = []
#     Mz = []
#     flipdir = -1
#     # simulate 10 TR
#     for tr_i in range(nTR):
#         # excitaion
#         flipdir = flipdir*(-1) # alternating the RF phase
#         R = rotation_matrix_z2x(flip*flipdir)
#         Mag = np.matmul(R,Mag)
#         # print(Mag)

#         # free precession
#         for t in np.linspace(0,TR,100):
#             # print(t)
#             M_fp = free_precession(Mag,t,T1,T2,df)
#             Mx.append(M_fp[0])
#             My.append(M_fp[1])
#             Mz.append(M_fp[2])

#         Mag = M_fp
#     fig,axs = plt.subplots()
#     plt.plot(Mx,label='x')
#     plt.plot(My,label='y')
#     plt.plot(Mz,label='z')
#     # plt.show()
#     plt.savefig('tmp.png')
#     plt.close(fig)
#     return

def bSSFP_steadystate_signal_revolution(flip,TR,T1,T2,df,nTR=10,
    alternating_rf_phase=False,
    spoiling_gradient_case='none',
    spoiling_gradient=0*np.pi,
    spoiling_rf='none',
    spoiling_rf_phase=0*np.pi,
    savepicname='tmp.png'):
    '''how signal change until steady state

    input:
        flip: e.g., 1/*np.pi
        TR: (ms)
        T1: (ms)
        T2: (ms)
        df: (Hz) off-resonance
        nTR: number of TR
        alternating_rf_phase:
        spoiling_gradient_case: what spoiling want to use
        spoiling_gradient: e.g., np.pi (if has)
        spoiling_rf: 
        spoiling_rf_phase: e.g., np.pi (if has)
    output:
        Mx:
        My:
        Mz:
    '''
    # T1 = 600 # (ms)
    # T2 = 100 # (ms)
    # df = 0 # (Hz)
    # flip = 60/180*np.pi
    # TR = 500 #(ms)
    # 
    Mag = np.array([0,0,1])
    Mx = np.zeros(nTR)
    My = np.zeros(nTR)
    Mz = np.zeros(nTR)
    # simulate 10 TR
    flipdir = 1
    for tr_i in range(nTR):
        # excitaion
        if alternating_rf_phase:
            flipdir = flipdir*(-1) # alternating RF phase
        R = rotation_matrix_z2x(flip*flipdir)
        Mag = np.matmul(R,Mag)
        if spoiling_rf=='linear':
            # rf_phase = (TR_i+1)*(TR_i+2)/2*spoiling_rf_phase
            rf_phase = (tr_i+1)*spoiling_rf_phase
            R = rotation_matrix_x2y(rf_phase)
            Mag = np.matmul(R,Mag)
        elif spoiling_rf=='quadratic':
            rf_phase = (tr_i+1)*(tr_i+2)/2*spoiling_rf_phase
            # rf_phase = (TR_i+1)*spoiling_rf_phase
            R = rotation_matrix_x2y(rf_phase)
            Mag = np.matmul(R,Mag)
        else:
            # 'none'
            pass

        # free precession
        for t in np.linspace(0,TR,100):
            # print(t)
            M_fp = free_precession(Mag,t,T1,T2,df)


            Mx[tr_i] = M_fp[0]
            My[tr_i] = M_fp[1]
            Mz[tr_i] = M_fp[2]
        
        # spoiling
        if spoiling_gradient_case=='none':
            pass
        elif spoiling_gradient_case=='perfect-gradient-spoiling':
            M_fp[0] = 0
            M_fp[1] = 0
        elif spoiling_gradient_case=='gradient-spoiling':
            Rspoiling = rotation_matrix_x2y(spoiling_gradient)
            Mfp = np.matmul(Rspoiling,M_fp)
        else:
            print('no this spoiling method !')
            print('Error !')
            exit(1)

        Mag = M_fp
    # plot the signal
    if savepicname!='none':
        fig,axs = plt.subplots()
        plt.plot(Mx,label='x')
        plt.plot(My,label='y')
        plt.plot(Mz,label='z')
        plt.plot(np.sqrt(Mx**2+My**2),label='|Mxy|')
        plt.legend()
        plt.xlabel('#TR')
        # plt.show()
        plt.savefig('tmp.png')
        plt.close(fig)
    return Mx,My,Mz

def SSFP_complex_steadystate_mag(flip,TR,TE,T1,T2,df,nTR=10,
    alternating_rf_phase=False,
    spoiling_gradient_case='none',
    spoiling_gradient=0*np.pi,
    spoiling_rf='none',
    spoiling_rf_phase=0*np.pi):
    '''simulate the steady state magnetization with sequence of RF

    input:
        flip: (rad)
        TR: (ms)
        TE: (ms)
        T1: (ms)
        T2: (ms)
        df: (Hz) off-resonance effect
        nTR: number of TR to get steady-state
        alternating_rf_phase:
        spoiling_gradient_case: what kind of spoiling added
        spoiling_gradient: how large the spoilng gradient is  (if has)
    output:
        Mag: (1*3) magnetization at TE
    '''
    Mag = np.array([0,0,1])

    # nTR = 10
    # simulate nTR response, it should get the steady state
    flipdir = 1
    for TR_i in range(nTR):
        # assume the flip is from z to x
        if alternating_rf_phase:
            flipdir = (-1)*flipdir # alternating the RF phase
        R = rotation_matrix_z2x(flip*flipdir)
        Mag = np.matmul(R,Mag)
        if spoiling_rf=='linear':
            # rf_phase = (TR_i+1)*(TR_i+2)/2*spoiling_rf_phase
            rf_phase = (TR_i+1)*spoiling_rf_phase
            R = rotation_matrix_x2y(rf_phase)
            Mag = np.matmul(R,Mag)
        elif spoiling_rf=='quadratic':
            rf_phase = (TR_i+1)*(TR_i+2)/2*spoiling_rf_phase
            # rf_phase = (TR_i+1)*spoiling_rf_phase
            R = rotation_matrix_x2y(rf_phase)
            Mag = np.matmul(R,Mag)
        else:
            # 'none'
            pass
        

        # free precession
        Mfp = free_precession(Mag,TR,T1,T2,df)

        # spoiling
        if spoiling_gradient_case=='perfect-gradient-spoiling':
            # perfect spoiling eliminate the transverse magnetization
            Mfp[0] = 0
            Mfp[1] = 0
        elif spoiling_gradient_case=='none':
            pass
        elif spoiling_gradient_case=='gradient-spoiling':
            Rspoiling = rotation_matrix_x2y(spoiling_gradient)
            Mfp = np.matmul(Rspoiling,Mfp)
        else:
            print('no this spoiling method !')
            print('Error !')
            exit(1)

        Mag = Mfp
    # then in next TR give the signal at TE
    if alternating_rf_phase:
        flipdir = flipdir*(-1) # alternating the RF phase
    R = rotation_matrix_z2x(flip*flipdir)
    Mag = np.matmul(R,Mag)
    if spoiling_rf=='linear':
        # rf_phase = (TR_i+1)*(TR_i+2)/2*spoiling_rf_phase
        rf_phase = (TR_i+1)*spoiling_rf_phase
        R = rotation_matrix_x2y(rf_phase)
        Mag = np.matmul(R,Mag)
    elif spoiling_rf=='quadratic':
        rf_phase = (TR_i+1)*(TR_i+2)/2*spoiling_rf_phase
        # rf_phase = (TR_i+1)*spoiling_rf_phase
        R = rotation_matrix_x2y(rf_phase)
        Mag = np.matmul(R,Mag)
    else:
        # 'none'
        pass
    Mag = free_precession(Mag,TE,T1,T2,df)
    return Mag



def bSSFP_steadystate_mag(flip,TR,TE,T1,T2,df,nTR=10):
    '''simulate the steady state magnetization with sequence of RF

    input:
        flip: (rad)
        TR: (ms)
        TE: (ms)
        T1: (ms)
        T2: (ms)
        df: (Hz) off-resonance effect
        nTR: number of TR to get steady-state
    output:
        Mag: (1*3) magnetization at TE
    '''
    Mag = np.array([0,0,1])

    # nTR = 10
    # simulate nTR response, it should get the steady state
    flipdir = -1
    for TR_i in range(nTR):
        # assume the flip is from z to x
        flipdir = (-1)*flipdir # alternating the RF phase
        R = rotation_matrix_z2x(flip*flipdir)
        Mag = np.matmul(R,Mag)

        # free precession
        Mfp = free_precession(Mag,TR,T1,T2,df)
        Mag = Mfp
    # then in next TR give the signal at TE
    flipdir = (-1)*flipdir
    R = rotation_matrix_z2x(flip*flipdir)
    Mag = np.matmul(R,Mag)
    Mag = free_precession(Mag,TE,T1,T2,df)
    return Mag


def bSSFP_frequency_response(flip,TR,TE,T1,T2,offres_range,nTR=10):
    '''frequence response of bSSFF
    
    input: 
        flip: 
        TR: (ms)
        TE: (ms)
        T1: (ms)
        T2: (ms)
        offres_range: (1*nf)
    output:
        Mend: (nf*3) 
    '''
    nf = offres_range.size
    M_response = np.zeros((nf,3))
    for k in range(nf):
        df = offres_range[k]
        M = bSSFP_steadystate_mag(flip,TR,TE,T1,T2,df,nTR)
        # M = bSSFP_gradientspoiling_steadystate_mag(flip,TR,TE,T1,T2,df,nTR,
        #                                            spoiling_gradient_case='none')
        M_response[k,:] = M
    return M_response






# =============================================================
# a more complicated bloch simulation function

def bloch_sim_np(M_init, Beff, Nt, dt, T1, T2):
    """Bloch simulation for one spin (in numpy)

    input:
        M_init: (1*3) initial magnetization
        Beff: (3:Nt)(mT)
        Nt: number of timepoints in simulation
        dt: (ms)
        T1: (ms)
        T2: (ms)
        df: (Hz) off-resonance effect
    output:
        M: (1*3) final magnetization
        M_hist: (3*Nt+1) 
    """
    M_hist = np.zeros((3,Nt+1))
    # M[:,0] = torch.tensor([1.,0.,0.],device=device)
    M = M_init
    M_hist[:,0] = M

    E1 = np.exp(-dt/T1)
    E2 = np.exp(-dt/T2)
    E = np.array([[E2,0.,0.],
                [0.,E2,0.],
                [0.,0.,E1]])
    e = np.array([0.,0.,1-E1])
    # print(E)
    # print(e)

    # Beff_hist = torch.zeros((3,Nt),device=device)*1.0
    # Beff_hist[0:2,:] = rf*spin.kappa # mT
    # Beff_hist[2,:] = spin.get_loc()@gr*1e-2 + spin.df/spin.gamma*1e-3 # mT/cm*cm = mT, Hz/(MHz/T)=1/1e6*1e3mT=1e-3*mT
    # # print('Beff_hist:',Beff_hist.shape)

    for t in range(Nt):
        # at time point t
        Beff_t = Beff[:,t]

        # get rotation axis
        Beff_norm = np.linalg.norm(Beff_t,2)
        if Beff_norm == 0:
            Beff_unit = np.zeros(3)
        else:
            Beff_unit = Beff_t/np.linalg.norm(Beff_t,2)

        # the rotation ----------
        phi = -(2*np.pi*GAMMA)*Beff_norm*dt # Caution: what is the sign here!> 2*pi*MHz/T*mT*ms=rad
        R1 = np.cos(phi)*np.eye(3) + (1-np.cos(phi))*np.outer(Beff_unit,Beff_unit)
        # print(phi)
        # print(R1)

        # the relaxation ----------
        # M_temp = R1@M_hist[:,k] + torch.sin(phi)*torch.cross(Beff_unit,M_hist[:,k])
        # print(M_temp)
        # M_hist[:,k+1] = E@M_temp + e
        M = np.matmul(R1,M) + np.sin(phi)*np.cross(Beff_unit,M)
        M = np.matmul(E,M) + e
        M_hist[:,t+1] = M

        # if k%50==0:
        # 	print('k =',k)
        # 	print(M.shape)
        # 	print(M.norm())

    return M, M_hist


def simulate_rf(rf,dt,df,T1,T2, M_init = np.array([0,0,1]),rephasing_gradient=False):
    '''simulate the result magnetization of rf pulse
    (only constant off-resonance effect is considered)

    input:
        rf: (1*Nt) (mT)
        dt: (ms)
        df: (Hz)
        T1: (ms)
        T2: (ms)
    output:
        M: (1*3) end magnetization
        M_hist: 
    '''
    Nt = len(rf)

    # M_init = np.array([0,0,1])
    Mag = M_init

    if False:
        print('simulation | dt: {} ms | max rf: {} mT | dur: {} ms | df: {} Hz |'.format(dt,
                np.max(np.abs(rf)),Nt*dt, df))

    # print('Nt:',Nt, 'rf', rf.shape)

    # define a simulation function here
    if False:
        for t in range(Nt):
            # each dt
            b1 = rf[t]
            # compute the rotation given by the rf pulse
            flip = dt*b1*GAMMA*2*np.pi  # ms*mT*MHz/T = Hz
            R_b1 = rotation_matrix_z2x(flip)
            rf_phase = np.angle(b1)
            R_rfphase = rotation_matrix_x2y(rf_phase)
            # 
            Mag = np.matmul(R_b1,Mag)
            Mag = np.matmul(R_rfphase,Mag)

            # the rotation given by the off-resonance
            # phi = df*dt*1e-3*2*np.pi  # Hz*ms
            # R_off = rotation_matrix_x2y(phi)
            # Mag = np.matmul(R_off,Mag)

            # compute the relaxation given by T1 and T2, and df
            Mag = free_precession(Mag,dur=dt,T1=T1,T2=T2,df=df)
        M_end = Mag


    # OR try different function for simulation
    # compute the effective B-field
    if True:
        Beff = np.zeros((3,Nt))  #(mT)
        Beff[0,:] = np.real(rf)
        Beff[1,:] = np.imag(rf)
        Beff[2,:] = df/GAMMA*1e-3  #(mT)

        # simulation
        M_end,M_hist_1 = bloch_sim_np(M_init,Beff,Nt,dt,T1,T2)
    
    if rephasing_gradient == True:
        #  additional rephasing gradient
        Beff = np.zeros((3,Nt))  #(mT)
        Beff[2,:] = -(df/2)/GAMMA*1e-3  #(mT) # here use this to equivalently represent rephasing

        M_end,M_hist_2 = bloch_sim_np(M_end,Beff,Nt,dt,T1,T2)
        M_hist = np.concatenate((M_hist_1,M_hist_2[:,1:]),axis=1)
    else:
        M_hist = M_hist_1

    return M_end,M_hist


def test_function():
	print('test of Phy.py')
	return


if __name__=='__main__':
    pass

    # test free precession
    # -------------------------
    # tlist = np.arange(1000) # (ms)
    # Mx = []
    # My = []
    # Mz = []
    # for t in tlist:
    # 	Mend = free_precession(np.array([1,0,0]),t,T1=600,T2=100,df=10)
    # 	Mx.append(Mend[0])
    # 	My.append(Mend[1])
    # 	Mz.append(Mend[2])
    # plt.figure()
    # plt.plot(tlist,Mx)
    # plt.plot(tlist,My)
    # plt.plot(tlist,Mz)
    # plt.show()

    # test
    # --------------------------
    # x = np.zeros((3,4))
    # y = np.linspace(1,4,4)
    # print(x)
    # print(y)
    # x[0,:] = y
    # print(x)
