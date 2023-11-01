# simulation functions
# jiayao



import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable



def epg_state_transfer_matrix(flip,phase):
    alpha = np.pi*flip/180
    phi = np.pi*phase/180
    R = np.zeros((3,3))*1j
    R[0,0] = np.cos(alpha/2)**2
    R[0][1] = np.exp(2*1j*phi)*((np.sin(alpha/2))**2)
    R[0][2] = -1j*np.exp(1j*phi)*np.sin(alpha)
    R[1][0] = np.exp(-2*1j*phi)*(np.sin(alpha/2)**2)
    R[1][1] = np.cos(alpha/2)**2
    R[1][2] = 1j*np.exp(-1j*phi)*np.sin(alpha)
    R[2][0] = -1j/2*np.exp(-1j*phi)*np.sin(alpha)
    R[2][1] = 1j/2*np.exp(1j*phi)*np.sin(alpha)
    R[2][2] = np.cos(alpha)
    return R

class ExtendedPhaseGraph:
    def __init__(self,dt=0.1,t_total=2,T1=np.array([1000]),T2=np.array([100]),
                 relaxation=False,diffusion=False,
                 unit_gradient_shift_time=1) -> None:
        '''
        dt: (ms)
        t_total: (ms)
        T1: (ms)
        T2: (ms)
        '''
        self.num = len(T1)
        self.dt = dt
        self.Nt = int(t_total/dt)
        print(self.Nt)
        if self.Nt > 5e4:
            print('[Nt = {}] too large'.format(self.Nt))
            print('[Error ! dt too small !]')
            return
        self.unit_gradient_shift_time = unit_gradient_shift_time #(ms)
        self.unit_state_len = int(self.unit_gradient_shift_time/dt)
        if np.abs(self.unit_state_len*dt - self.unit_gradient_shift_time)<1e-10:
            pass
        else:
            print('[Error !]')
            return
        self.relaxation = relaxation
        self.diffusion = diffusion
        self.T1 = T1
        self.T2 = T2
        # Intialize some members
        self.echo_states = np.zeros((self.num,self.Nt+1))*1j # for detecting echoes, save F0 states
        # self.states_idx = [0] # refers to the positive states
        # self.states_idx = [0]
        # self.states_coeff = np.array([[0],[0],[1]],dtype=np.cdouble)
        self.states_coeff_pos = np.zeros((self.num,self.Nt+1))*1j # [0,1,2,3,]
        self.states_coeff_neg = np.zeros((self.num,self.Nt+1))*1j # [0,1,2,3,]
        self.states_coeff_z = np.zeros((self.num,self.Nt+1))*1j
        self.states_coeff_z[:,0] = 1  # [0,1,2,3]
        # when the rf happens
        self.rf_events_flip = []
        self.rf_events_phase = []
        self.rf_events_time = []
        # some important simulation times
        self.key_times = []
        self.efficient_times = []
        self.efficient_echo_states = []
    # def initialize(self):
    #     '''initialization'''
    #     return
    def _state_reset(self):
        '''reset all the state, but not clean the saved rf events'''
        print('\n[states reset!]')
        self.states_coeff_pos = np.zeros((self.num,self.Nt+1))*1j
        self.states_coeff_neg = np.zeros((self.num,self.Nt+1))*1j
        self.states_coeff_z = np.zeros((self.num,self.Nt+1))*1j
        self.states_coeff_z[:,0] = 1
    def _states_transfer(self,rfflip,rfphase):
        '''the coefficients of state transfer to each other because of the rf rotation
        '''
        # print('rf:',rfflip,rfphase)
        # print(np.sum(np.abs(self.states_coeff_pos)>0)+np.sum(np.abs(self.states_coeff_neg)>0))
        # print()
        R = epg_state_transfer_matrix(flip=rfflip,phase=rfphase)
        # print((R.tolist()))
        if True:
            for n in range(self.Nt+1):
                state_n = np.zeros((3,self.num))*1j  # !! need to be complex
                state_n[0,:] = self.states_coeff_pos[:,n]
                state_n[1,:] = self.states_coeff_neg[:,n]
                state_n[2,:] = self.states_coeff_z[:,n]
                newstate = np.matmul(R,state_n)
                self.states_coeff_pos[:,n] = newstate[0,:]
                self.states_coeff_neg[:,n] = newstate[1,:]
                self.states_coeff_z[:,n] = newstate[2,:]
                # if np.linalg.norm(state_n) > 0:
                # if np.abs(state_n[0,:])+np.abs(state_n[1,:]) > 0:
                #     print('state {} ='.format(n),state_n,'new state:',newstate)
                # print('state {} ='.format(n),state_n,'new state:',newstate)
        # try to do state stranfer by RF together for all T1,T2
        if False:
            state_n = np.vstack([self.states_coeff_pos.reshape(-1),
                                 self.states_coeff_neg.reshape(-1),
                                 self.states_coeff_z.reshape(-1)])
            newstate = np.matmul(R,state_n)
            self.states_coeff_pos = newstate[0,:].reshape(self.num,self.Nt+1)
            self.states_coeff_neg = newstate[1,:].reshape(self.num,self.Nt+1)
            self.states_coeff_z = newstate[2,:].reshape(self.num,self.Nt+1)
        # print(R)
        # coeff = self.states_coeff
        # self.states_coeff = np.matmul(R,self.states_coeff)
    def _state_update(self,dt,count=1):
        '''
        dt: (ms) suppose to be the same the self.dt
        count: how many times update
        '''
        tdur = dt*count
        cc = count - 1
        # update the states
        self.states_coeff_pos[:,count:] = self.states_coeff_pos[:,:-count]
        self.states_coeff_pos[:,:count] = np.flip(np.conj(self.states_coeff_neg[:,1:(count+1)]),axis=1)
        self.states_coeff_neg[:,:-count] = self.states_coeff_neg[:,count:]
        self.states_coeff_neg[:,-count:] = 0
        # nothing need to do with Z-state, they reserve
        # self.states_coeff_z[:,1:] = self.states_coeff_z[:,:-1]
        # self.states_coeff_z[:,0] = 0  # TODO not sure about this
        # consider the relaxation, assume initial magnetization is 1
        if self.relaxation:
            E1 = np.exp(-tdur/self.T1).reshape(-1,1)
            E2 = np.exp(-tdur/self.T2).reshape(-1,1)
            self.states_coeff_pos = self.states_coeff_pos*E2
            self.states_coeff_neg = self.states_coeff_neg*E2
            self.states_coeff_z[:,0] = (self.states_coeff_z[:,0]-1)*E1.reshape(-1) + 1
            self.states_coeff_z[:,1:] = self.states_coeff_z[:,1:]*E1
        # consider the diffusion
    def _get_current_state_coeff(self,kth):
        '''get the full states coefficient attempt for EPG plot'''
        sig = np.hstack((np.flip(self.states_coeff_neg[kth,1:]),self.states_coeff_pos[kth]))
        return sig
    def _check_current_rf_event(self,t):
        '''check if there is a rf rotation in at current simulation time'''
        find_events = []
        for k in range(len(self.rf_events_time)):
            if self.rf_events_time[k] > t-self.dt/2:
                if self.rf_events_time[k] <= t+self.dt/2:
                    find_events.append(k)
        return find_events
    def _check_close_to_event(self,t,dist=1):
        '''chech if simulation time is close to some events'''
        close_events =False
        for k in range(len(self.rf_events_time)):
            # print(self.rf_events_time[k],end=',')
            if self.rf_events_time[k]>t-(dist+0.5)*self.dt:
                if self.rf_events_time[k] <= t+(dist+0.5)*self.dt:
                    close_events = True
                    # print(self.rf_events_time[k])
        for k in range(len(self.key_times)):
            if self.key_times[k]>t-(dist+0.5)*self.dt:
                if self.key_times[k] <= t+(dist+0.5)*self.dt:
                    close_events = True
        # print()
        return close_events
    def get_all_states(self):
        return self.states_coeff_pos,self.states_coeff_neg,self.states_coeff_z
    def set_all_states(self,states_coeff_pos,states_coeff_neg,states_coeff_z):
        self.states_coeff_pos = states_coeff_pos
        self.states_coeff_neg = states_coeff_neg
        self.states_coeff_z = states_coeff_z
        print('[manually set states !]')
    def detect_echo(self,t,terr=1):
        '''function that detects when echo happens'''
        # cautious about first excitation which can cause the appearance of self.echo_state
        # consider the errors in computation, find echo within an error range
        tidx = int(t/self.dt)
        check_idx = np.arange(2*terr+1)-terr+tidx
        check_idx = check_idx[np.nonzero(check_idx>0)]
        sig = self.echo_states[:,check_idx]
        # print(sig.shape)
        idx = np.argmax(np.abs(sig),axis=1)
        echo_dec = [sig[k,idx[k]] for k in range(self.num)]
        echo_dec = np.array(echo_dec)
        # echo_dec = sig[:,idx]
        # print(echo_dec.shape)
        return echo_dec
    def add_rf(self,flip,phase,t):
        '''
        t: (ms) the time to apply rf pulse
        flip: deg
        phase: deg
        '''
        self.rf_events_flip.append(flip)
        self.rf_events_phase.append(phase)
        self.rf_events_time.append(t)
        return
    def clean_all_events(self):
        print('[clean rf events !]')
        self.rf_events_flip = []
        self.rf_events_phase = []
        self.rf_events_time = []
        print('[clean all key times !]')
        self.key_times = []
    def add_key_times(self,times):
        for t in times:
            if t in self.key_times:
                pass
            else:
                self.key_times.append(t)
        return
    def simulate(self,efficient=False,reset=True):
        '''simulation of epg, the graph at different timesteps
        '''
        # first reset the state coefficient to avoids errors
        if reset:
            self._state_reset()
        # print a little info
        print('[ simulation ! Nt = {}]'.format(self.Nt))
        Nt = self.Nt
        dt = self.dt        
        # signalmatrix = np.zeros((2*Nt+1,Nt+1))*1j
    
        # check if there is a rf at time 0
        rf_events = self._check_current_rf_event(0)
        if len(rf_events) > 0:
            if len(rf_events) > 1:
                print('[Error !] dt too large')
            else:
                idx = rf_events[0]
                flip,phase = self.rf_events_flip[idx],self.rf_events_phase[idx]
                self._states_transfer(rfflip=flip,rfphase=phase)
        # signalmatrix[:,0] = self._get_current_state_coeff()

        # then start simulation for later events
        simu_t_total = 0
        accumulation = 1
        for t in range(Nt):
            if efficient==False:
                _ = print('[   t={} ...]'.format(t)) if t%20==0 else None
                simu_t_total = simu_t_total + dt

                # then the dephasing because of the constant gradient
                self._state_update(dt)

                # print(np.sum(np.abs(self.states_coeff_pos)>0))
                
                # if there happens a rf event at the end of the time dt
                rf_events = self._check_current_rf_event(simu_t_total)
                if len(rf_events) > 0:
                    if len(rf_events) > 1:
                        print('[Error !] dt too large')
                        return
                    else:
                        idx = rf_events[0]
                        flip,phase = self.rf_events_flip[idx],self.rf_events_phase[idx]
                        # the state coefficient change because of the rf
                        self._states_transfer(rfflip=flip,rfphase=phase)

                # add current results to save
                # signalmatrix[:,t+1] = self._get_current_state_coeff()
                self.echo_states[:,t+1] = self.states_coeff_pos[:,0]
            else:
                # perform simulation more efficient, not capable for epg graph plot
                # print('[simulation till {}ms]'.format(simu_t_total))
                simu_t_total = simu_t_total + dt
                accumulation = accumulation + 1
                rf_events = self._check_current_rf_event(simu_t_total)
                close_to_event = self._check_close_to_event(simu_t_total)
                if len(rf_events) > 1:
                    print('[Error !] dt too large')
                    return
                if close_to_event:
                    # perform simulation
                    print('[   t={} ...] [{:.2f}ms / {:.2f}ms]'.format(t,simu_t_total,self.Nt*self.dt))

                    # the dephasing because of the constant gradient
                    self._state_update(dt,count=accumulation)

                    # check if rf event happens
                    if len(rf_events) > 0:
                        idx = rf_events[0]
                        flip,phase = self.rf_events_flip[idx],self.rf_events_phase[idx]
                        self._states_transfer(rfflip=flip,rfphase=phase)

                    self.echo_states[:,t+1] = self.states_coeff_pos[:,0]
                    accumulation = 0
                else:
                    # print('[   t={} ...] skip'.format(t))
                    # print('accumulation: {}'.format(accumulation))

                    self.echo_states[:,t+1] = np.NaN

        return
    def get_epg_graph(self,kth=0,reset=True):
        '''simulation of epg, the graph at different timesteps
        
        output:
            epg_graph: 
        '''
        # first reset the state coefficient to avoids errors
        if reset:
            self._state_reset()
        # print a little info
        print('[ simulation ! Nt = {}]'.format(self.Nt))
        Nt = self.Nt
        dt = self.dt        
        epg_graph = np.zeros((2*Nt+1,Nt+1))*1j
    
        # check if there is a rf at time 0
        rf_events = self._check_current_rf_event(0)
        if len(rf_events) > 0:
            if len(rf_events) > 1:
                print('[Error !] dt too large')
                return
            else:
                idx = rf_events[0]
                flip,phase = self.rf_events_flip[idx],self.rf_events_phase[idx]
                self._states_transfer(rfflip=flip,rfphase=phase)
        epg_graph[:,0] = self._get_current_state_coeff(kth=kth)


        # print('state pos:',np.abs(self.states_coeff_pos[0,:10]))
        # print('state neg:',np.abs(self.states_coeff_neg[0,:10]))
        # print('state z:',np.abs(self.states_coeff_z[0,:10]))
        # self.states_coeff_neg[:,3]=1
        # print(np.nonzero(self.states_coeff_pos),np.nonzero(self.states_coeff_neg))
        # print(np.nonzero)

        # then start simulation for later events
        simu_t_total = 0
        for t in range(Nt):
            simu_t_total = simu_t_total + dt
            _ = print('[   t={} ...] [{:.2f}ms / {:.2f}ms]'.format(t,simu_t_total,self.Nt*self.dt)) if t%100==0 else None
            # print('\nt={} | {:.2f}ms'.format(t,simu_t_total))

            # then the dephasing because of the constant gradient
            self._state_update(dt)
            # print('state pos:',np.abs(self.states_coeff_pos[0,:10]))
            # print('state neg:',np.abs(self.states_coeff_neg[0,:10]))
            # print('state z:',np.abs(self.states_coeff_z[0,:10]))
            # print(np.sum(np.abs(self.states_coeff_pos)>0) + np.sum(np.abs(self.states_coeff_neg[:,1:])>0))
            # print(np.nonzero(self.states_coeff_pos),np.nonzero(self.states_coeff_neg))
            
            # if there happens a rf event at the end of the time dt
            rf_events = self._check_current_rf_event(simu_t_total)
            if len(rf_events) > 0:
                if len(rf_events) > 1:
                    print('[Error !] dt too large')
                else:
                    # print('[rf event]')
                    idx = rf_events[0]
                    flip,phase = self.rf_events_flip[idx],self.rf_events_phase[idx]
                    # the state coefficient change because of the rf
                    self._states_transfer(rfflip=flip,rfphase=phase)

            # add current results to save
            epg_graph[:,t+1] = self._get_current_state_coeff(kth=kth)
            self.echo_states[:,t+1] = self.states_coeff_pos[:,0]

        return epg_graph
    def plot_epg_graph(self,kth=0,state_lim=None,starttime=0,dur=None,reset=True,picname='tmp.png',savefig=False):
        '''dur:(ms)'''
        signalmatrix = self.get_epg_graph(kth,reset=reset)
        signalmatrix = np.abs(signalmatrix)
        if state_lim != None:
            # try to cut some parts of the graph, make it more clear
            n = int(self.Nt*state_lim)
            signalmatrix = signalmatrix[self.Nt-n:self.Nt+n,:]
        nstart = int(starttime/self.dt)
        signalmatrix = signalmatrix[:,nstart:]
        if dur!=None:
            n = int(dur/self.dt)
            signalmatrix = signalmatrix[:,:n]
        signalmatrix = np.flip(signalmatrix,0)
        # signalmatrix = np.random.rand(50,50)

        # signalmatrix = signalmatrix[::2,:]
        Nt = signalmatrix.shape[1]

        # plot
        fig,ax = plt.subplots(figsize=(10,5))
        im = plt.imshow(signalmatrix,cmap='gray')
        xaxis_n = np.arange(5)/5*Nt
        xlabels = ['{:.2f}'.format(k*self.dt) for k in xaxis_n]
        ax.set_xticks(xaxis_n)
        ax.set_xticklabels(xlabels)
        ax.set_yticks([])
        ax.set_aspect(1)
        plt.xlabel('time (ms)')
        plt.ylabel('state')
        plt.title('T1 = {} ms, T2 = {} ms'.format(self.T1[kth],self.T2[kth]))
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right",
                                    size="10%",
                                    pad=0.1)
        plt.colorbar(im,cax=cax)
        if savefig:
            plt.savefig(picname)
            plt.close(fig)
        else:
            plt.show()
        return
    def plot_echo_states(self,kth=0,efficient=False,reset=True,starttime=0,dur=None,picname='tmp.png',savefig=False):
        '''plot of echoes and thier amplitudes'''
        # get echo states
        self.simulate(efficient=efficient,reset=reset)
        echo_states = self.echo_states[kth,:]
        nstart = int(starttime/self.dt)
        echo_states = echo_states[nstart:]
        if dur!=None:
            n = int(dur/self.dt)
            echo_states = echo_states[:n]
        # plot
        fig,ax = plt.subplots(figsize=(8,4))
        ax.stem(np.abs(echo_states),markerfmt='.')
        # ax.plot(np.abs(echo_states),marker='.',linewidth=1)
        Nt = len(echo_states)
        xaxis_n = np.arange(5)/5*Nt
        xlabels = ['{:.2f}'.format(k*self.dt) for k in xaxis_n]
        ax.set_xticks(xaxis_n)
        ax.set_xticklabels(xlabels)
        plt.xlabel('time (ms)')
        plt.ylabel('echo amplitude')
        plt.title('T1 = {} ms, T2 = {} ms'.format(self.T1[kth],self.T2[kth]))
        if savefig:
            plt.savefig(picname)
            plt.close(fig)
        else:
            plt.show()
        return
    def show_info(self):
        print(''.center(50,'-')+'\nExtended phase graph')
        print('  simulation time={} ms | T1,T2 pairs={} | #time={}'.format(self.Nt*self.dt,self.num,self.Nt))
        # print('  T1 = {} ms'.format(self.T1))
        # print('  T2 = {} ms'.format(self.T2))
        print('  relaxation={} | diffusion={}'.format(self.relaxation,self.diffusion))
        # print('  states: {} |'.format(len(self.states)),self.states)
        # print(self.states_coeff)
        print(''.center(50,'-'))
        return




if __name__=='__main__':
    pass

    epg = ExtendedPhaseGraph(dt=0.1,t_total=10,relaxation=True)
    epg.add_rf(flip=90,phase=0,t=0)
    alpha = 120
    flip1 = alpha + 90 - alpha/2
    epg.add_rf(flip=flip1,phase=90,t=1)
    epg.add_rf(flip=alpha,phase=90,t=3)
    epg.add_rf(flip=alpha,phase=90,t=5)
    epg.add_rf(flip=alpha,phase=90,t=7)
    epg.add_rf(flip=alpha,phase=90,t=9)
    epg.add_rf(flip=alpha,phase=90,t=11)
    epg.add_rf(flip=alpha,phase=90,t=13)
    epg.show_info()
    print(epg.rf_events_time)
    epg.add_key_times([2,6,8])

    # timesteps = np.arange(10)*0.1 # (ms)
    # epg.simulate(efficient=True)
    epg.plot_epg_graph(state_lim=0.5,savefig=True,picname='tmp-2.png')

    # R = epg_state_transfer_matrix(10,10)
    epg.plot_echo_states(efficient=False,savefig=True)
    # print(R)
