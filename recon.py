import numpy as np
import matplotlib.pyplot as plt

def fft_recon(kdata):
    '''direct fourier transform reconstruction
    '''
    x = np.fft.fftshift(np.fft.ifft2(kdata))
    return x

def fft_recon_multicoil(kdata):
    '''direct Fourier reconstruction, considering multiple coils
    
    kdata: (nx*ny*ncoils)
    '''
    ncoils = kdata.shape[2]
    imgs = np.zeros_like(kdata)
    for coil_itr in range(ncoils):
        imgs[:,:,coil_itr] = fft_recon(kdata[:,:,coil_itr])
    return imgs


def POCS_recon(kdata,sampling_matr,phase_estimate_matr):
    '''POCS method of reconstruction

    input:
        kdata: zero-filled kspace data
        sampling_matr: indicate the sampled entries
        phase_estimate_matr: indicate the data for estimating phase
    '''
    # recon the low-resol image, take the center k-space data
    # and estimate the phase
    kdata_center = kdata*phase_estimate_matr
    img_lowres = fft_recon(kdata_center)
    phase_esti = np.angle(img_lowres)

    # iteratively project and recon the image
    niters = 10
    for iter_i in range(niters):
        pass

    return

def SENSE_recon(kdatas,coilmaps):
    '''SENSE reconstruction method
    '''
    return


def plot_img(x,title=None,picname='xxx.png',savefig=False):
    fig,ax = plt.subplots()
    x = np.transpose(x)
    ax.imshow(x,cmap='gray')
    if title!=None:
        ax.set_title(title)
    if savefig:
        plt.savefig(picname)
        plt.close(fig)
    else:
        plt.show()
    return

def plot_imgs(x,title=None,picname='xxx.png',savefig=False):
    nx,ny,nimgs = x.shape
    
    # nrow = int(np.ceil(np.sqrt(nimgs)))
    # ncol = nrow
    # ncol = ncol + 1 if nrow*ncol < nimgs else ncol
    
    nrow = 2
    ncol = 4
        
    fig, axs = plt.subplots(nrow, ncol, figsize=(10, 6), layout='constrained')
    img_itr = 0
    for ax in axs.flat:
        # ax.set_title(f'markevery={markevery}')
        ax.imshow(x[:,:,img_itr],cmap='gray')
        img_itr = img_itr + 1
        if img_itr == nimgs:
            break

    if savefig:
        plt.savefig(picname)
        plt.close(fig)
    else:
        plt.show()
    return