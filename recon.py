import numpy as np
import matplotlib.pyplot as plt

def fft_recon(kdata):
    '''direct fourier transform reconstruction
    '''
    x = np.fft.fftshift(np.fft.ifft2(kdata))
    return x

def fft_recon_multicoil(kdata):
    '''direct Fourier reconstruction, considering multiple coils
    
    kdata: (_*_*ncoils)
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
        sampling_matr: indicate the sampled entries (elements 0,1)
        phase_estimate_matr: indicate the data for estimating phase (elements 0,1)
    '''
    # recon the low-resol image, take the center k-space data
    # and estimate the phase
    kdata_center = kdata*phase_estimate_matr
    img_lowres = fft_recon(kdata_center)
    phase_esti = np.angle(img_lowres)

    # iteratively project and recon the image
    img = fft_recon(kdata) # Recon with zero-filled
    niters = 10
    for iter_i in range(niters):
        print('[POCS iteration = {}]'.format(iter_i))
        # Substitute the image phase
        img = np.abs(img)*np.exp(1j*phase_esti)
        kdata_esti = np.fft.fft2(np.fft.fftshift(img))

        # Replace the true kspace data
        kdata_esti = kdata + kdata_esti*(1-sampling_matr)

        # New recon
        img = fft_recon(kdata_esti)
    return img

def coil_combination(imgs,coilmaps):
    '''coil combination for the assignment

    imgs: (_*_*ncoils)
    coilmaps: (_*_*ncoils)
    '''
    ncoils = imgs.shape[2]
    img = 0
    for coil_itr in range(ncoils):
        img = img + np.conj(coilmaps[:,:,coil_itr])*imgs[:,:,coil_itr]
    return img

def SENSE_recon(kdatas,coilmaps,R:int):
    '''SENSE reconstruction method

    kdatas: (ny*nx*ncoils)
    R: acceleration factor
    '''
    print('[SNESE recon]')
    # assume the acceleration is in the 1st dimension
    # print(kdatas.shape)
    ny,nx,ncoils = kdatas.shape
    imgs_alised = fft_recon_multicoil(kdatas)
    for coil_itr in range(ncoils):
        imgs_alised[:,:,coil_itr] = np.fft.fftshift(imgs_alised[:,:,coil_itr],axes=0)
    # print(imgs_alised.shape)
    img_unalised = np.zeros((R,ny,nx),dtype=imgs_alised.dtype)
    # print(img_unalised.shape)

    # Recon images pixel-by-pixel
    for y in range(ny):
        for x in range(nx):
            px_alised = imgs_alised[y,x,:]
            C = np.zeros((ncoils,R),dtype=imgs_alised.dtype)
            for r in range(R):
                loc_in_coil = (y+r*ny)%(R*ny)
                C[:,r] = coilmaps[loc_in_coil,x,:]
            invC = np.linalg.pinv(C)
            px_unalised = np.matmul(invC,px_alised)
            img_unalised[:,y,x] = px_unalised

    # Compose the full FOV images
    img = img_unalised[0]
    for k in range(R-1):
        img = np.vstack((img,img_unalised[k+1]))
    return img


def plot_img(x,title=None,picname='xxx.png',savefig=False):
    fig,ax = plt.subplots()
    # x = np.transpose(x)
    im = ax.imshow(x,cmap='gray')
    fig.colorbar(im)
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
        ax.set_title('coil {}'.format(img_itr+1))
        img_itr = img_itr + 1
        if img_itr == nimgs:
            break

    if savefig:
        plt.savefig(picname)
        plt.close(fig)
    else:
        plt.show()
    return