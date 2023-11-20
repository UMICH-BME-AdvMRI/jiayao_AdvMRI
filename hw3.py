# bme 599 hw3
# author: jiayao

import numpy as np
import mat73
import scipy.io as spio
import matplotlib.pyplot as plt
import sys
sys.path.append('/scratch/Projects/um_bme599_hws/')

import recon

if __name__=='__main__':
    # problem 1 - a
    # -----------------------------------------------------------------
    if True:
        print('\n\nProblem 1:')
        data = mat73.loadmat('hw3/Data_Assignment3_Problem1.mat')
        print(data.keys())
        kdata = data['kspaceData_SingleCoil']
        ny,nx = kdata.shape


        recon.plot_img(np.abs(kdata),title='fully sampled kspace',picname='hw3/p1-kspace-mag.png',savefig=True)
        recon.plot_img(np.angle(kdata),title='fully sampled kspace',picname='hw3/p1-kspace-phase.png',savefig=True)

        img_true = recon.fft_recon(kdata)
        recon.plot_img(np.abs(img_true),title='fully sampled image (mag)',picname='hw3/p1-fully-sampled-image.png',savefig=True)
        recon.plot_img(np.angle(img_true),title='fully sampled image (phase)',picname='hw3/p1-fully-sampled-image-phase.png',savefig=True)


        # Retrospectively undersample this data using a partial Fourier factor of 5/8
        ny_sampled = int(np.ceil(ny*5/8))
        sample_matr = np.zeros((ny,nx))
        sample_matr[:ny_sampled,:] = 1
        kdata_pf = np.zeros_like(kdata)
        kdata_pf = kdata_pf + kdata*sample_matr
        # kdata_pf[:,:ny_sampled] = kdata[:,:ny_sampled]
        # recon.plot_img(np.abs(kdata_pf>0),savefig=True)
        center_sample_matr = np.zeros((ny,nx))
        center_sample_matr[(ny-ny_sampled):ny_sampled,:] = 1
        recon.plot_img(np.abs(kdata_pf),picname='hw3/p1-partialfourier-kspace-mag.png',savefig=True)
        recon.plot_img(np.angle(kdata_pf),picname='hw3/p1-partialfourier-kspace-phase.png',savefig=True)


        # Zero-filled reconstruction
        img_zerofill = recon.fft_recon(kdata_pf)
        recon.plot_img(np.abs(img_zerofill),title='zero-filled image (mag)',picname='hw3/p1-zero-filled-image.png',savefig=True)
        recon.plot_img(np.angle(img_zerofill),title='zero-filled image (phase)',picname='hw3/p1-zero-filled-image-phase.png',savefig=True)



        # Difference between images
        img_diff = img_true - img_zerofill
        recon.plot_img(np.abs(img_diff),title='zero-filled vs. fully sampled (mag)',picname='hw3/p1-partial-fourier-difference-image.png',savefig=True)
        recon.plot_img(np.angle(img_diff),title='zero-filled vs. fully sampled (phase)',picname='hw3/p1-partial-fourier-difference-image-phase.png',savefig=True)



        # Problem 1 - b
        # -----------------------------------------------------------------
        # Conjugate phase reconstruction
        img_pocs = recon.POCS_recon(kdata_pf,
                                    sampling_matr=sample_matr,
                                    phase_estimate_matr=center_sample_matr)
        recon.plot_img(np.abs(img_pocs),title='POCS image (mag)',picname='hw3/p1-pocs-image.png',savefig=True)
        recon.plot_img(np.angle(img_pocs),title='POCS image (phase)',picname='hw3/p1-pocs-image-phase.png',savefig=True)

        # Compute the difference
        img_diff = img_true - img_pocs
        recon.plot_img(np.abs(img_diff),title='POCS vs. fully sampled (mag)',picname='hw3/p1-pocs-difference-image.png',savefig=True)
        recon.plot_img(np.angle(img_diff),title='POCS vs. fully sampled (phase)',picname='hw3/p1-pocs-difference-image-phase.png',savefig=True)



    # Problem 2 
    # -----------------------------------------------------------------
    if True:
        print('\n\nProblem 2:')
        # Load the data
        data = mat73.loadmat('hw3/Data_Assignment3_Problem2.mat')
        # print(data.keys())
        coilmaps = data['coilmaps']
        kdata = data['kspaceData']
        print(coilmaps.shape,kdata.shape)

        recon.plot_imgs(np.abs(coilmaps),picname='hw3/p2-coilmaps.png',savefig=True)

        # Question a: Fully sampled images
        imgs_true = recon.fft_recon_multicoil(kdata)
        recon.plot_imgs(np.abs(imgs_true),picname='hw3/p2-fully-sampled-mag.png',savefig=True)
        recon.plot_imgs(np.angle(imgs_true),picname='hw3/p2-fully-sampled-phase.png',savefig=True)
        img_true_comb = recon.coil_combination(imgs_true,coilmaps)
        recon.plot_img(np.abs(img_true_comb),title='fully sampled (mag)',picname='hw3/p2-fully-sampled-coilcombined-mag.png',savefig=True)
        recon.plot_img(np.angle(img_true_comb),title='fully sampled (phase)',picname='hw3/p2-fully-sampled-coilcombined-phase.png',savefig=True)


        # Question b: Undersampled images R=2
        # Undersampling
        ny,nx,ncoils = kdata.shape
        kdata_undersampled = kdata[::2,:]
        # Recon of aliased images
        imgs_undersampled = recon.fft_recon_multicoil(kdata_undersampled)
        recon.plot_imgs(np.abs(imgs_undersampled),picname='hw3/p2-R=2-under-sampled-mag.png',savefig=True)
        recon.plot_imgs(np.angle(imgs_undersampled),picname='hw3/p2-R=2-under-sampled-phase.png',savefig=True)


        # Question c: R=2 SENSE reconed images
        img_R2 = recon.SENSE_recon(kdata_undersampled,coilmaps,R=2)
        recon.plot_img(np.abs(img_R2),title='SENSE(R=2) (mag)',picname='hw3/p2-R=2-sense-mag.png',savefig=True)

        img_diff = img_true - img_R2
        recon.plot_img(np.abs(img_diff),title='SENSE(R=2) vs. fully sampled (mag)',picname='hw3/p2-R=2-difference-image.png',savefig=True)
        recon.plot_img(np.angle(img_diff),title='SENSE(R=2) vs. fully sampled (phase)',picname='hw3/p2-R=2-difference-image-phase.png',savefig=True)


        # # Question d: R=4 SENSE reconed images
        # Compute the undersampled kspace data
        kdata_undersampled = kdata[::4,:]
        # Recon of aliased images
        imgs_undersampled = recon.fft_recon_multicoil(kdata_undersampled)
        recon.plot_imgs(np.abs(imgs_undersampled),picname='hw3/p2-R=4-under-sampled-mag.png',savefig=True)
        recon.plot_imgs(np.angle(imgs_undersampled),picname='hw3/p2-R=4-under-sampled-phase.png',savefig=True)
        # SENSE recon
        img_R4 = recon.SENSE_recon(kdata_undersampled,coilmaps,R=4)
        recon.plot_img(np.abs(img_R4),title='SENSE(R=4) (mag)',picname='hw3/p2-R=4-sense-mag.png',savefig=True)

        img_diff = img_true - img_R4
        recon.plot_img(np.abs(img_diff),title='SENSE(R=4) vs. fully sampled (mag)',picname='hw3/p2-R=4-difference-image.png',savefig=True)
        recon.plot_img(np.angle(img_diff),title='SENSE(R=4) vs. fully sampled (phase)',picname='hw3/p2-R=4-difference-image-phase.png',savefig=True)