import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

def calculate_gate_stats(predictions, poses):
    # display averages
    mean_pred = np.mean(predictions, axis=0)
    mean_pose = np.mean(poses, axis=0)
    print('Means (prediction, GT) : R({} , {}) Psi({} , {}) Phi_rel({} , {})'.format( # took theta out of print statement
        mean_pred[0], mean_pose[0], mean_pred[1], mean_pose[1], mean_pred[2], mean_pose[2])) # took out mean_pred[3] as will be length 3 not 4
    # display mean absolute error
    abs_diff = np.abs(predictions-poses)
    mae = np.mean(abs_diff, axis=0)
    mae[1:] = mae[1:] * 180/np.pi
    print('MAE : R({}) Psi({}) Phi_rel({})'.format(mae[0], mae[1], mae[2])) # took out theta and mae[3]
    # display standard deviation of error
    std = np.std(abs_diff, axis=0) / np.sqrt(abs_diff.shape[0])
    std[1:] = std[1:] * 180 / np.pi
    print('Standard error: R({}) Psi({}) Phi_rel({})'.format(std[0], std[1], std[2])) # took out theta and std[3]
    # display max errors
    max_diff = np.max(abs_diff, axis=0)
    print('Max error : R({}) Psi({}) Phi_rel({})'.format(max_diff[0], max_diff[1], max_diff[2])) # took out theta and max_diff[3]

    fig, axs = plt.subplots(1, 3, tight_layout=True)
    weights = np.ones(len(abs_diff[:, 0]))/len(abs_diff[:, 0])

    axs[0].hist(abs_diff[:, 0], bins=30, range=(0,2.0), weights=weights, density=False)
    #axs[1].hist(abs_diff[:, 1]*180/np.pi, bins=30, range=(0, np.pi/20*180/np.pi), weights=weights, density=False)
    axs[1].hist(abs_diff[:, 1]*180/np.pi, bins=30, range=(0, np.pi/20*180/np.pi), weights=weights, density=False)
    axs[2].hist(abs_diff[:, 2]*180/np.pi, bins=50, range=(0, np.pi/4*180/np.pi), weights=weights, density=False) # took out line for theta and adjusted indexing

    for idx in range(3):
        axs[idx].yaxis.set_major_formatter(PercentFormatter(xmax=1))

    axs[0].set_title(r'$r$')
    #axs[1].set_title(r'$\theta$')
    axs[1].set_title(r'$\phi$')
    axs[2].set_title(r'$\psi$')

    axs[0].set_xlabel('[m]')
    #axs[1].set_xlabel(r'[deg]')
    axs[1].set_xlabel(r'[deg]')
    axs[2].set_xlabel(r'[deg]')
    # axs[1].set_xlabel(r'[$^{\circ}$]')
    # axs[2].set_xlabel(r'[$^{\circ}$]')
    # axs[3].set_xlabel(r'[$^{\circ}$]')

    axs[0].set_ylabel('Error Density')

    fig.savefig(os.path.join('/home/campus.ncl.ac.uk/b3024896/Projects/gym-donkeytrack/logs/cmvae/run_1/', 'gate_stats_error_histograms.png'))

    plt.show()

    # fig, axs = plt.subplots(1, 4, tight_layout=True)
    # N, bins, patches = axs[0].hist(abs_diff[:, 0], bins=100, range=(0,3), density=True)

    # plt.title("R MAE histogram")
    # _ = plt.hist(abs_diff[:, 0], np.linspace(0.0, 10.0, num=1000))
    # plt.show()
    # plt.title("Theta MAE histogram")
    # _ = plt.hist(abs_diff[:, 1], np.linspace(0.0, np.pi, num=1000))
    # plt.show()
    # plt.title("Phi MAE histogram")
    # _ = plt.hist(abs_diff[:, 2], np.linspace(0.0, np.pi, num=1000))
    # plt.show()
    # plt.title("Phi_rel MAE histogram")
    # _ = plt.hist(abs_diff[:, 3], np.linspace(0.0, np.pi, num=100))
    # plt.show()


def calculate_v_stats(predictions, v_gt):
    # display averages
    mean_pred = np.mean(predictions, axis=0)
    mean_v = np.mean(v_gt, axis=0)
    print('Means (prediction, GT) : R({} , {}) Theta({} , {}) Psi({} , {}) Phi_rel({} , {})'.format(
        mean_pred[0], mean_v[0], mean_pred[1], mean_v[1], mean_pred[2], mean_v[2], mean_pred[3], mean_v[3]))
    # display mean absolute error
    abs_diff = np.abs(predictions-v_gt)
    mae = np.mean(abs_diff, axis=0)
    print('Absolute errors : Vx({}) Vy({}) Vz({}) Vyaw({})'.format(mae[0], mae[1], mae[2], mae[3]))
    # display max errors
    max_diff = np.max(abs_diff, axis=0)
    print('Max error : Vx({}) Vy({}) Vz({}) Vyaw({})'.format(max_diff[0], max_diff[1], max_diff[2], max_diff[3]))
    plt.title("Vx Absolute Error histogram")
    _ = plt.hist(abs_diff[:, 0], np.linspace(0.0, 10.0, num=1000))
    plt.show()
    plt.title("Vy Absolute Error histogram")
    _ = plt.hist(abs_diff[:, 1], np.linspace(0.0, 3, num=1000))
    plt.show()
    plt.title("Vz Absolute Error histogram")
    _ = plt.hist(abs_diff[:, 2], np.linspace(0.0, 3, num=1000))
    plt.show()
    plt.title("Vyaw Absolute Error histogram")
    _ = plt.hist(abs_diff[:, 3], np.linspace(0.0, 3, num=1000))
    plt.show()
