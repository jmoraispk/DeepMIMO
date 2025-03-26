import os
import time
import pickle
import scipy
import numpy as np
import pandas as pd
# import pandas as pd
# from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# import seaborn as sns
import xgboost as xgb
import multiprocessing

import testing_distances

import DeepMIMOv3

from tqdm import tqdm

from sklearn.model_selection import GridSearchCV # for model search
# from sklearn.model_selection import train_test_split 
# Classification metrics
# from sklearn.metrics import accuracy_score, balanced_accuracy_score, top_k_accuracy_score
# Regression metrics (requires at least version 1.4)
# from sklearn.metrics import root_mean_squared_error, mean_absolute_percentage_error, max_error
# XGBoost expects classes starting at 0
from sklearn.preprocessing import LabelEncoder


from sklearn.cluster import KMeans
from scipy.stats import wasserstein_distance, wasserstein_distance_nd
from sklearn.metrics import silhouette_score


def plot_coverage(rxs, cov_map, dpi=300, figsize=(10,10), cbar_title=None, title=False,
                  scat_sz=20, tx_pos=None, tx_ori=None, legend=False, lims=None,
                  proj_3D=False, equal_aspect=False, tight=True, cmap='tab20'):
    
    plt_params = {'cmap': cmap}
    if lims:
        plt_params['vmin'], plt_params['vmax'] = lims[0], lims[1]
    
    n = 3 if proj_3D else 2 # n coordinates to consider 2 = xy | 3 = xyz
    
    xyz = {'x': rxs[:,0], 'y': rxs[:,1]}
    if proj_3D:
        xyz['zs'] = rxs[:,2]
        
    fig, ax = plt.subplots(dpi=dpi, figsize=figsize,
                           subplot_kw={'projection': '3d'} if proj_3D else {})
    
    im = plt.scatter(**xyz, c=cov_map, s=scat_sz, marker='s', **plt_params)

    # cbar = plt.colorbar(im, label='' if not cbar_title else cbar_title)
    
    # plt.xlabel('x (m)')
    # plt.ylabel('y (m)')
    
    # TX position
    if tx_pos is not None:
        ax.scatter(*tx_pos[:n], marker='P', c='r', label='TX')
    
    # TX orientation
    if tx_ori is not None and tx_pos is not None: # ori = [azi, el]
        # positive azimuths point left (like positive angles in a unit circle)    
        # positive elevations point up
        r = 30 # ref size of pointing direction
        tx_lookat = np.copy(tx_pos)
        tx_lookat[:2] += r * np.array([np.cos(tx_ori[2]), np.sin(tx_ori[2])]) # azimuth
        tx_lookat[2] += r * np.sin(tx_ori[1]) # elevation
        
        line_components = [[tx_pos[i], tx_lookat[i]] for i in range(n)]
        line = {key:val for key,val in zip(['xs', 'ys', 'zs'], line_components)}
        if n == 2:
            ax.plot(line_components[0], line_components[1], c='k', alpha=1, zorder=3)
        else:
            ax.plot(**line, c='k', alpha=1, zorder=3)
        # TODO: find arguments to plot the line in 3D
        # TODO: maintain scale in 3D plot
        
    # if title:
        # ax.set_title(title)
    # plt.axis('off')
    ax.set_xticklabels([])  # This hides x-axis tick labels
    ax.set_yticklabels([])  # This hides y-axis tick labels
    
    if legend:
        plt.legend(loc='upper center', ncols=10, framealpha=.5)
    
    if tight:
        s = 1
        mins, maxs = np.min(rxs, axis=0)-s, np.max(rxs, axis=0)+s
        # TODO: change to set (simpler and to change space)
        if not proj_3D:
            plt.xlim([mins[0], maxs[0]])
            plt.ylim([mins[1], maxs[1]])
        else:
            ax.axes.set_xlim3d([mins[0], maxs[0]])
            ax.axes.set_ylim3d([mins[1], maxs[1]])
            if tx_pos is None:
                ax.axes.set_zlim3d([mins[2], maxs[2]])
            else:
                ax.axes.set_zlim3d([np.min([mins[2], tx_pos[2]]),
                                    np.max([mins[2], tx_pos[2]])])
    
    if equal_aspect and not proj_3D: # disrups the plot
        plt.axis('scaled')
    
    
    return fig, ax #, cbar


def select_by_idx(dataset, idxs):
    
    dataset_t = [] # trimmed
    for bs_idx in range(len(dataset)):
        dataset_t.append({})
        for key in dataset[bs_idx].keys():
            dataset_t[bs_idx]['location'] = dataset[bs_idx]['location']
            dataset_t[bs_idx]['user'] = {}
            for key in dataset[bs_idx]['user']:
                dataset_t[bs_idx]['user'][key] = dataset[bs_idx]['user'][key][idxs]
        
    return dataset_t


# TODO: add info field to dataset, and make these params func defaults
def uniform_sampling(sampling_div, n_rows, users_per_row):

    cols = np.arange(users_per_row, step=sampling_div[0])
    rows = np.arange(n_rows, step=sampling_div[1])
    uniform_idxs = np.array([j + i*users_per_row for i in rows for j in cols])
    
    return uniform_idxs


def steering_vec(array, phi=0, theta=0, kd=np.pi):
    # phi = azimuth
    # theta = elevation
    idxs = DeepMIMOv3.ant_indices(array)
    resp = DeepMIMOv3.array_response(idxs, phi, theta+np.pi/2, kd)
    return resp / np.linalg.norm(resp)


def get_bins(hist_min, hist_max, binwidth):
    return np.arange(hist_min, hist_max + binwidth, binwidth)


def get_bins_from_feature(feature):
    return get_bins(feature['xlim'][0], feature['xlim'][1], feature['binwidth'])


def search_best_XGBoost(x_train, y_train, verbose=False):
    xgb_model = xgb.XGBClassifier(n_jobs=multiprocessing.cpu_count() // 2, tree_method="hist")
    
    # Model Search
    clf = GridSearchCV(xgb_model, {"max_depth": [4, 6, 8], "n_estimators": [50, 100]}, verbose=1, n_jobs=2)
    
    t = time.time()
    clf.fit(x_train, y_train)
    if verbose:
        print(f'Search time = {time.time()-t:.1f}s')
        print(f'Classifier Accuracy: {clf.best_score_:.2f}') # default accuracy
        print(f'Best parameters: {clf.best_params_}')
    return clf.best_estimator_


def get_idxs_good_classes(data, min_instances=5):
    """ Finds the idxs of data to keep, i.e. all the data with more than min_instances of its class"""
    
    uniques, counts = np.unique(data, return_counts=True)
    
    uniques_to_remove = [val for idx,val in enumerate(uniques) if counts[idx] < min_instances]
    
    # mask will be False when we have to remove one 
    mask_to_keep = np.ones_like(data, dtype=bool)
    for unique in uniques_to_remove: 
        mask_to_keep = np.logical_and(mask_to_keep, data != unique)
    
    return np.where(mask_to_keep)[0]


def get_feat_weights(model, importance_type='gain', normalize=True, plot=False, feature_labels=None):
    
    if importance_type in ['gain', ]:
        weights = model.get_booster().get_score(importance_type=importance_type)
    else:
        raise NotImplementedError
        # TODO: SHAPLEY AND RANDOM PERMUTATION HERE!
    
    if normalize:
        norm_val = sum([weights[key] for key in weights.keys()])
        for key in weights.keys():
            weights[key] /= norm_val
    
    if feature_labels and len(weights.keys()) != len(feature_labels):
        new_weights = {}
        for feat_idx in range(len(feature_labels)):
            feat_name = f'f{feat_idx}'
            new_weights[feat_name] = weights[feat_name] if feat_name in weights.keys() else 0
        weights = new_weights
    
    if plot:
        keys = list(weights.keys())
        values = list(weights.values())
        
        plt.figure(dpi=200)
        plt.barh(feature_labels if feature_labels else keys, values)
        plt.grid()
        plt.title(f'Feature weights (importance_type = {importance_type})')
        plt.show()
    
    return weights


def dataset_distance(datasetA_idxs, datasetB_idxs, full_features,                 
                     uniform_dist_name='Wasserstein', model_approach='joint'):
    # Assumption 1: the sets of indices A and B will index the positions and features
    # Assumption 2: the first feature in full_features is the labels (outputs to predict)
    
    n_feat = len(full_features)-1
    feat_labels = [full_features[i]['xlabel'] for i in range(1,n_feat+1)]
    
    # Compute distances between feature distributions
    uniform_distance_func = testing_distances.get_dist_funcs(uniform_dist_name)
    feature_distances = np.zeros(n_feat)
    for feat_idx, feature in enumerate(full_features[1:]):
        
        bins = get_bins_from_feature(feature)
        dist = uniform_distance_func(feature['data'][datasetA_idxs], 
                                     feature['data'][datasetB_idxs], bins)
        feature_distances[feat_idx] = dist
    
    X = np.vstack(tuple(full_features[feat_idx]['data'] for feat_idx in range(1,n_feat+1))).T
    y = full_features[0]['data'] # best beam
    
    le = LabelEncoder()
    if 'return' in model_approach:
        models = []
    
    if 'separate' in model_approach or 'independent' in model_approach:
        acc_weights = np.zeros(n_feat)
        
        # Select indices for the datasets
        for idxs in [datasetA_idxs, datasetB_idxs]:
            y2 = y[idxs].astype(int)
            
            # filter classes with too little elements to avoid k-fold warning
            idxs_to_keep = get_idxs_good_classes(y2, min_instances=5)
            
            X3 = X[idxs,:][idxs_to_keep,:] # this is not the same as logical_and
            y3 = y2[idxs_to_keep]
            
            # class adaptation: class numbers must start from 0
            y3_encoded = le.fit_transform(y3)
            
            best_model = search_best_XGBoost(X3, y3_encoded)
            
            if 'return' in model_approach:
                models.append(best_model)
            
            feat_weights = get_feat_weights(best_model, importance_type='gain', 
                                            plot=True, feature_labels=feat_labels)
            
            acc_weights += np.array(list(feat_weights.values()))
        
        weights = acc_weights / 2
    elif 'joint' in model_approach:
        joint_idxs = np.unique(np.hstack((datasetA_idxs, datasetB_idxs)))
        y2 = y[joint_idxs].astype(int)
        
        # filter classes with too little elements to avoid k-fold warning
        idxs_to_keep = get_idxs_good_classes(y2, min_instances=5)
        
        X3 = X[joint_idxs,:][idxs_to_keep,:]
        y3 = y2[idxs_to_keep]
        
        # class adaptation: class numbers must start from 0
        y3_encoded = le.fit_transform(y3)
        
        best_model = search_best_XGBoost(X3, y3_encoded)
        
        if 'return' in model_approach:
            models.append(best_model)
            
        feat_weights = get_feat_weights(best_model, importance_type='gain', 
                                        plot=True, feature_labels=feat_labels)
        weights = np.array(list(feat_weights.values()))
    elif 'uniform' in model_approach:
        weights = np.ones(n_feat) / n_feat # equal, uniform weights (same as computing the mean)
    else:
        print(f'Invalid model approach {model_approach}.')
        return
    print(weights)
    final_distance = np.dot(weights, feature_distances)
    
    return (final_distance, models) if 'return' in model_approach else final_distance
    

class Area():
    def __init__(self, idxs=None, name='', center=''):
        # idxs inside the area
        self.idxs = idxs
        self.name = name
        self.center = center
    
    def __repr__(self):
        s =  f'name = {self.name}\n'
        s += f'center = {self.center}\n'
        s += f'Number of idxs = {len(self.idxs)}\n'
        s += f'idxs = {self.idxs}'
        return s


class Rectangle(Area):
    def __init__(self, name, w, h, x, y, n_idxs=None, data=None):
        self.name = name
        self.w = w
        self.h = h
        self.cx = x
        self.cy = y
        self.corners = (x - w/2, x + w/2, y - h/2, y + h/2)
        
        self.n_idxs = n_idxs
        
        if data is not None:
            self.set_indices_no_nans(data)
        
    def get_corners(self):
        return self.corners
    
    def set_indices(self, idxs):
        self.idxs = idxs
        self.n_idxs = len(idxs)
    
    def set_indices_no_nans(self, data):
        
        idxs_with_nans = np.where(data['user']['LoS'] == -1)[0] 
        
        idx_set = get_idxs_in_xy_box(data['user']['location'], 
                                     *self.get_corners(), 
                                     only_non_nan=True,
                                     idxs_with_nans=idxs_with_nans)
        
        self.set_indices(idx_set)
    
    
    def __repr__(self):
        s = f'w,h = ({self.h}, {self.w}) | x,y = ({self.cx}, {self.cy})'
        a = '' if self.n_idxs is None else f' | n_idxs = {self.n_idxs}'
        return s + a


def get_idxs_in_xy_box(data_pos, x_min, x_max, y_min, y_max, only_non_nan=False, idxs_with_nans=None):

    idxs_x = np.where((x_min < data_pos[:, 0]) & (data_pos[:, 0] < x_max))[0]
    idxs_y = np.where((y_min < data_pos[:, 1]) & (data_pos[:, 1] < y_max))[0]
    
    idx_intersection = np.array(list(set(idxs_x).intersection(idxs_y)))
    
    if only_non_nan:
        if idxs_with_nans is None:
            raise('Insufficient information to determine idxs without NaNs.')
        idxs_final = np.array([i for i in idx_intersection if i not in idxs_with_nans])
    else:
        idxs_final = idx_intersection 
        
    return sorted(idxs_final)


def display_areas(dataset_to_display, under_map, areas_idxs, m=2):
    all_pos = dataset_to_display['user']['location']
    plot_coverage(all_pos, under_map, tx_pos=dataset_to_display['location'], dpi=300,
                  title= 'Best Beams', cbar_title='Best beam index', scat_sz=.06*m)
    
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
              'tab:brown', 'tab:pink', 'tab:grey', 'tab:olive', 'tab:cyan']
        
    for i, rect_idxs in enumerate(areas_idxs):
        plt.scatter(all_pos[rect_idxs,0], all_pos[rect_idxs,1], s=.01*m, c=colors[i])


def train_xgb(X_train, y_train):
    model = xgb.XGBClassifier(n_jobs=multiprocessing.cpu_count() // 2, 
                               tree_method="hist", eval_metric='rmse',
                               max_depth=8, n_estimators=100)
    
    model.fit(X_train, y_train)
    return model


def check_datasets_equal_labels(labels, sets, verbose=True):
    
    all_unique = True
    first_unique = np.unique(labels[sets[1]])
    for _, datast in sets.items():
        unique_labels = np.unique(labels[datast])
        if verbose:
            print(unique_labels)
        if np.array_equal(unique_labels, first_unique):
            continue
        else:
            all_unique = False
    
    return all_unique


def plot_dist_perf(x, y1, y2, x_label='', y1_label='', y2_label='', title=None, 
                   plots=['double_axis', 'scatter']):
    
    corr_val = np.round(np.corrcoef(y1,y2)[0,1],2)
    print(f'Correlation {y1_label} with {y2_label} = {corr_val}')
    
    x_vals = np.arange(len(x))
    color1, color2 = 'tab:blue', 'tab:orange'
    if 'double_axis' in plots:
        fig, ax1 = plt.subplots(dpi=300)
        ax2 = ax1.twinx()
        ax1.plot(x_vals, y1, color1)
        ax2.plot(x_vals, y2, color2)
        ax1.set_xlabel(x_label)
        ax1.set_xticks(x_vals)
        ax1.set_xticklabels([f'{x_val:.0f}' for x_val in list(x)])
        ax1.set_ylabel(y1_label, color=color1)
        ax2.set_ylabel(y2_label, color=color2)
        plt.grid()
        if title:
            plt.title(title)
        plt.show()
    
    if 'scatter' in plots:
        plt.figure(dpi=200)
        plt.scatter(y1, y2, label='points')
        m,b = np.polyfit(y1, y2, 1)
        plt.plot(y1, b + m*y1, color2, label='linear fit \n'+fr'$\eta$ = {corr_val}')
        plt.xlabel(y1_label)
        plt.ylabel(y2_label)
        plt.grid()
        plt.legend()
        if title:
            plt.title(title)
        plt.show()


def nmse(A, B):
    return (np.linalg.norm(A - B, 'fro') / np.linalg.norm(A, 'fro'))**2



def plot_CIR(deepMIMO_dataset, user_idx):
    paths = deepMIMO_dataset['user']['paths'][user_idx]
    
    plt.figure(dpi=200)
    plt.stem(paths['ToA']*6, paths['power'])
    plt.xlabel('Time of arrival [us]')
    plt.ylabel('Power per path [W]')
    plt.yscale('log')
    plt.grid()
    
    

def convert_channel_angle_delay(channel):
    """
    Requires a channel where:
        - The last dimension (-1) is subcarriers (frequency)
        - The second from last dimension (-2) is antennas (space)
        - Returns a channel like: ... x angle x delay 
    """
    # Inside FFT Conversion to Angle Domain: fft + shift across antennas (axis = 2)
    # Outside IFFT Conversion from Frequency (subcarriers) to Delay Domain (axis = 3)
    # Using singles leads to an error of 1e-14. The usual value of one channel entry is 1e-7. csingles are ok.
    return np.fft.ifft(np.fft.fftshift(np.fft.fft(channel, axis=-2), -2), axis=-1).astype(np.csingle)


def proportion(a):
    return np.cumsum(a) / a.sum()

def info_explained(a):
    cum_sum = np.cumsum(a**2)
    return cum_sum / cum_sum[-1] # -1 = np.sum(.)

def save_var(var, path):
    path_full = path if path.endswith('.p') else (path + '.pickle')    
    with open(path_full, 'wb') as handle:
        pickle.dump(var, handle)

    return

def load_var(path):
    path_full = path if path.endswith('.p') else (path + '.pickle')
    with open(path_full, 'rb') as handle:
        var = pickle.load(handle)
    
    return var


def train_val_test_split(n_users, train_val_test_split=[0.6, 0.2, 0.2], seed=None,
                         train_csv='train.csv', val_csv='val.csv', test_csv='test.csv'):
    
    data_idxs = np.arange(n_users)
    
    if seed:
        np.random.seed(seed)
        np.random.shuffle(data_idxs)
    
    train_val_sep = int(train_val_test_split[0] * n_users)
    val_test_sep  = int((train_val_test_split[0] + train_val_test_split[1]) * n_users)
    
    train_idx = data_idxs[0:train_val_sep]
    val_idx = data_idxs[train_val_sep:val_test_sep]
    test_idx = data_idxs[val_test_sep:]
    
    df1 = pd.DataFrame(train_idx, columns=["data_idx"])
    df2 = pd.DataFrame(val_idx, columns=["data_idx"])
    df3 = pd.DataFrame(test_idx, columns=["data_idx"])
    
    if train_val_test_split[0]:
        df1.to_csv(train_csv, index=False)
    if train_val_test_split[1]:
        df2.to_csv(val_csv, index=False)
    if train_val_test_split[2]:
        df3.to_csv(test_csv, index=False)
    
    return


def save_test_channels(chs, folder):
    
    os.makedirs(folder, exist_ok=True)
    
    scipy.io.savemat(folder + 'channel_ad_clip.mat', 
                     {'all_channel_ad_clip': np.swapaxes(chs, -1, -2)})
    
    train_val_test_split(chs.shape[0], train_val_test_split=[0, 0, 1],
                         test_csv=folder + 'test_data_idx.csv')
    return 'test_data_idx.csv'



def plot_ellipse(x, y, ax, n_std=2.0, facecolor='none', **kwargs):
    """
    Plot an ellipse based on the x and y data using their covariance matrix.
    `n_std` specifies the number of standard deviations the ellipse should represent.
    """
    cov = np.cov(x, y)
    mean_x, mean_y = np.mean(x), np.mean(y)

    # Eigenvalues and eigenvectors
    eig_vals, eig_vecs = np.linalg.eigh(cov)
    order = eig_vals.argsort()[::-1]
    eig_vals, eig_vecs = eig_vals[order], eig_vecs[:, order]
    angle = np.degrees(np.arctan2(*eig_vecs[:, 0][::-1]))

    # Width and height of the ellipse, scaled by the number of standard deviations
    width, height = 2 * n_std * np.sqrt(eig_vals)
    ellipse = Ellipse(xy=(mean_x, mean_y), width=width, height=height, angle=angle,
                      facecolor=facecolor, **kwargs)

    ax.add_patch(ellipse)
    return ax


def get_colors(n):
    colors1 = ["#4EACC5","#FF9C34", "#4E9A06", "#0033A0", "#FCD116", 
               # Light Blue, Vibrant Orange, Deep Green, Deep Blue, Yellow
               "#C8102E", "#6A1B9A", "#C6007E", "#008080", "#8BC34A"]
               # Crimson Red, Purple, Magenta, Teal, Lime Green
    colors2 = ['red', 'blue', 'green', 'purple', 'orange', 'yellow', 'cyan', 
               'magenta', 'lime', 'teal']
    all_colors = colors1 + colors2
    return all_colors[:n]


def minmax(a, return_minmax=False):
    a_min = np.nanmin(a)
    a_max = np.nanmax(a)
    a_minmax = (a - a_min) / (a_max - a_min)
    if return_minmax:
        return a_minmax, a_min, a_max
    else:
        return a_minmax

def revert_minmax(a_minmax, a_min, a_max):
    return a_minmax * (a_max - a_min) + a_min

def compute_DoA(channel, N, norm_ant_spacing=0.5, method_subcarriers='sum'):
    """
    expects the channel from one user in the form of "n_rx=1, n_tx, n_subcarriers"
    N = number of antenna elements
    norm_ant_spacing = element distance in wavelengths
    """
    # Create an array of bin indices
    n = np.arange(-N/2, N/2, 1)  # Adjusted for zero-centered FFT
    
    # Principle of DoA: the signal arriving from angle theta will be captured by each
    # antenna and have a constant phase difference across the elements. This phase difference
    # is proportional to the spacing between elements and the sin(angle of arrival). 
    # This phase shift will manifest itself as a (spatial) frequency when we take an FFT. 
    # Then we only need to see which bin (or frequency) has the most power and
    # convert that frequency to the angle of arrival. 
    
    # Calculate angles from bin indices
    theta = np.arcsin(n / (N * norm_ant_spacing))
    
    # Convert angles from radians to degrees
    theta_degrees = np.degrees(theta)
    
    # Assuming fft_results is your FFT output array
    if method_subcarriers == 'sum':
        f = np.sum
    elif method_subcarriers == 'mean':
        f = np.mean
    ch_ang = f(channel, axis=-1).squeeze()
    fft_results = np.fft.fftshift(np.fft.fft(ch_ang))
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(theta_degrees, np.abs(fft_results))
    plt.title('FFT Output vs. Angle')
    plt.xlabel('Angle (degrees)')
    plt.ylabel('Magnitude')
    plt.grid(True)
    plt.show()
    
    print(f'main direction of arrival = {theta_degrees[np.argmax(np.abs(fft_results))]:.2f}')



def plot_ang_delay(ch, n_ant=32, NC=32, title='', label_axis=True, bandwidth=50e6, spacing=.5):
    f, ax = plt.subplots(dpi=300)
    ax.imshow(np.squeeze(np.abs(ch))[:,:NC])
    # plt.imshow(np.squeeze(np.abs(ch2))[i][:,:50])
    #, extent=[0, 50, angles[0], angles[-1]]) # change limits!
    
    plt.title(title)
    plt.ylabel('angle bins')
    plt.xlabel('delay bins')
    
    if label_axis:
        # X-Axis
        n_xtickstep = 4
        plt.xlabel('delays bins [us]')
        delay_idxs = np.arange(NC)
        delay_labels = delay_idxs / (bandwidth) * 1e6
        ax.set_xticks(delay_idxs[::n_xtickstep])
        ax.set_xticklabels([f'{label:.1f}' for label in delay_labels[::n_xtickstep]])
        
        # Y-Axis
        n_ytickstep = 4
        plt.ylabel('angle bins [º]')
        # Create an array of bin indices
        n = np.arange(-n_ant/2, n_ant/2, 1)  # Adjusted for zero-centered FFT
        # Calculate angles from bin indices
        ang_degrees = np.degrees(np.arcsin(n / (n_ant * spacing)))
        ax.set_yticks(np.arange(n_ant)[::n_ytickstep])
        ax.set_yticklabels([f'{label:.0f}' for label in ang_degrees[::n_ytickstep]])
        
    plt.show()


def plot_areas(areas, all_pos, s=50, show=True):
    n_areas = len(areas)
    colors = plt.get_cmap('tab20', n_areas)  # Get 'tab20' colormap for n_areas distinct colors

    f = plt.figure(dpi=300, figsize=(10, 10))
    ax = f.add_subplot(111)
    for k in range(n_areas):
        cluster_center = areas[k].center
        idxs = areas[k].idxs
        # Use the colormap to get a unique color for each area
        area_color = colors(k / n_areas)
        plt.scatter(all_pos[idxs, 0], all_pos[idxs, 1], color=area_color, s=s)

        # Optional: Uncomment to display cluster centers and labels
        plt.plot(cluster_center[0], cluster_center[1], "o", 
                  markerfacecolor=area_color, markeredgecolor="k", markersize=6)
        plt.text(cluster_center[0]+5, cluster_center[1], f'{k}', fontdict={'fontsize':25},
                  bbox=dict(facecolor='white', alpha=0.3))

    # Set plot limits based on your data
    plt.ylim([np.min(all_pos[:, 1]), np.max(all_pos[:, 1])])
    plt.xlim([np.min(all_pos[:, 0]), np.max(all_pos[:, 0])])
    ax.set_xticklabels([])  # Hide x-axis tick labels
    ax.set_yticklabels([])  # Hide y-axis tick labels

    # Show plot if required
    if show:
        plt.show()

    return f, ax



def make_gif(folder, gif_name, empty_folder=False, fps=None):
    
    import imageio # for GIFs
    rel_names = sorted(os.listdir(folder))
    filenames = [os.path.join(folder, rel_name) for rel_name in rel_names]
    
    gif_params = {'mode':'I'}
    if fps:
        gif_params['fps'] = fps
        
    with imageio.get_writer(gif_name, **gif_params) as writer:
        for filename in filenames:
            image = imageio.v2.imread(filename)
            writer.append_data(image)
    
    print(f"GIF saved as {gif_name}")
    
    if empty_folder:
        # Remove temporary files
        for filename in filenames:
            os.remove(filename)

def draw_on_plot_and_get_coordinates(fig, ax):
    import tkinter as tk
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    class DrawingApp:
        def __init__(self, root, fig, ax):
            self.fig = fig
            self.ax = ax
            self.root = root
            self.drawing = False
            
            # Embed the plot in the tkinter canvas
            self.canvas = FigureCanvasTkAgg(self.fig, master=root)
            self.canvas.draw()
            self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
            
            self.coordinates = []
            
            # Bind mouse events for drawing
            self.canvas.mpl_connect("button_press_event", self.start_draw)
            self.canvas.mpl_connect("motion_notify_event", self.draw)
            
            self.start_x = None
            self.start_y = None
            
            self.root.bind("<d>", self.toggle_drawing)
            self.canvas.mpl_connect("motion_notify_event", self.draw)


        def start_draw(self, event):
            if event.inaxes and self.drawing:
                self.coordinates.append((event.xdata, event.ydata))
                
        def toggle_drawing(self, event):
            self.drawing = not self.drawing
            print(f"Drawing mode {'enabled' if self.drawing else 'disabled'}")
            
        def draw(self, event):
            if event.inaxes and self.drawing:
                self.ax.scatter(event.xdata, event.ydata, color='k', s=6)
                self.canvas.draw()
                self.coordinates.append((event.xdata, event.ydata))

        def get_coordinates(self):
            return np.array([*self.coordinates])

    root = tk.Tk()

    app = DrawingApp(root, fig, ax)
    root.mainloop()

    return app.get_coordinates()



def transform_coordinates(coords, lon_max, lon_min, lat_min, lat_max):
    lats = []
    lons = []
    x_min, y_min = np.min(coords, axis=0)[:2]
    x_max, y_max = np.max(coords, axis=0)[:2]
    for (x, y) in zip(coords[:,0], coords[:,1]):
        lons += [lon_min + ((x - x_min) / (x_max - x_min)) * (lon_max - lon_min)]
        lats += [lat_min + ((y - y_min) / (y_max - y_min)) * (lat_max - lat_min)]
    return lats, lons




def compute_wasserstein(cluster, data):
    return wasserstein_distance(cluster.ravel(), data.ravel())
    # return wasserstein_distance_nd(cluster, data)

# def compute_wasserstein(cluster, data):
    
#     cluster = np.array(cluster)
#     data = np.array(data)
#     dist = np.zeros(cluster.shape[1])
    
#     for i in range (cluster.shape[1]):
#         dist[i] = np.linalg.norm(wasserstein_distance(cluster[:,i].real,data[:,i].real)-
#                                  wasserstein_distance(cluster[:,i].imag,data[:,i].imag))
        
#     return np.mean(dist)

def silhouette(X):
    
    silhouette_scores = []
    for i in range(2, 5):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=100, n_init=10, random_state=0)
        kmeans.fit(X)
        score = silhouette_score(X, kmeans.labels_)
        silhouette_scores.append(score)
    
    optimal_clusters = np.argmax(silhouette_scores) + 2
    
    return optimal_clusters

def redistribute_samples(desired_samples, total_samples, distances):
    
    desired_samples = np.array(desired_samples)
    total_samples = np.array(total_samples)
    distances = np.array(distances)
    
    picked_samples = np.minimum(desired_samples, total_samples)

    deficit = desired_samples - picked_samples
    total_deficit = np.sum(deficit[deficit > 0])

    available_capacity = total_samples - picked_samples

    distance_weights = distances / np.sum(distances)
    additional_samples = (distance_weights * total_deficit).astype(int)

    additional_samples = np.minimum(additional_samples, available_capacity)
    picked_samples += additional_samples

    remaining_deficit = total_deficit - np.sum(additional_samples)
    while remaining_deficit > 0:
        max_dist_idx = np.argmax(available_capacity * distances)
        if available_capacity[max_dist_idx] > 0:
            picked_samples[max_dist_idx] += 1
            available_capacity[max_dist_idx] -= 1
            remaining_deficit -= 1
        else:
            break
        
    picked_samples = np.minimum(picked_samples,total_samples)
    
    return picked_samples



def clustering_general(chs_target, n_clusters):
    
    # k = silhouette(chs_target) # TODO: check silhouette in cuML
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(chs_target)
    labels = kmeans.labels_
    # centers = kmeans.cluster_centers_
    
    return labels

def clustering(target_dataset, ch_for_dist, n_clusters, plot=False):
    """ Requires ch_for_dist being a list of n_areas """
    
    # num_users = areas[target_dataset].idxs.shape[0]
    chs_target = ch_for_dist[target_dataset]
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(chs_target)
    labels = kmeans.labels_
    
    if plot:
        plt.figure(dpi=300)
        plt.scatter(chs_target[:, 0], chs_target[:, 1], 
                    c=labels, cmap='viridis', marker='o', 
                    label=f'zone {target_dataset}')
        plt.title('K-Means Clustering')
        plt.xlabel('PC 1')
        plt.ylabel('PC 2')
        plt.xlim([-7.5,12.5])
        plt.ylim([-8,17])
        plt.legend()
        plt.show()
    return labels

def cluster_based_data_selection(source_dataset, target_dataset, ch_for_dist, 
                                 n_clusters, retrain_ratio, plot=False):
    
    target_clusters = clustering(target_dataset,ch_for_dist, n_clusters)
    retrain_sample_num = int(retrain_ratio * len(target_clusters)) 
    chs_source = ch_for_dist[source_dataset]
    chs_target = ch_for_dist[target_dataset]
    
    dist = []
    for i in range(n_clusters):
        dist.append(compute_wasserstein(chs_target[target_clusters == i], chs_source))
        
    total_distance = np.sum(dist)
    
    # Select number of points from each cluster
    n = []
    for i in range(n_clusters):
        sample_percentage = dist[i]/total_distance
        n.append(int(sample_percentage * retrain_sample_num))
    
    # Attribute the remaining few points to meet our retraining quote from the largest cluster
    if np.sum(n) < retrain_sample_num:
        max_idx = np.argmax(n)
        n[max_idx] += retrain_sample_num - np.sum(n)
    
    actual_sizes = []
    for i in range(n_clusters):
        actual_sizes.append(len(target_clusters[target_clusters==i]))
        
    if any(x < 0 for x in np.array(actual_sizes)-np.array(n)):
        n = redistribute_samples(n, actual_sizes, dist)
           
    train_subset = np.array([])
    train_subset_v2 = []
    for i in range(n_clusters):
        idxs_from_cluster_i = np.where(target_clusters == i)[0].astype(int)
        selected_idxs = np.random.choice(idxs_from_cluster_i, n[i], replace=False)
        train_subset = np.hstack((train_subset, selected_idxs))
        train_subset_v2.append(selected_idxs)
    # train_subset_v2 = np.array(train_subset_v2)
    
    # len(target_clusters), train_subset
    if plot:
        plt.figure(dpi=300)
        for i in range(n_clusters):
            cluster_points = chs_target[target_clusters == i]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                        label=f'Cluster {i} (Dist: {dist[i]:.2f})', marker='o',
                        s=3, edgecolor='k', linewidth=.1, alpha = .5,zorder= 3)
            plt.scatter(chs_target[train_subset_v2[i], 0],
                        chs_target[train_subset_v2[i], 1], marker='o',
                        s=10, edgecolor='k', linewidth=.3, alpha = .5,zorder= 3, )
                        # label=f'selected from cluster {i}')
        
        plt.scatter(chs_source[:, 0], chs_source[:, 1], marker='x', color='k',
                    s=3, linewidth=.3, alpha = .2, zorder= 3, 
                    label='Source dataset')
            
        plt.title(f'Source: {source_dataset} | Target: {target_dataset}')
        plt.xlabel('PC 1')
        plt.ylabel('PC 2')
        plt.legend(markerscale=6, ncols=4, loc='upper center', framealpha=.7, fontsize=7,
                   handletextpad=0.8, handlelength=0.4, labelspacing=0.1)
        plt.grid()
        plt.show()
    
    return train_subset.astype(int)


def umap_distance(ch_encoded, method=1, n_clusters=3, labels=None):
    """
    
    Both the channels and the layers should be lists per area
    Parameters
    ----------
    ch_encoded : TYPE
        ch_encoded should be [encoded_ch_per_area]
        
    method : int, optional
        Spectrum from Wasserstein (1) to euclidean on avg. encoding (single centroid) (5)
    
        1- Wasserstein
        2- Single centroids (clustering only on target)
        3- Double centroids (clustering on target and source)
        5- Euclidean on avg. encoding
    n_clusters : TYPE, optional
        DESCRIPTION. The default is 3.

    Raises
    ------
    Exception
        DESCRIPTION.

    Returns
    -------
    dist : TYPE
        DESCRIPTION.

    """
    
    n_areas = len(ch_encoded)
    if method in [5,6]:
        
        unique_labels = np.unique(np.hstack(labels))
        all_dists = np.zeros((len(np.unique(np.hstack(labels))),n_areas,n_areas))
        
        for a_idx1 in range(n_areas): # source
            for a_idx2 in range(n_areas): # target
                
                for label_idx, unique_label in enumerate(unique_labels):  
        
                    a1_idxs = np.where(labels[a_idx1] == unique_label)[0] 
                    a2_idxs = np.where(labels[a_idx2] == unique_label)[0]
                    
                    if np.size(a1_idxs) == 0 or np.size(a2_idxs) == 0:
                        continue

                    if method == 6:
                        a1_chs = ch_encoded[a_idx1][a1_idxs]
                        a2_chs = ch_encoded[a_idx2][a2_idxs]
                        all_dists[label_idx,a_idx1,a_idx2] = compute_wasserstein(a1_chs, a2_chs)
                    else:
                        a1_chs = np.nanmean(ch_encoded[a_idx1][a1_idxs], axis=0)
                        a2_chs = np.nanmean(ch_encoded[a_idx2][a2_idxs], axis=0)
                        all_dists[label_idx,a_idx1,a_idx2] = np.linalg.norm(a1_chs - a2_chs)
    
    n_dim = ch_encoded[0].shape[-1]
    if method in [2,3]:
        cluster_means = np.zeros((n_areas, n_clusters, n_dim))
        for area_idx in range(n_areas):
            clustered_points = clustering(area_idx, ch_encoded, n_clusters) 
            
            for cluster_idx in range(n_clusters):
                idxs_to_avg = np.where(clustered_points == cluster_idx)[0]
                cluster_means[area_idx, cluster_idx, :] = np.mean(ch_encoded[area_idx][idxs_to_avg], axis=0)
    
    # if method in [5]:
    #     unique_labels = np.unique(np.hstack(labels))

    #     cluster_means = np.zeros((len(unique_labels), n_areas, n_clusters, n_dim))

    #     for area_idx in range(n_areas):
    #         for label_idx, unique_label in enumerate(unique_labels):    
    #             idxs = np.where(labels[area_idx] == unique_label)[0]
                
    #             if np.size(idxs) == 0:
    #                 cluster_means[label_idx, area_idx, :, :] = np.nan
    #                 continue
                    
    #             chs_to_cluster = ch_encoded[area_idx][idxs]
    #             clustered_points = clustering_general(chs_to_cluster, n_clusters) 
                
    #             for cluster_idx in range(n_clusters):
    #                 idxs_to_avg = np.where(clustered_points == cluster_idx)[0]
    #                 cluster_means[label_idx, area_idx, cluster_idx, :] = \
    #                     np.mean(chs_to_cluster[idxs_to_avg], axis=0)
    
    # flag = np.zeros((len(np.unique(np.hstack(labels))),n_areas,n_areas))
    # flag_2 = np.zeros((len(np.unique(np.hstack(labels))),n_areas,n_areas))
    dist = np.zeros((n_areas,n_areas))
    for a_idx1 in range(n_areas): # source
        for a_idx2 in range(n_areas): # target
            d_list = []
            # if a_idx1 == 2 and a_idx2 == 5:
            #     print('stop!')
            
            if method == 1: # Wasserstein between raw datasets
                for i in range(ch_encoded[0].shape[-1]):
                    d_list += [compute_wasserstein(ch_encoded[a_idx1][:,i],
                                                    ch_encoded[a_idx2][:,i])]
            
            if method == 2:
                # Method 2 : avg area 0 and compute n_clusters distances between this 
                #            and area 1 clusters, and avg
                avg_area1 = np.mean(cluster_means[a_idx1, :, :], axis=0)
                
                for i in range(n_clusters):
                    d_list += [np.linalg.norm(avg_area1 - cluster_means[a_idx2, i, :])]
            
            if method == 3:
                # Method 3 : Double centroid - compute n_clusters**2 distances and average across them
                for i in range(n_clusters):
                    for j in range(n_clusters):
                        d_list += [np.linalg.norm(cluster_means[a_idx1, i, :] - 
                                                  cluster_means[a_idx2, j, :])]
            
            if method == 4:
                # Method 4: euclidean distance between the centroids of a pair of areas
                d_list += [np.linalg.norm(np.mean(ch_encoded[a_idx1], axis=0) - 
                                          np.mean(ch_encoded[a_idx2], axis=0))]

            # for label_idx, unique_label in enumerate(unique_labels):
            if method == 5:
                # Method 5: label-aware: Avg. distance within each label-cluster
                # a1_chs = cluster_means[:, a_idx1, 0]
                # a2_chs = cluster_means[:, a_idx2, 0]
            
                # distance = np.nanmean(np.linalg.norm(a1_chs - a2_chs, axis=-1))
                
                # if np.isnan(distance):
                #     distance = 1000
                    
                # d_list += [distance]
                
                unique_labels = np.unique(np.hstack(labels))
                # print(unique_labels)
                distance = []
                cnt = 1
                for label_idx, unique_label in enumerate(unique_labels):  
 
                    a1_idxs = np.where(labels[a_idx1] == unique_label)[0] 
                    a2_idxs = np.where(labels[a_idx2] == unique_label)[0]
                    
                    if np.size(a1_idxs) == 0 and np.size(a2_idxs) == 0:
                        continue
                    elif np.size(a1_idxs) == 0 or np.size(a2_idxs) == 0:
                        distance.append(.5*cnt*np.max(all_dists))
                        continue 
                    
                    a1_chs = np.mean(ch_encoded[a_idx1][a1_idxs], axis=0)
                    a2_chs = np.mean(ch_encoded[a_idx2][a2_idxs], axis=0)
                    
                    distance.append(np.linalg.norm(a1_chs - a2_chs))
                
                d_list += [distance]    
                    
            if method == 6:
                # Method 6: label-aware: Wasserstein
                unique_labels = np.unique(np.hstack(labels))
                distance = []
                cnt = 1
                for label_idx, unique_label in enumerate(unique_labels):  
 
                    a1_idxs = np.where(labels[a_idx1] == unique_label)[0] 
                    a2_idxs = np.where(labels[a_idx2] == unique_label)[0]
                    
                    if np.size(a1_idxs) == 0 and np.size(a2_idxs) == 0:
                        continue
                    elif np.size(a1_idxs) == 0 or np.size(a2_idxs) == 0:
                        distance.append(.5*cnt*np.max(all_dists))
                        continue
                    
                    a1_chs = ch_encoded[a_idx1][a1_idxs]
                    a2_chs = ch_encoded[a_idx2][a2_idxs]
                    
                    distance.append(compute_wasserstein(a1_chs, a2_chs))

                d_list += [distance]

            dist[a_idx1, a_idx2] = np.mean(np.array(d_list))

    return dist


def gen_labels(data, areas, task, best_beams=None, index_by_area=False):
    
    """
    Returns labels ordered by the areas indices

    TASKS = ['CSI compression', 'LoS identification', 'beam prediction',
             'scenario identification', 'multipath count', 'multipath count2', 'received power', 'localization']
    """
    usrs_with_chs = np.where(data['user']['LoS'] != -1)[0]
    n_usrs_with_chs = len(usrs_with_chs)
    n_usrs = len(data['user']['LoS'])
    idxs = np.hstack([area.idxs for area in areas])
    
    if task == 0 or task == 'CSI compression':
        labels = None
        task = 0 # skip label post processing
    
    if task == 1 or task == 'LOS classification':
        # Binary LOS/NLOS identification (1D outputs = 2 discrete status)
        y = data['user']['LoS']
    if task == 2 or task == 'multipath count':
        # Count Multipath Components     (1D outputs = 5 discrete counts)
        n_paths = np.zeros(n_usrs, dtype=int)
        pwr_threshold = 0.9
        for i in tqdm(range(n_usrs_with_chs), desc='Computing number of paths per user'):
            idx = usrs_with_chs[i]
            pwrs = data['user']['paths'][idx]['power']
            total_pwr = np.sum(pwrs)
            cumulative_pwr = np.cumsum(pwrs)
            n_paths[idx] = np.sum(cumulative_pwr <= pwr_threshold * total_pwr) + 1    
        y = n_paths - 1
    
    if task == 2.5 or task == 'multipath count v2':
        # Count Multipath Components     (1D outputs = 5 discrete counts)
        n_paths = np.zeros(n_usrs, dtype=int) * np.nan
        db_thres = 10
        for i in tqdm(range(n_usrs_with_chs), desc='Computing number of paths per user'):
            idx = usrs_with_chs[i]
            pwrs = data['user']['paths'][idx]['power']
            pwrs_db = 10 * np.log10(pwrs)
            n_paths[idx] = len(np.where(pwrs_db[0] - db_thres < pwrs_db)[0])
        y = n_paths - 1
    if task == 3 or task == 'scenario identification':
        y = np.zeros(n_usrs_with_chs, dtype=int)
        cnt = 0
        for area_idx, area in enumerate (areas):
            n_usrs_area = len(area.idxs)
            y[cnt:cnt+n_usrs_area] = area_idx
            cnt += n_usrs_area
    
    # if task == 3.5 or task == 'scenario identification':
    #     y = np.zeros(n_usrs_with_chs)
    #     cnt = 0
    #     for area_idx, area in enumerate (areas):
    #         n_usrs_area = len(area.idxs)
    #         y[cnt:cnt+n_usrs_area] = area_idx
    #         cnt += n_usrs_area
    
    if task == 4 or task == 'beam prediction':
        # Beam Prediction                (1D outputs = 25 discrete beams)
        if best_beams is None:
            raise Exception('Best beams not provided')
        y = best_beams
    if task == 5 or task == 'received power':
        # Recv. Power                    (1D outputs = continuous received powers)
        total_pwr = np.zeros(n_usrs)
        for i in tqdm(range(n_usrs_with_chs), desc='Computing number of paths per user'):
            idx = usrs_with_chs[i]
            total_pwr[idx] = np.sum(data['user']['paths'][idx]['power'])
            if total_pwr[idx] == 0:
                print(idx)
        y = 10 * np.log10(total_pwr)
    if task == 6 or task == 'localization': 
        # Positions                      (2D outputs = 2 continuous positions) 
        # y = data['user']['location'][:, :2]
        pos = data['user']['location'][:,:2].astype(np.float32) # 2500 x 2
        y = np.copy(pos).astype(np.float32)
        y[:,0], x_min, x_max = minmax(pos[:,0], return_minmax=True)
        y[:,1], y_min, y_max = minmax(pos[:,1], return_minmax=True)
    
    if task:
        labels = y if (task == 3 or task == 'scenario identification') else y[idxs]
        labels = labels.astype(int) 
    
    # TODO: return scenario_identification (above, not organized by areas, 
    # so all tasks follow the same org at the end.
    
    # if index_by_area:
    #     labels = y if index_by_area else y[idxs]
    
    return labels






def plot_dist_vs_perf2(d_matrix, p_matrix, ignored_columns=[], ignored_indexes=[], save_name=None, ylabel=None):
    n_areas = d_matrix.shape[0]
    vect_dist = []
    vect_perfs = []
    already_distanced  = []
    plt.figure(dpi=300)
    for a_idx1 in range(n_areas):
        for a_idx2 in range(n_areas):
            if a_idx1 == a_idx2 or a_idx1 in ignored_columns or [a_idx1,a_idx2] in ignored_indexes or (a_idx1, a_idx2) in already_distanced:
                continue
            # stacked_matrices = np.stack((p_matrix[a_idx1, a_idx2],
            #                              p_matrix[a_idx2, a_idx1]))
            # perf_avg = np.nanmean(stacked_matrices, axis=0)
            # perf_avg = np.nanmean(p_matrix[a_idx1, a_idx2],
            #                       p_matrix[a_idx2, a_idx1])
            perf_avg = (p_matrix[a_idx1, a_idx2] + p_matrix[a_idx2, a_idx1]) / 2
            plt.scatter(d_matrix[a_idx1, a_idx2], perf_avg, 
                        label=f'{a_idx1} -> {a_idx2}')
            vect_dist += [d_matrix[a_idx1, a_idx2]]
            vect_perfs += [perf_avg] 
            already_distanced += [(a_idx1, a_idx2), (a_idx2, a_idx1)]
    
    plt.grid()
    plt.xlabel('distance')
    plt.ylabel(ylabel if ylabel else 'perf')
    corr = np.corrcoef(vect_dist, vect_perfs)[0,1]
    plt.title(f'Performance vs Distance (Pearson coef: {corr:.2f})')
    if save_name:
        plt.savefig(save_name)
    plt.show()
    
    return corr 


def compute_acc(y_pred, y_true, top_k=[1,3,5]):
    """ Computes top-k accuracy given prediction and ground truth labels."""
    n_top_k = len(top_k)
    total_hits = np.zeros(n_top_k)
    
    n_test_samples = len(y_true)
    if len(y_pred) != n_test_samples:
        raise Exception('Number of predicted beams does not match number of labels.')
    
    # For each test sample, count times where true beam is in k top guesses
    for samp_idx in range(len(y_true)):
        for k_idx in range(n_top_k):
            hit = np.any(y_pred[samp_idx,:top_k[k_idx]] == y_true[samp_idx, -1])
            total_hits[k_idx] += 1 if hit else 0
    
    # Average the number of correct guesses (over the total samples)
    return np.round(total_hits / len(y_true), 4)


def get_task_name(task_idx):
    return ['LOS classification', 'multipath count', 'scenario identification', 
            'beam prediction', 'received power', 'localization'][task_idx]


def transform_into_per_area_list(data_to_transform, idxs_of_each_area, one_dim=False):
    last_idx = 0
    transformed_data = []
    for idxs in idxs_of_each_area:
        area_rel_idxs = last_idx + np.arange(len(idxs))
        if one_dim:
            transformed_data += [data_to_transform[area_rel_idxs]]
        else:
            transformed_data += [data_to_transform[area_rel_idxs, :]]
        last_idx = area_rel_idxs[-1] + 1
    
    return transformed_data
    


def plot_dist_vs_perf(d_matrix, p_matrix, save_name=None):
    n_areas = len(d_matrix)
    plt.figure(dpi=300)
    for a_idx1 in range(n_areas):
        for a_idx2 in range(n_areas):
            plt.scatter(d_matrix[a_idx1, a_idx2], p_matrix[a_idx1, a_idx2], 
                        label=f'{a_idx1} -> {a_idx2}')
    # plt.legend()
    plt.grid()
    plt.xlabel('distance')
    plt.ylabel('transfer error (linear NMSE)')
    corr = np.corrcoef(d_matrix.flatten(), p_matrix.flatten())[0,1]
    plt.title(f'Performance vs Distance (Pearson coef: {corr:.2f})')
    if save_name:
        plt.savefig(save_name)
    plt.show()
    return corr
    

def plot_dist_vs_perf3(d_matrix, p_matrix, save_name=None, corr_title='', plot=True):
    n_areas = d_matrix.shape[0]
    vect_dist = []
    vect_perfs = []
    already_distanced  = []
    if plot:
        plt.figure(dpi=300)
    for a_idx1 in range(n_areas):
        for a_idx2 in range(n_areas):
            if a_idx1 == a_idx2 or (a_idx1, a_idx2) in already_distanced:
                continue
            perf_avg = (p_matrix[a_idx1, a_idx2] + p_matrix[a_idx2, a_idx1]) / 2
            if plot:
                plt.scatter(d_matrix[a_idx1, a_idx2], perf_avg, label=f'{a_idx1} -> {a_idx2}')
            vect_dist += [d_matrix[a_idx1, a_idx2]]
            vect_perfs += [perf_avg] 
            already_distanced += [(a_idx1, a_idx2), (a_idx2, a_idx1)]
    
    corr = np.corrcoef(vect_dist, vect_perfs)[0,1]
    if plot:
        plt.grid()
        plt.xlabel('distance')
        plt.ylabel('cross-performance')
        plt.title(corr_title+f'Pearson coef: {corr:.2f}')
        if save_name:
            plt.savefig(save_name)
        plt.show()
    
    return corr 





















