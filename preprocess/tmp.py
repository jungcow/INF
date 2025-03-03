import os
os.environ["OPENBLAS_NUM_THREADS"] = "16"

import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load one .npy file (assume it has been loaded into 'scan_data')
# scan_data shape: (N, 5) with column indices [0: beam_origin_x, 1: beam_origin_y, 2: phi, 3: theta, 4: distance]
scan_data = np.load('/home/interns/dev/INF/data/kitti-360_largerot/scans/0000.npy')
phi_values = scan_data[:, 2].reshape(-1, 1)  # Reshape for clustering

# Define number of channels (e.g., 16 for a 16-beam LiDAR)
num_channels = 64

# Apply KMeans clustering to group phi values
kmeans = KMeans(n_clusters=num_channels, random_state=0).fit(phi_values)
labels = kmeans.labels_
cluster_centers = np.sort(kmeans.cluster_centers_.flatten())

# Plotting histogram with cluster centers
plt.hist(phi_values, bins=100, alpha=0.6, color='g')
for center in cluster_centers:
    plt.axvline(center, color='r', linestyle='--')
plt.xlabel('Phi (radians)')
plt.ylabel('Frequency')
plt.title('Histogram of Phi Values with Inferred Channels')
plt.savefig('phi_histogram.png')
plt.close()


# Assign a channel index to each point based on the nearest cluster center (optional, if more refinement is needed)
def assign_channel(phi, centers):
    return np.argmin(np.abs(centers - phi))

channels = np.array([assign_channel(phi, cluster_centers) for phi in scan_data[:, 2]])

breakpoint()

# You now have a 'channels' array that indicates the inferred LiDAR channel for each point.
