import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from scipy.ndimage import gaussian_filter1d
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sp


def align_major_axis_2d(points, full_point_cloud=None):
    """
    Rotates a 2D point cloud (X, Z) so that its major axis is aligned with the X-axis.

    Parameters:
        points (ndarray): Nx2 array of (X, Z) coordinates.

    Returns:
        rotated_points (ndarray): Aligned point cloud.
        rotation_matrix (ndarray): 2x2 rotation matrix used.
    """
    # Compute mean and center the points
    mean = np.mean(points, axis=0)
    centered = points - mean
    if full_point_cloud is not None:
        centered_full = full_point_cloud - mean

    # Compute covariance matrix and PCA
    covariance = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)

    # The eigenvector with the largest eigenvalue is the major axis
    major_axis = eigenvectors[:, np.argmax(eigenvalues)]  # Largest eigenvector

    # Compute rotation angle (angle to X-axis)
    angle = np.arctan2(major_axis[1], major_axis[0])  # Angle w.r.t. X-axis
    rotation_matrix = np.array([[np.cos(-angle), -np.sin(-angle)], 
                                [np.sin(-angle), np.cos(-angle)]])

    if full_point_cloud is not None:
        # Rotate points from whole dataset around centroid
        full_rotated_points = (rotation_matrix @ centered_full.T).T + mean  # Apply rotation & shift back

        return full_rotated_points, rotation_matrix
    else:
        # Rotate points around centroid
        rotated_points = (rotation_matrix @ centered.T).T + mean

        return rotated_points, rotation_matrix



#-----------------------------------------------------------------------#


def smooth_signal_in_point_cloud(points, signal, iterations=10, radius=30):
    """
    Smooths the point cloud's XYZ coordinates and signal using iterative averaging.
    
    Args:
        points (np.ndarray): Shape (N, 3), 3D point cloud (XYZ).
        signal (np.ndarray): Shape (N,), the signal to smooth.
        iterations (int): Number of iterations for smoothing.
        radius (float): Search radius to find neighbors.
    
    Returns:
        np.ndarray: Smoothed points (unchanged in this version).
        np.ndarray: Smoothed signal values.
    """
    # Build a KDTree for efficient neighbor search
    tree = cKDTree(points)

    # Precompute neighbors once
    neighbors = tree.query_ball_point(points, radius)

    for _ in range(iterations):
        print(f'Iteration round: {_}')
        
        # Compute the smoothed signal using NumPy vectorized operations
        new_signal = np.array([
            np.mean(signal[idx]) if idx else signal[i]
            for i, idx in enumerate(neighbors)
        ])
        
        signal = new_signal  # Update signal
    
    return points, signal


def generate_embryo_curve(embryo_cells, x_col='Oriented_X', z_col='Z', y_col='Oriented_Y'):
    """
    Generate a smooth curve representing the embryo shape based on cell coordinates.

    Args:
        embryo_cells (pd.DataFrame): DataFrame containing cell coordinates.
        x_col (str): Column name for X-coordinates.
        z_col (str): Column name for Z-coordinates.
        y_col (str): Column name for Y-coordinates.
    
    Outputs:
        embryo_curve (pd.DataFrame): DataFrame with smoothed curve points.
        distance_along_curve (np.ndarray): Cumulative distance along the curve.
        arg_min (np.ndarray): Indices of closest points on the curve.
        min_distance (np.ndarray): Minimum distances from cells to the curve.

    """

    poly_degree = 4
    x_values = embryo_cells[x_col].values
    z_values = embryo_cells[z_col].values
    y_values = embryo_cells[y_col].values

    z_poly = np.poly1d(np.polyfit(x_values, z_values, poly_degree))
    y_poly = np.poly1d(np.polyfit(x_values, y_values, poly_degree))

    # Generate dense points along the curve
    dense_x = np.linspace(x_values.min() - 20, x_values.max() + 20, num=1000)
    dense_z = z_poly(dense_x)
    dense_y = y_poly(dense_x)

    # Create a new dense DataFrame
    embryo_curve = pd.DataFrame({
        x_col: dense_x,
        z_col: dense_z,
        y_col: dense_y
    })

    # Calculate distance along the curve
    distance_along_curve = np.cumsum(np.sqrt(np.diff(dense_x)**2 + np.diff(dense_z)**2))
    distance_along_curve = np.insert(distance_along_curve, 0, 0)  # Add starting point

    # Find the closest point on the curve to each cell
    distance = cdist(embryo_cells[[z_col, x_col]], embryo_curve[[z_col, x_col]])
    arg_min = np.argmin(distance, axis=1)
    min_distance = np.min(distance, axis=1)

    # Make distance negative if the point is below the curve
    below_curve = embryo_cells[z_col].values < embryo_curve[z_col].values[arg_min]
    min_distance[below_curve] = -min_distance[below_curve]

    return embryo_curve, distance_along_curve, arg_min, min_distance


def find_midline(epi_nuc, midline_channel,
                      x_col='Oriented_Y', 
                      y_col='AP_position', 
                      z_col='Apical_basal_distance',
                      percentile_threshold=0.8,
                      lim_dist=10,
                      window_size=10
                      ):
    
    """
    Identify and smooth the midline of an embryo based on cell data.

    Parameters:
    epi_nuc (pd.DataFrame): DataFrame containing nuclear information for cells.
    midline_channel (str): Column indicating midline signal intensity.
    x_col (str): Column representing the X-axis (Oriented Y-axis).
    y_col (str): Column representing the Y-axis (AP position).
    z_col (str): Column representing the Z-axis (Apical-basal distance).
    percentile_threshold (float): Percentile threshold for midline signal.
    lim_dist (float): Maximum allowed distance between centroids.
    window_size (float): Size of the sliding window along the Y-axis.

    """
    # Define the sliding window step size
    step = window_size/2
    centroid_whole = [epi_nuc[y_col].mean(), epi_nuc[x_col].mean()]
    
    # Slice the embryo along the Y-axis
    slice_e = epi_nuc[(epi_nuc[y_col] > centroid_whole[0] - window_size) & (epi_nuc[y_col] < centroid_whole[0] + window_size) ]

    # Find the top 10% of cells with highest midline signal
    top_midline = slice_e[slice_e[midline_channel] > slice_e[midline_channel].quantile(percentile_threshold)]

    #centroid of top 10% of cells with highest midline signal - this is the first point
    centroid = [top_midline[z_col].mean(), top_midline[y_col].mean(), top_midline[x_col].mean()]

    #add point to list
    centroid_list = []
    centroid_list.append(centroid)


    #Start in the middle of the embryo and move from either directions

    #loop from centre to one end of the embryo
    cen_top = np.arange(int(np.min(epi_nuc[y_col])), int(centroid_whole[0] + window_size), int(step))
    cen_bot = np.arange(centroid_whole[0], np.max(epi_nuc[y_col])+ window_size, int(step))

    for i in cen_top:
        slice_e = epi_nuc[(epi_nuc[y_col] > i - window_size) & (epi_nuc[y_col] < i + window_size) ]
        top_midline = slice_e[slice_e[midline_channel] > slice_e[midline_channel].quantile(percentile_threshold)]

        #find the middle cell in the top 10% of cells with highest midline signal in the slice
        top_midline_sorted = top_midline.sort_values(by=x_col)

        # Find the middle row
        middle_idx = len(top_midline_sorted) // 2

        # Get the middle row (if the number of rows is odd, this will give the exact middle row)
        if not top_midline_sorted.empty:
            middle_idx = len(top_midline_sorted) // 2
            centroid = top_midline_sorted.iloc[middle_idx][[z_col, y_col, x_col]].values
            if not top_midline.empty:

                # Check if centroid is too far from the previous one
                if np.abs(centroid_list[-1][2] - centroid[2]) > lim_dist:
                    sign = np.sign(centroid_list[-1][2] - centroid[2])
                    diff = np.abs(centroid_list[-1][2] - centroid[2])
                    centroid[2] = centroid[2] - sign * (lim_dist-diff)
            centroid_list.append(centroid)
        else:
            # Handle the case where top_midline_sorted is empty
            print("No data available in top_midline_sorted.")
            continue  


     #loop from centre to one end of the embryo
    for i in cen_bot:
        slice_e = epi_nuc[(epi_nuc[y_col] > i - window_size) & (epi_nuc[y_col] < i + window_size) ]
        top_midline = slice_e[slice_e[midline_channel] > slice_e[midline_channel].quantile(percentile_threshold)]

        #find the middle cell in the top 10% of cells with highest TBRA in the sli
        top_midline_sorted = top_midline.sort_values(by=x_col)

        #Find the middle row
        middle_idx = len(top_midline_sorted) // 2

        #Get the middle row (if the number of rows is odd, this will give the exact middle row)
        if not top_midline_sorted.empty:

            middle_idx = len(top_midline_sorted) // 2
            centroid = top_midline_sorted.iloc[middle_idx][[z_col, y_col, x_col]].values

            if not top_midline.empty:
                # Check if centroid is too far from the previous one
                if np.abs(centroid_list[-1][2] - centroid[2]) > lim_dist:
                    sign = np.sign(centroid_list[-1][2] - centroid[2])
                    diff = np.abs(centroid_list[-1][2] - centroid[2])
                    centroid[2] = centroid[2] - sign * (lim_dist-diff)
                    
            centroid_list.append(centroid)
        else:
            # Handle the case where top_midline_sorted is empty
            continue  

    # Convert list of centroids to numpy array
    centroid_list = np.asarray(centroid_list, dtype=float)

    # Remove NaN values
    mask = ~np.isnan(centroid_list).any(axis=1)
    centroid_list_clean = centroid_list[mask]

    #order the list
    centroid_list_clean = centroid_list_clean[centroid_list_clean[:, 1].argsort()]

    # Apply Gaussian smoothing to centroid X-values (column index 2)
    smoothed_x = gaussian_filter1d(centroid_list_clean[:, 2], sigma=1)  # Adjust sigma for more/less smoothing
    smoothed_z = centroid_list_clean[:, 0]
    xnew = centroid_list_clean[:, 1]  # Keep original X-values

    # Combine smoothed values with original X-values
    smoothed_line = np.column_stack((smoothed_z, smoothed_x, xnew))
    smoothed_line = pd.DataFrame(smoothed_line, columns=[z_col, x_col, y_col])
    
    #generate a dense line with more point intervals for later steps
    # Define polynomial degree (adjustable, 3-5 works well)
    poly_degree = 4

    # Extract original sparse points
    x_values = smoothed_line[y_col].values
    z_values = smoothed_line[z_col].values
    y_values = smoothed_line[x_col].values

    # Fit polynomials to Z and Y separately
    z_poly = np.poly1d(np.polyfit(x_values, z_values, poly_degree))
    y_poly = np.poly1d(np.polyfit(x_values, y_values, poly_degree))

    # Generate a dense set of X values (match the step size)
    dense_x = np.arange(x_values.min()-20, x_values.max()+20, step / 5)  # 5x denser than original

    # Predict new Z and Y values
    dense_z = z_poly(dense_x)
    dense_y = y_poly(dense_x)

    # Create a new dense DataFrame
    smoothed_lineXYZ_dense = pd.DataFrame({
        y_col: dense_x,
        z_col: dense_z,
        x_col: dense_y
    })

    return smoothed_lineXYZ_dense, centroid_list_clean

#-----------------------------------------------------------------------#



def compute_mean_edge_position(cells, mask = None, embryo_id = 'embryo', LR_col = 'AVLR', LR_limit = 0.95):
    """
    Computes the mean absolute positions of cells on the left and right side of an embryo,
    groups them into bins, and interpolates missing values.

    Parameters:
    - cells (pd.DataFrame): DataFrame containing embryo data with 'binned', 'Rel_LR_position', 'AVLR', and 'Epiblast'.
    - LR_limit (float): Left-right position threshold.

    Returns:
    - l_mean_values (pd.DataFrame): Mean absolute values for the left side, interpolated.
    - r_mean_values (pd.DataFrame): Mean absolute values for the right side, interpolated.
    """
    
    l_mean_values = []
    r_mean_values = []

    for embryo in cells[embryo_id].unique():
        if mask is not None:
            cells = cells[mask & (cells[embryo_id] == embryo)]
        else:
            cells = cells[cells[embryo_id] == embryo]

        # Separate left and right cells
        l_cells = cells[cells['Rel_LR_position'] > LR_limit]
        r_cells = cells[cells['Rel_LR_position'] < -LR_limit]

        # Compute mean absolute values per bin
        l_mean = l_cells.groupby('binned').agg(mean_Abs=(LR_col, 'mean'))
        r_mean = r_cells.groupby('binned').agg(mean_Abs=(LR_col, 'mean'))

        # Assign embryo label
        l_mean[embryo_id] = embryo
        r_mean[embryo_id] = embryo

        # Store results
        l_mean_values.append(l_mean)
        r_mean_values.append(r_mean)

    # Combine all embryos' data
    l_mean_values = pd.concat(l_mean_values).groupby('binned').agg(mean_Abss=('mean_Abs', 'mean'))
    r_mean_values = pd.concat(r_mean_values).groupby('binned').agg(mean_Abss=('mean_Abs', 'mean'))

    # Ensure all bins are covered
    full_bins = np.arange(l_mean_values.index.min(), l_mean_values.index.max() + 1, 1)

    # Reindex and interpolate missing values
    l_mean_values = l_mean_values.reindex(full_bins).interpolate(method='linear')
    r_mean_values = r_mean_values.reindex(full_bins).interpolate(method='linear')

    return l_mean_values, r_mean_values


##-----------------------------------------------------------------------#



class LR_flattening:
    """
    Class for flattening an embryo represented as a 3D point cloud based with a midline intensity channel.

    Arguments:
    ---------
    h : the search radius for the local center of mass and PCA during local curve fitting.
    step_size : the step size for moving along the curve.
    max_iter : the maximum number of iterations for curve fitting.
    tol : the tolerance for convergence.
    start_point : the starting point for curve fitting. Can be 'mean', 'left', 'right' or a specific point.
    end_point : the end point for curve fitting. If None, the curve will be grown until convergence.
    k : the exponent for the local PCA direction.
    circular : whether the curve is circular or not. This will also be determined by the end_point.
    smooth_h : the bandwidth for Gaussian smoothing of the curve.
    z_col : the column name for the Z coordinate in the input DataFrame.
    y_col : the column name for the Y coordinate in the input DataFrame.
    x_col : the column name for the X coordinate in the input DataFrame.

    Outputs:
    --------
    curve_points : a DataFrame with the smoothed curve points.
    distance_analysis : a DataFrame with the distances from the curve to the original points.

    """

    def __init__(self, h=20, step_size=20, max_iter=500, tol=1e-6, 
                 start_point ='mean', end_point=None,
                 k=2, circular=False, smooth_h=None,
                 z_col='Z', y_col='Oriented_Y', x_col='Oriented_X'):
        self.h = h
        self.step_size = step_size
        self.max_iter = max_iter
        self.tol = tol
        self.start = start_point
        self.end_point = end_point
        self.k = k
        self.circular = circular
        self.smooth_h = smooth_h if smooth_h is not None else h
        self.part_circular = False
        self.state = None
        self.z_col = z_col
        self.y_col = y_col
        self.x_col = x_col

    def _calculate_weights(self, x, X, bandwidth=None):
        """Calculate Gaussian weights with optional bandwidth parameter."""
        h = bandwidth if bandwidth is not None else self.h
        distances = cdist([x], X)[0]
        weights = np.exp(-0.5 * (distances / h) ** 2)
        return weights / np.sum(weights)
    
    def _smooth_curve(self, curve_points):
        """Apply local Gaussian smoothing to curve points."""
        curve_array = np.array(curve_points)
        smoothed_points = []
        
        for i in range(len(curve_points)):
            weights = self._calculate_weights(curve_array[i], curve_array, self.smooth_h)
            smoothed_point = np.average(curve_array, weights=weights, axis=0)
            smoothed_points.append(smoothed_point)
            
        return smoothed_points

    def _local_center_of_mass(self, x, X):
        """Calculate the local center of mass using Gaussian weights."""
        weights = self._calculate_weights(x, X)
        return np.average(X, weights=weights, axis=0)
    
    def _local_pca(self, x, X, prev_gamma=None):
        """
        Perform local PCA to find the direction of the curve.
        """
        # Calculate the covariance matrix using weighted points
        weights = self._calculate_weights(x, X)
        weighted_mean = np.average(X, weights=weights, axis=0)
        centered_X = X - weighted_mean
        
        # Calculate the covariance matrix
        cov_matrix = np.zeros((X.shape[1], X.shape[1]))
        for i in range(X.shape[0]):
            cov_matrix += weights[i] * np.outer(centered_X[i], centered_X[i])
            
        # Perform eigen decomposition
        eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)
        idx = eigenvals.argsort()[::-1]
        gamma = eigenvecs[:, idx[0]]
        
        # Ensure the direction is consistent with the previous gamma
        if prev_gamma is not None:
            cos_alpha = np.dot(prev_gamma, gamma)

            # If the angle is obtuse, flip the direction
            if cos_alpha < 0:
                gamma = -gamma
                cos_alpha = -cos_alpha
            
            # Apply the exponential decay to the direction
            a_x = abs(cos_alpha) ** self.k
            gamma = a_x * gamma + (1 - a_x) * prev_gamma
            gamma = gamma / np.linalg.norm(gamma)
                
        return gamma
    
    def _cumulated_distance(self, points):
        """
        Calculate the cumulative distance along the curve.
        
        Parameters:
        -----------
        points : pandas DataFrame or list of arrays
            DataFrame with z_col and y_col columns representing the curve
            or list of arrays with curve points
            
        Returns:
        --------
        distances : array-like
            Array of cumulative distances along the curve points
        """
        z_col = self.z_col
        y_col = self.y_col
        
        distances = np.zeros(len(points))
        #check if points are a dataframe
        if points.__class__.__name__ == 'DataFrame':
            for i in range(1, len(points)):
                distances[i] = distances[i - 1] + np.linalg.norm(points.iloc[i][[z_col, y_col]] - points.iloc[i - 1][[z_col, y_col]])
        else:
            for i in range(1, len(points)):
                distances[i] = distances[i - 1] + np.linalg.norm(points[i] - points[i - 1])
        
        return distances


    def _find_nearest_intersection(self, point, curve_points, cumulative_distances):
        z_col = self.z_col
        y_col = self.y_col
        
        distances = []
        intersections = []
        path_distances = []
        if isinstance(curve_points, pd.DataFrame):
            curve_points = curve_points[[z_col, y_col]].values  # Extract as array

        for i in range(len(curve_points) - 1):
            p1 = curve_points[i]
            p2 = curve_points[i + 1]
            
            # Calculate the nearest point on the line segment p1-p2 to the given point
            line_vec = p2 - p1
            point_vec = point - p1
            line_len = np.linalg.norm(line_vec)
            line_unitvec = line_vec / line_len
            point_vec_scaled = point_vec / line_len

            # Calculate the projection of point_vec onto line_unitvec
            t = np.dot(line_unitvec, point_vec_scaled)
            
            if t < 0.0:
                t = 0.0
            elif t > 1.0:
                t = 1.0
            
            # Calculate the nearest point on the line segment
            nearest = p1 + t * line_vec
            dist = np.linalg.norm(point - nearest)
            
            # Calculate path distance to intersection
            path_dist = cumulative_distances[i] + t * line_len
            
            distances.append(dist)
            intersections.append(nearest)
            path_distances.append(path_dist)
            
        if not distances:
            return None, None, None
            
        min_idx = np.argmin(distances)
        return intersections[min_idx], distances[min_idx], path_distances[min_idx], min_idx


    def _grow_curve(self, x, X, direction=1):

        curve_points = []
        prev_mu = None
        prev_gamma = None
        min_start_distance = 2 * self.h
        start_point = x
        loop_check_start_iter = 10
        circ_bool = self.circular
        reached_tolerance = False
        end_point = self.end_point
        
        #add start point to curve points
        curve_points.append(x)

        for i in range(self.max_iter):
            mu = self._local_center_of_mass(x, X)
            distance_from_start = np.linalg.norm(mu - start_point)
            
            if end_point is not None and np.linalg.norm(mu - end_point) < self.min_start_distance/4:
                break
    
            if i >= loop_check_start_iter and len(curve_points) > 0:
                if distance_from_start < min_start_distance:
                    circ_bool = True
                    break
            
            if prev_mu is not None and np.linalg.norm(mu - prev_mu) < self.tol:
                reached_tolerance = True
                break
            
            gamma = self._local_pca(mu, X, prev_gamma)
            prev_gamma = gamma
            x = mu + direction * self.step_size * gamma
            curve_points.append(mu)
            prev_mu = mu
            
        if reached_tolerance and prev_gamma is not None:
            final_point = curve_points[-1] + direction * 4 * self.h * prev_gamma
            curve_points.append(final_point)
        
        return curve_points, circ_bool
    
    def fit(self, df, start_point=None, smooth=True):
        z_col = self.z_col
        y_col = self.y_col        
        X = df.values
        
        if start_point is not None:
            x_start = np.array(start_point)
        else:
            if isinstance(self.start, str):
                if self.start == 'mean':
                    x_start = np.mean(X, axis=0)
                elif self.start == 'left':
                    x_start = X[np.argmin(X[:, 0])]
                elif self.start == 'right':
                    x_start = X[np.argmax(X[:, 0])]
            else:
                x_start = np.array(self.start)
        
        forward_points, forward_circular = self._grow_curve(x_start, X, direction=1)
        backward_points, backward_circular = self._grow_curve(x_start, X, direction=-1)
        
        self.circular = forward_circular or backward_circular
        all_points = backward_points[::-1] + forward_points
        
        if smooth:
            all_points = self._smooth_curve(all_points)
        
        return pd.DataFrame(all_points, columns=[z_col, y_col])
    
        
    def _analyze_path_distances(self, X, search_points, start_P_arg):
        z_col = self.z_col
        y_col = self.y_col
                
        points = X
        sp_arr = search_points[[z_col, y_col]].values

        comb_distances = self._cumulated_distance(sp_arr)
        comb_distances = comb_distances- comb_distances[start_P_arg]
        analysis = []

        for point in points:
            # Exclude current point and immediate neighbors from intersection search
            
            if len(search_points) > 1:
                intersection, perp_distance, path_dist, min_idx = self._find_nearest_intersection(
                    point, search_points, comb_distances)
                
                #if the curve is circular flip perp_distance if inside the circle
                if self.circular:
                    centroid = np.mean(sp_arr, axis=0)
                    self.centroid = centroid
                    #if the point is outside the circle, make the distance negative
                    if np.linalg.norm(point - centroid) > np.linalg.norm(centroid - intersection) :
                        perp_distance = -perp_distance                
                
                # if the curve is not circular or part circular 
                # flip the distance if the point under the closest point
                # in z
                if not self.circular and not self.part_circular:
                    if search_points.iloc[min_idx][z_col] > point[0]:
                        perp_distance = -perp_distance                

                analysis.append({
                    'Dist_from_midline': path_dist,
                    'intersection_distance': perp_distance
                })

        return pd.DataFrame(analysis)
    
    def classify_points(self, point_cloud, curve, normals):
        # 4️⃣ Assign Each Point to One Side of the Curve
        """Classifies points based on which side of the curve they lie on."""
        vor = sp.spatial.Voronoi(curve)  # Compute Voronoi regions

        classifications = []
        for point in point_cloud:
            # Find nearest curve point using Voronoi partitioning
            closest_idx = np.argmin(np.linalg.norm(curve - point, axis=1))
            closest_curve_point = curve[closest_idx]
            normal = normals[closest_idx]

            # Compute dot product to determine side
            d = np.dot(point - closest_curve_point, normal)
            classifications.append(1 if d > 0 else -1)

        return np.array(classifications)
    
    def compute_normals(self,curve):
        """Computes local normals using finite differences."""
        tangents = np.gradient(curve, axis=0)
        normals = np.column_stack((-tangents[:, 1], tangents[:, 0]))  # Rotate by 90°
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)  # Normalize
        return normals

    def fit_with_branches(self, df, start_point=None, smooth=True):
        
        z_col = self.z_col
        y_col = self.y_col

        X = df[[z_col, y_col]].values        
        
        if start_point is not None:
            x_start = np.array(start_point)
        else:
            x_start = np.array(self.start) if not isinstance(self.start, str) else np.mean(X, axis=0)



        forward_points, forward_circular = self._grow_curve(x_start, X, direction=1)
        backward_points, backward_circular = self._grow_curve(x_start, X, direction=-1)


        #If backward or forward points doens't work, 
        # just normalize the points around the start point


        # if self.end_point is not None and not self.circular:
        self.circular = forward_circular or backward_circular

        if smooth:
            forward_points = self._smooth_curve(forward_points)
            backward_points = self._smooth_curve(backward_points)



        if len(backward_points) < 15 or len(forward_points) < 15:

            # print('No points found, normalizing around start point')
            # print(df)
            df_copy = df.copy()
            df_copy[z_col] = df_copy[z_col] - x_start[0]
            df_copy[y_col] = df_copy[y_col] - x_start[1]

            #Rename the columns to same output of analyse distances
            df_copy.rename(columns={z_col: 'intersection_distance', y_col: 'Dist_from_midline'}, inplace=True)
            
            # Create points with original (z_col, y_col) values
            backward_points = [[x_start[0], df[y_col].min()],  # Minimum y_col value (original scale)
                            [x_start[0], x_start[1]]]       # Start point (original coordinates)

            forward_points = [[x_start[0], df[y_col].max()],  # Maximum y_col value (original scale)
                            [x_start[0], x_start[1]]]       # Start point (original coordinates)

            # Convert to DataFrames and add direction labels
            backward_df = pd.DataFrame(backward_points, columns=[z_col, y_col])
            backward_df['direction'] = 'backward'

            forward_df = pd.DataFrame(forward_points, columns=[z_col, y_col])
            forward_df['direction'] = 'forward'

            # Combine both into a single DataFrame
            comb_points = pd.concat([backward_df, forward_df], ignore_index=True)

            return comb_points, df_copy
        
        else:
            #remove the start point from the forward points and the end point from the backward points
            forward_points = forward_points[1:]
            backward_points = backward_points[1:]
            
            # Add end point to forward and backward branches if circular
            if self.end_point is not None and not self.circular:
                forward_points.append(self.end_point)
                backward_points.append(self.end_point)
        
            #Analye branches depending on if they are circular or not

            #If circular
            if self.end_point is None and self.circular:
                
                backward_cum_dist = self._cumulated_distance(backward_points)
                forward_cum_dist = self._cumulated_distance(forward_points)
                
                #averge the max distance of the two branches
                mid_point = (max(forward_cum_dist) + max(backward_cum_dist))/4
                self.max_dist = mid_point
                self.max_dist_f = max(forward_cum_dist)
                self.max_dist_b = max(backward_cum_dist)

                
                # find closest point to the max distance on forward branch
                closest_f_end_point = forward_points[np.argmin(np.abs(forward_cum_dist-mid_point))]
                
                #select points before closest point
                forward_points_sub = forward_points[:np.argmin(np.abs(forward_cum_dist-mid_point))+1]
                
                #find closest point of backward branch to the closest end point
                backward_points_sub = backward_points[:np.argmin(np.linalg.norm(np.array(backward_points) - np.array(closest_f_end_point), axis=1))]
                
                # forward_points_sub = forward_points[:np.argmin(np.abs(forward_cum_dist-max_dist))]
                self.end_point = closest_f_end_point
                
                #add end point to backward branch
                backward_points_sub.append(closest_f_end_point)
                
                
                fp_df = pd.DataFrame(forward_points_sub, columns=[z_col, y_col])
                fp_df['direction'] = 'forward'

                bp_df = pd.DataFrame(backward_points_sub, columns=[z_col, y_col])
                bp_df['direction'] = 'backward'
                            #reverse the backward points
                bp_df = bp_df[::-1]
                
                start_P_arg = len(bp_df)

                comb_points = pd.concat([bp_df, fp_df])
                
                # print(comb_points)
                distance_analysis = self._analyze_path_distances(X, comb_points, start_P_arg)

                return (
                    comb_points,
                    distance_analysis
                )
            

            #If not circular with no combined end point
            if self.end_point is None and not self.circular:

                # print('not circular')
                fp_df = pd.DataFrame(forward_points, columns=[z_col, y_col])
                fp_df['direction'] = 'forward'

                bp_df = pd.DataFrame(backward_points, columns=[z_col, y_col])
                bp_df['direction'] = 'backward'
                            #reverse the backward points
                bp_df = bp_df[::-1]

                comb_points = pd.concat([bp_df, fp_df])
                #sort points by Z and Oriented_Y
                # comb_points = comb_points.sort_values(by=[z_col, y_col])
                start_P_arg = len(bp_df)

                distance_analysis = self._analyze_path_distances(X, comb_points, start_P_arg)
                
                return (
                    comb_points,
                    distance_analysis
                )



    def plot_test(self, sub_slice, curve, distance_analysis, start_point):
        
        z_col = self.z_col  
        y_col = self.y_col
        x_col = self.x_col
        
        forward_curve = curve[curve['direction'] == 'forward']
        backward_curve = curve[curve['direction'] == 'backward']


        if sub_slice[y_col].max() < sub_slice[z_col].max():
            fig, axes = plt.subplots(1, 3, figsize=(10, 10), sharey=True)
            
        if sub_slice[y_col].max() > sub_slice[z_col].max():
            fig, axes = plt.subplots(3, 1, figsize=(20, 10), sharey=True)
        # Raw input data
        axes[0].scatter(sub_slice[y_col], sub_slice[z_col], alpha=0.5, label='Data Points')
        axes[0].scatter(start_point[1], start_point[0], color='green', s=100, label='Starting Point')
        axes[0].plot(forward_curve[y_col], forward_curve[z_col], 'r', label='Forward Branch')
        axes[0].plot(backward_curve[y_col], backward_curve[z_col], 'b-', label='Backward Branch')
        if self.end_point is not None:
            end_point = self.end_point
            axes[0].scatter(end_point[1], end_point[0], color='red', s=100, label='End Point')
        axes[0].set_xlabel('X')
        axes[0].set_ylabel('Z')
        axes[0].legend()
        axes[0].set_aspect('equal', adjustable='box')
        axes[0].grid(True)
        axes[0].set_title(f'Circular: {self.circular}')
        
        # LR distance color mapping
        divider = make_axes_locatable(axes[1])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        sc = axes[1].scatter(sub_slice[y_col], sub_slice[z_col], c=distance_analysis['intersection_distance'],
                            cmap=sns.color_palette("icefire", as_cmap=True), vmin=-(abs(distance_analysis['intersection_distance']).max()),
                            vmax=abs(distance_analysis['intersection_distance']).max())
        fig.colorbar(sc, cax=cax, label='New apical-basal distance')
        axes[1].scatter(start_point[1], start_point[0], color='green', s=100, label='Starting Point')
        axes[1].plot(forward_curve[y_col], forward_curve[z_col], 'r', label='Forward Branch')
        axes[1].plot(backward_curve[y_col], backward_curve[z_col], 'b-', label='Backward Branch')
        axes[1].set_xlabel('X')
        if self.end_point is not None:
            end_point = self.end_point
            axes[1].scatter(end_point[1], end_point[0], color='red', s=100, label='End Point')
        axes[1].set_ylabel('Z')
        axes[1].set_title('Apical-basal Distance')
        axes[1].legend()
        axes[1].set_aspect('equal', adjustable='box')
        axes[1].grid(True)

        # Perpendicular distance color mapping
        divider = make_axes_locatable(axes[2])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        sc = axes[2].scatter(sub_slice[y_col], sub_slice[z_col], c=distance_analysis['Dist_from_midline'],
                            cmap=sns.color_palette("icefire", as_cmap=True), vmin=-(abs(distance_analysis['Dist_from_midline']).max()),
                            vmax=abs(distance_analysis['Dist_from_midline']).max())
        fig.colorbar(sc, cax=cax, label='New left-right distance')
        axes[2].scatter(start_point[1], start_point[0], color='green', s=100, label='Starting Point')
        axes[2].plot(forward_curve[y_col], forward_curve[z_col], 'r', label='Forward Branch')
        axes[2].plot(backward_curve[y_col], backward_curve[z_col], 'b-', label='Backward Branch')
        if self.end_point is not None:
            end_point = self.end_point
            axes[2].scatter(end_point[1], end_point[0], color='red', s=100, label='End Point')
        axes[2].set_xlabel('X')
        axes[2].set_ylabel('Z')
        axes[2].legend()
        axes[2].set_title('Left-right Distance ')

        axes[2].set_aspect('equal', adjustable='box')
        axes[2].grid(True)

        # Make text size bigger
        for ax in axes:
            for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                        ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(20)
        plt.tight_layout()
        plt.show()
    
    def find_closest_points(self, smoothed_points, tree):
        distances, indices = tree.query(smoothed_points, k=1)
        return distances, indices

    def test_slice(self, df, smoothed_line, slice_lim,slice_buff, start_point=None):
        """
        Test the PRINGLE method on a single slice of the embryo.
        """
        z_col = self.z_col
        y_col = self.y_col
        x_col = self.x_col
        
        df = df[[z_col, y_col,x_col]]
        sub_slice = df[(df[x_col] > slice_lim) & (df[x_col] < (slice_lim + slice_buff))]

        sub_smoothed_line = smoothed_line[(smoothed_line[x_col] > slice_lim) & (smoothed_line[x_col] < (slice_lim + slice_buff))]
        start_point = np.array([sub_smoothed_line[z_col].median(), sub_smoothed_line[y_col].median()])
        
        curve, distance_analysis = self.fit_with_branches(sub_slice[[z_col,y_col]], start_point=start_point)
        self.plot_test(sub_slice, curve, distance_analysis, start_point)
        return curve, distance_analysis
    
