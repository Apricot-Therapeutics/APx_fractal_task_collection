from sklearn.neighbors import KernelDensity, NearestNeighbors
import numpy as np
from skimage.measure import regionprops_table
import pandas as pd

def measure_density(coordinates: np.array, img_dimensions: tuple,
                    bandwidth=0.01, kernel='gaussian'):
    """
    Measure the density of the given coordinates.
    Args:
        coordinates: np.array with shape (n, m), where n is the number of
        points and m is the number of dimensions (e.g. 2 for 2D, 3 for 3D, etc.)
        img_dimensions: tuple with the dimensions of the image 
        bandwidth: float, bandwidth of the kernel.
        kernel: str, kernel to use.

    Returns:
        density: np.array with shape (n,), where n is the number of points.

    """
    # rescale to max 1 per axis
    coordinates_rescaled = coordinates / img_dimensions

    kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian').fit(
        coordinates_rescaled)
    density = np.exp(kde.score_samples(coordinates_rescaled))

    return density / np.max(density)


def measure_neighbour_distance(coordinates: np.array, n_neighbours: int):
    """
    Measure the density of the given coordinates.
    Args:
        coordinates: np.array with shape (n, m), where n is the number of
        points and m is the number of dimensions (e.g. 2 for 2D, 3 for 3D, etc.)
        n_neighbours: int, number of neighbours to consider.

    Returns:
        distances: np.array with shape (n,), where n is the number of points.
            This contains the mean distance to the n_neighbours closest points.

    """

    neigh = NearestNeighbors(n_neighbors=n_neighbours)
    neigh.fit(coordinates)

    distances, indices = neigh.kneighbors(coordinates)
    distances = np.mean(distances, axis=1)

    return distances


def measure_neighbours(coordinates: np.array, radius: int):
    """
    Measure the density of the given coordinates.
    Args:
        coordinates: np.array with shape (n, m), where n is the number of
        points and m is the number of dimensions (e.g. 2 for 2D, 3 for 3D, etc.)
        radius: int, radius to consider.

    Returns:
        n_neighbours: np.array with shape (n,), where n is the number of points.
            This contains the number of neighbours within the given radius.
        distances: np.array with shape (n,), where n is the number of points.
            This contains the mean distance to neighbours in the given radius.

    """

    neigh = NearestNeighbors(radius=radius)
    neigh.fit(coordinates)

    distances, indices = neigh.radius_neighbors(coordinates, sort_results=True)
    distances = np.array([np.mean(d) for d in distances])
    n_neighbours = np.array([len(i) for i in indices])

    return n_neighbours, distances


def measure_population_features(
        label_image: np.array,
        n_neighbours: list = [5, 10, 25, 50, 100],
        radii: list = [100, 200, 300, 400, 500],
        bandwidths: list = [0.01, 0.02, 0.03, 0.04, 0.05, 0.2, 0.5, 1.0],
        kernel: str = 'gaussian'):
    """
    Measure population features for the given coordinates.
    Args:
        label_image: np.array containing objects.
        n_neighbours: list, number of neighbours to consider.
        radii: list, radii to consider.
        bandwidths: list, bandwidths of the kernel.
        kernel: str, kernel to use.

    Returns:
        population_features: pd.DataFrame with the population features.

    """

    label_image = np.squeeze(label_image)

    # get coordinates
    coordinates_df = pd.DataFrame(
        regionprops_table(
            label_image,
            properties=['label', 'centroid'],
        )
    )

    coordinates = coordinates_df.drop('label', axis=1).values
    img_dimensions = label_image.shape

    column_names = []
    features = []

    # measure density
    for bandwidth in bandwidths:
        density = measure_density(coordinates, img_dimensions,
                                  bandwidth, kernel)
        column_names.append(f'density_bw_{bandwidth}')
        features.append(density)

    # measure distance to nearest neighbours
    for n_neighbour in n_neighbours:
        distances = measure_neighbour_distance(coordinates, n_neighbour)
        column_names.append(f'mean_distance_nn_{n_neighbour}')
        features.append(distances)

    # measure number of neighbours and mean distance to neighbours
    for radius in radii:
        n_neigh, distances = measure_neighbours(coordinates, radius)
        column_names.append(f'n_neighbours_radius_{radius}')
        column_names.append(f'mean_distance_neighbours_radius_{radius}')
        features.append(n_neigh)
        features.append(distances)

    population_features = pd.DataFrame(
        np.array(features).T, columns=column_names)

    population_features.insert(0, 'label', coordinates_df['label'])

    return population_features