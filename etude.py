import shapely
from sklearn.neighbors import KNeighborsClassifier
from functools import cached_property
import pandas as pd
import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import cv2
from pyproj import CRS

from utils import *

matplotlib.rc("image", origin="upper")



# Fichier contenant les résultats du traitement.
output_file = "resultats.csv"

# Côté des carrés utilisés pour échantillonner la carte lors de la classification.
side = 30

# Polygone délimitant la France métropolitaine, en coordonnées GPS.
mainland_france = shapely.geometry.Polygon(
    [
        (-7.602539, 52.669720),
        (-7.470703, 41.244772),
        (8.613281, 42.358544),
        (9.975586, 51.536086),
    ]
)


# Numéro d'image maximal à considérer.
# On ignore les images 13 et 14.
MAX_IM = 12

CLASS_UNKNOWN = -1
CLASS_NO_COVERAGE = 0
CLASS_COVERED = 1

# Couleurs des points affichés dans les images, en fonction de la classe.
class_colors = {}
class_colors[CLASS_UNKNOWN] = (255, 0, 255)
class_colors[CLASS_NO_COVERAGE] = (0,255,0)
class_colors[CLASS_COVERED] = (255, 0, 0)


# Système de coordonnées Lambert93
lamb = CRS.from_epsg(2154)

# Système de coordonnées GPS.
maps = CRS.from_epsg(4326)


def compute_patch_indices(cities_coords):
    patch_indices = list(
        map(
            lambda c: (
                slice(c[0] - side // 2, c[0] + side // 2),
                slice(c[1] - side // 2, c[1] + side // 2),
                slice(0, 3),
            ),
            cities_coords.astype(int),
        )
    )
    return patch_indices



def patch_repr_color(p, n_sample=400, value_threshold=None, sat_threshold=0.2,n_sample_min=200):
    """
    Étant donné une section d'image p de dimensions (x,y,3),
    renvoie sa couleur représentative.
    
    Il s'agit de la moyenne des `n_sample` pixels les plus saturés, parmi les
    pixels de "value" HSV supérieure à `value_threshold`.

    S'il y a moins de `n_sample_min`
    """
    undecided = np.array([np.nan, np.nan, np.nan])
    
    if len(p) == 0:
        return undecided

    # Flatten
    arr = p.reshape((-1, 3))

    # Convert to HSV
    arr_hsv = matplotlib.colors.rgb_to_hsv(arr.astype(float) / 255.)

    if value_threshold is not None:
        # Only keep elements that are bright enough.
        value_mask = arr_hsv[:, 2] > value_threshold
        sat_mask = arr_hsv[:, 1] > sat_threshold

        mask = np.logical_or(value_mask, sat_mask)
        arr_hsv = arr_hsv[mask]
        # print(np.unique(value_mask, return_counts=True))

        if len(arr_hsv) < n_sample_min:
            return undecided

    # Average the most saturated ones.
    saturations = arr_hsv[:, 1]
    indices = np.argsort(saturations)[-n_sample:]

    return arr[indices].mean(axis=0)


def read_processed(i):
    """
    Charge l'image pré-traitée numéro `i`.
    """

    im = cv2.imread(f"processed/map-{i:02d}.png")
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return im


def load_ref(path):
    """
    Charge le fichier des relevés de coordonnées utilisé pour aligner les cartes.
    """

    df = pd.read_csv(path)
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lng, df.lat, crs=maps))
    return gdf



class Aligner:
    def __init__(self, population_threshold) -> None:
        self.population_threshold = population_threshold

    def load_measurements(self):
        # Measurements DataFrame
        self.mdf = pd.read_csv("ref.csv")

        # GeoDataFrame with GPS coordinates of the control points
        self.ref = load_ref("gps.csv")

    def load_images(self):
        self.processed = list(map(read_processed, range(1, MAX_IM + 1)))

    def load_mainland_cities(self):
        communes = pd.read_csv(
            "communes-departement-region.csv",
            usecols=[
                "latitude",
                "longitude",
                "nom_commune_postal",
                "code_departement",
                "code_commune_INSEE",
            ],
            dtype={
                "latitude": float,
                "longitude": float,
                "code_departement": str,  # Corse: 2A / 2B
                "nom_commune_postal": str,
                "code_commune_INSEE": str,
            },
        )
        communes.code_commune_INSEE = communes.code_commune_INSEE.str.pad(
            width=5, fillchar="0"
        )

        vf = pd.read_csv(
            "villes_france.csv",
            header=None,
            usecols=[2, 10, 14],
            names=["nom", "code_commune_INSEE", "pop"],
            dtype={
                "nom": str,
                "code_commune_INSEE": str,
                "pop": float,
            },
        )

        communes = communes.merge(
            vf[["pop", "code_commune_INSEE"]], on="code_commune_INSEE"
        )

        communes = communes[communes["pop"] > self.population_threshold]

        geo_com = gpd.GeoDataFrame(
            communes,
            crs=maps,
            geometry=gpd.points_from_xy(communes["longitude"], communes["latitude"]),
        )

        self.cities = geo_com[geo_com.within(mainland_france)]  # .iloc[:10]

    def build_classifiers(self):
        cols = pd.read_csv("colors.csv")

        self.clfs = dict()

        for num in range(1, MAX_IM + 1):
            neigh = KNeighborsClassifier(n_neighbors=1)
            arr = cols[cols.im == num][["degroupe", "r", "g", "b"]].to_numpy()
            x = arr[:, 1:].astype(float)
            y = arr[:, 0].astype(int)
            neigh.fit(x, y)

            self.clfs[num] = neigh

    def init(self):
        self.load_mainland_cities()
        print(f"Processing {len(self.cities)} cities")

        self.load_measurements()
        print("Loaded manual measurements")

        self.load_images()
        print("Loaded images")

        self.build_classifiers()
        print("Built classifiers")

        self.min_hsv_values = pd.read_csv("min_vals.csv", index_col="im")
        print("Loaded minimum HSV values")

    @cached_property
    def cps_lamb(self):
        return geoseries_to_np_xy(self.ref.geometry.to_crs(lamb))

    @cached_property
    def cities_lamb(self):
        return geoseries_to_np_xy(self.cities.geometry.to_crs(lamb))

    def cp_coords_in_im(self, num):
        tr = self.get_lamb_im_transform(num)
        x = np.zeros((len(self.cps_lamb), 3))
        x[:, :2] = self.cps_lamb
        x[:, 2] = 1

        return (tr.T @ x.T).T

    def plot_measurements(self, num):
        im = self.processed[num - 1].copy()

        color = (0, 0, 255)
        radius = 100
        thickness = 5
        for i, row in self.mdf[self.mdf.image == num].iterrows():
            coords = (int(row["x_image"]), int(row["y_image"]))
            cv2.circle(im, coords, radius, color, thickness)
            cv2.circle(im, coords, 5, (255, 0, 0), 4)

        color = (154, 120, 433)
        radius = 50
        thickness = 5
        for x in self.cp_coords_in_im(num):
            cv2.circle(im, tuple(x.astype(int)), radius, color, thickness)

        plot_im(im)
        plt.title(f"Control points for im-{num:02d}")

    def get_im_im_transform(self, ref_num, deriv_num):
        """
        Affine transform from [deriv_num]'s CS to [ref_num].
        """

        cols = ["x_image", "y_image"]
        ref_cps = self.mdf[self.mdf.image == ref_num]
        ref_coords = ref_cps[cols].to_numpy()

        deriv_cps = self.mdf[self.mdf.image == deriv_num]
        deriv_coords = deriv_cps[cols].to_numpy()

        return coordinate_transform(ref_coords, deriv_coords)

    def get_lamb_im_transform(self, num):
        """
        Affine transform from Lambert93 to [num]'s CS.
        """

        cols = ["x_image", "y_image"]
        deriv_cps = self.mdf[self.mdf.image == num]
        deriv_coords = deriv_cps[cols].to_numpy()

        return coordinate_transform(
            ref_coords=deriv_coords,
            deriv_coords=self.lamb_ref(),
        )

    def show_overlay(self, ref_num, deriv_num):
        ref = self.processed[ref_num - 1]
        deriv = self.processed[deriv_num - 1]
        x = self.get_im_im_transform(ref_num, deriv_num)

        warpMat = x.T

        dst_shape = ref.shape[1::-1]
        res = cv2.warpAffine(deriv, warpMat, dst_shape)
        # plot_im(res)
        blended = cv2.addWeighted(ref, 0.5, res, 0.5, 1)
        plot_im(blended)
        plt.title(f"Map {deriv_num} overlaid with map {ref_num}")

    def lamb_ref(self):
        """
        Returns the Lamber93 coordinates of the reference points, as a numpy array.
        """

        geom = self.ref.to_crs(lamb).geometry
        return geoseries_to_np_xy(geom)

    def cities_coords_in_im(self, num):
        tr = self.get_lamb_im_transform(num)

        cl = self.cities_lamb
        x = np.zeros((len(cl), 3))
        x[:, :2] = cl
        x[:, 2] = 1

        return (tr.T @ x.T).T


    def process_image(self, num):
        cc = self.cities_coords_in_im(num)
        cpis = compute_patch_indices(cc)
        cm = self.cities_reprs(num, cc, cpis)
        cov = self.classify_coverage(num, cm)
        self.cities[f"couverture-im-{num:02d}"] = cov

        print(f"Processed image {num:02d}")

        return {
            "im_num": num,
            "coordinates": cc,
            "patch_indices": cpis,
            "representative_colors": cm,
            "classification": cov
        }

    def process_all(self):
        for num in range(1, MAX_IM + 1):
            res = self.process_image(num)
            self.plot_cities_in_im(res, write_to_disk=True)

        self.cities.drop(["geometry"], axis=1).to_csv(output_file)
        print(f"Wrote results to {output_file}")

    def plot_cities_in_im(self, process_result, write_to_disk=False, plot=True):
        num = process_result["im_num"]
        cc = process_result["coordinates"]
        cpis = process_result["patch_indices"]
        cm = process_result["representative_colors"]
        cov = process_result["classification"]

        assert len(cc) == len(cm)
        assert len(cc) == len(cov)

        im = self.processed[num - 1].copy()

        radius = 1
        thickness = 2
        for i, x in enumerate(cc):
            classified_col = class_colors[cov[i]]
            mean_neigh_col = cm[i]
            cv2.circle(im, tuple(x.astype(int)), radius, classified_col, thickness)

            rstart = (cpis[i][0].start, cpis[i][1].start)
            rend = (cpis[i][0].stop, cpis[i][1].stop)
            cv2.rectangle(im, rstart, rend, mean_neigh_col.tolist())

        if write_to_disk:
            im_bgr = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
            file = f"analyzed/im-{num:02d}.png"
            cv2.imwrite(file, im_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 9])
            print(f"Wrote to {file}")
        if plot:
            plot_im(im)
            plt.title(f"Cities in im-{num:02d}")
            print(f"Annotated im-{num:02d}")


    def cities_patches(self, num, cc, pis):
        n = len(cc)

        patch_shape = (side, side, 3)
        patches = np.zeros((n,) + patch_shape, dtype=int)

        # Slow.
        for i, pi in enumerate(pis):
            # Beware of X/Y ordering!
            pi = (pi[1], pi[0], pi[2])

            p = to_shape(self.processed[num - 1][pi], (side, side, 3))
            assert patches[i, :, :, :].shape == p.shape
            patches[i, :, :, :] = p

        return patches

    def cities_reprs(self, num, cities_coordinates, patch_indices):
        patches = self.cities_patches(num, cities_coordinates, patch_indices)

        value_threshold = None
        try:
            value_threshold = self.min_hsv_values.loc[num]["min_val_percent"] / 100.
        except Exception as e:
            pass

        filtered_patches = np.array(list(map(lambda p: patch_repr_color(p, value_threshold=value_threshold), patches)))
        return filtered_patches

    def classify_coverage(self, num, repr_colors):
        """
        Étant donné un tableau `repr_colors` des couleurs RGB représentatives de chaque ville, renvoie un tableau 1D donnant la classification obtenue pour chaque ville:
        - 0 en l'absence de couverture.
        - 1 s'il y a une couverture haut débit.
        - -1 si des données sont manquantes.
        """

        clf = self.clfs[num]

        predicted = np.zeros((repr_colors.shape[0],), dtype=int)
        predicted.fill(-1)  # missing value

        non_na = ~np.any(np.isnan(repr_colors), axis=1)
        non_na_indices = np.argwhere(non_na).reshape(-1)

        predicted[non_na_indices] = clf.predict(repr_colors[non_na_indices])
        return predicted
