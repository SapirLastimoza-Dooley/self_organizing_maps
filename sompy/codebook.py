import numpy as np
import scipy as sp
import sys
sys.path.append(r"C:\DevSource\Shuyang-GEOG676\Projects\sompy")
from sklearn.decomposition import PCA
from decorators import timeit


class InvalidNodeIndexError(Exception):
    pass


class InvalidMapsizeError(Exception):
    pass

def generate_hex_lattice(n_rows, n_columns):
    x_coord = []
    y_coord = []
    for i in range(n_rows):
        for j in range(n_columns):
            x_coord.append(i*1.5)
            y_coord.append(np.sqrt(2/3)*(2*j+(1+i)%2))
    coordinates = np.column_stack([x_coord, y_coord])
    return coordinates


class Codebook(object):

    def __init__(self, mapsize, lattice='rect'):
        self.lattice = lattice

        if 2 == len(mapsize):
            _size = [1, np.max(mapsize)] if 1 == np.min(mapsize) else mapsize

        elif 1 == len(mapsize):
            _size = [1, mapsize[0]]
            print('input was considered as the numbers of nodes')
            print('map size is [{dlen},{dlen}]'.format(dlen=int(mapsize[0]/2)))
        else:
            raise InvalidMapsizeError(
                "Mapsize is expected to be a 2 element list or a single int")

        self.mapsize = _size
        self.nnodes = self.mapsize[0]*self.mapsize[1]
        self.matrix = np.asarray(self.mapsize)
        self.initialized = False

        if lattice == "hexa":
            n_rows, n_columns = mapsize
            coordinates = generate_hex_lattice(n_rows, n_columns)
            self.lattice_distances = (sp.spatial.distance_matrix(coordinates, coordinates)
                                      .reshape(n_rows * n_columns, n_rows, n_columns))

    @timeit()
    def random_initialization(self, data):
        mn = np.tile(np.min(data, axis=0), (self.nnodes, 1))
        mx = np.tile(np.max(data, axis=0), (self.nnodes, 1))
        self.matrix = mn + (mx-mn)*(np.random.rand(self.nnodes, data.shape[1]))
        self.initialized = True

    @timeit()
    def pca_linear_initialization(self, data):
        cols = self.mapsize[1]
        coord = None
        pca_components = None

        if np.min(self.mapsize) > 1:
            coord = np.zeros((self.nnodes, 2))
            pca_components = 2

            for i in range(0, self.nnodes):
                coord[i, 0] = int(i / cols)  # x
                coord[i, 1] = int(i % cols)  # y

        elif np.min(self.mapsize) == 1:
            coord = np.zeros((self.nnodes, 1))
            pca_components = 1

            for i in range(0, self.nnodes):
                coord[i, 0] = int(i % cols)  # y

        mx = np.max(coord, axis=0)
        mn = np.min(coord, axis=0)
        coord = (coord - mn)/(mx-mn)
        coord = (coord - .5)*2
        me = np.mean(data, 0)
        data = (data - me)
        tmp_matrix = np.tile(me, (self.nnodes, 1))

        pca = PCA(n_components=pca_components, svd_solver='randomized')

        pca.fit(data)
        eigvec = pca.components_
        eigval = pca.explained_variance_
        norms = np.sqrt(np.einsum('ij,ij->i', eigvec, eigvec))
        eigvec = ((eigvec.T/norms)*eigval).T

        for j in range(self.nnodes):
            for i in range(eigvec.shape[0]):
                tmp_matrix[j, :] = tmp_matrix[j, :] + coord[j, i]*eigvec[i, :]

        self.matrix = np.around(tmp_matrix, decimals=6)
        self.initialized = True

    def grid_dist(self, node_ind):
        if self.lattice == 'rect':
            return self._rect_dist(node_ind)

        elif self.lattice == 'hexa':
            return self._hexa_dist(node_ind)

    def _hexa_dist(self, node_ind):
        return self.lattice_distances[node_ind]

    def _rect_dist(self, node_ind):
        rows = self.mapsize[0]
        cols = self.mapsize[1]
        dist = None

        if 0 <= node_ind <= (rows*cols):
            node_col = int(node_ind % cols)
            node_row = int(node_ind / cols)
        else:
            raise InvalidNodeIndexError(
                "Node index '%s' is invalid" % node_ind)

        if rows > 0 and cols > 0:
            r = np.arange(0, rows, 1)[:, np.newaxis]
            c = np.arange(0, cols, 1)
            dist2 = (r-node_row)**2 + (c-node_col)**2

            dist = dist2.ravel()
        else:
            raise InvalidMapsizeError(
                "One or both of the map dimensions are invalid. "
                "Cols '%s', Rows '%s'".format(cols=cols, rows=rows))

        return dist