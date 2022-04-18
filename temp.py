import numpy as np 



points = np.random.random((5, 2))

distMatrix = np.sum((points[:, np.newaxis, :] - points[np.newaxis, :, :]) ** 2, axis=-1)