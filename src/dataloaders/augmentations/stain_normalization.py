import numpy as np


def normalize_stain_macenko(img, tli: int = 240, alpha: int = 1, 
                            beta: int = 0.15):
    """Normalize staining appearence of H&E stained images

    Parameters
    ----------
    img :
        RGB image.
    tli : int
        Transmitted light intensity.
    alpha : int
        Percentile.
    beta : int
        Transparency threshold.

    Returns
    -------
    img_norm:
        Stain normalized image.
    hem:
        hematoxylin image.
    eos:
        eosin image.
    """
    he_ref = np.array([[0.5626, 0.2159],
                       [0.7201, 0.8012],
                       [0.4062, 0.5581]])
    max_c_ref = np.array([1.9705, 1.0308])
    h, w = img.shape[:2]
    img = img.reshape((-1, 3))

    # Calculate optical density
    od = -np.log((img.astype(float)+1) / tli)

    # Remove transparent pixels
    od_hat = od[~np.any(od < beta, axis=1)]

    # Compute eigenvectors
    try:
        eigvals, eigvecs = np.linalg.eigh(np.cov(od_hat.T))
    except np.linalg.LinAlgError as err:
        print(f'Error in computing eigenvectors: {err}')
        raise

    # Project on the plane spanned by the eigenvectors corresponding 
    # to the two largest eigenvalues
    that = od_hat.dot(eigvecs[:, 1:3])
    phi = np.arctan2(that[:, 1], that[:, 0])
    min_phi = np.percentile(phi, alpha)
    max_phi = np.percentile(phi, 100-alpha)
    vMin = eigvecs[:, 1:3].dot(np.array([(np.cos(min_phi), np.sin(min_phi))]).T)
    vMax = eigvecs[:, 1:3].dot(np.array([(np.cos(max_phi), np.sin(max_phi))]).T)

    # Make the vector corresponding to hematoxylin first and
    # the one corresponding to eosin second
    if vMin[0] > vMax[0]:
        he = np.array((vMin[:,0], vMax[:,0])).T
    else:
        he = np.array((vMax[:,0], vMin[:,0])).T

    # Rows correspond to channels (RGB), columns to od values
    Y = np.reshape(od, (-1, 3)).T

    # Determine concentrations of the individual stains
    c = np.linalg.lstsq(he, Y, rcond=None)[0]

    # Normalize stain concentrations
    max_c = np.array([np.percentile(c[0, :], 99), np.percentile(c[1, :], 99)])
    tmp = np.divide(max_c, max_c_ref)
    c2 = np.divide(c, tmp[:, np.newaxis])

    # Recreate the image using reference mixing matrix
    img_norm = np.multiply(tli, np.exp(-he_ref.dot(c2)))
    img_norm[img_norm > 255] = 254
    img_norm = np.reshape(img_norm.T, (h, w, 3)).astype(np.uint8)

    # Unmix hematoxylin and eosin
    hem = np.multiply(tli, np.exp(np.expand_dims(-he_ref[:, 0], axis=1).dot(
        np.expand_dims(c2[0, :], axis=0))))
    hem[hem > 255] = 254
    hem = np.reshape(hem.T, (h, w, 3)).astype(np.uint8)

    eos = np.multiply(tli, np.exp(np.expand_dims(-he_ref[:, 1], axis=1).dot(
        np.expand_dims(c2[1, :], axis=0))))
    eos[eos > 255] = 254
    eos = np.reshape(eos.T, (h, w, 3)).astype(np.uint8)

    return img_norm, hem, eos

