import json
from loaders.doc3dwc_loader import doc3dwcLoader
from loaders.doc3dgie_loader import doc3dgieLoader
from loaders.doc3dgie_real_data_loader import doc3dgierealLoader
from loaders.doc3dbmp import doc3dbmpLoader
from loaders.doc3dbmnoimgc_loader import doc3dbmnoimgcLoader
from loaders.doc3dcrbm import doc3dcrLoader
from loaders.doc3dcrfm import doc3dcrfmLoader
from loaders.doc3dcrfm_synth import doc3dcrfmsynthLoader
from loaders.doc3dcrfm_classify import doc3dcrfmclassifyLoader
from loaders.doc3dcrfm_synth_v2 import doc3dcrfmsynthv2Loader
from loaders.doc3dcrfm_patch import doc3dcrfmpatchLoader


def get_loader(name):
    """get_loader

    :param name:
    """
    return {
        'doc3dwc':doc3dwcLoader,
        'doc3dgie':doc3dgieLoader,
        'doc3dgie_real':doc3dgierealLoader,
        'doc3dbmnic':doc3dbmnoimgcLoader,
        'doc3dbmp':doc3dbmpLoader,
        'doc3dcr':doc3dcrLoader,
        'doc3dcrfm':doc3dcrfmLoader,
        'doc3dcrfmsynth':doc3dcrfmsynthLoader,
        'doc3dcrfmclassify':doc3dcrfmclassifyLoader,
        'doc3dcrfmsynthv2':doc3dcrfmsynthv2Loader,
        'doc3dcrfmpatch':doc3dcrfmpatchLoader
    }[name]
