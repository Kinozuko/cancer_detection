import cv2
import numpy as np
import dicomsdl

RESIZE = (512,512) # Tamano al cual se creara la imagen png

def dicom_to_array(path: str):
    dcm_file = dicomsdl.open(path) # Se lee el archivo dcm
    data = dcm_file.pixelData() # Se extrae la data de los pixeles

    data = (data - data.min()) / (data.max() - data.min()) # Se normalizan

    if dcm_file.getPixelDataInfo()['PhotometricInterpretation'] == "MONOCHROME1":
        # se aplica un resize a la imagen
        data = 1 - data

    data = cv2.resize(data, RESIZE) # Se hace un resize al formato que lo deseamos
    data = (data * 255).astype(np.uint8) # Se multiplica por la cantidad de pixeles a usar
    return data
