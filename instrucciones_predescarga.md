# Instrucciones — Dataset `plant_leaves` en YUCA

Actividad 03: Transfer Learning con `plant_leaves` (4,502 imágenes, 22 clases).
Este documento asume que YUCA sí tiene internet. La descarga se hace una vez y el entrenamiento solo lee el dataset desde disco.

## 1. Verificar el entorno

```bash
source activate pinguinos
TF_CPP_MIN_LOG_LEVEL=3 python -c "import tensorflow as tf, tensorflow_datasets as tfds; print('TF', tf.__version__, 'TFDS', tfds.__version__); print(tfds.list_versions('plant_leaves'))"
```

- El aviso `Could not find cuda drivers... GPU will not be used` es normal en el login node o en un nodo CPU. La GPU solo se ve dentro de un job con GPU.
- Anotar las versiones y las versiones disponibles de `plant_leaves`.

## 2. Instalar TFDS si falta

Si `pinguinos` ya tiene `tensorflow_datasets`, no instalar nada.

Si falta y hay internet:

```bash
python -c "import tensorflow_datasets" 2>/dev/null || pip install tensorflow-datasets
```

Si el entorno es compartido y no hay permisos:

```bash
pip install --user tensorflow-datasets
```

Cuidado: si `pip` intenta cambiar TensorFlow o `numpy`, detenerse. Eso puede romper `pinguinos`.

## 3. Descargar el dataset en YUCA

Elegir una ruta con espacio (home o scratch):

```bash
export TFDS_DATA_DIR=$HOME/tfds_data   # o /scratch/$USER/tfds_data
mkdir -p "$TFDS_DATA_DIR"
```

Descargar y preparar:

```bash
python - << 'PY'
import os
import tensorflow_datasets as tfds
tfds.load("plant_leaves", split="train", data_dir=os.environ["TFDS_DATA_DIR"])
PY
```

- Descarga: 6.56 GiB. Dataset preparado: 6.81 GiB. Reservar ~15 GB.
- Hacerlo en el login node si tiene internet, para no gastar tiempo de cómputo.
- Si falla alguna imagen, ejecutar de nuevo; reanuda.

Verificar:

```bash
ls "$TFDS_DATA_DIR/plant_leaves"
```

## 4. Usar el dataset en el job

En el `.slrm`:

```bash
export TFDS_DATA_DIR=$HOME/tfds_data
```

En Python:

```python
import os
import tensorflow_datasets as tfds

ds = tfds.load(
    "plant_leaves",
    split="train",
    data_dir=os.environ["TFDS_DATA_DIR"],
)
```

Si se indica `data_dir` y el dataset ya existe, TFDS no descarga nada.

## 5. GPU en el `.slrm`

Confirmar la partición GPU:

```bash
sinfo -o "%P %G %l %D"
```

Agregar al script:

```bash
#SBATCH --partition=<particion_gpu>
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=plant_leaves_%j.out
#SBATCH --error=plant_leaves_%j.err
```

Dentro del job:

```bash
nvidia-smi
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

Si `nvidia-smi` funciona pero TensorFlow no ve la GPU, cargar el módulo de CUDA correspondiente o revisar el entorno; avisar con la salida.

## 6. Si los nodos de cómputo no tienen internet

No hay problema si el login node sí tiene internet: la descarga del paso 3 se hace ahí y el job solo lee el dataset.

Si ni el login tiene internet, usar el respaldo: descargar en un equipo local, empaquetar `plant_leaves/` y subirlo con `scp`/`rsync`.

## 7. Datos que faltan

- Salida real de versiones (la del mensaje anterior no mostró las líneas `TF`/`TFDS`).
- `sinfo -o "%P %G %l %D"`.
- Ruta y cuota de disco disponibles (`$HOME` vs scratch).
- Tiempo máximo permitido por job.
