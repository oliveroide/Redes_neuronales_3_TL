# Diagnóstico Agrícola Inteligente: Clasificación de 22 Enfermedades en Hojas de Plantas usando Transfer Learning y MobileNetV2

*Cómo adaptar una red neuronal preentrenada en ImageNet para detectar patologías foliares con alta precisión y baja huella de memoria para aplicaciones móviles.*

---

## 🌿 Introducción: El Desafío de la Salud Vegetal en la Era Digital

La detección temprana de patologías en cultivos es uno de los mayores retos de la agricultura moderna. Una plaga o enfermedad foliar no diagnosticada a tiempo puede devastar cosechas enteras en cuestión de semanas. Tradicionalmente, la inspección fitosanitaria requiere agrónomos expertos en campo; sin embargo, con los avances recientes en **Visión por Computadora** y **Aprendizaje Profundo (Deep Learning)**, hoy es posible llevar diagnósticos confiables al bolsillo de cualquier agricultor mediante un teléfono móvil.

En este artículo exploramos cómo entrenar un clasificador robusto sobre el dataset **`plant_leaves`** (4,502 imágenes distribuidas en 22 clases), implementando una estrategia completa de **Transfer Learning** y **Fine-Tuning** basada en **MobileNetV2**.

---

## 📦 1. El Dataset: `plant_leaves`

El conjunto de datos proviene de [TensorFlow Datasets (TFDS)](https://www.tensorflow.org/datasets/catalog/plant_leaves) y presenta particularidades técnicas desafiantes:
* **Volumen:** 4,502 imágenes fotográficas reales.
* **Gran número de clases:** 22 categorías botánicas y patológicas (diferenciando hojas saludables de aquellas afectadas por hongos, bacterias o deficiencias nutricionales).
* **Resolución masiva:** Las fotografías crudas tienen dimensiones de hasta **$4000 \times 6000$ píxeles**. Procesar estas imágenes directamente desbordaría la memoria GPU/RAM de cualquier entorno; por ello, es indispensable construir un pipeline de entrada altamente eficiente.

### Pipeline de Datos Optimizado (`tf.data`)
Para manejar los datos sin cuellos de botella de I/O:
1. Dividimos el dataset en **80% Entrenamiento**, **10% Validación** y **10% Prueba**.
2. Redimensionamos cada imagen a **$224 \times 224 \times 3$**.
3. Encadenamos `.cache()` (para retener tensores procesados en RAM tras la primera época) y `.prefetch(buffer_size=tf.data.AUTOTUNE)` (para permitir que la CPU prepare el lote siguiente mientras la GPU entrena el actual).

---

## 🏗️ 2. ¿Por qué MobileNetV2 para este Problema?

Entrenar una red convolucional profunda desde cero sobre 4,502 imágenes provocaría dos problemas graves: sobreajuste inmediato (*overfitting*) y semanas de entrenamiento. La solución natural es **Transfer Learning**: reutilizar los filtros visuales que una red ya aprendió al ser entrenada sobre **ImageNet-1K** ($1.3$ millones de imágenes).

Entre todas las arquitecturas disponibles en `tf.keras.applications`, seleccionamos **MobileNetV2** por razones estratégicas:
* **Peso y Memoria:** Con apenas **$\approx 3.5$ millones de parámetros** y un tamaño en disco de **$13.5\text{ MB}$**, es drásticamente más ligera que gigantes como VGG16 ($138\text{M}$ parámetros, $>500\text{ MB}$).
* **Convoluciones Separables en Profundidad:** Factoriza la convolución estándar en un filtrado espacial por canal y una mezcla puntual $1\times 1$, reduciendo el cómputo en más de un 80% sin sacrificar capacidad representativa.
* **Idoneidad para Aplicaciones Móviles:** Permite exportar a **TensorFlow Lite (TFLite)** o **TensorFlow.js**, posibilitando inferencia en tiempo real en la cámara de un smartphone sin necesidad de conexión a internet en el campo.

---

## ⚙️ 3. La Estrategia en Dos Fases: Feature Extraction y Fine-Tuning

### Regla de Oro: La trampa de `BatchNormalization`
Cuando se utiliza una red con capas de normalización por lotes (`BatchNormalization`), no basta con fijar `base_model.trainable = False`. Si se llama al modelo base sin especificar `training=False`, BatchNorm continuará actualizando sus medias y varianzas móviles con los nuevos lotes, destruyendo las representaciones preentrenadas. En nuestro diseño usamos la **API Funcional de Keras**:

```python
inputs = keras.Input(shape=(224, 224, 3))
x = data_augmentation(inputs)
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
x = base_model(x, training=False)  # Mantiene BatchNorm en modo inferencia
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(22, activation='softmax', kernel_initializer='glorot_uniform')(x)
model = keras.Model(inputs, outputs)
```

### Fase 1: Extracción de Características (*Feature Extraction*)
* El modelo base permanece completamente congelado.
* Entrenamos únicamente el cabezal clasificador denso (`GlobalAveragePooling2D` + `Dropout(0.2)` + `Dense(22)`).
* Optimizador: `Adam(learning_rate=0.001)`.
* En pocas épocas, el clasificador aprende a asociar los mapas de características preexistentes con las 22 enfermedades de hojas.

### Fase 2: Ajuste Fino (*Fine-Tuning*)
* Una vez estabilizado el cabezal, descongelamos las capas convolucionales superiores del modelo base (a partir de la capa 100), manteniendo congeladas las capas iniciales (que detectan bordes y texturas genéricas).
* **Paso crítico:** Reducir la tasa de aprendizaje a `1e-5` ($100\times$ menor). Una tasa grande sobre capas descongeladas alteraría bruscamente los pesos preentrenados (*catastrophic forgetting*).

---

## 📈 4. Evaluación y Resultados

* **Curvas de Aprendizaje:** Monitoreamos simultáneamente la pérdida (*Loss*) y exactitud (*Accuracy*) en entrenamiento y validación, evidenciando una convergencia suave tras el punto de inicio del *Fine-Tuning*.
* **Matriz de Confusión ($22 \times 22$):** Evaluada sobre el 10% de prueba reservado con `shuffle=False`. Permite auditar qué patologías presentan patrones visuales similares (por ejemplo, manchas tempranas por hongos frente a clorosis bacteriana) y verificar que no existan clases con tasa de error desmedida.

---

## 🚀 Conclusiones y Próximos Pasos

El Aprendizaje por Transferencia demuestra ser la técnica más eficiente para abordar problemas complejos de clasificación de imágenes en dominios especializados. Gracias a la arquitectura liviana de **MobileNetV2**, el modelo resultante es tanto preciso como ideal para su empaquetado en una aplicación móvil offline.

El código fuente completo, el notebook interactivo y los scripts de entrenamiento están disponibles en GitHub:  
👉 **[Repositorio en GitHub: Redes_neuronales_3_TL](https://github.com/oliveroide/Redes_neuronales_3_TL)**

---

*Proyecto desarrollado como parte del curso de Redes Neuronales y Aprendizaje Profundo.*
