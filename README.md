# 🏆 Copa FutBotMX – Capítulo Visión por Computadora
## [NOMBRE_DEL_EQUIPO_O_SOLUCIÓN]

Este repositorio contiene la solución desarrollada por **[NOMBRE_DEL_EQUIPO]** para la **Copa FutBotMX – Capítulo Visión por Computadora** (cumpliendo con la sección 3.5.4 de las bases oficiales). Nuestro pipeline utiliza el modelo **SAM 3 (Segment Anything Model 3)** de Meta para la segmentación, rastreo (tracking) y análisis de videos de partidos de fútbol robótico.

---

## 📖 1. Descripción del Enfoque

Nuestra solución procesa videos de los partidos de robótica y extrae métricas clave (posesión, mapas de calor, eventos) utilizando una arquitectura basada en **SAM 3**.

- **Uso de SAM 3**: Se emplea para la segmentación precisa de los robots, la cancha y el balón.
- **Prompts utilizados**: Se utilizan [TEXTO / CAJAS (BBOX) / PUNTOS] para guiar a SAM 3 durante el análisis frame a frame.
- **Fine-Tuning**: [SE MENCIONA SI HUBO FINE-TUNING EN DATASETS ESPECÍFICOS COMO ROBOCUP, O SI SE USÓ ZERO-SHOT].
- **Integración con Trackers**: La salida de SAM 3 se acopla con **[ByteTrack / DeepSORT / Otro]** para mantener la identidad de los robots de manera consistente a lo largo del tiempo.
- **Post-procesamiento**: Los datos crudos del tracking se filtran con [FILTRO DE KALMAN U OTRO MÉTODO] para suavizar trayectorias y generar mapas de calor precisos.

---

## 💻 2. Requisitos de Hardware y Software

Para ejecutar este proyecto, se recomienda el siguiente entorno:

- **Sistema Operativo**: [Linux Ubuntu 22.04 / Windows 11 / macOS]
- **GPU Recomendada**: [NVIDIA RTX 3060 o superior con al menos 8GB de VRAM / CPU-only si aplica]
- **Lenguaje**: Python [3.10 o superior]
- **Dependencias Principales**:
  - `torch` >= [VERSIÓN]
  - `opencv-python`
  - `supervision` (para el manejo de bounding boxes y métricas)
  - `ultralytics` (para soporte SAM 3)
  - [OTRAS LIBRERÍAS CLAVE]

---

## ⚙️ 3. Instalación y Ejecución

Sigue estos pasos para reproducir nuestra solución en tu equipo local:

### 3.1 Clonar el repositorio
```bash
git clone [https://github.com/kevin4moe/Computer-Vision-Copa-FutBotMX.git](https://github.com/kevin4moe/Computer-Vision-Copa-FutBotMX.git)
cd Computer-Vision-Copa-FutBotMX