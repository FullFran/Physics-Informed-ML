**Capítulo X+4: Diseño de Funciones de Pérdida en Aprendizaje Automático Informado por Física**  
**Título: Integración de Principios Físicos mediante Funciones de Pérdida**  

---

### 1. Introducción  
La función de pérdida es el núcleo que guía el entrenamiento de modelos de aprendizaje automático (ML). En el contexto de ML informado por física, diseñar funciones de pérdida que incorporen conocimientos físicos (e.g., ecuaciones diferenciales, conservación de energía) es crucial para garantizar generalización y precisión. Este capítulo detalla métodos para construir funciones de pérdida que integren física, desde redes neuronales informadas hasta regularización por espacialidad.

---

### 2. Redes Neuronales Informadas por Física (PINNs)  

#### **2.1 Estructura de la Función de Pérdida**  
- **Componentes**:  
  - **Pérdida de Datos ($(\mathcal{L}_{\text{data}}$))**: Mide el error entre predicciones y datos observados.  
  - **Pérdida Física ($(\mathcal{L}_{\text{physics}}$))**: Penaliza violaciones de ecuaciones gobernantes (e.g., Navier-Stokes).  
- **Formulación**:  
  $$
  \mathcal{L} = \mathcal{L}_{\text{data}} + \lambda \mathcal{L}_{\text{physics}},
  $$
  donde $(\lambda$) balancea la importancia de la física vs. ajuste a datos.  

#### **2.2 Ejemplo: Ecuación de Navier-Stokes**  
- **Variables**: Velocidad (\(u, v\)), presión (\(p\)).  
- **Pérdida Física**:  
  $$
  \mathcal{L}_{\text{physics}} = \frac{1}{N} \sum_{i=1}^N \left( \frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} + v \frac{\partial u}{\partial y} + \frac{\partial p}{\partial x} - \nu \nabla^2 u \right)^2 + \text{Términos para } v \text{ y } \nabla \cdot \mathbf{u}.
  $$
- **Implementación en PyTorch**:  
  ```python
  def physics_loss(u, v, p, x, y, t, nu):
      u_t = grad(u, t, create_graph=True)
      u_x = grad(u, x, create_graph=True)
      u_y = grad(u, y, create_graph=True)
      # Calcular términos de Navier-Stokes
      momentum_x = u_t + u*u_x + v*u_y + grad(p, x) - nu*(grad(grad(u, x), x) + grad(grad(u, y), y))
      return torch.mean(momentum_x**2)
  ```

#### **2.3 Caso de Estudio: Flujo en Tubería**  
- **Resultados**: PINNs reducen error en presión un 30% vs. modelos sin física.  
- **Hiperparámetros**: $(\lambda = 0.1-1.0$), optimizador Adam con tasa de aprendizaje $(10^{-3}$).

---

### 3. Redes Lagrangiana y Hamiltonianas  

#### **3.1 Conservación de Energía en Pérdida**  
- **Lagrangiana ($(\mathcal{L}$))**:  
  $$
  \mathcal{L}(q, \dot{q}) = T(\dot{q}) - V(q).
  $$
- **Pérdida**:  
  $$
  \mathcal{L}_{\text{Hamilton}} = \left( \frac{\partial H}{\partial p} - \dot{q} \right)^2 + \left( \frac{\partial H}{\partial q} + \dot{p} \right)^2.
  $$

#### **3.2 Implementación en TensorFlow**  
```python
def hamiltonian_loss(q, p, dq_dt, dp_dt):
    with tf.GradientTape() as tape:
        H = model_H(q, p)
    dH_dq = tape.gradient(H, q)
    dH_dp = tape.gradient(H, p)
    loss = tf.reduce_mean((dH_dp - dq_dt)**2 + (dH_dq + dp_dt)**2)
    return loss
```

---

### 4. Esparcidad y Parsimonia con SINDy  

#### **4.1 Pérdida Regularizada L1**  
- **Formulación**:  
  $$
  \mathcal{L} = \underbrace{\|\dot{\mathbf{X}} - \mathbf{\Theta}(\mathbf{X})\mathbf{\Xi}\|_2}_{\text{Error de datos}} + \lambda \underbrace{\|\mathbf{\Xi}\|_1}_{\text{Esparcidad}}.
  $$
- **Biblioteca ($(\mathbf{\Theta}$))**: Incluye polinomios, funciones trigonométricas, etc.  

#### **4.2 Ejemplo: Péndulo Caótico**  
- **Ecuación Descubierta**:  
  $$
  \dot{x} = \sigma(y - x), \quad \dot{y} = x(\rho - z) - y, \quad \dot{z} = xy - \beta z.
  $$
- **Implementación**:  
  ```python
  from sklearn.linear_model import Lasso
  model = Lasso(alpha=0.1)
  model.fit(Theta, X_dot)
  xi = model.coef_
  ```

---

### 5. Invarianzas y Simetrías en la Pérdida  

#### **5.1 Pérdida para Invarianza Rotacional**  
- **Transformación**: $( \mathbf{x}' = R(\theta)\mathbf{x}$).  
- **Pérdida**:  
  $$
  \mathcal{L}_{\text{sym}} = \|f(R\mathbf{x}) - R f(\mathbf{x})\|^2.
  $$

#### **5.2 Ejemplo: Clasificación de Imágenes Rotadas**  
```python
def symmetry_loss(images, model, rotation_fn):
    rotated_images = rotation_fn(images)
    pred_original = model(images)
    pred_rotated = model(rotated_images)
    loss = torch.mean((rotated_images - rotation_fn(pred_original))**2)
    return loss
```

---

### 6. Funciones de Pérdida Multiobjetivo  

#### **6.1 Balance entre Objetivos**  
- **Ponderación Adaptativa**: Ajustar $(\lambda$) dinámicamente durante el entrenamiento.  
- **Ejemplo**:  
  $$
  \lambda(t) = \lambda_0 \cdot e^{-kt}.
  $$

#### **6.2 Caso de Estudio: Diseño Aerodinámico**  
- **Objetivos**: Minimizar arrastre ($(C_d$)), maximizar sustentación ($(C_l$)).  
- **Pérdida**:  
  $$
  \mathcal{L} = 0.7 \cdot C_d + 0.3 \cdot (1 - C_l).
  $$

---

### 7. Retos y Mejores Prácticas  

#### **7.1 Dificultades Comunes**  
- **Desbalance de Pérdidas**: Usar normalización o ponderación adaptativa.  
- **Derivadas Numéricas**: Emplear diferenciación automática para precisión.  

#### **7.2 Consejos para Implementación**  
1. **Inicialización**: Preentrenar con $(\mathcal{L}_{\text{data}}$) antes de introducir $(\mathcal{L}_{\text{physics}}$).  
2. **Hiperparámetros**: Grid-search para $(\lambda$) en rango logarítmico (e.g., $(10^{-3}$) a $(10^{1}$)).  
3. **Verificación**: Validar físicamente (e.g., conservación de masa en CFD).

---

### 8. Caso de Estudio Integrado: Modelado de Olas Marinas  

#### **8.1 Problema**: Predecir altura de olas (\(h\)) con Ecuación de Aguas Someras.  
#### **8.2 Pérdida**:  
$$
\mathcal{L} = \underbrace{\frac{1}{N} \sum (h_{\text{pred}} - h_{\text{obs}})^2}_{\text{Datos}} + \lambda \underbrace{\left( \frac{\partial h}{\partial t} + \nabla \cdot (h \mathbf{u}) \right)^2}_{\text{Conservación de Masa}}.
$$
#### **8.3 Resultados**: Error RMSE reducido un 40% vs. modelo sin física.

---

**Conclusión**  
Las funciones de pérdida informadas por física transforman modelos de ML en herramientas robustas para aplicaciones científicas. Desde PINNs hasta regularización por espacialidad, integrar conocimiento físico mejora generalización y reduce necesidad de datos. Al combinar múltiples objetivos y técnicas de optimización, es posible diseñar modelos que no solo predicen, sino que respetan las leyes fundamentales de la naturaleza.
