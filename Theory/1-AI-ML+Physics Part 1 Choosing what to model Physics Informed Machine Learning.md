**Chapter X: Problem Formulation in Physics-Informed Machine Learning**  
**Title: Choosing What to Model: Integrating Physics and Data-Driven Objectives**

https://youtu.be/ARMk955pGbg?si=F_3toznMTz410To4

---

### 1. Introduction  
The foundational stage of any machine learning (ML) pipeline—defining the problem to model—is critical in physics-informed ML. This chapter explores how the integration of physical principles and engineering design processes guides the selection of modeling tasks, ensuring alignment with scientific goals and computational feasibility. The iterative nature of problem formulation is emphasized, where refining objectives based on data constraints, model performance, and downstream applications is essential.

---

### 2. The Role of Problem Formulation in Scientific ML  
Problem formulation bridges traditional scientific inquiry and data-driven modeling. Key principles include:  
- **Iterative Refinement**: Analogous to experimental design, initial hypotheses are revised using insights from data, architecture limitations, and model validation.  
- **Purpose-Driven Modeling**: Objectives must align with practical applications (e.g., design optimization, discovering unknown physics) rather than abstract curiosity.  
- **Human-Centric Design**: As emphasized by Picasso’s adage, “Computers are useless—they can only give answers,” the modeler defines the problem’s scope, fidelity, and utility.

---

### 3. Key Considerations for Problem Selection  
#### 3.1 When to Use Machine Learning  
- **Unknown Physics**: Systems with incomplete first-principles models (e.g., turbulence closure, neural activity, cloud formation).  
- **Expensive Simulations**: Surrogate models for accelerating computational tasks (e.g., fluid dynamics, molecular dynamics).  
- **Multiphysics Complexity**: Systems requiring hybrid models (e.g., climate science, where atmospheric dynamics couple with microphysical processes).  
- **Adaptive Systems**: Digital twins updated with real-time data (e.g., predictive maintenance, control systems).  

#### 3.2 When to Avoid Machine Learning  
- **Overkill Solutions**: Problems solvable with simpler methods (e.g., linear regression for low-dimensional dynamics).  
- **Misaligned Objectives**: Tasks where data lacks causal relevance (e.g., astrology vs. orbital mechanics).  

---

### 4. Methodological Approaches  
#### 4.1 Task-Specific Modeling  
- **Super-Resolution**: Enhancing experimental or simulation data fidelity while preserving conservation laws (e.g., jet flow reconstruction).  
- **Physics Discovery**: Extracting interpretable laws from data (e.g., rediscovering Newtonian mechanics or novel plasma dynamics).  
- **Digital Twins**: Hierarchical models combining low/high-fidelity data for predictive optimization (e.g., wind turbine design).  
- **Shape Optimization**: Leveraging automatic differentiation for adjoint-free optimization (e.g., aerodynamic surfaces).  

#### 4.2 Mathematical Frameworks  
- **Continuous vs. Discrete Dynamics**: Choosing between neural ODEs (continuous) and ResNet-style discretizations.  
- **Hamiltonian/Lagrangian Systems**: Encoding energy conservation directly into architectures.  
- **Uncertainty Quantification**: Probabilistic predictions for chaotic systems (e.g., climate modeling).  

---

### 5. Challenges and Best Practices  
- **Chaos and Sensitivity**: Long-term predictions in chaotic systems (e.g., Lorenz attractor) require probabilistic approaches.  
- **Benchmarking**: Standardized tasks (e.g., fluid mechanics benchmarks) to evaluate model generalizability.  
- **Symmetry and Invariance**: Embedding physical constraints (e.g., rotational invariance in turbulence models).  
- **Data-Physics Synergy**: Balancing data availability with physical priors (e.g., Reynolds-averaged Navier-Stokes closures).  

---

### 6. Case Studies  
#### 6.1 Turbulence Modeling (Julia Ling et al.)  
- **Challenge**: Reynolds stress closure in turbulent flows.  
- **Solution**: ML models constrained by Galilean invariance and conservation laws.  
- **Impact**: Demonstrated ML’s capability to extend decades-old analytical efforts.  

#### 6.2 Climate Science  
- **Task**: Multiscale cloud modeling using hybrid physics-ML frameworks.  
- **Outcome**: Improved parameterization of subgrid-scale processes in climate simulations.  

#### 6.3 Materials Discovery  
- **Objective**: Accelerated drug/protein design via surrogate models.  
- **Tool**: Differentiable ML architectures for gradient-based optimization.  

---

### 7. Conclusion  
Problem formulation in physics-informed ML is both an art and science. It demands domain expertise to identify gaps where data-driven methods complement physical principles, while avoiding overcomplication. Success hinges on iterative refinement, alignment with real-world applications, and adherence to the scientific method. Future work must prioritize benchmarking and interdisciplinary collaboration to advance the field’s maturity.

---

**Key Takeaways**  
- Define clear, purpose-driven objectives informed by physical knowledge.  
- Iterate between problem formulation, data curation, and model validation.  
- Leverage ML for tasks where physics is incomplete, computationally prohibitive, or adaptive.  
- Avoid ML when simpler models suffice or tasks lack scientific grounding.  

This chapter underscores that the choice of *what to model* is as critical as the modeling itself, shaping the trajectory of scientific discovery and engineering innovation.