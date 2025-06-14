# PV-Defect-ML-Dataset
High-quality, processed SCAPS-1D simulation dataset for ZnO/CdS/CIGS solar cells. Features ~4100 unique J-V/defect configurations, enabling machine learning to predict microscopic defects from macroscopic performance and supporting advanced visualization.
________________________________________
SCAPS-1D Simulated ZnO/CdS/CIGS Solar Cell Defect Dataset for AI/ML and Advanced Data Visualization Applications
This repository hosts a meticulously curated dataset derived from high-throughput SCAPS-1D simulations of ZnO/CdS/CIGS thin-film solar cells. This dataset is specifically designed to bridge the critical gap between macroscopic device performance and microscopic defect mechanisms, enabling advanced data visualization and machine learning applications in photovoltaic research.
________________________________________
1. Introduction and Motivation
The advancement of solar photovoltaic technologies, particularly thin-film CIGS, is crucial for renewable energy. While macroscopic J-V measurements provide performance parameters, they often fail to directly reveal underlying microscopic mechanisms, such as bulk and interface defects, which critically determine device performance. Although techniques like DLTS and admittance spectroscopy can probe defect states, their cost and complexity limit widespread adoption. This creates a strong motivation for developing methods that can extract comprehensive defect information from routine J-V characterization, providing deeper material insights without specialized instrumentation.
Recent advancements in Artificial Intelligence (AI) and Machine Learning (ML) offer a transformative opportunity to achieve this. ML algorithms have shown remarkable success in identifying hidden patterns for defect classification, performance prediction, and design optimization in the photovoltaic domain. A key novelty of this study is the aim to correlate J-V response patterns with specific defect configurations.
However, two primary challenges hinder the development of robust AI models for solar cell diagnostics:
1.	Scarcity of high-quality, well-annotated experimental datasets linking defect properties to J-V characteristics. This dataset directly addresses this critical challenge, as obtaining such relevant and diverse experimental data through traditional means is often practically impossible.
2.	The inherent noise and variability in real-world measurements which complicate model training.
To overcome these data limitations, this work proposes a computational framework integrating high-throughput solar cell simulations with AI-driven data analysis, using CIGS solar cells as a case study. This integrated approach, believed to be the first of its kind, enables defect diagnostics in thin-film photovoltaics through AI/ML algorithms based on routine J-V curves. SCAPS-1D simulations serve as an accurate tool to generate synthetic, yet physically realistic, J-V datasets across controlled defect configurations, thereby overcoming experimental limitations and offering the necessary granularity for advanced data visualization, analysis, and interpretable ML model training.
________________________________________
2. Simulated ZnO/CdS/CIGS Solar Cell Structure
The SCAPS-1D simulated thin-film solar cell consists of n-ZnO window, n-CdS buffer, and p-CIGS absorber layers. To mimic real-world performance, defect states were included in all layers and interfaces. Table 1 details the general layer parameters and optical properties. Table 2 outlines the specific defect configurations chosen to represent key recombination pathways, including CIGS bulk defects (shallow Se and deep Cu vacancies) and interface states at the ZnO/CdS and CdS/CIGS junctions.
Table 1: General layer parameters and optical properties for solar cell simulation.
Parameter	n-ZnO	n-CdS	p-CIGS
Thickness (nm)	50	100	3000
Bandgap (eV)	3.3	2.4	1.15
Electron Affinity (eV)	4.4	4.2	4.5
Doping (cm⁻³)	1×10¹⁸ (n)	1×10¹⁷ (n)	1×10¹⁶ (p)
Mobility (cm²/Vs)	μₙ:100, μₚ:25	μₙ:50, μₚ:10	μₙ:30, μₚ:5
Optical Model	Forouhi-Bloomer	Adachi	Mudryi
Table 2: Defect Configurations and Recombination Effects.
| No. | Defect | Defect Type | Energy Level | Density | Potential impact on performance |
| :-- | :----------------- | :------------- | :------------- | :-------------------------- | :-------------------------------------------------- |
| 1 | CIGS (VSe) | L1-defect1 | EV + 0.3 eV | 1×10¹⁴ cm⁻³ | Shallow acceptor; limits hole collection. |
| 2 | CIGS (VCu) | L1-defect2 | EC – 0.6 eV | 5×10¹³ cm⁻³ | Dominant VOC loss via SRH recombination. |
| 3 | CdS/CIGS Interface | I1 | Mid-gap (0.8 eV) | 1×10¹² cm⁻² eV⁻¹ | Carrier trapping; reduces JSC and FF. |
| 4 | CdS Bulk (Donor) | L2 | EC – 0.3 eV | 5×10¹⁵ cm⁻³ | Reduces minority carrier lifetime in buffer layer. |
| 5 | ZnO/CdS Interface | I2 | Mid-gap (0.9 eV) | 1×10¹² cm⁻² eV⁻¹ | Increases interface recombination; reduces FF. |
| 6 | ZnO Bulk (Donor) | L3 | EC – 0.2 eV | 1×10¹⁶ cm⁻³ | Enhances conductivity but may cause tunneling losses. |
________________________________________
3. Single Simulation - Device Performance Characteristics
A single simulated ZnO/CdS/CIGS solar cell, using the parameters in Table 1 and defects in Table 2, showed suboptimal performance: Power Conversion Efficiency (PCE) of 3.43%, Open-Circuit Voltage (VOC) of 0.675 V, Short-Circuit Current Density (JSC) of 22.62 mA/cm², and Fill Factor (FF) of 22.46%. Analysis of the J-V and EQE curves (Figure 1, Table 3) revealed three primary loss mechanisms contributing to these limitations. These mechanisms, along with their correlative evidence, potential mitigation strategies, and comparison with existing literature, are detailed in Table 3.
Table 3: Performance Limitations and Potential Origins.
Parameter	Simulated Value	Probable Loss Mechanism	Correlative Evidence	Potential Mitigation
JSC
22.62 mA/cm²	Bulk recombination in CIGS (Nt=1×10¹⁴ cm⁻³ (VSe), 5×10¹³ cm⁻³ (VCu)).
EQE plateau at ~34% (500–760 nm), indicating insufficient minority carrier diffusion length (Ln<0.5μm). This aligns with reduced current collection due to increased bulk recombination rates.	Reduce CIGS bulk defect density (Nt<10¹³ cm⁻³). To achieve high efficiency cells (>20% PCE).
VOC
0.675 V	Interface recombination at ZnO/CdS interface traps (Dit=1×10¹² cm⁻² eV⁻¹, Et=EC−0.9 eV, asymmetric capture cross-sections σₙ=10⁻¹⁵ cm², σₚ=10⁻¹⁶ cm²).	While VOC is close to typical values, interface traps can still contribute to recombination. High Dit with asymmetric capture cross-sections are known to impact VOC and overall device characteristics.	Implement passivating buffer layers (e.g., Zn(O,S)) to reduce interface trap density (Dit<10¹¹ cm⁻² eV⁻¹).
FF	22.46%	High series resistance (Rs≈5Ω·cm²) from: (i) Non-optimized Mo/CIGS back contact, (ii) Limited ZnO window layer conductivity (ND=10¹⁸ cm⁻³).	J-V curve "kink" at V>0.6 V, indicating significant resistive losses. This is consistent with non-ideal back contacts and potentially insufficient ZnO conductivity.	Integrate MoSe₂ interlayer at Mo/CIGS interface and optimize ZnO doping and conductivity.
________________________________________
4. SCAPS-1D Generated Defect Datasets for Data Visualization and AI-Driven Photovoltaics Research
In solar cell research, as in other experimental disciplines, complex multidimensional datasets are routinely generated in both research laboratories and industrial R&D facilities. While conventional data visualization is common, the inherent complexity of these datasets often necessitates more advanced analytical techniques for proper professional visualization and interpretation, which are less frequently employed. However, the critical bottleneck remains the general lack of properly categorized data – a fundamental requirement for implementing sophisticated analysis methods. Indeed, accessing suitable data that enables effective data visualization, advanced analytical techniques, and robust machine learning/AI applications is often difficult, if not practically impossible. Without such systematically classified datasets, which can only be produced through professional-grade data interpretation and analysis pipelines, the application of cutting-edge techniques like machine learning and artificial intelligence for novel solar cell characterizations becomes severely constrained. This limitation not only impedes the validation of emerging analytical approaches but also can restrict the development of next-generation photovoltaic diagnostic methodologies.
To address these limitations, a systematic solution has been developed through the implementation of advanced batch simulations in SCAPS-1D. High-quality defect datasets have been generated by combining physically realistic numerical modeling with automated Python scripting protocols. Within this framework, the defect tolerance landscape of CIGS solar cells has been comprehensively mapped through controlled variation of six key defect parameters across technologically relevant ranges (Table 2), encompassing both bulk and interface defects. This methodology has yielded a rigorously structured 6-dimensional parameter space containing 4,096 unique configurations (46, Table 4), designed specifically to enable both advanced data interpretation and machine learning applications while maintaining physical validity.
Table 4: Defect Parameters for Simulation.
Defect Name	Defect Type	Location	Range (cm⁻³ or cm⁻²)	Steps	Potential Effect on Efficiency
L1-defect1	VSe (Donor)	CIGS Bulk	10¹² to 10¹⁵	×10 (4 steps)	Large decrease
L1-defect2	VCu (Acceptor)	CIGS Bulk	10¹² to 10¹⁵	×10 (4 steps)	Large decrease
L2	VS (Donor)
CdS Bulk	10¹⁰ to 10¹³	×10 (4 steps)	Medium decrease
L3	VO (Donor)
ZnO Bulk	10¹⁴ to 10¹⁷	×10 (4 steps)	Small decrease
I1	Interface States	CdS/CIGS	10¹⁰ to 10¹³	×10 (4 steps)	Large decrease
I2	Interface States	ZnO/CdS	10¹⁰ to 10¹³	×10 (4 steps)	Medium decrease
________________________________________
5. Dataset Description and Usage
The accompanying Excel file contains the processed dataset used in this study. The raw output of 4,096 simulated J-V curves was systematically processed through a custom Python pipeline to generate a comprehensive 16-column dataset. This included ten columns extracted from J-V curve characteristics (six standard parameters: VOC, JSC, FF, η, VMPP, JMPP, and four computationally derived metrics: Dynamic Resistance, Series Resistance, Shunt Resistance, and Maximum Power) and six additional columns recording the predefined defect densities parsed from the simulation output.
A crucial post-processing step was undertaken to optimize the dataset for machine learning. The initial 4,000-row dataset, while comprehensive, contained significant redundancy, with many rows exhibiting identical or near-identical defect parameters. To prevent overfitting and ensure robust model generalization, a custom Python script was employed to refine this dataset, retaining only rows with unique combinations of all six defect parameters. 
This refined, high-quality dataset is intended to serve dual purposes: as a potential resource for defect identification research and a provisional benchmark for methodological developments in emerging data visualization techniques and machine learning applications. By establishing a connection between theoretical simulations and practical data science implementations, this framework may provide researchers with a platform for evaluating new algorithms, while potentially contributing to the understanding of defect-property relationships in photovoltaic materials.
The dataset has been made publicly available here to promote transparency and collaborative research. This accessibility could benefit the broader research community by facilitating cross-study comparisons and supporting progress in solar cell characterization through reproducible, data-driven approaches.
________________________________________
6. Further Details
For a more in-depth explanation of the methodology, simulation parameters, and the complete analysis, please refer to our full scientific article, which is currently under review by relevant journals.
________________________________________

