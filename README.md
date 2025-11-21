# Software-Defined SAR Imaging Pipeline for Surveillance Drones

---
## Abstract

A modular, Python-based SAR (Synthetic Aperture Radar) imaging pipeline developed for cost-effective surveillance drones, blending precise signal processing, simulation, image enhancement, and quality assessment. Designed to address real-world military and disaster response needs, and recognized as a top project at the Savitribai Phule Pune University Aavishkar Research Competition 2025-26.

---

## Table of Contents

| Section                  | Description                                                                 |
|--------------------------|-----------------------------------------------------------------------------|
| Motivation               | Project inspiration and real-world problem statement                        |
| Achievements             | Competition standings and recognition                                       |
| Pipeline Overview        | Stepwise summary of the SAR imaging process                                 |
| File Structure           | Key directories, notebooks, and modules                                     |
| Quick Start              | Getting set up, running demos and notebooks                                 |
| Data & Results           | Types of test data and sample outputs                                       |
| Quality Metrics          | SAR image quality evaluation parameters                                     |
| Roadmap & Future Work    | Planned improvements and open challenges                                    |
| Collaboration & Contact  | Contribution guidelines, team details                                      |

---

## Motivation

**Real-World Need:**  
Recent interactions with Indian Army officials and defense researchers (Defense Expo 2025) highlighted the urgent requirement for:
- All-weather, day-night surveillance capability on lightweight aerial platforms.
- Quantitative and actionable image quality for tactical decision-making.
- Modular software suitable for rapid customization and deployment on embedded platforms.

This project was initiated in response to their feedback, leveraging knowledge gained at Army Institute of Technology and developed with a focus on scalability and adaptability for both synthetic data and actual SAR mission datasets.

---

## Achievements

| Competition                   | Level     | Achievement                                         |
|-------------------------------|-----------|-----------------------------------------------------|
| SPPU Aavishkar 2025-26        | Zonal     | Cleared Zonal Round                                 |
| SPPU Aavishkar 2025-26        | University| Shortlisted in Top Ten, Selected for Workshop        |
| Defense Expo 2025             | National  | Demonstrated to Army, Received Technical Feedback    |

**Summary:**  
Shortlisted among top ten university research teams. Recognized for technical innovation, practical impact, and presented to Indian defense organizations.

---

## Pipeline Overview

| Step                | Function                                                                                  |
|---------------------|------------------------------------------------------------------------------------------|
| Chirp Generation    | Create LFM radar pulses; tune bandwidth for resolution and duration for SNR.              |
| Data Simulation     | Model drone flight over terrain; embed synthetic targets and noise for algorithm testing. |
| Range Compression   | Use FFT-based matched filtering for optimal range focusing.                               |
| Azimuth Compression | Exploit Doppler signatures for synthetic aperture and high azimuth resolution.            |
| RCMC                | Correct for target migration (wide swath scenarios).                                      |
| Despeckling         | Suppress SAR-specific speckle noise via adaptive Lee filtering.                          |
| Enhancement         | Log scaling, contrast stretching, histogram EQ for interpretable outputs.                 |
| Feature Detection   | Run edge/point/region detectors for mapping & change analysis.                           |
| Metrics Assessment  | Calculate PSLR, ISLR, ENL, SNR and visualize outputs at each stage.                      |

---

## File Structure

| Directory / File            | Purpose                                                                  |
|-----------------------------|--------------------------------------------------------------------------|
| `README.md`                 | Project documentation                                                    |
| `requirements.txt`          | Dependency listing for Python modules                                    |
| `data/synthetic`            | Simulated I/Q radar scenes                                               |
| `data/satellite`            | Satellite SAR test images                                                |
| `notebooks/`                | Jupyter-based algorithm demos and tutorials                              |
| `src/`                      | Modular Python source code for all algorithms                            |
| `tests/`                    | Validation scripts/unit tests                                            |
| `results/`                  | Saved output images and metric visualizations                            |

---

## Quick Start
---
- git clone https://github.com/yourusername/sar-drone-imaging.git
- cd sar-drone-imaging
- pip install -r requirements.txt
- jupyter notebook notebooks/demo_end_to_end.ipynb
---


---

## Data & Results

- **Test Data:** Synthetic and open-source satellite scenes (protected for defense/academic use).
- **Results:** Range-compressed images, focused SAR outputs, despeckled/enhanced visualizations.
- **Metrics:** PSLR, ISLR, ENL plots and overlays for quantifiable validation.

---

## Quality Metrics

| Metric   | Description                                      | Typical Purpose        |
|----------|--------------------------------------------------|-----------------------|
| PSLR     | Main peak vs highest sidelobe                    | Sharpness/Clarity     |
| ISLR     | Main lobe vs total sidelobe energy               | Clutter assessment    |
| ENL      | (Mean^2)/Variance in uniform regions             | Speckle reduction     |
| SNR      | Signal to noise measurement                      | Detection assurance   |

---

## Roadmap & Future Work

| Phase              | Goal                                                        |
|--------------------|------------------------------------------------------------|
| Data Integration   | Testing with real drone SAR sensor datasets                |
| Hardware Porting   | Optimizing modules for embedded computer platforms         |
| Advanced Analytics | Deep learning for speckle removal and automated detection  |
| Real Deployment    | Pilot testing in field with military co-development        |

---

## Collaboration & Contact

- This work is open-source for academic and non-commercial defense research.
- Feedback, code contributions, and scientific partnerships welcome.
- For access to test data, collaboration proposals, or technical questions, contact:
    -Abhijit Rai
    - Email: abhijit.airosys@gmail.com


---

## Acknowledgments

- Indian Army officials & AIT mentors for technical direction.
- SPPU Internal Quality Assurance Cell, Aavishkar Research Competition organizers.
- Open research community for algorithm references and data sources.

---

**This project represents the intersection of national defense needs, rapid prototyping, and scientific rigor. We aim to advance open, robust SAR imaging for the next generation of aerial surveillance platforms.**

