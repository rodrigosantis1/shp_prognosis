# SHP Prognosis

This repository contains the code and methodology used in research on prognosis methods for predicting the Remaining Useful Life (RUL) of rolling bearing systems in Small Hydroelectric Plants (SHPs). This research leverages a data-driven framework, combining the Tsfresh feature extraction library and machine learning survival models, as presented in the following paper:

> **de Santis, R. B., Gontijo, T. S., & Costa, M. A. (2022).**  
> *A data-driven framework for small hydroelectric plant prognosis using Tsfresh and machine learning survival models.*  
> Sensors, 23(1), 12.

## Description

The data-driven framework in this repository was designed to predict the Remaining Useful Life of SHP rolling bearing systems. It utilizes Tsfresh for automated feature extraction from time-series data, followed by survival analysis using machine learning models. This approach enables accurate predictions of fault times, helping in the preventive maintenance and effective operation of SHPs.

### Key Features

- **Automated Feature Extraction**: Uses the Tsfresh library to generate relevant features from time-series data without manual selection.
- **Survival Models**: Implements machine learning survival models for prognostic analysis, allowing estimation of RUL.
- **Application to SHPs**: Tailored to the specific needs of SHPs, focusing on the prognosis of rolling bearing systems.

## Citation

If you use this code in your research, please cite the following reference:

```bibtex
@article{de2022data,
  title={A data-driven framework for small hydroelectric plant prognosis using Tsfresh and machine learning survival models},
  author={de Santis, Rodrigo Barbosa and Gontijo, Tiago Silveira and Costa, Marcelo Azevedo},
  journal={Sensors},
  volume={23},
  number={1},
  pages={12},
  year={2022},
  publisher={MDPI}
}
```
