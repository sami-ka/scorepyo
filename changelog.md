# Changelog
This changelog is based on https://keepachangelog.com/en/1.0.0/.

## [Unreleased]
## Added
### Changed
### Deprecated
### Removed
### Fixed
### Security

## [0.2.3] - 2023/07/31
## Added
- ranker based on mrmr
### Changed
### Deprecated
### Removed
### Fixed
- index of binarized features dataframe set to original dataframe
### Security

## [0.2.2] - 2023/03/21
## Added
### Changed
### Deprecated
### Removed
### Fixed
- column in risk score model summary
### Security

## [0.2.1] - 2023/03/21
### Added
- added checks in calibrator and ranker classes for future developments
### Changed
### Deprecated
### Removed
### Fixed
- signature of parent class abstract function for mypy (ranker and calibrator)
- typos in docs
### Security

## [0.2.0] - 2023/03/16
### Added
- define architecture of risk score model: binary features ranker, enumeration metric, calibration method.
- Add several rankers
- Add several calibration methods
- add dask usage for binary features metric enumeration
### Changed
- architecture of risk score model
- script naming
- design of EBMBinarizer
- update docs
### Deprecated
### Removed
- Usage of Optuna and OptunaRiskScore
### Fixed
### Security

## [0.1.1] - 2022/12/07
### Fixed
- add manifest file for wheel building
- modified setup.cfg for better package information


## [0.1.0] - 2022/12/07
### Added
- Add automatic binary featurizer based on EBM
- Add Optuna based risk score model
- Add documentation
- Add devops utils (local+github CI)
### Changed
### Deprecated
### Removed
### Fixed
### Security
