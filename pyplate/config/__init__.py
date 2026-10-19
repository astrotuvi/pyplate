"""Configuration: TOML files read into frozen dataclasses (DESIGN.md section 3).

Old INI configurations are converted with ``python -m
pyplate.config.ini2toml``.
"""
from .loader import ConfigError, config_from_dict, load_config
from .sections import (ArchiveConfig, CatalogConfig, Config, ConnectionConfig,
                       CrossmatchConfig, CsvConfig, DatabaseConfig,
                       ExtractionConfig, FilesConfig, HeaderConfig,
                       ImageConfig, MetadataConfig, PipelineConfig,
                       ProgramsConfig, SolveConfig, WfpdbConfig)

__all__ = ['Config', 'ConfigError', 'load_config', 'config_from_dict',
           'ArchiveConfig', 'CatalogConfig', 'ConnectionConfig',
           'CrossmatchConfig', 'CsvConfig', 'DatabaseConfig',
           'ExtractionConfig', 'FilesConfig', 'HeaderConfig', 'ImageConfig',
           'MetadataConfig', 'PipelineConfig', 'ProgramsConfig',
           'SolveConfig', 'WfpdbConfig']
