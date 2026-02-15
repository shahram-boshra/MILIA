"""
HPO Analysis Module

Provides analysis utilities for HPO studies including parameter importance
analysis, convergence tracking, visualization data preparation, statistical
analysis, and export functionality.

This module exports the StudyAnalyzer class and related configuration
classes for comprehensive HPO study analysis.

Exports:
    StudyAnalyzer: Main analyzer class for HPO studies
    AnalysisConfig: Configuration for analysis settings
    ImportanceMethod: Enum for parameter importance calculation methods
    ExportFormat: Enum for supported export formats

Usage:
    >>> from milia_pipeline.models.hpo.analysis import StudyAnalyzer
    >>>
    >>> # Create analyzer from HPOManager
    >>> analyzer = StudyAnalyzer.from_manager(hpo_manager)
    >>>
    >>> # Get parameter importance
    >>> importance = analyzer.get_parameter_importance()
    >>>
    >>> # Export results
    >>> analyzer.export_results("results.json")

Author: Milia Team
Version: 1.0.0
"""

from .study_analyzer import (
    AnalysisConfig,
    ExportFormat,
    ImportanceMethod,
    StudyAnalyzer,
)

__all__ = [
    "StudyAnalyzer",
    "AnalysisConfig",
    "ImportanceMethod",
    "ExportFormat",
]
