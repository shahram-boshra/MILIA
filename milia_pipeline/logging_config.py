# logging_config.py - Enhanced for Transformation System Integration

"""
This module configures the logging system for the application with full support
for dataset handler strategy pattern migration operations, and Transformation System Integration
transformation system with experimental setup support.

It sets up console and file handlers, ensuring logs are written to both
standard output and a dedicated log file within the script's directory.
It also manages third-party library log levels to prevent excessive output
and provides specialized logging support for dataset handlers, migration
operations, transformation operations, and experimental setups.

Transformation System Integration Enhancements:
- Transformation system logging configuration
- Experimental setup tracking and logging
- Transform validation and composition logging support
- Enhanced error context for transformation failures
- Performance logging for transform operations
- Structured logging for systematic experimentation workflows
"""
import logging
import sys
import functools
import hashlib
import inspect
from pathlib import Path
from typing import TextIO, Optional, Dict, Any, List
from datetime import datetime

from milia_pipeline.exceptions import (
    LoggingConfigurationError, 
    BaseProjectError, 
    HandlerError,
    HandlerOperationError,
    MigrationError,
    HandlerValidationError,
    TransformConfigurationError,
    TransformValidationError,
    TransformCompositionError,
    ExperimentalSetupError
)



class HandlerLoggerAdapter(logging.LoggerAdapter):
    """
    Custom logger adapter for dataset handlers that adds handler-specific context.
    
    This adapter automatically includes handler type, operation context, and other
    handler-specific information in all log messages, making it easier to debug
    handler operations and track handler performance.
    """
    
    def __init__(self, logger: logging.Logger, handler_type: str, extra: Optional[Dict[str, Any]] = None):
        """
        Initialize the handler logger adapter.
        
        Args:
            logger: Base logger instance
            handler_type: Type of dataset handler (e.g., "DFT", "DMC")
            extra: Additional context to include in all log messages
        """
        self.handler_type = handler_type
        base_extra = {'handler_type': handler_type}
        if extra:
            base_extra.update(extra)
        super().__init__(logger, base_extra)
    
    def process(self, msg: str, kwargs: Dict[str, Any]) -> tuple:
        """
        Process log message to include handler context.
        
        Args:
            msg: Log message
            kwargs: Logging keyword arguments
            
        Returns:
            Tuple of (processed_message, processed_kwargs)
        """
        # Add handler prefix to message
        processed_msg = f"[{self.handler_type}] {msg}"
        
        # Add molecule context if present
        if 'molecule_index' in kwargs:
            processed_msg = f"[Mol:{kwargs['molecule_index']}] {processed_msg}"
        
        return processed_msg, kwargs


class MigrationLoggerAdapter(logging.LoggerAdapter):
    """
    Custom logger adapter for migration operations that adds migration-specific context.
    
    This adapter tracks migration phases, steps, and provides structured logging
    for the Handler-Based Pattern Development migration from legacy code to handler pattern.
    """
    
    def __init__(self, logger: logging.Logger, migration_phase: str, extra: Optional[Dict[str, Any]] = None):
        """
        Initialize the migration logger adapter.
        
        Args:
            logger: Base logger instance
            migration_phase: Current migration phase identifier
            extra: Additional context to include in all log messages
        """
        self.migration_phase = migration_phase
        base_extra = {'migration_phase': migration_phase}
        if extra:
            base_extra.update(extra)
        super().__init__(logger, base_extra)
    
    def process(self, msg: str, kwargs: Dict[str, Any]) -> tuple:
        """
        Process log message to include migration context.
        
        Args:
            msg: Log message
            kwargs: Logging keyword arguments
            
        Returns:
            Tuple of (processed_message, processed_kwargs)
        """
        processed_msg = f"[Migration-{self.migration_phase}] {msg}"
        
        # Add migration step if present
        if 'migration_step' in kwargs:
            processed_msg = f"[Step:{kwargs['migration_step']}] {processed_msg}"
        
        return processed_msg, kwargs


class TransformLoggerAdapter(logging.LoggerAdapter):
    """
    Custom logger adapter for transformation operations.
    
    This adapter tracks experimental setups, transform compositions, and provides
    structured logging for systematic transformation experimentation workflows.
    """
    
    def __init__(self, logger: logging.Logger, 
                 experimental_setup: Optional[str] = None,
                 transform_context: Optional[str] = None,
                 extra: Optional[Dict[str, Any]] = None):
        """
        Initialize the transformation logger adapter.
        
        Args:
            logger: Base logger instance
            experimental_setup: Name of experimental setup being used
            transform_context: Context of transformation operation (e.g., "validation", "composition")
            extra: Additional context to include in all log messages
        """
        self.experimental_setup = experimental_setup
        self.transform_context = transform_context
        
        base_extra = {}
        if experimental_setup:
            base_extra['experimental_setup'] = experimental_setup
        if transform_context:
            base_extra['transform_context'] = transform_context
        if extra:
            base_extra.update(extra)
            
        super().__init__(logger, base_extra)
    
    def process(self, msg: str, kwargs: Dict[str, Any]) -> tuple:
        """
        Process log message to include transformation context.
        
        Args:
            msg: Log message
            kwargs: Logging keyword arguments
            
        Returns:
            Tuple of (processed_message, processed_kwargs)
        """
        prefix_parts = []
        
        if self.experimental_setup:
            prefix_parts.append(f"Setup:{self.experimental_setup}")
        
        if self.transform_context:
            prefix_parts.append(f"Context:{self.transform_context}")
        
        # Add transform name if present in kwargs
        if 'transform_name' in kwargs:
            prefix_parts.append(f"Transform:{kwargs['transform_name']}")
        
        prefix = "[" + "|".join(prefix_parts) + "] " if prefix_parts else ""
        processed_msg = f"{prefix}{msg}"
        
        return processed_msg, kwargs


def setup_logging(enable_handler_logging: bool = True,
                 enable_migration_logging: bool = True,
                 enable_transform_logging: bool = True,
                 log_level: str = "INFO") -> logging.Logger:
    """
    Configures and initializes the application's logging system with handler pattern support
    and transformation system integration.

    Sets up a root logger with both console (stdout) and file handlers.
    Logs are written to a file named after the main script, located in the
    same directory as the script. Prevents handler duplication on
    multiple calls. Also silences RDKit's logger to 'ERROR' level.

    Transformation System Integration Enhancements:
    - Configures transformation system logging
    - Sets up experimental setup tracking
    - Adds structured logging for transform validation and composition
    - Enhanced error context capture for transform failures

    Args:
        enable_handler_logging: Whether to enable enhanced handler logging
        enable_migration_logging: Whether to enable migration phase logging
        enable_transform_logging: Whether to enable transformation system logging (NEW)
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    Raises:
        LoggingConfigurationError: If there's an issue setting up
                                   file logging (e.g., permissions) or
                                   any other unexpected error during setup.

    Returns:
        logging.Logger: The configured application logger instance.
    """
    logger: logging.Logger = logging.getLogger(__name__)
    
    # Convert string log level to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(numeric_level)

    # Ensure handlers are not duplicated if called multiple times
    if not logger.handlers:
        try:
            # Get the name of the script without the .py extension
            script_path: Path = Path(inspect.getfile(inspect.currentframe()))
            log_file_name: str = script_path.with_suffix('.log').name
            log_file_path: Path = script_path.parent / log_file_name

            # Create enhanced formatter that includes more context
            if enable_handler_logging or enable_migration_logging or enable_transform_logging:
                # Enhanced format with additional context fields
                formatter: logging.Formatter = logging.Formatter(
                    '%(asctime)s - %(levelname)s - %(name)s:%(lineno)d - %(message)s'
                )
            else:
                # Standard format for backward compatibility
                formatter: logging.Formatter = logging.Formatter(
                    '%(asctime)s - %(levelname)s - %(message)s'
                )

            # Console Handler
            console_handler: logging.StreamHandler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

            # File Handler
            try:
                file_handler: logging.FileHandler = logging.FileHandler(log_file_path)
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)
                
                log_msg = f"Logging initialized. Output will be written to console and '{log_file_path}'"
                if enable_handler_logging:
                    log_msg += " (Handler logging: ENABLED)"
                if enable_migration_logging:
                    log_msg += " (Migration logging: ENABLED)"
                if enable_transform_logging:
                    log_msg += " (Transform logging: ENABLED)"
                    
                logger.info(log_msg)
                
            except OSError as e:
                raise LoggingConfigurationError(
                    message=f"Failed to set up file logging to '{log_file_path}'.",
                    details=f"OS Error: {e}"
                ) from e

        except Exception as e:
            logger.error(f"An unexpected error occurred during logging setup: {e}", exc_info=True)
            raise LoggingConfigurationError(
                message="An unexpected error prevented logging from being fully initialized.",
                details=str(e)
            ) from e

    # Configure third-party library loggers
    _configure_third_party_loggers()
    
    # Set up handler and migration specific loggers if enabled
    if enable_handler_logging:
        _setup_handler_loggers(logger)
    
    if enable_migration_logging:
        _setup_migration_loggers(logger)
    
    # Set up transformation system loggers
    if enable_transform_logging:
        _setup_transform_loggers(logger)

    return logger


def _configure_third_party_loggers() -> None:
    """
    Configure third-party library loggers to prevent excessive output.
    
    This includes RDKit and other scientific libraries that tend to be verbose.
    Enhanced to handle additional libraries used by handlers.
    """
    # Silence RDKit warnings by setting its logger level to ERROR
    rdkit_logger: logging.Logger = logging.getLogger('rdkit')
    rdkit_logger.setLevel(logging.ERROR)
    
    # Additional third-party loggers that may be verbose
    verbose_loggers = [
        'matplotlib',
        'PIL',
        'urllib3',
        'requests',
        'torch',
        'torch_geometric'
    ]
    
    for logger_name in verbose_loggers:
        third_party_logger = logging.getLogger(logger_name)
        third_party_logger.setLevel(logging.WARNING)


def _setup_handler_loggers(base_logger: logging.Logger) -> None:
    """
    Set up specialized loggers for dataset handlers.
    
    Args:
        base_logger: Base logger to derive handler loggers from
    """
    # Create handler-specific loggers
    handler_types = ['DFT', 'DMC', 'GENERIC']
    
    for handler_type in handler_types:
        handler_logger_name = f"handler.{handler_type.lower()}"
        handler_logger = logging.getLogger(handler_logger_name)
        handler_logger.setLevel(base_logger.level)
        
        # Handler loggers inherit from base logger handlers
        handler_logger.parent = base_logger
        handler_logger.propagate = True


def _setup_migration_loggers(base_logger: logging.Logger) -> None:
    """
    Set up specialized loggers for migration operations.
    
    Args:
        base_logger: Base logger to derive migration loggers from
    """
    # Create migration phase loggers
    migration_phases = ['6F', '6G', '6H', '6I', '6J']
    
    for phase in migration_phases:
        migration_logger_name = f"migration.phase_{phase}"
        migration_logger = logging.getLogger(migration_logger_name)
        migration_logger.setLevel(base_logger.level)
        
        # Migration loggers inherit from base logger handlers
        migration_logger.parent = base_logger
        migration_logger.propagate = True


def _setup_transform_loggers(base_logger: logging.Logger) -> None:
    """
    Set up specialized loggers for transformation operations.
    
    Args:
        base_logger: Base logger to derive transformation loggers from
    """
    # Create transformation operation loggers
    transform_operations = [
        'registry',      # Transform discovery and registration
        'validation',    # Parameter validation
        'composition',   # Transform sequence composition
        'experimental'   # Experimental setup management
    ]
    
    for operation in transform_operations:
        transform_logger_name = f"transform.{operation}"
        transform_logger = logging.getLogger(transform_logger_name)
        transform_logger.setLevel(base_logger.level)
        
        # Transform loggers inherit from base logger handlers
        transform_logger.parent = base_logger
        transform_logger.propagate = True


def create_handler_logger(handler_type: str, 
                         base_logger: Optional[logging.Logger] = None,
                         extra_context: Optional[Dict[str, Any]] = None) -> HandlerLoggerAdapter:
    """
    Create a specialized logger adapter for dataset handlers.
    
    This function creates a logger adapter that automatically includes
    handler-specific context in all log messages.
    
    Args:
        handler_type: Type of dataset handler (e.g., "DFT", "DMC")
        base_logger: Base logger to adapt (uses default if None)
        extra_context: Additional context to include in log messages
        
    Returns:
        HandlerLoggerAdapter: Configured handler logger adapter
        
    Example:
        >>> handler_logger = create_handler_logger("DFT", extra_context={"batch_id": "batch_001"})
        >>> handler_logger.info("Processing molecule")
        # Output: [DFT] Processing molecule
    """
    if base_logger is None:
        base_logger = logging.getLogger(f"handler.{handler_type.lower()}")
    
    return HandlerLoggerAdapter(base_logger, handler_type, extra_context)


def create_migration_logger(migration_phase: str,
                           base_logger: Optional[logging.Logger] = None,
                           extra_context: Optional[Dict[str, Any]] = None) -> MigrationLoggerAdapter:
    """
    Create a specialized logger adapter for migration operations.
    
    This function creates a logger adapter for tracking migration progress
    and debugging migration issues.
    
    Args:
        migration_phase: Migration phase identifier (e.g., "6F", "6G", "6H", "6I", "6J")
        base_logger: Base logger to adapt (uses default if None)
        extra_context: Additional context to include in log messages
        
    Returns:
        MigrationLoggerAdapter: Configured migration logger adapter
        
    Example:
        >>> migration_logger = create_migration_logger("6F", extra_context={"module": "property_enrichment.py"})
        >>> migration_logger.info("Starting property enrichment migration")
        # Output: [Migration-6F] Starting property enrichment migration
    """
    if base_logger is None:
        base_logger = logging.getLogger(f"migration.phase_{migration_phase}")
    
    return MigrationLoggerAdapter(base_logger, migration_phase, extra_context)


def create_transform_logger(experimental_setup: Optional[str] = None,
                           transform_context: Optional[str] = None,
                           base_logger: Optional[logging.Logger] = None,
                           extra_context: Optional[Dict[str, Any]] = None) -> TransformLoggerAdapter:
    """
    PHASE 1: Create a specialized logger adapter for transformation operations.
    
    This function creates a logger adapter for tracking transformation operations,
    experimental setups, and systematic experimentation workflows.
    
    Args:
        experimental_setup: Name of experimental setup being used
        transform_context: Context of transformation operation (e.g., "validation", "composition")
        base_logger: Base logger to adapt (uses default if None)
        extra_context: Additional context to include in log messages
        
    Returns:
        TransformLoggerAdapter: Configured transformation logger adapter
        
    Example:
        >>> transform_logger = create_transform_logger(
        ...     experimental_setup="baseline",
        ...     transform_context="validation",
        ...     extra_context={"config_source": "yaml"}
        ... )
        >>> transform_logger.info("Validating transform configuration")
        # Output: [Setup:baseline|Context:validation] Validating transform configuration
    """
    if base_logger is None:
        # Determine appropriate base logger based on context
        if transform_context:
            base_logger = logging.getLogger(f"transform.{transform_context}")
        else:
            base_logger = logging.getLogger("transform.experimental")
    
    return TransformLoggerAdapter(base_logger, experimental_setup, transform_context, extra_context)


def log_handler_operation(operation_name: str):
    """
    Decorator to automatically log handler operations with timing and error handling.
    
    This decorator provides consistent logging for all handler operations,
    including execution time, success/failure status, and error details.
    
    Args:
        operation_name: Name of the operation being logged
        
    Returns:
        Decorator function
        
    Example:
        @log_handler_operation("validate_molecule")
        def validate_molecule_data(self, ...):
            # method implementation
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get handler instance from first argument (self)
            handler_instance = args[0] if args else None
            handler_type = getattr(handler_instance, 'get_dataset_type', lambda: 'UNKNOWN')()
            
            # Create handler logger for this operation
            logger = create_handler_logger(handler_type)
            
            # Extract molecule index if present in kwargs
            molecule_index = kwargs.get('molecule_index')
            if molecule_index is not None:
                logger = HandlerLoggerAdapter(logger.logger, handler_type, {'molecule_index': molecule_index})
            
            start_time = datetime.now()
            logger.debug(f"Starting {operation_name}")
            
            try:
                result = func(*args, **kwargs)
                execution_time = (datetime.now() - start_time).total_seconds()
                logger.debug(f"Completed {operation_name} in {execution_time:.3f}s")
                return result
                
            except HandlerError as e:
                execution_time = (datetime.now() - start_time).total_seconds()
                logger.error(f"Handler error in {operation_name} after {execution_time:.3f}s: {e}")
                raise
                
            except Exception as e:
                execution_time = (datetime.now() - start_time).total_seconds()
                logger.error(f"Unexpected error in {operation_name} after {execution_time:.3f}s: {e}", exc_info=True)
                raise
                
        return wrapper
    return decorator


def log_migration_step(step_name: str, migration_phase: str):
    """
    Decorator to automatically log migration steps with progress tracking.
    
    This decorator provides consistent logging for migration operations,
    including step timing, success/failure status, and rollback information.
    
    Args:
        step_name: Name of the migration step
        migration_phase: Migration phase identifier (e.g., "6F", "6G")
        
    Returns:
        Decorator function
        
    Example:
        @log_migration_step("migrate_property_enrichment", "6F")
        def migrate_property_enrichment_module():
            # migration implementation
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = create_migration_logger(migration_phase, extra_context={'step': step_name})
            
            start_time = datetime.now()
            logger.info(f"Starting migration step: {step_name}")
            
            try:
                result = func(*args, **kwargs)
                execution_time = (datetime.now() - start_time).total_seconds()
                logger.info(f"Completed migration step: {step_name} in {execution_time:.3f}s")
                return result
                
            except MigrationError as e:
                execution_time = (datetime.now() - start_time).total_seconds()
                logger.error(f"Migration error in {step_name} after {execution_time:.3f}s: {e}")
                if e.rollback_available:
                    logger.info(f"Rollback available for {step_name}")
                else:
                    logger.warning(f"No rollback available for {step_name}")
                raise
                
            except Exception as e:
                execution_time = (datetime.now() - start_time).total_seconds()
                logger.error(f"Unexpected error in migration step {step_name} after {execution_time:.3f}s: {e}", exc_info=True)
                raise
                
        return wrapper
    return decorator


def log_transform_operation(operation_name: str, 
                           experimental_setup: Optional[str] = None,
                           transform_context: Optional[str] = None):
    """
    Decorator to automatically log transformation operations with timing and error handling.
    
    This decorator provides consistent logging for all transformation operations,
    including validation, composition, and experimental setup management.
    
    Args:
        operation_name: Name of the operation being logged
        experimental_setup: Optional experimental setup name
        transform_context: Optional context (e.g., "validation", "composition")
        
    Returns:
        Decorator function
        
    Example:
        @log_transform_operation("validate_parameters", transform_context="validation")
        def validate_transform_config(self, config):
            # validation implementation
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Extract experimental setup from kwargs if not provided
            setup = experimental_setup or kwargs.get('experimental_setup')
            
            # Create transform logger for this operation
            logger = create_transform_logger(
                experimental_setup=setup,
                transform_context=transform_context or operation_name
            )
            
            # Extract transform name if present in kwargs
            transform_name = kwargs.get('transform_name') or kwargs.get('name')
            if transform_name:
                logger = TransformLoggerAdapter(
                    logger.logger, 
                    setup, 
                    transform_context,
                    {'transform_name': transform_name}
                )
            
            start_time = datetime.now()
            logger.debug(f"Starting {operation_name}")
            
            try:
                result = func(*args, **kwargs)
                execution_time = (datetime.now() - start_time).total_seconds()
                logger.debug(f"Completed {operation_name} in {execution_time:.3f}s")
                return result
                
            except (TransformConfigurationError, TransformValidationError, 
                    TransformCompositionError, ExperimentalSetupError) as e:
                execution_time = (datetime.now() - start_time).total_seconds()
                logger.error(f"Transform error in {operation_name} after {execution_time:.3f}s: {e}")
                raise
                
            except Exception as e:
                execution_time = (datetime.now() - start_time).total_seconds()
                logger.error(f"Unexpected error in {operation_name} after {execution_time:.3f}s: {e}", exc_info=True)
                raise
                
        return wrapper
    return decorator


def log_handler_performance(handler_logger: logging.Logger,
                          operation: str,
                          molecule_count: int,
                          execution_time: float,
                          error_count: int = 0,
                          additional_metrics: Optional[Dict[str, Any]] = None) -> None:
    """
    Log handler performance metrics in a structured format.
    
    This function provides standardized performance logging for handler operations,
    making it easier to monitor handler efficiency and identify bottlenecks.
    
    Args:
        handler_logger: Handler logger instance
        operation: Name of the operation
        molecule_count: Number of molecules processed
        execution_time: Total execution time in seconds
        error_count: Number of errors encountered
        additional_metrics: Additional metrics to log
        
    Example:
        >>> handler_logger = create_handler_logger("DFT")
        >>> log_handler_performance(
        ...     handler_logger, "batch_validation", 100, 45.2, error_count=3,
        ...     additional_metrics={"memory_used_mb": 128.5}
        ... )
    """
    metrics = {
        'operation': operation,
        'molecule_count': molecule_count,
        'execution_time_sec': round(execution_time, 3),
        'molecules_per_sec': round(molecule_count / execution_time, 2) if execution_time > 0 else 0,
        'error_count': error_count,
        'success_rate': round((molecule_count - error_count) / molecule_count * 100, 2) if molecule_count > 0 else 0
    }
    
    if additional_metrics:
        metrics.update(additional_metrics)
    
    # Format metrics as readable string
    metrics_str = ", ".join([f"{k}={v}" for k, v in metrics.items()])
    handler_logger.info(f"Performance metrics - {metrics_str}")


def log_transform_performance(transform_logger: logging.Logger,
                             operation: str,
                             transform_count: int,
                             execution_time: float,
                             error_count: int = 0,
                             experimental_setup: Optional[str] = None,
                             additional_metrics: Optional[Dict[str, Any]] = None) -> None:
    """
    Log transformation performance metrics in a structured format.
    
    This function provides standardized performance logging for transformation operations,
    making it easier to monitor transform efficiency and experimental setup performance.
    
    Args:
        transform_logger: Transform logger instance
        operation: Name of the operation
        transform_count: Number of transforms processed
        execution_time: Total execution time in seconds
        error_count: Number of errors encountered
        experimental_setup: Name of experimental setup
        additional_metrics: Additional metrics to log
        
    Example:
        >>> transform_logger = create_transform_logger(experimental_setup="baseline")
        >>> log_transform_performance(
        ...     transform_logger, "composition", 5, 0.023, error_count=0,
        ...     experimental_setup="baseline",
        ...     additional_metrics={"cache_hits": 2}
        ... )
    """
    metrics = {
        'operation': operation,
        'transform_count': transform_count,
        'execution_time_sec': round(execution_time, 3),
        'transforms_per_sec': round(transform_count / execution_time, 2) if execution_time > 0 else 0,
        'error_count': error_count,
        'success_rate': round((transform_count - error_count) / transform_count * 100, 2) if transform_count > 0 else 0
    }
    
    if experimental_setup:
        metrics['experimental_setup'] = experimental_setup
    
    if additional_metrics:
        metrics.update(additional_metrics)
    
    # Format metrics as readable string
    metrics_str = ", ".join([f"{k}={v}" for k, v in metrics.items()])
    transform_logger.info(f"Transform performance - {metrics_str}")


def get_handler_logger_by_type(handler_type: str) -> logging.Logger:
    """
    Get the specialized logger for a specific handler type.
    
    Args:
        handler_type: Type of handler (e.g., "DFT", "DMC")
        
    Returns:
        logging.Logger: Handler-specific logger
    """
    logger_name = f"handler.{handler_type.lower()}"
    return logging.getLogger(logger_name)


def get_migration_logger_by_phase(migration_phase: str) -> logging.Logger:
    """
    Get the specialized logger for a specific migration phase.
    
    Args:
        migration_phase: Migration phase identifier (e.g., "6F", "6G")
        
    Returns:
        logging.Logger: Migration phase-specific logger
    """
    logger_name = f"migration.phase_{migration_phase}"
    return logging.getLogger(logger_name)


def get_transform_logger_by_operation(operation: str) -> logging.Logger:
    """
    Get the specialized logger for a specific transformation operation.
    
    Args:
        operation: Transformation operation (e.g., "validation", "composition", "registry")
        
    Returns:
        logging.Logger: Transform operation-specific logger
    """
    logger_name = f"transform.{operation}"
    return logging.getLogger(logger_name)


def log_exception_with_context(logger: logging.Logger,
                              exception: BaseException,
                              operation: str,
                              context: Optional[Dict[str, Any]] = None) -> None:
    """
    Log exceptions with enhanced context information.
    
    This function provides structured exception logging that includes
    handler-specific context, molecule information, and recovery suggestions.
    
    Args:
        logger: Logger instance to use
        exception: Exception to log
        operation: Operation that caused the exception
        context: Additional context information
        
    Example:
        >>> try:
        ...     handler.validate_molecule_data(...)
        ... except HandlerValidationError as e:
        ...     log_exception_with_context(
        ...         handler_logger, e, "molecule_validation",
        ...         context={"molecule_index": 42, "handler_type": "DFT"}
        ...     )
    """
    # Build context string
    context_parts = [f"operation={operation}"]
    
    if context:
        context_parts.extend([f"{k}={v}" for k, v in context.items()])
    
    # Add exception-specific information
    if isinstance(exception, HandlerError):
        if hasattr(exception, 'handler_type') and exception.handler_type:
            context_parts.append(f"handler_type={exception.handler_type}")
        if hasattr(exception, 'handler_operation') and exception.handler_operation:
            context_parts.append(f"handler_operation={exception.handler_operation}")
    
    # Add transform-specific information
    if isinstance(exception, (TransformConfigurationError, TransformValidationError, 
                             TransformCompositionError, ExperimentalSetupError
                              )):
        if hasattr(exception, 'experimental_setup') and exception.experimental_setup:
            context_parts.append(f"experimental_setup={exception.experimental_setup}")
        if hasattr(exception, 'config_source') and exception.config_source:
            context_parts.append(f"config_source={exception.config_source}")
        if hasattr(exception, 'transform_name') and exception.transform_name:
            context_parts.append(f"transform_name={exception.transform_name}")
    
    if hasattr(exception, 'molecule_index') and exception.molecule_index is not None:
        context_parts.append(f"molecule_index={exception.molecule_index}")
    
    context_str = ", ".join(context_parts)
    
    # Log the exception with full context
    logger.error(f"Exception in {operation}: {exception} (Context: {context_str})", exc_info=True)
    
    # Log recovery suggestions if available
    from milia_pipeline.exceptions import get_exception_recovery_suggestions
    suggestions = get_exception_recovery_suggestions(exception)
    if suggestions:
        logger.info(f"Recovery suggestions for {operation}: {suggestions}")


def log_experimental_setup_switch(logger: logging.Logger,
                                  from_setup: Optional[str],
                                  to_setup: str,
                                  success: bool,
                                  transform_count: Optional[int] = None,
                                  error: Optional[Exception] = None) -> None:
    """
    Log experimental setup switching operations.
    
    This function provides structured logging for experimental setup switches,
    tracking the transition between different transformation configurations.
    
    Args:
        logger: Logger instance to use
        from_setup: Previous experimental setup (None if initial setup)
        to_setup: New experimental setup
        success: Whether the switch was successful
        transform_count: Number of transforms in new setup
        error: Exception if switch failed
        
    Example:
        >>> transform_logger = create_transform_logger()
        >>> log_experimental_setup_switch(
        ...     transform_logger, "baseline", "augmented", True, transform_count=5
        ... )
    """
    if success:
        msg = f"Experimental setup switch successful: {from_setup or 'None'} → {to_setup}"
        if transform_count is not None:
            msg += f" ({transform_count} transforms)"
        logger.info(msg)
    else:
        msg = f"Experimental setup switch failed: {from_setup or 'None'} → {to_setup}"
        if error:
            msg += f" - {type(error).__name__}: {error}"
        logger.error(msg)


def log_transform_validation_results(logger: logging.Logger,
                                    transform_name: str,
                                    validation_results: Dict[str, Any],
                                    experimental_setup: Optional[str] = None) -> None:
    """
    Log transform validation results in structured format.
    
    Args:
        logger: Logger instance to use
        transform_name: Name of transform being validated
        validation_results: Dictionary containing validation results
        experimental_setup: Optional experimental setup name
        
    Example:
        >>> transform_logger = create_transform_logger(experimental_setup="baseline")
        >>> validation_results = {
        ...     'valid': True,
        ...     'warnings': ['Parameter X may cause performance issues'],
        ...     'errors': []
        ... }
        >>> log_transform_validation_results(
        ...     transform_logger, "RandomRotate", validation_results, "baseline"
        ... )
    """
    setup_prefix = f"[Setup:{experimental_setup}] " if experimental_setup else ""
    
    if validation_results.get('valid', False):
        logger.info(f"{setup_prefix}Transform '{transform_name}' validation passed")
        
        warnings = validation_results.get('warnings', [])
        if warnings:
            for warning in warnings:
                logger.warning(f"{setup_prefix}Transform '{transform_name}' validation warning: {warning}")
    else:
        logger.error(f"{setup_prefix}Transform '{transform_name}' validation failed")
        
        errors = validation_results.get('errors', [])
        for error in errors:
            logger.error(f"{setup_prefix}Transform '{transform_name}' validation error: {error}")


def log_transform_composition_summary(logger: logging.Logger,
                                     composition_results: Dict[str, Any],
                                     experimental_setup: Optional[str] = None) -> None:
    """
    Log transform composition summary with sequence analysis.
    
    Args:
        logger: Logger instance to use
        composition_results: Dictionary containing composition results
        experimental_setup: Optional experimental setup name
        
    Example:
        >>> transform_logger = create_transform_logger(experimental_setup="baseline")
        >>> composition_results = {
        ...     'transform_count': 5,
        ...     'composition_time': 0.023,
        ...     'cache_hit': True,
        ...     'warnings': ['ToUndirected appears multiple times'],
        ...     'transform_sequence': ['AddSelfLoops', 'ToUndirected', 'GCNNorm']
        ... }
        >>> log_transform_composition_summary(
        ...     transform_logger, composition_results, "baseline"
        ... )
    """
    setup_prefix = f"[Setup:{experimental_setup}] " if experimental_setup else ""
    
    transform_count = composition_results.get('transform_count', 0)
    composition_time = composition_results.get('composition_time', 0)
    cache_hit = composition_results.get('cache_hit', False)
    
    logger.info(f"{setup_prefix}Transform composition completed: {transform_count} transforms in {composition_time:.3f}s (cache: {'HIT' if cache_hit else 'MISS'})")
    
    # Log transform sequence
    transform_sequence = composition_results.get('transform_sequence', [])
    if transform_sequence:
        sequence_str = " → ".join(transform_sequence)
        logger.debug(f"{setup_prefix}Transform sequence: {sequence_str}")
    
    # Log warnings
    warnings = composition_results.get('warnings', [])
    for warning in warnings:
        logger.warning(f"{setup_prefix}Composition warning: {warning}")
    
    # Log performance insights
    if transform_count > 0 and composition_time > 0:
        transforms_per_sec = transform_count / composition_time
        logger.debug(f"{setup_prefix}Composition performance: {transforms_per_sec:.2f} transforms/sec")


def configure_debug_logging_for_handlers() -> None:
    """
    Configure debug-level logging specifically for handler development and debugging.
    
    This function enables detailed debug logging for all handler operations,
    useful during development and troubleshooting of handler pattern migration.
    """
    # Set handler loggers to debug level
    handler_logger_names = [
        'handler.dft',
        'handler.dmc', 
        'handler.generic'
    ]
    
    for logger_name in handler_logger_names:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.DEBUG)
    
    # Set migration loggers to debug level
    migration_logger_names = [
        'migration.phase_6f',
        'migration.phase_6g',
        'migration.phase_6h',
        'migration.phase_6i',
        'migration.phase_6j'
    ]
    
    for logger_name in migration_logger_names:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.DEBUG)
    
    root_logger = logging.getLogger()
    root_logger.info("Debug logging enabled for all handlers and migration operations")


def configure_debug_logging_for_transforms() -> None:
    """
    Configure debug-level logging specifically for transformation system development.
    
    This function enables detailed debug logging for all transformation operations,
    useful during development and troubleshooting of the transformation system.
    """
    # Set transform loggers to debug level
    transform_logger_names = [
        'transform.registry',
        'transform.validation',
        'transform.composition',
        'transform.experimental'
    ]
    
    for logger_name in transform_logger_names:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.DEBUG)
    
    root_logger = logging.getLogger()
    root_logger.info("Debug logging enabled for all transformation operations")


def disable_verbose_third_party_logging() -> None:
    """
    Disable verbose logging from third-party libraries.
    
    This function sets restrictive logging levels for libraries that
    tend to produce excessive output, keeping logs focused on application logic.
    """
    # Libraries to silence
    libraries_to_silence = [
        ('rdkit', logging.ERROR),
        ('matplotlib', logging.WARNING),
        ('PIL', logging.WARNING),
        ('urllib3', logging.WARNING),
        ('requests', logging.WARNING),
        ('torch', logging.WARNING),
        ('torch_geometric', logging.WARNING),
        ('numpy', logging.WARNING),
        ('scipy', logging.WARNING)
    ]
    
    for library_name, log_level in libraries_to_silence:
        library_logger = logging.getLogger(library_name)
        library_logger.setLevel(log_level)
    
    root_logger = logging.getLogger()
    root_logger.info("Verbose third-party logging has been disabled")


def create_experimental_setup_hash(setup_config: List[Dict[str, Any]]) -> str:
    """
    Create a deterministic hash for an experimental setup configuration.
    
    This is useful for caching and tracking experimental setup variations.
    
    Args:
        setup_config: List of transform specifications
        
    Returns:
        str: Hexadecimal hash of the configuration
        
    Example:
        >>> setup_config = [
        ...     {'name': 'AddSelfLoops'},
        ...     {'name': 'RandomRotate', 'kwargs': {'degrees': 180}}
        ... ]
        >>> hash_val = create_experimental_setup_hash(setup_config)
    """
    # Create deterministic string representation
    config_str = str(sorted([
        (spec.get('name', ''), tuple(sorted(spec.get('kwargs', {}).items())))
        for spec in setup_config
    ]))
    
    # Generate hash
    return hashlib.md5(config_str.encode()).hexdigest()[:12]


def log_experimental_setup_summary(logger: logging.Logger,
                                  setup_name: str,
                                  setup_config: List[Dict[str, Any]],
                                  validation_results: Optional[Dict[str, Any]] = None) -> None:
    """
    Log comprehensive summary of an experimental setup configuration.
    
    Args:
        logger: Logger instance to use
        setup_name: Name of the experimental setup
        setup_config: List of transform specifications
        validation_results: Optional validation results
        
    Example:
        >>> transform_logger = create_transform_logger(experimental_setup="baseline")
        >>> setup_config = [
        ...     {'name': 'AddSelfLoops'},
        ...     {'name': 'ToUndirected'}
        ... ]
        >>> log_experimental_setup_summary(
        ...     transform_logger, "baseline", setup_config
        ... )
    """
    logger.info(f"=== Experimental Setup Summary: '{setup_name}' ===")
    logger.info(f"Transform count: {len(setup_config)}")
    
    # Log transform sequence
    transform_names = [spec.get('name', 'unknown') for spec in setup_config]
    logger.info(f"Transform sequence: {' → '.join(transform_names)}")
    
    # Log individual transforms with parameters
    for i, spec in enumerate(setup_config, 1):
        transform_name = spec.get('name', 'unknown')
        kwargs = spec.get('kwargs', {})
        
        if kwargs:
            kwargs_str = ", ".join([f"{k}={v}" for k, v in kwargs.items()])
            logger.debug(f"  {i}. {transform_name}({kwargs_str})")
        else:
            logger.debug(f"  {i}. {transform_name}()")
    
    # Log configuration hash for tracking
    config_hash = create_experimental_setup_hash(setup_config)
    logger.debug(f"Configuration hash: {config_hash}")
    
    # Log validation results if provided
    if validation_results:
        if validation_results.get('valid', False):
            logger.info("Setup validation: PASSED")
        else:
            logger.warning("Setup validation: FAILED")
            
        warnings = validation_results.get('warnings', [])
        errors = validation_results.get('errors', [])
        
        if warnings:
            logger.info(f"Validation warnings: {len(warnings)}")
            for warning in warnings[:3]:  # Show first 3 warnings
                logger.warning(f"  - {warning}")
                
        if errors:
            logger.error(f"Validation errors: {len(errors)}")
            for error in errors[:3]:  # Show first 3 errors
                logger.error(f"  - {error}")


# Backward compatibility aliases
def setup_basic_logging() -> logging.Logger:
    """
    Backward compatibility function that sets up basic logging without handler enhancements.
    
    Returns:
        logging.Logger: Basic configured logger
    """
    return setup_logging(
        enable_handler_logging=False, 
        enable_migration_logging=False,
        enable_transform_logging=False
    )


# Enhanced logging_config.py now includes:
# ========================================

# 1. **TransformLoggerAdapter**: Custom logger adapter for transformation operations
#    - Tracks experimental setups
#    - Includes transform context (validation, composition, registry)
#    - Provides structured logging for systematic experimentation
#
# 2. **Transform Logger Creation Functions**:
#    - create_transform_logger(): Create transform-specific logger adapters
#    - get_transform_logger_by_operation(): Get logger for specific operations
#
# 3. **Transform Operation Decorators**:
#    - @log_transform_operation: Automatic logging for transform operations
#    - Tracks execution time, errors, and recovery
#
# 4. **Transform-Specific Logging Functions**:
#    - log_transform_performance(): Performance metrics for transform operations
#    - log_experimental_setup_switch(): Track setup switching
#    - log_transform_validation_results(): Structured validation logging
#    - log_transform_composition_summary(): Composition analysis logging
#    - log_experimental_setup_summary(): Comprehensive setup summaries
#
# 5. **Enhanced Exception Logging**:
#    - Extended log_exception_with_context() to handle transform exceptions
#    - Includes experimental setup, config source, and transform name context
#
# 6. **Configuration and Setup**:
#    - _setup_transform_loggers(): Initialize transform operation loggers
#    - configure_debug_logging_for_transforms(): Enable debug mode for transforms
#    - Updated setup_logging() with enable_transform_logging parameter
#
# 7. **Utility Functions**:
#    - create_experimental_setup_hash(): Generate deterministic hashes for setups
#    - Support for tracking and caching experimental configurations
#
# These enhancements provide comprehensive logging support for:
# - Transform discovery and registration (TransformRegistry)
# - Parameter validation (TransformValidator)
# - Transform composition (TransformComposer)
# - Experimental setup management and switching
# - Systematic experimentation workflows
# - Performance monitoring and optimization
#
# The logging system now fully supports the transformation system
# while maintaining backward compatibility and integration with existing
# handler and migration logging infrastructure.
